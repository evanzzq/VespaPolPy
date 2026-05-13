"""
Download and rotate Earth data for a source-array experiment.

Source-array geometry means one receiver station and many source locations.
This script can either search a catalog by event/source location or use an
explicit event list. It writes SAC files plus rotated R/T components and a CSV
manifest that preserves the event identity for later TAPIR preparation.

Edit CONFIG below, then run for example:

    python scripts/source_array_download_rotate.py

or provide an event-list text file:

    python scripts/source_array_download_rotate.py --event-list my_events.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from obspy import Stream, UTCDateTime, read
from obspy.clients.fdsn import Client
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.io.sac import SACTrace
from obspy.signal.rotate import rotate_ne_rt
from obspy.taup import TauPyModel


CONFIG = {
    "client": "IRIS",
    "output_root": r"H:/My Drive/Research/VespaPolPy/data_download",
    "run_name": "IU_RSSD_locbox_-30_-20_-65_-55_depth_600_T_600_750",
    "data_subdir": "source_array_event_download",
    "noise_subdir": "source_array_noise_download",
    # One receiver station.
    "network": "IU",
    "station": "RSSD",
    "preferred_location": "00",
    "location_fallback": True,
    "channels": ["BH?"],
    # Catalog search mode. Used when event_lines is empty and --event-list is
    # not supplied.
    "catalog": {
        "enabled": True,
        "start_time": "2000-01-01T00:00:00",
        "end_time": "2026-01-01T00:00:00",
        "min_magnitude": 6.0,
        # Event/source location filter. mode: "box", "radius", "both", or "none".
        "filter_mode": "box",
        "locbox": [-30.0, -20.0, -65.0, -55.0],  # minlat, maxlat, minlon, maxlon
        # If radius_center is None, the station coordinates are used.
        "radius_center": None,  # [lat, lon] or None
        "minradius": 0.0,
        "maxradius": 180.0,
    },
    # Explicit event-list mode. Lines can look like:
    # 1994JUL13 11:45:23-7.532 127.77
    # Depth/magnitude are resolved from the catalog when possible.
    "event_list_file": None,
    "event_lines": [],
    "event_list_time_tolerance_sec": 600,
    "event_list_location_tolerance_deg": 2.0,
    "default_depth_km": 0.0,
    "default_magnitude": np.nan,
    # Optional event depth filter in km, applied after catalog/event-list
    # resolution. Use None for either bound to leave it open.
    "min_depth_km": 300,
    "max_depth_km": 700,
    # Waveform window relative to origin time.
    "tstart": 0.0,
    "tend": 1800.0,
    # If true, print phase arrivals before each event download and ask for the
    # download window relative to origin time.
    "prompt_time_window": True,
    # Source-array runs usually use one window for all events. When true, the
    # script prints phase times for every event before downloading, then asks
    # once for a global window.
    "review_all_phase_times_before_download": True,
    "single_time_window_for_all_events": True,
    "download_noise": True,
    # Noise window ending before the first requested phase arrival.
    "noise_phase": "P",
    "noise_gap_before_phase": 20.0,
    # Processing.
    "response_output": "VEL",
    "downsample": True,
    "sampling_rate": 5.0,
    "ignore_existing": True,
    "rotate_after_download": True,
    "phase_list_for_report": ["P"],
}


MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True)
class EventSpec:
    time: UTCDateTime
    latitude: float
    longitude: float


@dataclass(frozen=True)
class OriginInfo:
    event_id: str
    time: UTCDateTime
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float
    source: str


def parse_event_line(line: str) -> EventSpec | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    pattern = re.compile(
        r"^(\d{4})([A-Za-z]{3})(\d{1,2})\s+"
        r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)\s*"
        r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)$"
    )
    match = pattern.match(line)
    if not match:
        raise ValueError(f"Could not parse event-list line: {line!r}")

    year, mon, day, clock, lat, lon = match.groups()
    month = MONTHS[mon.upper()]
    date_text = f"{int(year):04d}-{month:02d}-{int(day):02d}T{clock}"
    return EventSpec(UTCDateTime(date_text), float(lat), float(lon))


def parse_event_lines(lines: Iterable[str]) -> list[EventSpec]:
    events = []
    for line in lines:
        spec = parse_event_line(line)
        if spec is not None:
            events.append(spec)
    return events


def event_id(origin_time: UTCDateTime) -> str:
    return origin_time.strftime("%Y%m%d%H%M%S")


def get_single_station_inventory(client: Client, config: dict, when: UTCDateTime):
    return client.get_stations(
        network=config["network"],
        station=config["station"],
        location="*",
        channel=",".join(config["channels"]),
        starttime=when,
        endtime=when + 3600,
        level="response",
    )


def first_channel_coordinates(inventory) -> tuple[float, float]:
    for net in inventory:
        for sta in net:
            if sta.channels:
                return sta.latitude, sta.longitude
            return sta.latitude, sta.longitude
    raise ValueError("Station inventory is empty.")


def filter_catalog_by_radius(events, center_lat, center_lon, minradius, maxradius):
    kept = []
    for event in events:
        origin = event.preferred_origin() or event.origins[0]
        dist = locations2degrees(center_lat, center_lon, origin.latitude, origin.longitude)
        if (minradius is None or dist >= minradius) and (maxradius is None or dist <= maxradius):
            kept.append(event)
    return kept


def catalog_search(client: Client, config: dict, station_lat: float, station_lon: float) -> list[OriginInfo]:
    cat_cfg = config["catalog"]
    kwargs = {
        "starttime": UTCDateTime(cat_cfg["start_time"]),
        "endtime": UTCDateTime(cat_cfg["end_time"]),
        "minmagnitude": cat_cfg["min_magnitude"],
    }
    mode = cat_cfg["filter_mode"].lower()
    locbox = cat_cfg["locbox"]

    if mode in {"box", "both"}:
        kwargs.update(
            {
                "minlatitude": locbox[0],
                "maxlatitude": locbox[1],
                "minlongitude": locbox[2],
                "maxlongitude": locbox[3],
            }
        )
    if mode == "radius":
        center = cat_cfg["radius_center"] or [station_lat, station_lon]
        kwargs.update(
            {
                "latitude": center[0],
                "longitude": center[1],
                "minradius": cat_cfg["minradius"],
                "maxradius": cat_cfg["maxradius"],
            }
        )

    cat = list(client.get_events(**kwargs))
    if mode == "both":
        center = cat_cfg["radius_center"] or [station_lat, station_lon]
        cat = filter_catalog_by_radius(
            cat,
            center[0],
            center[1],
            cat_cfg["minradius"],
            cat_cfg["maxradius"],
        )

    origins = [origin_from_event(ev, "catalog") for ev in cat]
    origins.sort(key=lambda item: item.time)
    return origins


def origin_from_event(event, source: str) -> OriginInfo:
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)
    depth_km = float(origin.depth or 0.0) / 1000.0
    mag = float(magnitude.mag) if magnitude is not None and magnitude.mag is not None else np.nan
    return OriginInfo(
        event_id(origin.time),
        origin.time,
        float(origin.latitude),
        float(origin.longitude),
        depth_km,
        mag,
        source,
    )


def resolve_event_spec(client: Client, spec: EventSpec, config: dict) -> OriginInfo:
    try:
        cat = client.get_events(
            starttime=spec.time - config["event_list_time_tolerance_sec"],
            endtime=spec.time + config["event_list_time_tolerance_sec"],
            latitude=spec.latitude,
            longitude=spec.longitude,
            maxradius=config["event_list_location_tolerance_deg"],
        )
        if len(cat):
            events = sorted(
                cat,
                key=lambda ev: abs((ev.preferred_origin() or ev.origins[0]).time - spec.time),
            )
            return origin_from_event(events[0], "event_list_catalog")
    except Exception as exc:
        print(f"[WARN] Catalog resolve failed for {spec.time}: {exc}")

    return OriginInfo(
        event_id(spec.time),
        spec.time,
        spec.latitude,
        spec.longitude,
        float(config["default_depth_km"]),
        float(config["default_magnitude"]),
        "event_list_input",
    )


def choose_location_code(station, preferred: str, allow_fallback: bool) -> str:
    locs = sorted({ch.location_code or "" for ch in station.channels})
    if not locs:
        raise ValueError(f"No channels found for station {station.code}")
    if preferred in locs:
        return preferred
    if allow_fallback:
        return locs[0]
    raise ValueError(f"Preferred location {preferred!r} not available; found {locs}")


def process_stream(stream, config: dict):
    stream.merge(method=1, fill_value=0)
    stream.remove_response(output=config["response_output"])
    stream.detrend("linear")
    stream.taper(type="cosine", max_percentage=0.05)
    if config["downsample"]:
        sr = float(config["sampling_rate"])
        stream.filter("lowpass", freq=0.4 * sr, zerophase=True)
        stream.resample(sampling_rate=sr)
        stream.detrend("demean")
        stream.detrend("linear")
        stream.taper(type="cosine", max_percentage=0.05)
    return stream


def write_trace_as_sac(trace, channel, origin: OriginInfo, out_path: Path):
    dist_m, az, baz = gps2dist_azimuth(
        origin.latitude,
        origin.longitude,
        channel.latitude,
        channel.longitude,
    )
    gcarc = locations2degrees(
        origin.latitude,
        origin.longitude,
        channel.latitude,
        channel.longitude,
    )

    sac = SACTrace.from_obspy_trace(trace)
    sac.stla = float(channel.latitude)
    sac.stlo = float(channel.longitude)
    sac.stel = float(channel.elevation or 0.0)
    sac.evla = origin.latitude
    sac.evlo = origin.longitude
    sac.evdp = origin.depth_km
    if not np.isnan(origin.magnitude):
        sac.mag = origin.magnitude
    sac.dist = dist_m / 1000.0
    sac.az = az
    sac.baz = baz
    sac.gcarc = gcarc
    sac.o = 0.0
    sac.cmpaz = float(channel.azimuth or 0.0)
    sac.cmpinc = float(channel.dip or 0.0) + 90.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sac.write(str(out_path))


def download_event(
    client: Client,
    origin: OriginInfo,
    config: dict,
    data_dir: Path,
    noise_dir: Path,
    inventory=None,
):
    if inventory is None:
        try:
            inventory = get_single_station_inventory(client, config, origin.time)
        except Exception as exc:
            print(f"[WARN] Skipping {origin.event_id}: station request failed: {exc}")
            return []
    downloaded = []
    tstart = float(config["tstart"])
    tend = float(config["tend"])
    duration = tend - tstart

    for net in inventory:
        for sta in net:
            use_loc = choose_location_code(
                sta,
                config["preferred_location"],
                config["location_fallback"],
            )
            channels = [ch for ch in sta.channels if (ch.location_code or "") == use_loc]
            for channel in channels:
                net_code = net.code
                sta_code = sta.code
                loc_code = use_loc
                chan_code = channel.code
                loc_token = loc_code if loc_code else "--"
                name = f"{origin.event_id}_{net_code}.{sta_code}.{loc_token}.{chan_code}.sac"
                out_path = data_dir / name
                if config["ignore_existing"] and out_path.exists():
                    downloaded.append(out_path)
                    continue

                try:
                    st = client.get_waveforms(
                        network=net_code,
                        station=sta_code,
                        location=loc_code,
                        channel=chan_code,
                        starttime=origin.time + tstart,
                        endtime=origin.time + tend,
                        attach_response=True,
                    )
                    st = process_stream(st, config)
                    tr = st[0]
                    write_trace_as_sac(tr, channel, origin, out_path)
                    downloaded.append(out_path)
                    print(f"Saved: {out_path}")
                except Exception as exc:
                    print(f"[WARN] Failed {origin.event_id} {net_code}.{sta_code}.{chan_code}: {exc}")

                if config["download_noise"]:
                    try:
                        noise_path = noise_dir / name.replace(".sac", ".noise.sac")
                        if config["ignore_existing"] and noise_path.exists():
                            continue
                        p_time = first_phase_time(origin, channel, config["noise_phase"])
                        st_noise = client.get_waveforms(
                            network=net_code,
                            station=sta_code,
                            location=loc_code,
                            channel=chan_code,
                            starttime=origin.time + p_time - config["noise_gap_before_phase"] - duration,
                            endtime=origin.time + p_time - config["noise_gap_before_phase"],
                            attach_response=True,
                        )
                        st_noise = process_stream(st_noise, config)
                        write_trace_as_sac(st_noise[0], channel, origin, noise_path)
                        print(f"Saved: {noise_path}")
                    except Exception as exc:
                        print(
                            f"[WARN] Failed noise {origin.event_id} "
                            f"{net_code}.{sta_code}.{chan_code}: {exc}"
                        )
    return downloaded


def first_phase_time(origin: OriginInfo, channel, phase: str) -> float:
    model = TauPyModel("iasp91")
    dist_deg = locations2degrees(
        origin.latitude,
        origin.longitude,
        channel.latitude,
        channel.longitude,
    )
    arrivals = model.get_travel_times(
        source_depth_in_km=origin.depth_km,
        distance_in_degree=dist_deg,
        phase_list=[phase],
    )
    if not arrivals:
        raise ValueError(f"No {phase} arrival for {origin.event_id}")
    return float(arrivals[0].time)


def phase_report(inventory, phase_list: list[str], origin: OriginInfo):
    times = phase_times_for_event(inventory, phase_list, origin)
    for phase in phase_list:
        vals = times.get(phase, [])
        if vals:
            print(
                f"{origin.event_id} {phase} travel time: "
                f"min={min(vals):.2f}s max={max(vals):.2f}s"
            )
    return times


def phase_times_for_event(inventory, phase_list: list[str], origin: OriginInfo):
    if not phase_list:
        return {}
    model = TauPyModel("iasp91")
    times = defaultdict(list)
    for net in inventory:
        for sta in net:
            dist_deg = locations2degrees(origin.latitude, origin.longitude, sta.latitude, sta.longitude)
            arrivals = model.get_travel_times(
                source_depth_in_km=origin.depth_km,
                distance_in_degree=dist_deg,
                phase_list=phase_list,
            )
            for arrival in arrivals:
                times[arrival.name].append(float(arrival.time))
    return times


def print_phase_summary(all_phase_times: dict[str, list[float]]):
    print("\nPhase travel-time range across retained events:")
    for phase, vals in all_phase_times.items():
        if vals:
            print(f"{phase}: min={min(vals):.2f}s max={max(vals):.2f}s")
        else:
            print(f"{phase}: no arrivals found")


def source_event_distances(inventory, origin: OriginInfo) -> list[float]:
    distances = []
    for net in inventory:
        for sta in net:
            distances.append(
                locations2degrees(origin.latitude, origin.longitude, sta.latitude, sta.longitude)
            )
    return distances


def print_distance_summary(distances: list[float]):
    if not distances:
        print("Epicentral distance: no station/event distances available")
        return
    print(f"Epicentral distance: min={min(distances):.2f}deg max={max(distances):.2f}deg")


def prompt_time_window(origin: OriginInfo, default_tstart: float, default_tend: float) -> tuple[float, float]:
    print(f"\nDownload window for {origin.event_id} relative to origin time.")
    while True:
        start_text = input(f"Enter start time [default: {default_tstart}]: ").strip()
        end_text = input(f"Enter end time [default: {default_tend}]: ").strip()
        try:
            tstart = default_tstart if not start_text else float(start_text)
            tend = default_tend if not end_text else float(end_text)
        except ValueError:
            print("Please enter numeric times in seconds, or press Enter for defaults.")
            continue
        if tend <= tstart:
            print("End time must be greater than start time.")
            continue
        return tstart, tend


def prompt_global_time_window(default_tstart: float, default_tend: float) -> tuple[float, float]:
    print("\nDownload window for all events, relative to each event origin time.")
    while True:
        start_text = input(f"Enter start time [default: {default_tstart}]: ").strip()
        end_text = input(f"Enter end time [default: {default_tend}]: ").strip()
        try:
            tstart = default_tstart if not start_text else float(start_text)
            tend = default_tend if not end_text else float(end_text)
        except ValueError:
            print("Please enter numeric times in seconds, or press Enter for defaults.")
            continue
        if tend <= tstart:
            print("End time must be greater than start time.")
            continue
        return tstart, tend


def component_suffix(channel: str) -> str:
    return channel[-1].upper()


def horizontal_to_ne(tr1, tr2) -> tuple[np.ndarray, np.ndarray]:
    az1 = np.deg2rad(float(tr1.stats.sac.cmpaz))
    az2 = np.deg2rad(float(tr2.stats.sac.cmpaz))
    matrix = np.array(
        [
            [np.cos(az1), np.sin(az1)],
            [np.cos(az2), np.sin(az2)],
        ],
        dtype=float,
    )
    det = np.linalg.det(matrix)
    if abs(det) < 1e-6:
        raise ValueError(f"Horizontal components are nearly singular (det={det})")
    data = np.vstack([tr1.data, tr2.data])
    ne = np.linalg.solve(matrix, data)
    return ne[0], ne[1]


def trim_to_common_length(traces):
    npts = min(tr.stats.npts for tr in traces)
    for tr in traces:
        if tr.stats.npts != npts:
            tr.data = tr.data[:npts]
            tr.stats.npts = npts


def rotate_event_files(data_dir: Path, origin: OriginInfo, config: dict, is_noise: bool = False):
    files = sorted(data_dir.glob(f"{origin.event_id}_*.sac"))
    if is_noise:
        files = [path for path in files if path.name.endswith(".noise.sac")]
    else:
        files = [path for path in files if not path.name.endswith(".noise.sac")]
    try:
        if not files:
            raise FileNotFoundError(f"No matching SAC files in {data_dir}")
        stream = Stream()
        for path in files:
            stream += read(str(path), debug_headers=True)
    except Exception as exc:
        label = "noise" if is_noise else "event"
        print(f"[WARN] Could not read {label} files for rotation {origin.event_id}: {exc}")
        return

    groups = defaultdict(list)
    for tr in stream:
        if tr.stats.channel.upper().endswith(("R", "T")):
            continue
        if tr.stats.network != config["network"] or tr.stats.station != config["station"]:
            continue
        key = (tr.stats.network, tr.stats.station, tr.stats.location, tr.stats.channel[:2])
        groups[key].append(tr)

    if not groups:
        label = "noise " if is_noise else ""
        print(
            f"[WARN] No unrotated {label}SAC files found for configured station "
            f"{config['network']}.{config['station']} and event {origin.event_id}."
        )
        return

    for (net, sta, loc, band), traces in sorted(groups.items()):
        comps = {component_suffix(tr.stats.channel): tr for tr in traces}
        z_tr = comps.get("Z")
        n_tr = comps.get("N")
        e_tr = comps.get("E")
        h1_tr = comps.get("1")
        h2_tr = comps.get("2")

        try:
            if n_tr is not None and e_tr is not None:
                trim_to_common_length([n_tr, e_tr, z_tr] if z_tr is not None else [n_tr, e_tr])
                n_data, e_data = n_tr.data, e_tr.data
                template = z_tr or n_tr
            elif h1_tr is not None and h2_tr is not None:
                trim_to_common_length([h1_tr, h2_tr, z_tr] if z_tr is not None else [h1_tr, h2_tr])
                n_data, e_data = horizontal_to_ne(h1_tr, h2_tr)
                template = z_tr or h1_tr
            else:
                print(f"[WARN] Missing horizontal pair for {origin.event_id} {net}.{sta}.{band}?")
                continue

            baz = float(template.stats.sac.baz)
            r_data, t_data = rotate_ne_rt(n=n_data, e=e_data, ba=baz)

            loc_token = loc if loc else "--"
            suffix = ".noise" if is_noise else ""
            r_path = data_dir / f"{origin.event_id}_{net}.{sta}.{loc_token}.{band}R{suffix}.sac"
            t_path = data_dir / f"{origin.event_id}_{net}.{sta}.{loc_token}.{band}T{suffix}.sac"

            tr_r = template.copy()
            tr_r.stats.channel = f"{band}R"
            tr_r.data = r_data
            tr_t = template.copy()
            tr_t.stats.channel = f"{band}T"
            tr_t.data = t_data

            sac_r = SACTrace.from_obspy_trace(tr_r)
            sac_r.cmpaz = baz
            sac_r.cmpinc = 90.0
            sac_r.write(str(r_path))

            sac_t = SACTrace.from_obspy_trace(tr_t)
            sac_t.cmpaz = (baz + 90.0) % 360.0
            sac_t.cmpinc = 90.0
            sac_t.write(str(t_path))

            print(f"Saved: {r_path}")
            print(f"Saved: {t_path}")
        except Exception as exc:
            print(f"[WARN] Rotation failed for {origin.event_id} {net}.{sta}.{band}?: {exc}")


def write_manifest(manifest_path: Path, origins: list[OriginInfo]):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "time", "evla", "evlo", "evdp_km", "mag", "source"])
        for origin in origins:
            writer.writerow(
                [
                    origin.event_id,
                    str(origin.time),
                    origin.latitude,
                    origin.longitude,
                    origin.depth_km,
                    origin.magnitude,
                    origin.source,
                ]
            )
    print(f"Saved manifest: {manifest_path}")


def load_origins(client: Client, config: dict, event_list_file: str | None) -> list[OriginInfo]:
    if event_list_file:
        lines = Path(event_list_file).read_text().splitlines()
        specs = parse_event_lines(lines)
        return [resolve_event_spec(client, spec, config) for spec in specs]

    if config["event_lines"]:
        specs = parse_event_lines(config["event_lines"])
        return [resolve_event_spec(client, spec, config) for spec in specs]

    if config["catalog"]["enabled"]:
        station_probe_time = UTCDateTime(config["catalog"]["start_time"])
        inventory = get_single_station_inventory(client, config, station_probe_time)
        station_lat, station_lon = first_channel_coordinates(inventory)
        return catalog_search(client, config, station_lat, station_lon)

    raise ValueError("No event_lines, --event-list, or enabled catalog search configured.")


def filter_origins_by_depth(origins: list[OriginInfo], min_depth_km, max_depth_km) -> list[OriginInfo]:
    if min_depth_km is None and max_depth_km is None:
        return origins

    kept = []
    for origin in origins:
        if min_depth_km is not None and origin.depth_km < min_depth_km:
            continue
        if max_depth_km is not None and origin.depth_km > max_depth_km:
            continue
        kept.append(origin)
    print(
        f"Depth filter retained {len(kept)} of {len(origins)} event(s) "
        f"for range [{min_depth_km if min_depth_km is not None else '-inf'}, "
        f"{max_depth_km if max_depth_km is not None else 'inf'}] km."
    )
    return kept


def main(config: dict | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-list",
        help="Text file containing event lines. Overrides CONFIG['event_list_file'].",
    )
    parser.add_argument("--run-name", help="Override CONFIG['run_name'].")
    parser.add_argument("--network", help="Override CONFIG['network'].")
    parser.add_argument("--station", help="Override CONFIG['station'].")
    parser.add_argument("--location", help="Override CONFIG['preferred_location'].")
    parser.add_argument(
        "--channels",
        help='Comma-separated channel patterns, e.g. "BH?" or "BH?,HH?".',
    )
    parser.add_argument("--output-root", help="Override CONFIG['output_root'].")
    parser.add_argument("--min-depth-km", type=float, help="Minimum event depth in km.")
    parser.add_argument("--max-depth-km", type=float, help="Maximum event depth in km.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve events but do not download.")
    parser.add_argument(
        "--rotate-only",
        action="store_true",
        help="Rotate existing event/noise SAC files for the resolved events, then exit.",
    )
    parser.add_argument(
        "--download-noise",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable downloading/rotating matching noise windows.",
    )
    parser.add_argument("--no-rotate", action="store_true", help="Download only; skip R/T rotation.")
    parser.add_argument(
        "--no-prompt-time-window",
        action="store_true",
        help="Use configured tstart/tend without asking before each event.",
    )
    parser.add_argument(
        "--prompt-time-window",
        action="store_true",
        help="Print configured phase arrivals and ask for tstart/tend before each event download.",
    )
    args = parser.parse_args()

    if args.prompt_time_window and args.no_prompt_time_window:
        parser.error("Use only one of --prompt-time-window or --no-prompt-time-window.")
    if args.rotate_only and args.no_rotate:
        parser.error("Use only one of --rotate-only or --no-rotate.")

    cfg = dict(CONFIG if config is None else config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.network:
        cfg["network"] = args.network
    if args.station:
        cfg["station"] = args.station
    if args.location:
        cfg["preferred_location"] = args.location
    if args.channels:
        cfg["channels"] = [item.strip() for item in args.channels.split(",") if item.strip()]
    if args.output_root:
        cfg["output_root"] = args.output_root
    if args.download_noise is not None:
        cfg["download_noise"] = args.download_noise
    if args.min_depth_km is not None:
        cfg["min_depth_km"] = args.min_depth_km
    if args.max_depth_km is not None:
        cfg["max_depth_km"] = args.max_depth_km
    if args.no_rotate:
        cfg["rotate_after_download"] = False
    if args.no_prompt_time_window:
        cfg["prompt_time_window"] = False
    if args.prompt_time_window:
        cfg["prompt_time_window"] = True

    client = Client(cfg["client"])
    event_list_file = args.event_list or cfg.get("event_list_file")
    origins = load_origins(client, cfg, event_list_file)
    origins = filter_origins_by_depth(origins, cfg["min_depth_km"], cfg["max_depth_km"])
    if not origins:
        raise ValueError("No events found.")

    root = Path(cfg["output_root"])
    data_dir = root / cfg["data_subdir"] / cfg["run_name"]
    noise_dir = root / cfg["noise_subdir"] / cfg["run_name"]

    print(f"Resolved {len(origins)} event(s):")
    print(f"Configured station: {cfg['network']}.{cfg['station']}")
    print(f"Output data directory: {data_dir}")
    for idx, origin in enumerate(origins):
        print(
            f"[{idx:03d}] {origin.event_id} {origin.time} "
            f"lat={origin.latitude:.3f} lon={origin.longitude:.3f} "
            f"depth={origin.depth_km:.1f}km mag={origin.magnitude}"
        )

    write_manifest(data_dir / "source_array_manifest.csv", origins)
    if args.dry_run:
        return
    if args.rotate_only:
        for origin in origins:
            rotate_event_files(data_dir, origin, cfg)
            if cfg["download_noise"]:
                rotate_event_files(noise_dir, origin, cfg, is_noise=True)
        return

    inventories = {}
    retained_origins = []
    skipped_origins = []
    global_cfg = dict(cfg)
    if cfg["review_all_phase_times_before_download"]:
        all_phase_times = {phase: [] for phase in cfg["phase_list_for_report"]}
        all_distances = []
        print("\nCalculating phase travel times for all events...")
        for origin in origins:
            try:
                inventory = get_single_station_inventory(client, cfg, origin.time)
            except Exception as exc:
                print(f"[WARN] Skipping {origin.event_id}: station request failed: {exc}")
                skipped_origins.append(origin)
                continue
            inventories[origin.event_id] = inventory
            retained_origins.append(origin)
            event_phase_times = phase_times_for_event(
                inventory,
                cfg["phase_list_for_report"],
                origin,
            )
            all_distances.extend(source_event_distances(inventory, origin))
            for phase in cfg["phase_list_for_report"]:
                all_phase_times[phase].extend(event_phase_times.get(phase, []))

        print_phase_summary(all_phase_times)
        print_distance_summary(all_distances)

        if skipped_origins:
            print(f"[WARN] Skipped {len(skipped_origins)} event(s) before download.")

        if not retained_origins:
            raise ValueError("No events retained after station availability checks.")

        if cfg["prompt_time_window"] and cfg["single_time_window_for_all_events"]:
            tstart, tend = prompt_global_time_window(float(cfg["tstart"]), float(cfg["tend"]))
            global_cfg["tstart"] = tstart
            global_cfg["tend"] = tend
    else:
        retained_origins = origins

    for origin in retained_origins:
        inventory = inventories.get(origin.event_id)
        if inventory is None:
            try:
                inventory = get_single_station_inventory(client, cfg, origin.time)
            except Exception as exc:
                print(f"[WARN] Skipping {origin.event_id}: station request failed: {exc}")
                continue
            phase_report(inventory, cfg["phase_list_for_report"], origin)

        event_cfg = dict(global_cfg)
        if (
            cfg["prompt_time_window"]
            and not cfg["single_time_window_for_all_events"]
        ):
            tstart, tend = prompt_time_window(origin, float(cfg["tstart"]), float(cfg["tend"]))
            event_cfg["tstart"] = tstart
            event_cfg["tend"] = tend

        try:
            downloaded = download_event(
                client,
                origin,
                event_cfg,
                data_dir,
                noise_dir,
                inventory=inventory,
            )
        except Exception as exc:
            print(f"[WARN] Skipping {origin.event_id}: download failed: {exc}")
            continue
        if not downloaded:
            print(f"[WARN] No waveform files saved for {origin.event_id}; continuing.")
            continue
        if cfg["rotate_after_download"]:
            rotate_event_files(data_dir, origin, event_cfg)
            if event_cfg["download_noise"]:
                rotate_event_files(noise_dir, origin, event_cfg, is_noise=True)


if __name__ == "__main__":
    main()
