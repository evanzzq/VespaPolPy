from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetValidation:
    dataset_dir: Path
    n_samples: int
    n_traces: int
    components: tuple[str, ...]
    sampling_rate_hz: float
    metadata_format: str


def _load_csv(path: Path, *, skiprows: int = 0) -> np.ndarray:
    try:
        values = np.asarray(np.loadtxt(path, delimiter=",", skiprows=skiprows), dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{path.name} must contain finite numeric values.")
    return values


def validate_dataset(
    dataset_dir: str | Path,
    *,
    is3c: bool = True,
    component: str = "Z",
    cdopt: int = 0,
    source_array: bool = False,
    is_mars: bool = False,
    manual_stf: bool = False,
) -> DatasetValidation:
    """Validate a prepared TAPIR dataset without changing it."""
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    if cdopt not in {0, 3}:
        raise ValueError("cdopt must be 0 or 3.")

    component = str(component).upper()
    if component not in {"Z", "R", "T"}:
        raise ValueError("component must be Z, R, or T.")
    components = ("Z", "R", "T") if is3c else (component,)
    waveform_paths = [dataset_dir / f"U{name}.csv" for name in components]
    if not is3c and (dataset_dir / "U.csv").is_file():
        waveform_paths = [dataset_dir / "U.csv"]
        components = ("U",)

    required = [dataset_dir / "time.csv", dataset_dir / "eventinfo.csv", *waveform_paths]
    metadata_format = "distbaz" if (is_mars or source_array) else "latlon"
    metadata_path = dataset_dir / (
        "station_metadata_db.csv" if metadata_format == "distbaz" else "station_metadata.csv"
    )
    required.append(metadata_path)
    if manual_stf:
        required.append(dataset_dir / "stf.csv")
    if cdopt == 3:
        required.extend(dataset_dir / f"CD_U{name}_fit.csv" for name in components)

    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing required file(s): " + ", ".join(sorted(set(missing))))

    time = np.atleast_1d(_load_csv(dataset_dir / "time.csv"))
    if time.ndim != 1 or time.size < 2:
        raise ValueError("time.csv must be a one-dimensional vector with at least two samples.")
    intervals = np.diff(time)
    if np.any(intervals <= 0):
        raise ValueError("time.csv must be strictly increasing.")
    dt = float(np.median(intervals))
    if not np.allclose(intervals, dt, rtol=1e-6, atol=max(1e-12, abs(dt) * 1e-9)):
        raise ValueError("time.csv must be uniformly sampled.")

    waveforms = []
    for path in waveform_paths:
        values = _load_csv(path)
        if values.ndim == 1:
            values = values[:, np.newaxis]
        if values.ndim != 2 or values.shape[0] != time.size:
            raise ValueError(
                f"{path.name} must have {time.size} rows; got shape {values.shape}."
            )
        waveforms.append(values)
    shape = waveforms[0].shape
    if any(values.shape != shape for values in waveforms[1:]):
        raise ValueError("All waveform component files must have identical shapes.")

    metadata = np.atleast_2d(_load_csv(metadata_path, skiprows=1))
    if metadata.shape != (shape[1], 2):
        raise ValueError(
            f"{metadata_path.name} must have one two-value row per trace; "
            f"expected {(shape[1], 2)}, got {metadata.shape}."
        )
    if metadata_format == "latlon":
        if np.any(np.abs(metadata[:, 0]) > 90) or np.any(np.abs(metadata[:, 1]) > 360):
            raise ValueError("Latitude/longitude metadata are outside valid ranges.")
    else:
        if np.any(metadata[:, 0] < 0) or np.any((metadata[:, 1] < 0) | (metadata[:, 1] >= 360)):
            raise ValueError("Distance must be non-negative and azimuth must be in [0, 360).")

    eventinfo = _load_csv(dataset_dir / "eventinfo.csv", skiprows=1).reshape(-1)
    if eventinfo.size != 2:
        raise ValueError("eventinfo.csv must contain exactly one latitude/longitude pair.")
    if abs(eventinfo[0]) > 90 or abs(eventinfo[1]) > 360:
        raise ValueError("eventinfo.csv coordinates are outside valid ranges.")

    if manual_stf:
        stf = np.atleast_2d(_load_csv(dataset_dir / "stf.csv", skiprows=1))
        if stf.shape[1] != 2 or stf.shape[0] < 2 or np.any(np.diff(stf[:, 0]) <= 0):
            raise ValueError("stf.csv must contain increasing time and amplitude columns.")

    if cdopt == 3:
        for name in components:
            covariance = _load_csv(dataset_dir / f"CD_U{name}_fit.csv")
            if covariance.shape != (time.size, time.size):
                raise ValueError(
                    f"CD_U{name}_fit.csv must have shape {(time.size, time.size)}; "
                    f"got {covariance.shape}."
                )
            if not np.allclose(covariance, covariance.T, rtol=1e-7, atol=1e-10):
                raise ValueError(f"CD_U{name}_fit.csv must be symmetric.")

    return DatasetValidation(
        dataset_dir=dataset_dir,
        n_samples=time.size,
        n_traces=shape[1],
        components=components,
        sampling_rate_hz=1.0 / dt,
        metadata_format=metadata_format,
    )
