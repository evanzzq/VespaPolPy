from __future__ import annotations

import argparse

from .config import load_yaml_mapping
from .runner import run_config
from .utils import prepare_inputs_from_sac


def _parse_pair(value: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must be provided as 'a,b'.")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numeric values.") from exc


def _parse_int_list(value: str) -> list[int]:
    if not value.strip():
        return []
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("outliers must be a comma-separated list of integers.") from exc


def _normalize_pair(value, name: str):
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_pair(value, name)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"{name} must be given as a two-item sequence or 'a,b' string.")


def _normalize_int_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_int_list(value)
    return [int(item) for item in value]


def _resolve_prep_earth_args(args) -> dict:
    config = load_yaml_mapping(args.config) if args.config else {}
    cli_values = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "noise_dir": args.noise_dir,
        "bandpass": args.bandpass,
        "downsample_hz": args.downsample_hz,
        "time_window": args.time_window,
        "snr_component": args.snr_component,
        "snr_threshold": args.snr_threshold,
        "outliers": args.outliers,
    }
    resolved = {}
    for key, cli_value in cli_values.items():
        resolved[key] = cli_value if cli_value is not None else config.get(key)

    missing = [key for key in ("data_dir", "output_dir") if not resolved.get(key)]
    if missing:
        raise ValueError(
            "prep-earth requires the following settings either via --config or CLI flags: "
            + ", ".join(missing)
        )

    resolved["bandpass"] = _normalize_pair(resolved.get("bandpass"), "bandpass")
    resolved["time_window"] = _normalize_pair(resolved.get("time_window"), "time-window")
    resolved["outliers"] = _normalize_int_list(resolved.get("outliers"))
    resolved["snr_component"] = resolved.get("snr_component", "UZ")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tapir", description="TAPIR command-line interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one or more experiments from a YAML config")
    run_parser.add_argument(
        "--config",
        default="parameter_setup.yaml",
        help="Path to a YAML config file",
    )

    prep_parser = subparsers.add_parser(
        "prep-earth",
        help="Prepare Earth SAC files into TAPIR CSV inputs",
    )
    prep_parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML file for Earth preprocessing settings",
    )
    prep_parser.add_argument("--data-dir", default=None, help="Directory containing input SAC files")
    prep_parser.add_argument("--output-dir", default=None, help="Directory to write TAPIR-ready CSV files")
    prep_parser.add_argument(
        "--noise-dir",
        default=None,
        help="Optional directory containing matching noise SAC files",
    )
    prep_parser.add_argument(
        "--bandpass",
        type=lambda value: _parse_pair(value, "bandpass"),
        default=None,
        help="Optional bandpass as 'freqmin,freqmax'",
    )
    prep_parser.add_argument(
        "--downsample-hz",
        type=float,
        default=None,
        help="Optional target sampling rate in Hz",
    )
    prep_parser.add_argument(
        "--time-window",
        type=lambda value: _parse_pair(value, "time-window"),
        default=None,
        help="Optional trim window in seconds from trace start as 't0,t1'",
    )
    prep_parser.add_argument(
        "--snr-component",
        default="UZ",
        help="Component to threshold on when noise data are provided: UZ, UR, UT, or min",
    )
    prep_parser.add_argument(
        "--snr-threshold",
        type=float,
        default=None,
        help="Optional SNR threshold for outlier rejection when noise data are provided",
    )
    prep_parser.add_argument(
        "--outliers",
        type=_parse_int_list,
        default=None,
        help="Optional comma-separated trace indices to remove manually",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_config(args.config)
    elif args.command == "prep-earth":
        prep_args = _resolve_prep_earth_args(args)
        prepare_inputs_from_sac(
            data_dir=prep_args["data_dir"],
            isbp=prep_args["bandpass"] is not None,
            isds=prep_args["downsample_hz"],
            freqs=prep_args["bandpass"],
            noise_dir=prep_args["noise_dir"],
            output_dir=prep_args["output_dir"],
            snr_component=prep_args["snr_component"],
            snr_threshold=prep_args["snr_threshold"],
            outliers_manual=prep_args["outliers"],
            twin=prep_args["time_window"],
        )
