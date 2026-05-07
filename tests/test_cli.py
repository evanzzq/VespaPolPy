from argparse import Namespace
from pathlib import Path

import pytest

from vespainv.cli import _resolve_prep_earth_args


def test_resolve_prep_earth_args_allows_optional_processing_settings(tmp_path: Path):
    config_path = tmp_path / "prep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_dir: /tmp/data",
                "output_dir: /tmp/out",
                "bandpass: null",
                "downsample_hz: null",
                "time_window: null",
                "snr_component: min",
                "snr_threshold: null",
                "plot_summary: true",
            ]
        ),
        encoding="utf-8",
    )

    args = Namespace(
        config=str(config_path),
        data_dir=None,
        output_dir=None,
        noise_dir=None,
        bandpass=None,
        downsample_hz=None,
        time_window=None,
        snr_component=None,
        snr_threshold=None,
        plot_summary=None,
    )

    resolved = _resolve_prep_earth_args(args)

    assert resolved["bandpass"] is None
    assert resolved["downsample_hz"] is None
    assert resolved["time_window"] is None
    assert resolved["snr_component"] == "min"
    assert resolved["plot_summary"] is True


def test_resolve_prep_earth_args_rejects_non_positive_downsample():
    args = Namespace(
        config=None,
        data_dir="/tmp/data",
        output_dir="/tmp/out",
        noise_dir=None,
        bandpass=None,
        downsample_hz=0,
        time_window=None,
        snr_component=None,
        snr_threshold=None,
        plot_summary=None,
    )

    with pytest.raises(ValueError, match="downsample-hz must be positive"):
        _resolve_prep_earth_args(args)


def test_resolve_prep_earth_args_defaults_plot_summary_to_false():
    args = Namespace(
        config=None,
        data_dir="/tmp/data",
        output_dir="/tmp/out",
        noise_dir=None,
        bandpass=None,
        downsample_hz=None,
        time_window=None,
        snr_component=None,
        snr_threshold=None,
        plot_summary=None,
    )

    resolved = _resolve_prep_earth_args(args)

    assert resolved["plot_summary"] is False


def test_plot_prep_can_take_output_dir_from_config(tmp_path: Path):
    config_path = tmp_path / "prep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_dir: /tmp/data",
                f"output_dir: {tmp_path / 'prepared'}",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "data_dir": "/tmp/data",
        "output_dir": str(tmp_path / "prepared"),
    }

    assert config["output_dir"] in config_path.read_text(encoding="utf-8")
