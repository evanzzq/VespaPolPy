from argparse import Namespace
from pathlib import Path

import pytest

from vespainv.cli import _resolve_prep_earth_args, _resolve_prep_source_earth_args


def test_resolve_prep_earth_args_allows_optional_processing_settings(tmp_path: Path):
    config_path = tmp_path / "prep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  input_dir: /tmp/data",
                "  output_dir: /tmp/out",
                "processing:",
                "  bandpass: null",
                "  downsample_hz: null",
                "  time_window: null",
                "qc:",
                "  snr_component: min",
                "  snr_threshold: null",
                "  plot_summary: true",
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
                "dataset:",
                "  input_dir: /tmp/data",
                f"  output_dir: {tmp_path / 'prepared'}",
            ]
        ),
        encoding="utf-8",
    )

    config = {
        "data_dir": "/tmp/data",
        "output_dir": str(tmp_path / "prepared"),
    }

    assert config["output_dir"] in config_path.read_text(encoding="utf-8")


def test_resolve_prep_source_earth_args_uses_config(tmp_path: Path):
    config_path = tmp_path / "prep_source.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  input_dir: /tmp/source_sac",
                "  output_dir: /tmp/source_out",
                "  noise_dir: /tmp/source_noise",
                "processing:",
                "  bandpass: [0.02, 0.5]",
                "  downsample_hz: 5",
                "  time_window: [0, 100]",
                "qc:",
                "  snr_component: min",
                "  snr_threshold: 1.5",
                "  plot_summary: false",
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

    resolved = _resolve_prep_source_earth_args(args)

    assert resolved["data_dir"] == "/tmp/source_sac"
    assert resolved["output_dir"] == "/tmp/source_out"
    assert resolved["noise_dir"] == "/tmp/source_noise"
    assert resolved["bandpass"] == (0.02, 0.5)
    assert resolved["downsample_hz"] == 5
    assert resolved["time_window"] == (0, 100)
    assert resolved["snr_component"] == "min"
    assert resolved["snr_threshold"] == 1.5
    assert resolved["plot_summary"] is False
