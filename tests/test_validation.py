from pathlib import Path

import numpy as np
import pytest

from vespainv.validation import validate_dataset


def _write_minimal_dataset(path: Path):
    time = np.arange(5, dtype=float) * 0.2
    waveform = np.column_stack((np.sin(time), np.cos(time)))
    np.savetxt(path / "time.csv", time, delimiter=",")
    for component in "ZRT":
        np.savetxt(path / f"U{component}.csv", waveform, delimiter=",")
    np.savetxt(
        path / "station_metadata.csv",
        np.array([[40.0, -120.0], [41.0, -121.0]]),
        delimiter=",", header="lat,lon", comments="",
    )
    np.savetxt(
        path / "eventinfo.csv", np.array([[10.0, 20.0]]),
        delimiter=",", header="lat,lon", comments="",
    )


def test_validate_receiver_array_dataset(tmp_path: Path):
    _write_minimal_dataset(tmp_path)

    result = validate_dataset(tmp_path)

    assert result.n_samples == 5
    assert result.n_traces == 2
    assert result.components == ("Z", "R", "T")
    assert result.metadata_format == "latlon"
    assert result.sampling_rate_hz == pytest.approx(5.0)


def test_validate_dataset_detects_trace_metadata_mismatch(tmp_path: Path):
    _write_minimal_dataset(tmp_path)
    np.savetxt(
        tmp_path / "station_metadata.csv", np.array([[40.0, -120.0]]),
        delimiter=",", header="lat,lon", comments="",
    )

    with pytest.raises(ValueError, match="one two-value row per trace"):
        validate_dataset(tmp_path)


def test_validate_mars_uses_distance_azimuth_metadata(tmp_path: Path):
    _write_minimal_dataset(tmp_path)
    np.savetxt(
        tmp_path / "station_metadata_db.csv", np.array([[20.0, 30.0], [21.0, 31.0]]),
        delimiter=",", header="dist_deg,baz", comments="",
    )

    result = validate_dataset(tmp_path, is_mars=True)

    assert result.metadata_format == "distbaz"
