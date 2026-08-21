from pathlib import Path

import numpy as np
import pytest

from vespainv.runner import _filter_stf, _resolve_stf_bandpass


def test_filter_stf_applies_bandpass_and_renormalizes():
    time = np.arange(-10.0, 10.0, 0.05)
    amplitude = np.exp(-(time / 0.8) ** 2)
    stf = np.column_stack((time, amplitude))

    filtered = _filter_stf(stf, (0.2, 0.8))

    assert filtered.shape == stf.shape
    assert np.max(np.abs(filtered[:, 1])) == pytest.approx(1.0)
    assert not np.allclose(filtered[:, 1], stf[:, 1])


def test_resolve_stf_bandpass_reads_earth_manifest(tmp_path: Path):
    (tmp_path / "dataset.yaml").write_text(
        "processing:\n  bandpass: [0.02, 0.4]\n", encoding="utf-8"
    )

    assert _resolve_stf_bandpass({}, str(tmp_path)) == (0.02, 0.4)


def test_resolve_stf_bandpass_rejects_mismatch(tmp_path: Path):
    (tmp_path / "dataset.yaml").write_text(
        "processing:\n  bandpass: [0.02, 0.4]\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match"):
        _resolve_stf_bandpass({"bandpass": [0.1, 0.5]}, str(tmp_path))
