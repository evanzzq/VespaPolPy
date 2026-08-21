from pathlib import Path

import numpy as np

from vespainv.rjmcmc import _initialize_progress_outputs, _write_progress_checkpoint


def test_progress_checkpoint_appends_traces_and_writes_figure(tmp_path: Path):
    _initialize_progress_outputs(tmp_path)
    log_likelihood = [-10.0, -9.0, -8.0]
    loge = [0.0, 0.1, 0.2]
    nphase = [1, 2, 2]

    _write_progress_checkpoint(
        tmp_path, 0, log_likelihood, loge, nphase, 2, 3, 0.5
    )
    _write_progress_checkpoint(
        tmp_path, 3, log_likelihood, loge, nphase, 3, 3, 1.0
    )

    assert np.loadtxt(tmp_path / "log_likelihood.txt").tolist() == log_likelihood
    assert np.loadtxt(tmp_path / "loge.txt").tolist() == loge
    assert np.loadtxt(tmp_path / "Nphase.txt").tolist() == nphase
    assert (tmp_path / "likelihood_phase_count_progress.png").stat().st_size > 0
    assert "Step 3/3" in (tmp_path / "progress.txt").read_text()
