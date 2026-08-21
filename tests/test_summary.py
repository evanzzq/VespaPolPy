from pathlib import Path

import numpy as np

from vespainv.summary import summarize_run


def _write_chain(path: Path, offset: float):
    path.mkdir(parents=True)
    np.savetxt(path / "log_likelihood.txt", [-10.0 + offset, -8.0 + offset])
    np.savetxt(path / "Nphase.txt", [1, 2])


def test_summarize_multichain_run(tmp_path: Path):
    _write_chain(tmp_path / "chain_0", 0.0)
    _write_chain(tmp_path / "chain_1", 1.0)

    summaries = summarize_run(tmp_path)

    assert [summary.name for summary in summaries] == ["chain_0", "chain_1"]
    assert summaries[0].steps == 2
    assert summaries[1].best_log_likelihood == -7.0
    assert (tmp_path / "run_summary.png").stat().st_size > 0


def test_summarize_single_chain_run(tmp_path: Path):
    np.savetxt(tmp_path / "log_likelihood.txt", [-3.0, -2.0])
    np.savetxt(tmp_path / "Nphase.txt", [2, 3])

    summaries = summarize_run(tmp_path)

    assert len(summaries) == 1
    assert summaries[0].name == "chain_0"
