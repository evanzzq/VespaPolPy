from __future__ import annotations

import pickle
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChainSummary:
    name: str
    steps: int
    final_log_likelihood: float
    best_log_likelihood: float
    median_phase_count: float


def _chain_directories(run_dir: Path) -> list[Path]:
    chains = sorted(
        (path for path in run_dir.glob("chain_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    return chains or [run_dir]


def _load_trace(path: Path, name: str) -> np.ndarray:
    trace_path = path / name
    if not trace_path.is_file():
        raise ValueError(f"Missing {name} in {path}.")
    values = np.atleast_1d(np.asarray(np.loadtxt(trace_path), dtype=float))
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{trace_path} must contain finite values.")
    return values


def _burn_in_steps(run_dir: Path) -> int:
    bookkeeping_path = run_dir / "Bookkeeping.pkl"
    if not bookkeeping_path.is_file():
        return 0
    with bookkeeping_path.open("rb") as handle:
        bookkeeping = pickle.load(handle)
    return max(0, int(getattr(bookkeeping, "burnInSteps", 0)))


def summarize_run(run_dir: str | Path, output: str | Path | None = None) -> list[ChainSummary]:
    """Summarize completed or in-progress TAPIR chain traces and save a figure."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    chain_dirs = _chain_directories(run_dir)
    burn_in = _burn_in_steps(run_dir)

    traces = []
    summaries = []
    for chain_dir in chain_dirs:
        log_likelihood = _load_trace(chain_dir, "log_likelihood.txt")
        nphase = _load_trace(chain_dir, "Nphase.txt")
        if log_likelihood.shape != nphase.shape:
            raise ValueError(f"Trace length mismatch in {chain_dir}.")
        post_burn = nphase[min(burn_in, nphase.size - 1):]
        summaries.append(
            ChainSummary(
                name=chain_dir.name if chain_dir != run_dir else "chain_0",
                steps=log_likelihood.size,
                final_log_likelihood=float(log_likelihood[-1]),
                best_log_likelihood=float(np.max(log_likelihood)),
                median_phase_count=float(np.median(post_burn)),
            )
        )
        traces.append((summaries[-1].name, log_likelihood, nphase))

    cache_root = Path(tempfile.gettempdir()) / "tapir-cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, log_likelihood, nphase in traces:
        steps = np.arange(1, log_likelihood.size + 1)
        axes[0].plot(steps, log_likelihood, linewidth=0.8, label=name)
        axes[1].plot(steps, nphase, linewidth=0.8, label=name)
    if burn_in:
        for axis in axes:
            axis.axvline(burn_in, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Log likelihood")
    axes[1].set_ylabel("Number of phases")
    axes[1].set_xlabel("Step")
    for axis in axes:
        axis.grid(alpha=0.25)
        if len(traces) > 1:
            axis.legend(fontsize="small")
    fig.suptitle(run_dir.name)
    fig.tight_layout()
    output_path = Path(output).expanduser().resolve() if output else run_dir / "run_summary.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return summaries
