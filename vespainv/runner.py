from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np

from .config import load_config, load_dataset_manifest
from .model import Bookkeeping, Prior, Prior3c
from .utils import (
    calc_array_center,
    create_stf,
    create_stf_gaussian,
    est_dom_freq,
    est_stf_wid,
    prep_data,
)
from .validation import validate_dataset


def _run_chain(chain_id: int, exp_vars: dict) -> list:
    from .rjmcmc import rjmcmc_run, rjmcmc_run3c

    data_name = exp_vars["dataset"]
    runname = exp_vars["runname"]
    is3c = exp_vars["is3c"]
    num_chains = exp_vars["num_chains"]
    runs_root = exp_vars["runs_root"]

    if num_chains == 1:
        save_dir = os.path.join(runs_root, data_name, runname)
    else:
        save_dir = os.path.join(
            runs_root,
            data_name,
            runname,
            f"chain_{chain_id}",
        )
    os.makedirs(save_dir, exist_ok=True)

    if is3c:
        samples, logl_trace, loge_trace, nphase = rjmcmc_run3c(
            exp_vars["U_obs"],
            exp_vars["CDinv"],
            exp_vars["CD_sqrt_inv"],
            exp_vars["metadata"],
            exp_vars["Utime"],
            exp_vars["stf"],
            exp_vars["prior"],
            exp_vars["bookkeeping"],
            save_dir,
        )
    else:
        samples, logl_trace, loge_trace, nphase = rjmcmc_run(
            exp_vars["U_obs"],
            exp_vars["CDinv"],
            exp_vars["CD_sqrt_inv"],
            exp_vars["metadata"],
            exp_vars["Utime"],
            exp_vars["stf"],
            exp_vars["prior"],
            exp_vars["bookkeeping"],
            save_dir,
        )

    with open(os.path.join(save_dir, "ensemble.pkl"), "wb") as handle:
        pickle.dump(samples, handle)
    np.savetxt(os.path.join(save_dir, "log_likelihood.txt"), logl_trace)
    np.savetxt(os.path.join(save_dir, "loge.txt"), loge_trace)
    np.savetxt(os.path.join(save_dir, "Nphase.txt"), nphase)

    return samples


def _normalize_bandpass(value):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("bandpass must contain exactly [frequency_min, frequency_max].")
    fmin, fmax = map(float, value)
    if not (0 < fmin < fmax):
        raise ValueError("bandpass frequencies must satisfy 0 < fmin < fmax.")
    return fmin, fmax


def _resolve_stf_bandpass(params: dict, dataset_dir: str):
    configured = _normalize_bandpass(params.get("bandpass"))
    manifest = load_dataset_manifest(dataset_dir)
    prepared = _normalize_bandpass(manifest.get("processing", {}).get("bandpass"))
    if configured is not None and prepared is not None and not np.allclose(configured, prepared):
        raise ValueError(
            f"Run bandpass {configured} does not match the prepared dataset bandpass {prepared}."
        )
    return configured if configured is not None else prepared


def _filter_stf(stf, bandpass_hz):
    if bandpass_hz is None:
        return stf
    from scipy.signal import butter, sosfiltfilt

    stf = np.asarray(stf, dtype=float).copy()
    if stf.ndim != 2 or stf.shape[1] != 2 or stf.shape[0] < 4:
        raise ValueError("Bandpass filtering requires stf.csv with at least four time/amplitude rows.")
    intervals = np.diff(stf[:, 0])
    stf_dt = float(np.median(intervals))
    if np.any(intervals <= 0) or not np.allclose(intervals, stf_dt, rtol=1e-6):
        raise ValueError("STF time samples must be strictly increasing and uniform.")
    fmin, fmax = bandpass_hz
    nyquist = 0.5 / stf_dt
    if fmax >= nyquist:
        raise ValueError(
            f"STF bandpass upper frequency {fmax:g} Hz must be below Nyquist {nyquist:g} Hz."
        )
    sos = butter(2, (fmin, fmax), btype="bandpass", fs=1.0 / stf_dt, output="sos")
    padlen = min(3 * (2 * len(sos) + 1), stf.shape[0] - 1)
    filtered = sosfiltfilt(sos, stf[:, 1], padlen=padlen)
    scale = float(np.max(np.abs(filtered)))
    if not np.isfinite(scale) or scale == 0:
        raise ValueError("Bandpass filtering produced an invalid zero-amplitude STF.")
    stf[:, 1] = filtered / scale
    return stf


def _prepare_stf(data_dir: str, man_stf: bool, stfshape: str, U_obs, is3c, dt, bandpass_hz=None):
    if man_stf:
        stf = np.loadtxt(os.path.join(data_dir, "stf.csv"), delimiter=",", skiprows=1)
    elif stfshape == "dGaussian":
        stf = create_stf(est_dom_freq(U_obs if not is3c else U_obs[:, :, 0], 1 / dt), dt)
    elif stfshape == "Gaussian":
        stf = create_stf_gaussian(est_dom_freq(U_obs if not is3c else U_obs[:, :, 0], 1 / dt), dt)
    else:
        raise ValueError(f"Unsupported stfshape: {stfshape}")
    return _filter_stf(stf, bandpass_hz)


def _prepare_prior(is3c: bool, stf_wid: float, maxN: int, sigma: float, Utime, ampRange, slwRange, distDiffRange, bazDiffRange):
    kwargs = {
        "minSpace": stf_wid,
        "maxN": maxN,
        "sigma": sigma,
        "timeRange": (Utime[0], Utime[-1]),
        "ampRange": ampRange,
        "slwRange": slwRange,
        "distDiffRange": distDiffRange,
        "bazDiffRange": bazDiffRange,
    }
    if is3c:
        return Prior3c(**kwargs)
    return Prior(**kwargs)


def _physical_cpu_count() -> int:
    try:
        import psutil

        return psutil.cpu_count(logical=False) or mp.cpu_count()
    except ImportError:
        return mp.cpu_count()


def _validate_run_parameters(params: dict) -> None:
    total_steps = int(params["totalSteps"])
    burn_in_steps = int(params["burnInSteps"])
    n_save_models = int(params["nSaveModels"])
    num_chains = int(params["num_chains"])
    actions_per_step = int(params["actionsPerStep"])
    max_n = int(params["maxN"])
    norm_opt = int(params["normOpt"])
    if total_steps <= 0:
        raise ValueError("totalSteps must be positive.")
    if not 0 <= burn_in_steps < total_steps:
        raise ValueError("burnInSteps must satisfy 0 <= burnInSteps < totalSteps.")
    if n_save_models <= 0 or n_save_models > total_steps - burn_in_steps:
        raise ValueError("nSaveModels must be positive and no larger than post-burn-in steps.")
    if num_chains <= 0:
        raise ValueError("num_chains must be positive.")
    if actions_per_step <= 0:
        raise ValueError("actionsPerStep must be positive.")
    if max_n <= 0:
        raise ValueError("maxN must be positive.")
    if norm_opt not in {1, 2}:
        raise ValueError("normOpt must be 1 (L1) or 2 (L2).")


def run_experiment(params: dict) -> None:
    _validate_run_parameters(params)
    data_name = params.get("dataset")
    if not data_name:
        raise ValueError("Each experiment must define 'dataset'.")
    runname = params["runname"]
    is3c = params["is3c"]
    comp = params["comp"]
    num_chains = params["num_chains"]
    totalSteps = int(params["totalSteps"])
    burnInSteps = int(params["burnInSteps"])
    nSaveModels = params["nSaveModels"]
    actionsPerStep = params["actionsPerStep"]
    maxN = params["maxN"]
    sigma = params["sigma"]
    man_stf = params["man_stf"]
    stfshape = params["stfshape"]
    ampRange = tuple(params["ampRange"])
    slwRange = tuple(params["slwRange"])
    minSpace = params["minSpace"]
    CDopt = params["CDopt"]
    locDiff = params["locDiff"]
    distDiffRange = tuple(params["distDiffRange"])
    bazDiffRange = tuple(params["bazDiffRange"])
    fitAtts = params["fitAtts"]
    fitLoge = params["fitLoge"]
    fitPhase = params["fitPhase"]
    normOpt = params["normOpt"]
    isMars = params["isMars"]
    srcArray = params["srcArray"]
    pref = params.get("pref", 0.0)
    fstVp = params.get("fstVp")
    fstVs = params.get("fstVs")
    ref_manual = params["ref_manual"]
    refLat = params["refLat"]
    refLon = params["refLon"]
    refBaz = params["refBaz"]

    if isMars:
        srcArray = True

    data_root = params.get("data_root") or params.get("paths", {}).get("real_data_root")
    runs_root = params.get("runs_root") or params.get("paths", {}).get("runs_root")
    if not data_root or not runs_root:
        raise ValueError(
            "Run config must provide data_root and runs_root, either directly or via workspace paths."
        )

    print(f"\n=== Running experiment: {data_name} / {runname} ===")

    dataset_dir = os.path.join(data_root, data_name)
    validation = validate_dataset(
        dataset_dir,
        is3c=is3c,
        component=comp,
        cdopt=CDopt,
        source_array=srcArray,
        is_mars=isMars,
        manual_stf=man_stf,
    )
    print(
        f"Validated dataset: {validation.n_samples} samples, "
        f"{validation.n_traces} traces, {validation.sampling_rate_hz:g} Hz."
    )

    U_obs, Utime, CDinv, CD_sqrt_inv, metadata, is3c = prep_data(
        data_root, data_name, is3c, comp, CDopt, is_mars=isMars, src_array=srcArray
    )
    dt = Utime[1] - Utime[0]

    stf_bandpass = _resolve_stf_bandpass(params, dataset_dir)
    stf = _prepare_stf(
        dataset_dir, man_stf, stfshape, U_obs, is3c, dt,
        bandpass_hz=stf_bandpass,
    )
    if not man_stf:
        stf_path = os.path.join(dataset_dir, "stf.csv")
        np.savetxt(stf_path, stf, delimiter=",", header="time,stf", comments="")
    stf_wid = minSpace if minSpace is not None else est_stf_wid(stf)

    prior = _prepare_prior(
        is3c, stf_wid, maxN, sigma, Utime, ampRange, slwRange, distDiffRange, bazDiffRange
    )
    save_dir = os.path.join(runs_root, data_name, runname)
    os.makedirs(save_dir, exist_ok=True)
    np.savetxt(
        os.path.join(save_dir, "stf_used.csv"), stf,
        delimiter=",", header="time,stf", comments="",
    )
    prior_path = os.path.join(save_dir, "Prior.pkl")
    if not os.path.exists(prior_path):
        with open(prior_path, "wb") as handle:
            pickle.dump(prior, handle)

    srcLat, srcLon = np.loadtxt(os.path.join(dataset_dir, "eventinfo.csv"), delimiter=",", skiprows=1)
    if not ref_manual:
        refLat, refLon, _, refBaz = calc_array_center(
            metadata,
            srcLat,
            srcLon,
            srcArray,
            metadata_format="distbaz" if (isMars or srcArray) else "latlon",
        )

    bookkeeping = Bookkeeping(
        totalSteps=totalSteps,
        burnInSteps=burnInSteps,
        nSaveModels=nSaveModels,
        actionsPerStep=actionsPerStep,
        locDiff=locDiff,
        fitAtts=fitAtts,
        fitLoge=fitLoge,
        fitPhase=fitPhase,
        normOpt=normOpt,
        isMars=isMars,
        srcArray=srcArray,
        srcLat=srcLat,
        srcLon=srcLon,
        refLat=refLat,
        refLon=refLon,
        refBaz=refBaz,
        pref=pref,
        fstVp=fstVp,
        fstVs=fstVs,
    )
    bk_path = os.path.join(save_dir, "Bookkeeping.pkl")
    if not os.path.exists(bk_path):
        with open(bk_path, "wb") as handle:
            pickle.dump(bookkeeping, handle)

    cpu_cores = _physical_cpu_count()
    print(f"Detected {cpu_cores} physical cores.")
    if num_chains >= 2:
        threads_per_chain = max(1, cpu_cores // min(num_chains, cpu_cores))
        os.environ["OMP_NUM_THREADS"] = str(threads_per_chain)
        os.environ["MKL_NUM_THREADS"] = str(threads_per_chain)
        print(f"Setting {threads_per_chain} threads per chain to avoid oversubscription.")
    else:
        print("Single chain: allow full multithreading.")

    exp_vars = {
        "U_obs": U_obs,
        "CDinv": CDinv,
        "CD_sqrt_inv": CD_sqrt_inv,
        "metadata": metadata,
        "Utime": Utime,
        "stf": stf,
        "prior": prior,
        "bookkeeping": bookkeeping,
        "dataset": data_name,
        "runname": runname,
        "is3c": is3c,
        "num_chains": num_chains,
        "runs_root": runs_root,
    }

    # Diagnostic plotting needs writable cache locations on restricted systems.
    cache_root = Path(tempfile.gettempdir()) / "tapir-cache"
    matplotlib_cache = cache_root / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))

    start = time.time()
    if num_chains == 1:
        _run_chain(0, exp_vars)
        print(f"Total elapsed time: {time.time() - start:.2f} seconds")
        return

    ctx = mp.get_context("spawn")
    batch_size = cpu_cores
    total_batches = (num_chains + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_chain = batch_idx * batch_size
        end_chain = min(start_chain + batch_size, num_chains)
        batch_chain_ids = list(range(start_chain, end_chain))
        print(f"Running batch {batch_idx + 1}/{total_batches} with chains {batch_chain_ids}")
        with ctx.Pool(processes=len(batch_chain_ids)) as pool:
            pool.starmap(_run_chain, [(cid, exp_vars) for cid in batch_chain_ids])

    print(f"Total elapsed time: {time.time() - start:.2f} seconds")


def run_config(config_path: str | Path) -> None:
    config_path = Path(config_path)
    config = load_config(config_path)
    defaults = config.get("defaults", {})
    workspace_paths = config.get("paths", {})
    if workspace_paths:
        defaults.setdefault("data_root", workspace_paths.get("real_data_root"))
        defaults.setdefault("runs_root", workspace_paths.get("runs_root"))
        defaults.setdefault("paths", workspace_paths)
    if "filedir" in defaults:
        raise ValueError("The 'filedir' setting is no longer supported. Use workspace paths instead.")
    for experiment in config["experiments"]:
        params = {**defaults, **experiment}
        if "filedir" in experiment:
            raise ValueError("Experiment-level 'filedir' is no longer supported. Use workspace paths instead.")
        params.setdefault("paths", workspace_paths)
        run_experiment(params)
