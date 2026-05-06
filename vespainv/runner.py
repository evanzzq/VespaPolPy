from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .model import Bookkeeping, Prior, Prior3c
from .utils import (
    calc_array_center,
    create_stf,
    create_stf_gaussian,
    est_dom_freq,
    est_stf_wid,
    prep_data,
)


def _run_chain(chain_id: int, exp_vars: dict) -> list:
    from .rjmcmc import rjmcmc_run, rjmcmc_run3c

    filedir = exp_vars["filedir"]
    modname = exp_vars["modname"]
    runname = exp_vars["runname"]
    is3c = exp_vars["is3c"]
    is_syn = exp_vars["isSyn"]
    num_chains = exp_vars["num_chains"]

    if num_chains == 1:
        save_dir = os.path.join(filedir, "runs/syn" if is_syn else "runs/data", modname, runname)
    else:
        save_dir = os.path.join(
            filedir,
            "runs/syn" if is_syn else "runs/data",
            modname,
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


def _prepare_stf(datadir: str, modname: str, is_syn: bool, man_stf: bool, stfshape: str, U_obs, is3c, dt):
    if is_syn or man_stf:
        return np.loadtxt(os.path.join(datadir, modname, "stf.csv"), delimiter=",", skiprows=1)
    if stfshape == "dGaussian":
        return create_stf(est_dom_freq(U_obs if not is3c else U_obs[:, :, 0], 1 / dt), dt)
    if stfshape == "Gaussian":
        return create_stf_gaussian(est_dom_freq(U_obs if not is3c else U_obs[:, :, 0], 1 / dt), dt)
    raise ValueError(f"Unsupported stfshape: {stfshape}")


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


def run_experiment(params: dict) -> None:
    filedir = params["filedir"]
    modname = params["modname"]
    runname = params["runname"]
    is_syn = params["isSyn"]
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
    ref_manual = params["ref_manual"]
    refLat = params["refLat"]
    refLon = params["refLon"]
    refBaz = params["refBaz"]

    if isMars:
        srcArray = True

    print(f"\n=== Running experiment: {modname} / {runname} ===")

    datadir = os.path.join(filedir, "SynData") if is_syn else os.path.join(filedir, "RealData")
    U_obs, Utime, CDinv, CD_sqrt_inv, metadata, is3c = prep_data(
        datadir, modname, is3c, comp, CDopt, is_mars=isMars
    )
    dt = Utime[1] - Utime[0]

    stf = _prepare_stf(datadir, modname, is_syn, man_stf, stfshape, U_obs, is3c, dt)
    if not (is_syn or man_stf):
        stf_path = os.path.join(datadir, modname, "stf.csv")
        np.savetxt(stf_path, stf, delimiter=",", header="time,stf", comments="")
    stf_wid = minSpace if minSpace is not None else est_stf_wid(stf)

    prior = _prepare_prior(
        is3c, stf_wid, maxN, sigma, Utime, ampRange, slwRange, distDiffRange, bazDiffRange
    )
    save_dir = os.path.join(filedir, "runs/syn" if is_syn else "runs/data", modname, runname)
    os.makedirs(save_dir, exist_ok=True)
    prior_path = os.path.join(save_dir, "Prior.pkl")
    if not os.path.exists(prior_path):
        with open(prior_path, "wb") as handle:
            pickle.dump(prior, handle)

    if is_syn:
        with open(os.path.join(datadir, modname, "Bookkeeping_0.pkl"), "rb") as handle:
            bookkeeping_0 = pickle.load(handle)
        srcLat = bookkeeping_0.srcLat
        srcLon = bookkeeping_0.srcLon
        refLat = bookkeeping_0.refLat
        refLon = bookkeeping_0.refLon
        refBaz = bookkeeping_0.refBaz
    else:
        srcLat, srcLon = np.loadtxt(os.path.join(datadir, modname, "eventinfo.csv"), delimiter=",", skiprows=1)
        if not ref_manual:
            refLat, refLon, _, refBaz = calc_array_center(
                metadata,
                srcLat,
                srcLon,
                srcArray,
                metadata_format="distbaz" if isMars else "latlon",
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

    ctx = mp.get_context("spawn")
    batch_size = cpu_cores
    total_batches = (num_chains + batch_size - 1) // batch_size
    exp_vars = {
        "U_obs": U_obs,
        "CDinv": CDinv,
        "CD_sqrt_inv": CD_sqrt_inv,
        "metadata": metadata,
        "Utime": Utime,
        "stf": stf,
        "prior": prior,
        "bookkeeping": bookkeeping,
        "filedir": filedir,
        "modname": modname,
        "runname": runname,
        "is3c": is3c,
        "isSyn": is_syn,
        "num_chains": num_chains,
    }

    start = time.time()
    for batch_idx in range(total_batches):
        start_chain = batch_idx * batch_size
        end_chain = min(start_chain + batch_size, num_chains)
        batch_chain_ids = list(range(start_chain, end_chain))
        print(f"Running batch {batch_idx + 1}/{total_batches} with chains {batch_chain_ids}")
        with ctx.Pool(processes=len(batch_chain_ids)) as pool:
            pool.starmap(_run_chain, [(cid, exp_vars) for cid in batch_chain_ids])

    print(f"Total elapsed time: {time.time() - start:.2f} seconds")


def run_config(config_path: str | Path) -> None:
    config = load_config(config_path)
    defaults = config.get("defaults", {})
    for experiment in config["experiments"]:
        run_experiment({**defaults, **experiment})
