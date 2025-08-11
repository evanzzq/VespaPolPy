import os
from parameter_setup import *
# Limit BLAS threads early to avoid oversubscription
if num_chains > 1:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import pickle, time
import numpy as np
import multiprocessing as mp
from vespainv.model import Bookkeeping, Prior, Prior3c
from vespainv.utils import calc_array_center, create_stf, est_stf_wid, est_dom_freq, prep_data
from vespainv.model import Bookkeeping

start = time.time()

# ---- Load static data once in parent ----
datadir = os.path.join(filedir, "SynData") if isSyn else os.path.join(filedir, "RealData")

U_obs, Utime, CDinv, metadata, is3c = prep_data(
    datadir, modname, is3c, comp, CDopt, isbp, freqs, isds
)
dt = Utime[1] - Utime[0]

# STF: load if synthetic, else compute and save once
if isSyn:
    stf = np.loadtxt(os.path.join(datadir, modname, "stf.csv"), delimiter=",", skiprows=1)
else:
    stf = create_stf(est_dom_freq(U_obs if not is3c else U_obs[:, :, 0], 1/dt), dt)
    stf_path = os.path.join(datadir, modname, "stf.csv")
    if not os.path.exists(stf_path):
        np.savetxt(stf_path, stf, delimiter=",", header="time,stf", comments="")

stf_wid = minSpace if minSpace is not None else est_stf_wid(stf)

# Load or create prior
if isSyn:
    with open(os.path.join(datadir, modname, "Prior.pkl"), "rb") as f:
        prior = pickle.load(f)
    prior.arrStd = 5.0
else:
    srcLat, srcLon = np.loadtxt(os.path.join(datadir, modname, "eventinfo.csv"), delimiter=",", skiprows=1)
    refLat, refLon, refBaz = calc_array_center(metadata, srcLat, srcLon)
    if is3c:
        prior = Prior3c(
            refLat=refLat, refLon=refLon, refBaz=refBaz, srcLat=srcLat, srcLon=srcLon,
            minSpace=stf_wid, maxN=maxN,
            timeRange=(Utime[0], Utime[-1]), ampRange=ampRange,
            slwRange=slwRange, bazRange=bazRange, distDiffRange=distDiffRange, bazDiffRange=bazDiffRange
        )
    else:
        prior = Prior(
            refLat=refLat, refLon=refLon, refBaz=refBaz, srcLat=srcLat, srcLon=srcLon,
            minSpace=stf_wid, maxN=maxN,
            timeRange=(Utime[0], Utime[-1]), ampRange=ampRange,
            slwRange=slwRange, bazRange=bazRange, distDiffRange=distDiffRange, bazDiffRange=bazDiffRange
        )
    prior_path = os.path.join(datadir, modname, "Prior.pkl")
    if not os.path.exists(prior_path):
        with open(prior_path, "wb") as f:
            pickle.dump(prior, f)

# Load model if synthetic
model = None
if isSyn:
    with open(os.path.join(datadir, modname, "Model.pkl"), "rb") as f:
        model = pickle.load(f)

# Create bookkeeping once
bookkeeping = Bookkeeping(
    totalSteps=totalSteps,
    burnInSteps=burnInSteps,
    nSaveModels=nSaveModels,
    actionsPerStep=actionsPerStep,
    phaseBaz=phaseBaz,
    locDiff=locDiff,
    fitAtts=fitAtts,
    fitPhase=fitPhase
)

# -------- Chain runner function --------
def run_chain(chain_id):
    from vespainv.rjmcmc import rjmcmc_run, rjmcmc_run3c

    # Setup per-chain save directory
    if num_chains == 1:
        save_dir = os.path.join(filedir, "runs/syn" if isSyn else "runs/data", modname, runname)
    else:
        save_dir = os.path.join(filedir, "runs/syn" if isSyn else "runs/data", modname, runname, f"chain_{chain_id}")
    os.makedirs(save_dir, exist_ok=True)

    if is3c:
        samples, logL_trace = rjmcmc_run3c(
            U_obs, CDinv, metadata, Utime,
            stf, prior, bookkeeping, save_dir
        )
    else:
        samples, logL_trace = rjmcmc_run(
            U_obs, CDinv, metadata, Utime,
            stf, prior, bookkeeping, save_dir
        )

    # Save chain outputs
    with open(os.path.join(save_dir, "ensemble.pkl"), "wb") as f:
        pickle.dump(samples, f)
    np.savetxt(os.path.join(save_dir, "log_likelihood.txt"), logL_trace)

    return samples

# -------- Main execution --------
if __name__ == "__main__":
    # Get number of physical cores
    try:
        import psutil
        cpu_cores = psutil.cpu_count(logical=False) or mp.cpu_count()
    except ImportError:
        cpu_cores = mp.cpu_count()
    
    print(f"Detected {cpu_cores} physical cores.")
    
    # Limit threads per chain if running multiple chains
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

    start = time.time()
    for batch_idx in range(total_batches):
        start_chain = batch_idx * batch_size
        end_chain = min(start_chain + batch_size, num_chains)
        batch_chain_ids = list(range(start_chain, end_chain))
        print(f"Running batch {batch_idx+1}/{total_batches} with chains {batch_chain_ids}")

        with ctx.Pool(processes=len(batch_chain_ids)) as pool:
            pool.map(run_chain, batch_chain_ids)

    end = time.time()
    print(f"Total elapsed time: {end - start:.2f} seconds")