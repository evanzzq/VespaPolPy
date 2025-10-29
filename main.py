import os, time, pickle, argparse, yaml
import numpy as np
import multiprocessing as mp
from vespainv.model import Bookkeeping, Prior, Prior3c
from vespainv.utils import calc_array_center, create_stf, est_stf_wid, est_dom_freq, prep_data

# ---- Parse config file ----
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="parameter_setup.yaml", help="YAML config file")
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

defaults = config.get("defaults", {})
experiments = config["experiments"]

# -------- Chain runner function --------
def run_chain(chain_id, exp_vars):
    from vespainv.rjmcmc import rjmcmc_run, rjmcmc_run3c

    # Unpack experiment variables
    U_obs = exp_vars["U_obs"]
    CDinv = exp_vars["CDinv"]
    CD_sqrt_inv = exp_vars["CD_sqrt_inv"]
    metadata = exp_vars["metadata"]
    Utime = exp_vars["Utime"]
    stf = exp_vars["stf"]
    prior = exp_vars["prior"]
    bookkeeping = exp_vars["bookkeeping"]
    filedir = exp_vars["filedir"]
    modname = exp_vars["modname"]
    runname = exp_vars["runname"]
    is3c = exp_vars["is3c"]
    isSyn = exp_vars["isSyn"]
    num_chains = exp_vars["num_chains"]

    # Setup per-chain save directory
    if num_chains == 1:
        save_dir = os.path.join(filedir, "runs/syn" if isSyn else "runs/data", modname, runname)
    else:
        save_dir = os.path.join(filedir, "runs/syn" if isSyn else "runs/data", modname, runname, f"chain_{chain_id}")
    os.makedirs(save_dir, exist_ok=True)

    # Run RJMCMC
    if is3c:
        samples, logL_trace = rjmcmc_run3c(U_obs, CDinv, CD_sqrt_inv, metadata, Utime,
                                           stf, prior, bookkeeping, save_dir)
    else:
        samples, logL_trace = rjmcmc_run(U_obs, CDinv, CD_sqrt_inv, metadata, Utime,
                                         stf, prior, bookkeeping, save_dir)

    # Save outputs
    with open(os.path.join(save_dir, "ensemble.pkl"), "wb") as f:
        pickle.dump(samples, f)
    np.savetxt(os.path.join(save_dir, "log_likelihood.txt"), logL_trace)

    return samples


# -------- Main execution --------
if __name__ == "__main__":
    for exp in experiments:
        # Merge defaults + experiment
        params = {**defaults, **exp}

        # Unpack parameters
        filedir    = params["filedir"]
        modname    = params["modname"]
        runname    = params["runname"]
        isSyn      = params["isSyn"]
        is3c       = params["is3c"]
        comp       = params["comp"]
        num_chains = params["num_chains"]
        totalSteps = int(params["totalSteps"])
        burnInSteps = int(params["burnInSteps"])
        nSaveModels = params["nSaveModels"]
        actionsPerStep = params["actionsPerStep"]
        maxN       = params["maxN"]

        ampRange   = tuple(params["ampRange"])
        slwRange   = tuple(params["slwRange"])
        minSpace   = params["minSpace"]

        CDopt      = params["CDopt"]
        isbp       = params["isbp"]
        freqs      = tuple(params["freqs"])
        isds       = params["isds"]

        bazRange   = tuple(params["bazRange"])
        locDiff    = params["locDiff"]
        distDiffRange = tuple(params["distDiffRange"])
        bazDiffRange  = tuple(params["bazDiffRange"])

        fitAtts    = params["fitAtts"]
        fitPhase   = params["fitPhase"]
        normOpt    = params["normOpt"]
        isMars     = params["isMars"]
        srcArray   = params["srcArray"]

        print(f"\n=== Running experiment: {modname} / {runname} ===")

        # ---- Load static data ----
        datadir = os.path.join(filedir, "SynData") if isSyn else os.path.join(filedir, "RealData")

        U_obs, Utime, CDinv, CD_sqrt_inv, metadata, is3c = prep_data(
            datadir, modname, is3c, comp, CDopt
        )
        dt = Utime[1] - Utime[0]

        # STF
        if isSyn:
            stf = np.loadtxt(os.path.join(datadir, modname, "stf.csv"), delimiter=",", skiprows=1)
        else:
            stf = create_stf(est_dom_freq(U_obs if not is3c else U_obs[:, :, 0], 1/dt), dt)
            stf_path = os.path.join(datadir, modname, "stf.csv")
            np.savetxt(stf_path, stf, delimiter=",", header="time,stf", comments="")

        stf_wid = minSpace if minSpace is not None else est_stf_wid(stf)

        # Prior
        if isSyn:
            with open(os.path.join(datadir, modname, "Prior.pkl"), "rb") as f:
                prior = pickle.load(f)
            prior.arrStd = 5.0
        else:
            if is3c:
                prior = Prior3c(
                    minSpace=stf_wid, maxN=maxN,
                    timeRange=(Utime[0], Utime[-1]), ampRange=ampRange,
                    slwRange=slwRange, bazRange=bazRange, distDiffRange=distDiffRange, bazDiffRange=bazDiffRange
                )
            else:
                prior = Prior(
                    timeRange=(Utime[0], Utime[-1]), ampRange=ampRange,
                    slwRange=slwRange, bazRange=bazRange, distDiffRange=distDiffRange, bazDiffRange=bazDiffRange
                )
            save_dir = os.path.join(filedir, "runs/syn" if isSyn else "runs/data", modname, runname)
            os.makedirs(save_dir, exist_ok=True)
            prior_path = os.path.join(save_dir, "Prior.pkl")
            if not os.path.exists(prior_path):
                with open(prior_path, "wb") as f:
                    pickle.dump(prior, f)

        # Model (synthetic only)
        model = None
        if isSyn:
            with open(os.path.join(datadir, modname, "Model.pkl"), "rb") as f:
                model = pickle.load(f)

        # Bookkeeping
        srcLat, srcLon = np.loadtxt(os.path.join(datadir, modname, "eventinfo.csv"), delimiter=",", skiprows=1)
        refLat, refLon, _, refBaz = calc_array_center(metadata, srcLat, srcLon, srcArray)
        bookkeeping = Bookkeeping(
            totalSteps=totalSteps,
            burnInSteps=burnInSteps,
            nSaveModels=nSaveModels,
            actionsPerStep=actionsPerStep,
            locDiff=locDiff,
            fitAtts=fitAtts,
            fitPhase=fitPhase,
            isMars=isMars,
            srcArray=srcArray,
            srcLat=srcLat,
            srcLon=srcLon,
            refLat=refLat,
            refLon=refLon,
            refBaz=refBaz
        )
        bk_path = os.path.join(save_dir, "Bookkeeping.pkl")
        if not os.path.exists(bk_path):
            with open(bk_path, "wb") as f:
                pickle.dump(bookkeeping, f)

        # -------- Multiprocessing setup --------
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

        # Pass experiment variables to the chain function
        exp_vars = {
            "U_obs": U_obs, "CDinv": CDinv, "CD_sqrt_inv": CD_sqrt_inv,
            "metadata": metadata, "Utime": Utime, "stf": stf,
            "prior": prior, "bookkeeping": bookkeeping,
            "filedir": filedir, "modname": modname, "runname": runname,
            "is3c": is3c, "isSyn": isSyn, "num_chains": num_chains
        }

        # Run all chains in batches
        start = time.time()
        for batch_idx in range(total_batches):
            start_chain = batch_idx * batch_size
            end_chain = min(start_chain + batch_size, num_chains)
            batch_chain_ids = list(range(start_chain, end_chain))
            print(f"Running batch {batch_idx+1}/{total_batches} with chains {batch_chain_ids}")

            with ctx.Pool(processes=len(batch_chain_ids)) as pool:
                pool.starmap(run_chain, [(cid, exp_vars) for cid in batch_chain_ids])

        end = time.time()
        print(f"Total elapsed time: {end - start:.2f} seconds")
