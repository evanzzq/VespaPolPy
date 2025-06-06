import pickle, time, os
import numpy as np
from vespainv.model import Bookkeeping, Prior, Prior3c
from vespainv.rjmcmc import rjmcmc_run, rjmcmc_run3c
from vespainv.utils import calc_array_center, create_stf, est_stf_wid, est_dom_freq, prep_data
from parameter_setup import *

# ---- Parameter setup ----
# ---- Now done in parameter_setup.py ----

# ---- NO EDITS NEEDED BELOW ----
# ---- Define and make directories ----
if isSyn:
    datadir = os.path.join(filedir, "SynData")
    saveDir = os.path.join(filedir, "runs/syn/", modname, runname)
else:
    datadir = os.path.join(filedir, "RealData")
    saveDir = os.path.join(filedir, "runs/data/", modname, runname)

os.makedirs(saveDir, exist_ok=True)

start = time.time()

# ---- Load and prepare data ----
if isSyn: isbp = False
U_obs, Utime, CDinv, metadata, is3c = prep_data(datadir, modname, is3c, comp, isbp, freqs, isds)
dt = Utime[1] - Utime[0]

# ---- Load (for synthetic) or prepare and save (for data) stf ----
if isSyn:
    stf = np.loadtxt(os.path.join(datadir, modname, "stf.csv"), delimiter=",", skiprows=1)  # columns: stf_time, stf
else:
    stf = create_stf(est_dom_freq(U_obs if not is3c else U_obs[:,:,0], 1/dt), dt)
    np.savetxt(os.path.join(datadir, modname, "stf.csv"), stf, delimiter=",", header="time,stf", comments="")

stf_wid = minSpace if minSpace is not None else est_stf_wid(stf)

# ---- Load (for synthetic) or prepare and save (for data) prior ----
if isSyn:
    with open(os.path.join(datadir, modname, "Prior.pkl"), "rb") as f:
        prior = pickle.load(f)
    prior.arrStd = 5.0
else:
    srcLat, srcLon = np.loadtxt(os.path.join(datadir, modname, "eventinfo.csv"), delimiter=",", skiprows=1)
    refLat, refLon, refBaz = calc_array_center(metadata, srcLat, srcLon)
    if is3c:
        prior = Prior3c(
            refLat=refLat, refLon=refLon, refBaz=refBaz, srcLat=srcLat, srcLon=srcLon, minSpace=stf_wid, maxN=maxN,
            timeRange=(Utime[0],Utime[-1]), ampRange=ampRange, slwRange=slwRange, distRange=distRange, bazRange=bazRange
            )
    else:
        prior = Prior(
            refLat=refLat, refLon=refLon, refBaz=refBaz, srcLat=srcLat, srcLon=srcLon, minSpace=stf_wid, maxN=maxN,
            timeRange=(Utime[0],Utime[-1]), ampRange=ampRange, slwRange=slwRange, distRange=distRange, bazRange=bazRange
            )
    with open(os.path.join(datadir, modname, "Prior.pkl"), "wb") as f:
        pickle.dump(prior, f)


# ---- Bookkeeping ----
bookkeeping = Bookkeeping(
    totalSteps=totalSteps,
    burnInSteps=burnInSteps,
    nSaveModels=nSaveModels,
    actionsPerStep=actionsPerStep,
    locDiff=locDiff,
    fitNoise=fitNoise,
    fitAtts=fitAtts
)

# ---- Run RJMCMC ----
if is3c:
    samples, logL_trace = rjmcmc_run3c(U_obs, CDinv, metadata, Utime, stf, prior, bookkeeping, saveDir)
else:
    samples, logL_trace = rjmcmc_run(U_obs, CDinv, metadata, Utime, stf, prior, bookkeeping, saveDir)

# ---- Save output ----
with open(os.path.join(saveDir, "ensemble.pkl"), "wb") as f:
    pickle.dump(samples, f)
np.savetxt(os.path.join(saveDir, "log_likelihood.txt"), logL_trace)

end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")