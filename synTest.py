import pickle, time, os
import numpy as np
from vespainv.model import Bookkeeping, Prior, Prior3c
from vespainv.rjmcmc import rjmcmc_run, rjmcmc_run3c
from vespainv.utils import calc_array_center, bandpass, create_stf, est_dom_freq

# ---- Parameter setup ----
isSyn = False
is3c = False # for synthetic this will be overriden
comp = "R" # only applies to real data

modname = "200705062111"
runname = "run2_R"
totalSteps = int(3e5)

burnInSteps = int(2e5)
nSaveModels = 100
actionsPerStep = 2

ampRange = (-1., 1.) # only applies to real data
slwRange = (0., 10.) # only applies to real data

isbp = True
freqs = (0.02, 1.0)

locDiff = False
distRange = (-5., -5.)
bazRange = (-5., -5.)

Temp = 0.1 # acceptance rate for worse models lower if smaller

# ---- NO EDITS NEEDED BELOW ----
if isSyn:
    datadir = "./SynData"
    saveDir = os.path.join("./runs/syn/", modname, runname)
else:
    datadir = "./RealData"
    saveDir = os.path.join("./runs/data/", modname, runname)

os.makedirs(saveDir, exist_ok=True)

start = time.time()

# ---- Load and prepare data ----
if os.path.isfile(os.path.join(datadir, modname, "U.csv")):
    is3c = False if isSyn else is3c
    U_obs = np.loadtxt(os.path.join(datadir, modname, "U.csv"), delimiter=",")  # columns: data
else:
    is3c = True if isSyn else is3c
    if is3c:
        Z_obs = np.loadtxt(os.path.join(datadir, modname, "UZ.csv"), delimiter=",")  # columns: data
        R_obs = np.loadtxt(os.path.join(datadir, modname, "UR.csv"), delimiter=",")  # columns: data
        T_obs = np.loadtxt(os.path.join(datadir, modname, "UT.csv"), delimiter=",")  # columns: data
        U_obs = np.stack([Z_obs, R_obs, T_obs], axis=-1)
    else:
        Uname = "U"+comp+".csv"
        U_obs = np.loadtxt(os.path.join(datadir, modname, Uname), delimiter=",")  # columns: data

U_obs /= np.max(np.abs(U_obs)) # normalize

Utime  = np.loadtxt(os.path.join(datadir, modname, "time.csv"), delimiter=",")  # columns: time
metadata = np.loadtxt(os.path.join(datadir, modname, "station_metadata.csv"), delimiter=",", skiprows=1)  # columns: distance, baz
dt = Utime[1] - Utime[0]

if isbp:
    U_obs = bandpass(U_obs, 1/dt, freqs[0], freqs[1])

# ---- Load (for synthetic) or prepare (for data) stf ----
if isSyn:
    stf = np.loadtxt(os.path.join(datadir, modname, "stf.csv"), delimiter=",", skiprows=1)  # columns: stf_time, stf
else:
    stf = create_stf(est_dom_freq(U_obs if not is3c else U_obs[:,:,0], 1/dt), dt)

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
            refLat=refLat, refLon=refLon, refBaz=refBaz, srcLat=srcLat, srcLon=srcLon, 
            timeRange=(Utime[0],Utime[-1]), ampRange=ampRange, slwRange=slwRange, distRange=distRange, bazRange=bazRange
            )
    else:
        prior = Prior(
            refLat=refLat, refLon=refLon, refBaz=refBaz, srcLat=srcLat, srcLon=srcLon, 
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
    Temp=Temp
)

# ---- Run RJMCMC ----
if is3c:
    samples, logL_trace = rjmcmc_run3c(U_obs, metadata, Utime, stf, prior, bookkeeping, saveDir)
else:
    samples, logL_trace = rjmcmc_run(U_obs, metadata, Utime, stf, prior, bookkeeping, saveDir)

# ---- Save output ----
with open(os.path.join(saveDir, "ensemble.pkl"), "wb") as f:
    pickle.dump(samples, f)
np.savetxt(os.path.join(saveDir, "log_likelihood.txt"), logL_trace)

end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")