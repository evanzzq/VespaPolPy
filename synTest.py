import pickle, time, os
import numpy as np
from vespainv.model import Bookkeeping
from vespainv.waveformBuilder import create_U_from_model
from vespainv.rjmcmc import rjmcmc_run, rjmcmc_run3c
from vespainv.utils import generate_arr

modname = "model3"
runname = "run2.1_T=0.1_freq"
totalSteps = int(3e5)

burnInSteps = int(2e5)
nSaveModels = 100
actionsPerStep = 2

saveDir = os.path.join("./runs/syn/", modname, runname)
os.makedirs(saveDir, exist_ok=True)

start = time.time()

# ---- Load and prepare data ----
if os.path.isfile(os.path.join("./SynData/", modname, "U_syn.csv")):
    is3c = False
    U_obs = np.loadtxt(os.path.join("./SynData/", modname, "U_syn.csv"), delimiter=",")  # columns: data
else:
    is3c = True
    Z_obs = np.loadtxt(os.path.join("./SynData/", modname, "UZ_syn.csv"), delimiter=",")  # columns: data
    R_obs = np.loadtxt(os.path.join("./SynData/", modname, "UR_syn.csv"), delimiter=",")  # columns: data
    T_obs = np.loadtxt(os.path.join("./SynData/", modname, "UT_syn.csv"), delimiter=",")  # columns: data
    U_obs = np.stack([Z_obs, R_obs, T_obs], axis=-1)

Utime  = np.loadtxt(os.path.join("./SynData/", modname, "time_syn.csv"), delimiter=",")  # columns: time
stf   = np.loadtxt(os.path.join("./SynData/", modname, "stf_syn.csv"), delimiter=",", skiprows=1)  # columns: stf_time, stf
metadata = np.loadtxt(os.path.join("./SynData/", modname, "station_metadata.csv"), delimiter=",", skiprows=1)  # columns: distance, baz
# ---- Load prior ----
with open(os.path.join("./SynData/", modname, "synPrior.pkl"), "rb") as f:
    prior = pickle.load(f)
prior.arrStd = 5.0

# ---- Bookkeeping ----
bookkeeping = Bookkeeping(
    totalSteps=totalSteps,
    burnInSteps=burnInSteps,
    nSaveModels=nSaveModels,
    actionsPerStep=actionsPerStep
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