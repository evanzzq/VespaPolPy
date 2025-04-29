from vespainv.visualization import plot_ensemble_vespagram
import pickle, os
import numpy as np

isSyn = False
modname = "200705062111"
runname = "run2_R"

dataDir = "./SynData/" if isSyn else "./RealData/"
resDir = "./runs/syn/" if isSyn else "./runs/data/"

with open(os.path.join(resDir, modname, runname, "ensemble.pkl"), "rb") as f:
    ensemble = pickle.load(f)
with open(os.path.join(dataDir, modname, "Prior.pkl"), "rb") as f:
    prior = pickle.load(f)
if isSyn:
    with open(os.path.join(dataDir, modname, "Model.pkl"), "rb") as f:
        model = pickle.load(f)
Utime  = np.loadtxt(os.path.join(dataDir, modname, "time.csv"), delimiter=",")  # columns: time

plot_ensemble_vespagram(ensemble, Utime, prior, amp_weighted=True, true_model=model if isSyn else None)