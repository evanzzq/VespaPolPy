from vespainv.visualization import plot_ensemble_vespagram, plot_seismogram_compare
from vespainv.utils import prep_data
import pickle, os
import numpy as np
import matplotlib.pyplot as plt
from parameter_setup import *

datadir = os.path.join(filedir, "SynData") if isSyn else os.path.join(filedir, "RealData")
resdir = os.path.join(filedir, "runs/syn") if isSyn else os.path.join(filedir, "runs/data")

with open(os.path.join(resdir, modname, runname, "ensemble.pkl"), "rb") as f:
    ensemble = pickle.load(f)
with open(os.path.join(datadir, modname, "Prior.pkl"), "rb") as f:
    prior = pickle.load(f)
if isSyn:
    with open(os.path.join(datadir, modname, "Model.pkl"), "rb") as f:
        model = pickle.load(f)

U_obs, Utime, metadata, is3c = prep_data(datadir, modname, is3c, comp, isbp, freqs, isds)
stf = np.loadtxt(os.path.join(datadir, modname, "stf.csv"), delimiter=",", skiprows=1)

plot_ensemble_vespagram(ensemble, Utime, prior, amp_weighted=False, true_model=model if isSyn else None, is3c=is3c)
plot_seismogram_compare(U=U_obs, time=Utime, offset=1.5, ensemble=ensemble, prior=prior, metadata=metadata, stf=stf)
plot_seismogram_compare(U=U_obs, time=Utime, offset=1.5, ensemble=[ensemble[-1]], prior=prior, metadata=metadata, stf=stf)
plt.show()