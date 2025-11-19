from vespainv.visualization import plot_ensemble_vespagram, plot_seismogram_compare
from vespainv.utils import prep_data
import pickle, os, re, argparse, yaml
import numpy as np
import matplotlib.pyplot as plt

# ---- File Dir (Mac/PC) override ----
# filedir = "H:\My Drive\Research\VespaPolPy"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

# ---- Moveout correction click ----
third_click = False

# ---- Manually input model and run OR select from yaml setup file? ----
use_manual = False

# The following will be overridden if use_manual == False
modname    = "201111221848_P_45_48"
runname    = "run5_3c_CD_fit_multichain_L1_maxN50_action1"
isSyn      = False
is3c       = True # for synthetic this will be overriden
comp       = "Z" # only applies to real data
CDopt      = 3 # 0 - False (single Sigma value), 1 - Empirical, 2 - Robust, 3 - Fit
isbp       = False
freqs      = (0.02, 0.5)    # Bandpass frequencies
fitAtts    = False
fitPhase   = True

# -------- Selection options --------
chains_to_plot = None          # Example: [0, 2] to select specific chains by index
likelihood_threshold = None #-5.5e4     # Example: -5000 to select chains with final LL > threshold

if not use_manual:
    # ---- Parse config file ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="parameter_setup.yaml", help="YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    experiments = config["experiments"]

    all_exp_names = [None] * len(experiments)

    for iexp, exp in enumerate(experiments):
        # Merge defaults + experiment
        params = {**defaults, **exp}
        # Unpack parameters
        modname = params["modname"]
        runname = params["runname"]

        all_exp_names[iexp] = f"{modname}: {runname}"

    # ---- Print and ask for selection ----
    print("\nAvailable experiments:")
    for i, name in enumerate(all_exp_names):
        print(f"{i}: {name}")

    choice = int(input("\nSelect an experiment index to plot: "))
    if choice < 0 or choice >= len(experiments):
        raise ValueError("Invalid choice!")

    # ---- Selected experiment ----
    selected_params = {**defaults, **experiments[choice]}

    # Unpack parameters
    modname    = selected_params["modname"]
    runname    = selected_params["runname"]
    isSyn      = selected_params["isSyn"]
    is3c       = selected_params["is3c"]
    comp       = selected_params["comp"]
    CDopt      = selected_params["CDopt"]
    fitAtts    = selected_params["fitAtts"]

print(f"Selected: {modname}, {runname}")

# ---- Paths ----
datadir = os.path.join(filedir, "SynData") if isSyn else os.path.join(filedir, "RealData")
resdir = os.path.join(filedir, "runs/syn") if isSyn else os.path.join(filedir, "runs/data")
run_path = os.path.join(resdir, modname, runname)

# ---- Detect chains ----
chain_dirs = sorted(
    [os.path.join(run_path, d) for d in os.listdir(run_path)
     if os.path.isdir(os.path.join(run_path, d)) and re.match(r"chain_\d+", d)]
)

# ---- Select chains ----
selected_dirs = []
if chain_dirs:
    candidate_dirs = chain_dirs

    # By index
    if chains_to_plot is not None:
        candidate_dirs = []
        for cid in chains_to_plot:
            cpath = os.path.join(run_path, f"chain_{cid}")
            if os.path.isdir(cpath):
                candidate_dirs.append(cpath)
            else:
                print(f"Warning: Chain {cid} not found, skipping.")

    # By likelihood threshold
    if likelihood_threshold is not None:
        filtered_dirs = []
        for cdir in candidate_dirs:
            ll_path = os.path.join(cdir, "log_likelihood.txt")
            if os.path.isfile(ll_path):
                try:
                    ll_values = np.loadtxt(ll_path)
                    final_ll = ll_values[-1] if ll_values.ndim == 1 else ll_values[-1, 0]
                    if final_ll > likelihood_threshold:
                        filtered_dirs.append(cdir)
                except Exception as e:
                    print(f"Warning: Could not read likelihood from {cdir}: {e}")
            else:
                print(f"Warning: log_likelihood.txt not found in {cdir}, skipping threshold check.")
        candidate_dirs = filtered_dirs

    selected_dirs = candidate_dirs
else:
    selected_dirs = []

# ---- Report selected chains ----
if selected_dirs:
    print("\nSelected chains for plotting:")
    for cdir in selected_dirs:
        cid = int(re.search(r"chain_(\d+)", cdir).group(1))
        ll_path = os.path.join(cdir, "log_likelihood.txt")
        if os.path.isfile(ll_path):
            ll_values = np.loadtxt(ll_path)
            final_ll = ll_values[-1] if ll_values.ndim == 1 else ll_values[-1, 0]
            print(f"  Chain {cid}: final log-likelihood = {final_ll:.4f}")
        else:
            print(f"  Chain {cid}: (no log_likelihood.txt)")
else:
    print("\nNo multi-chain directories found. Using single-chain results.")

# ---- Load ensemble(s) ----
ensembles = []
if selected_dirs:
    for cdir in selected_dirs:
        with open(os.path.join(cdir, "ensemble.pkl"), "rb") as f:
            ensembles.append(pickle.load(f))
else:
    # single-chain case
    with open(os.path.join(run_path, "ensemble.pkl"), "rb") as f:
        ensembles.append(pickle.load(f))

# Combine all ensembles into one list
ensemble = sum(ensembles, [])

# ---- Load prior and (if synthetic) true model ----
with open(os.path.join(run_path, "Prior.pkl"), "rb") as f:
    prior = pickle.load(f)
model = None
if isSyn:
    with open(os.path.join(datadir, modname, "Model.pkl"), "rb") as f:
        model = pickle.load(f)

# ---- Load observed data & STF ----
U_obs, Utime, _, _, metadata, is3c_flag = prep_data(datadir, modname, is3c, comp, CDopt)
stf = np.loadtxt(os.path.join(datadir, modname, "stf.csv"), delimiter=",", skiprows=1)

# ---- Load Bookkeeping ----
with open(os.path.join(run_path, "Bookkeeping.pkl"), "rb") as f:
    bookkeeping = pickle.load(f)

# ---- Plot ----
moveout_pt = plot_ensemble_vespagram(
    ensemble, Utime, prior,
    amp_weighted=True,
    true_model=model,
    is3c=is3c_flag,
    third_click=third_click
)
plot_seismogram_compare(
    U=U_obs, time=Utime, offset=1.5,
    ensemble=ensemble, prior=prior, metadata=metadata,
    stf=stf, bookkeeping=bookkeeping, moveout_pt=moveout_pt
)
plot_seismogram_compare(
    U=U_obs, time=Utime, offset=1.5,
    ensemble=[ensemble[4]], prior=prior, metadata=metadata,
    stf=stf, bookkeeping=bookkeeping, moveout_pt=moveout_pt
)

plt.show()
