from vespainv.visualization import (
    plot_ensemble_vespagram,
    plot_seismogram_compare,
    phase_count_distribution_by_model,
    plot_chain_convergence_by_region,
)
from vespainv.config import load_config, load_workspace
from vespainv.utils import prep_data
import pickle, os, re, argparse
import numpy as np
import matplotlib.pyplot as plt

workspace = load_workspace("configs/workspace.yaml")
data_root = workspace["paths"]["real_data_root"]
runs_root = workspace["paths"]["runs_root"]

# ---- Moveout correction click ----
third_click = False

# ---- Plot modes ----
run_standard_plots = True
run_convergence_analysis = False

# ---- Manually input model and run OR select from yaml setup file? ----
use_manual = True
seis_mode = "P" # P, S, or All - show selected wave type in waveforms

# The following will be overridden if use_manual == False
dataset    = "IU_RSSD_locbox_-30_-20_-65_-55_depth_600_T_600_750"
runname    = "test_run_8c"
is3c       = True
comp       = "Z"
CDopt      = 3
isMars     = False
isbp       = False
freqs      = (0.02, 0.5)    # Bandpass frequencies
fitAtts    = True
fitPhase   = True

# -------- Selection options --------
chains_to_plot = None          # Example: [0, 2] to select specific chains by index; None uses all chains
likelihood_threshold = None #-5.5e4     # Example: -5000 to select chains with final LL > threshold

# ---- Code-defined convergence boxes ----
# Set any bound to None to use the corresponding prior range. For 3C inversions,
# wave_type can be "P", "S", or None.
convergence_regions = [
    {"name": "S", "tmin": 20.0, "tmax": 40.0, "pmin": 10., "pmax": 12., "wave_type": "S"},
    {"name": "SdS", "tmin": 35.0, "tmax": 50.0, "pmin": 8., "pmax": 10., "wave_type": "S"},
    {"name": "ScS", "tmin": 50.0, "tmax": 65.0, "pmin": 7., "pmax": 10., "wave_type": "S"},
    # {"name": "PcP", "tmin": 45, "tmax": 60.0, "pmin": 3., "pmax": 5., "wave_type": "P"}
]
convergence_bins = 20
save_convergence_plots = False

if not use_manual:
    # ---- Parse config file ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="parameter_setup.yaml", help="YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    defaults = config.get("defaults", {})
    experiments = config["experiments"]
    paths = config.get("paths", {})
    data_root = defaults.get("data_root", paths.get("real_data_root", data_root))
    runs_root = defaults.get("runs_root", paths.get("runs_root", runs_root))

    all_exp_names = [None] * len(experiments)

    for iexp, exp in enumerate(experiments):
        # Merge defaults + experiment
        params = {**defaults, **exp}
        # Unpack parameters
        dataset = params["dataset"]
        runname = params["runname"]

        all_exp_names[iexp] = f"{dataset}: {runname}"

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
    dataset    = selected_params["dataset"]
    runname    = selected_params["runname"]
    is3c       = selected_params["is3c"]
    comp       = selected_params["comp"]
    CDopt      = selected_params["CDopt"]
    isMars     = selected_params["isMars"]
    fitAtts    = selected_params["fitAtts"]

print(f"Selected: {dataset}, {runname}")

# ---- Paths ----
datadir = data_root
resdir = runs_root
run_path = os.path.join(resdir, dataset, runname)

# ---- Load Bookkeeping ----
with open(os.path.join(run_path, "Bookkeeping.pkl"), "rb") as f:
    bookkeeping = pickle.load(f)

# ---- Detect chains ----
chain_dirs = sorted(
    [os.path.join(run_path, d) for d in os.listdir(run_path)
     if os.path.isdir(os.path.join(run_path, d)) and re.match(r"chain_\d+", d)],
    key=lambda d: int(re.search(r"chain_(\d+)", d).group(1))
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
chain_labels = []
if selected_dirs:
    for cdir in selected_dirs:
        with open(os.path.join(cdir, "ensemble.pkl"), "rb") as f:
            ensembles.append(pickle.load(f))
        chain_labels.append(int(re.search(r"chain_(\d+)", cdir).group(1)))
else:
    # single-chain case
    with open(os.path.join(run_path, "ensemble.pkl"), "rb") as f:
        ensembles.append(pickle.load(f))
    chain_labels.append("single")

# Combine all ensembles into one list
ensemble = sum(ensembles, [])

# ---- Load prior ----
with open(os.path.join(run_path, "Prior.pkl"), "rb") as f:
    prior = pickle.load(f)
model = None

# ---- Load observed data & STF only when needed for standard plots ----
if run_standard_plots:
    U_obs, Utime, _, _, metadata, is3c_flag = prep_data(
        datadir, dataset, is3c, comp, CDopt, is_mars=bookkeeping.isMars, src_array=bookkeeping.srcArray
    )
    stf = np.loadtxt(os.path.join(datadir, dataset, "stf.csv"), delimiter=",", skiprows=1)

# ---- Extract burn-in and total steps ONCE (same for all chains) ----
burn = bookkeeping.burnInSteps
tot  = bookkeeping.totalSteps

# ---- Plot log-likelihood, Nphase vs steps, and histogram of Nphase (post burn-in) ----
fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(8, 10),
    gridspec_kw={'height_ratios': [1, 1, 1]}
)

all_nop_postburn = []  # collect Nphase after burn-in from all chains

if selected_dirs:
    # ---- Multi-chain case ----
    for cdir in selected_dirs:
        ll_path = os.path.join(cdir, "log_likelihood.txt")
        nop_path = os.path.join(cdir, "Nphase.txt")

        # log-likelihood
        if os.path.isfile(ll_path):
            ll = np.loadtxt(ll_path)
            if ll.ndim > 1:
                ll = ll[:, 0]
            steps_ll = np.arange(len(ll))
            ax1.plot(steps_ll, ll, 'k-')  # logL, black solid
        else:
            print(f"Warning: log_likelihood.txt not found in {cdir}, skipping logL.")

        # Nphase
        if os.path.isfile(nop_path):
            nop = np.loadtxt(nop_path)
            steps_nop = np.arange(len(nop))
            ax2.plot(steps_nop, nop, 'b-')  # Nphase, blue solid

            # collect post-burnin Nphase
            i0 = min(burn, len(nop))
            i1 = min(tot,  len(nop))
            if i1 > i0:
                all_nop_postburn.append(nop[i0:i1])
        else:
            print(f"Warning: Nphase.txt not found in {cdir}, skipping Nphase.")

else:
    # ---- Single-chain case ----
    ll_path = os.path.join(run_path, "log_likelihood.txt")
    nop_path = os.path.join(run_path, "Nphase.txt")

    # log-likelihood
    if os.path.isfile(ll_path):
        ll = np.loadtxt(ll_path)
        if ll.ndim > 1:
            ll = ll[:, 0]
        steps_ll = np.arange(len(ll))
        ax1.plot(steps_ll, ll, 'k-')
    else:
        print("Warning: log_likelihood.txt not found for single-chain run.")

    # Nphase
    if os.path.isfile(nop_path):
        nop = np.loadtxt(nop_path)
        steps_nop = np.arange(len(nop))
        ax2.plot(steps_nop, nop, 'b-')

        # collect post-burnin Nphase
        i0 = min(burn, len(nop))
        i1 = min(tot,  len(nop))
        if i1 > i0:
            all_nop_postburn.append(nop[i0:i1])
    else:
        print("Warning: Nphase.txt not found for single-chain run.")

# ---- Axis labels / titles ----
ax1.set_ylabel("Log Likelihood")
ax1.set_title("Log-likelihood vs. MCMC step")
ax1.grid(True)

ax2.set_ylabel("Nphase")
ax2.set_title("Number of Phases vs. MCMC step")
ax2.grid(True)
ax2.set_xlabel("Step")

# ---- Optional: log-scale on x-axis for step ----
ax1.set_xscale("log")
# ax2.set_xscale("log")

# ---- Histogram of Nphase after burn-in ----
if all_nop_postburn:
    nop_post = np.concatenate(all_nop_postburn)
    bins = np.arange(nop_post.min() - 0.5, nop_post.max() + 1.5, 1)
    ax3.hist(nop_post, bins=bins, edgecolor='k', alpha=0.7)
    ax3.set_xlabel("Nphase (post burn-in)")
    ax3.set_ylabel("Count")
    ax3.set_title("Histogram of Nphase after burn-in")
    ax3.grid(True)
else:
    ax3.text(0.5, 0.5, "No Nphase data post burn-in",
             ha='center', va='center', transform=ax3.transAxes)
    ax3.set_axis_off()

fig.tight_layout()
# (No saving here)

# # ---- Optional statistics ----
# summary = phase_count_distribution_by_model(
#     ensemble,
#     tmin=20, tmax=40,
#     pmin=10, pmax=12,
#     is3c=True,          # if you have wvtype and want filtering
#     wave_type="S",      # or None to include all
#     make_plots=True
# )

# print(summary)

if run_convergence_analysis:
    convergence_save_dir = (
        os.path.join(run_path, "convergence_plots")
        if save_convergence_plots else None
    )
    convergence_summary = plot_chain_convergence_by_region(
        ensembles=ensembles,
        chain_labels=chain_labels,
        regions=convergence_regions,
        prior=prior,
        is3c=is3c,
        bins=convergence_bins,
        save_dir=convergence_save_dir,
    )

if run_standard_plots:
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
        stf=stf, bookkeeping=bookkeeping, moveout_pt=moveout_pt, mode=seis_mode
    )
# plot_seismogram_compare(
#     U=U_obs, time=Utime, offset=1.5,
#     ensemble=[ensemble[4]], prior=prior, metadata=metadata,
#     stf=stf, bookkeeping=bookkeeping, moveout_pt=moveout_pt
# )

plt.show()
