import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from vespainv.model import VespaModel, Prior, VespaModel3c, Prior3c, Bookkeeping
from vespainv.waveformBuilder import (
    create_U_from_model_freqdomain,
    create_U_from_model_3c_freqdomain,
)
from vespainv.utils import dest_point


# ============================================================
# User parameters
# ============================================================
filedir = "H:/My Drive/Research/VespaPolPy"
# filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

modname = "model_test_P_5_long0"
Nphase = 5
maxN = 20
is3c = True
ampRange = (0, 1)
slwRange = (-2.0, 2.0)

# STF / time axis
f0 = 0.5
dt = 0.2
tmax = 80

# Array geometry
srcLat = 0.0
srcLon = 0.0
base_dist = 110.0
base_baz = 30.0
base_az = (base_baz + 180) % 360
Ntrace = 5
refLat, refLon = dest_point(srcLat, srcLon, base_az, base_dist)

# Optional location perturbation for source-array / Mars cases
locDiff = False
distDiff = np.random.uniform(-3.0, 3.0, Ntrace)
bazDiff = np.random.uniform(-3.0, 3.0, Ntrace)

# Model definition
# If False, a random model is drawn from the prior
# If True, the arrays below are used directly
# NOTE: This script no longer saves Prior/Bookkeeping objects with the dataset.
defAll = True
arr = np.array([20,30,40,50,60])
slw = np.array([0,0,0,0,0])
amp = np.array([0.5,-0.3,0.8,0.4,-0.5])
azi = np.array([0,0,0,0,0])
ph_hh = np.array([0,0,0,0,0])
ph_vh = np.array([0,0,0,0,0])
atts = np.array([1,1,1,1,1])
wvtype = np.array([1,1,1,1,1])

# Noise setup
# base_sigma is the sigma you will use in the inversion.
# true_loge is the true hyperparameter used to scale the noise level.
# The actual noise scale added here is:
#     sigma_eff = base_sigma * exp(0.5 * true_loge)
# This matches the current RJMCMC scaling:
#   L2 : CDinv      -> exp(-loge)
#   L1 : CD_sqrtinv -> exp(-0.5 * loge)
make_clean_dataset = True
make_noisy_datasets = True
noise_types = ("L2", "L1")  # any subset of {"L2", "L1"}
base_sigma = 0.02
true_loge = 1.0
noise_seed = 42

# For testing: keep False unless you explicitly want identical noise on all traces.
# False -> each trace/component/sample gets an independent realization.
# True  -> same realization is broadcast to all traces (still separate between components for 3C).
same_noise_across_traces = False

# Plotting
show_plot = True
save_plot = False


# ============================================================
# Helpers
# ============================================================
def build_stf(f0, dt):
    stf_time_0 = np.arange(-4.0 / f0, 4.0 / f0 + dt, dt)
    stf_0 = np.exp(-stf_time_0**2 / (2.0 * (1.0 / (2.0 * np.pi * f0)) ** 2))
    stf_time = stf_time_0[:-1]
    stf = np.diff(stf_0) / np.diff(stf_time_0)
    stf = stf / np.max(np.abs(stf))
    return stf_time, stf


def generate_station_metadata(srcLat, srcLon, base_dist, base_baz, Ntrace):
    dists = base_dist + np.random.uniform(-2.0, 2.0, Ntrace)
    bazs = base_baz + np.random.uniform(-2.0, 2.0, Ntrace)
    stlas = np.zeros_like(dists)
    stlos = np.zeros_like(dists)

    for ista in range(Ntrace):
        stlas[ista], stlos[ista] = dest_point(
            srcLat, srcLon, (bazs[ista] + 180) % 360, dists[ista]
        )

    idx = np.argsort(dists)
    dists = dists[idx]
    bazs = bazs[idx]
    stlas = stlas[idx]
    stlos = stlos[idx]

    station_metadata_db = np.column_stack((dists, bazs))
    station_metadata = np.column_stack((stlas, stlos))
    return station_metadata, station_metadata_db



def make_model(time):
    if is3c:
        prior = Prior3c(
            maxN=maxN,
            timeRange=(time[0], time[-1]),
            ampRange=ampRange,
            slwRange=slwRange,
        )
        if defAll:
            model = VespaModel3c(
                Nphase=Nphase,
                Ntrace=Ntrace,
                arr=arr,
                slw=slw,
                amp=amp,
                azi=azi,
                ph_hh=ph_hh,
                ph_vh=ph_vh,
                atts=atts,
                wvtype=wvtype,
                loge=true_loge,
            )
        else:
            model = VespaModel3c.create_random(
                Nphase=Nphase,
                Ntrace=Ntrace,
                time=time,
                prior=prior,
                arr=arr if arr is not None else None,
            )
            model.loge = true_loge
    else:
        prior = Prior(
            maxN=maxN,
            timeRange=(time[0], time[-1]),
            ampRange=ampRange,
            slwRange=slwRange,
        )
        if defAll:
            model = VespaModel(
                Nphase=Nphase,
                Ntrace=Ntrace,
                arr=arr,
                slw=slw,
                amp=amp,
                atts=atts,
                loge=true_loge,
            )
        else:
            model = VespaModel.create_random(
                Nphase=Nphase,
                Ntrace=Ntrace,
                time=time,
                prior=prior,
                arr=arr if arr is not None else None,
            )
            model.loge = true_loge

        if locDiff:
            model.distDiff = distDiff.copy()
            model.bazDiff = bazDiff.copy()

    return model, prior



def forward_model(model, station_metadata, time, stf_time, stf):
    bookkeeping = Bookkeeping(
        refLat=refLat,
        refLon=refLon,
        refBaz=base_baz,
        srcLat=srcLat,
        srcLon=srcLon,
        locDiff=locDiff,
        fitAtts=False,
        fitPhase=True,
        fitLoge=True,
        normOpt=1,
        isMars=False,
    )

    if is3c:
        return create_U_from_model_3c_freqdomain(
            model, station_metadata, time, stf_time, stf, bookkeeping
        )
    return create_U_from_model_freqdomain(
        model, station_metadata, time, stf_time, stf, bookkeeping
    )



def effective_sigma(base_sigma, loge):
    return float(base_sigma * np.exp(0.5 * loge))



def _broadcast_same_noise(base_noise, full_shape):
    if len(full_shape) == 3:
        return np.broadcast_to(base_noise[:, np.newaxis, :], full_shape).copy()
    return np.broadcast_to(base_noise[:, np.newaxis], full_shape).copy()



def generate_noise(shape, noise_type, base_sigma, true_loge, rng, same_across_traces=False):
    sigma_eff = effective_sigma(base_sigma, true_loge)
    noise_type = noise_type.upper()

    if noise_type == "L2":
        if same_across_traces:
            base_shape = (shape[0], shape[2]) if len(shape) == 3 else (shape[0],)
            base_noise = rng.normal(loc=0.0, scale=sigma_eff, size=base_shape)
            noise = _broadcast_same_noise(base_noise, shape)
        else:
            noise = rng.normal(loc=0.0, scale=sigma_eff, size=shape)

        info = {
            "noise_type": "L2",
            "distribution": "Gaussian",
            "base_sigma_for_inversion": base_sigma,
            "true_loge": true_loge,
            "effective_sigma_added": sigma_eff,
            "same_noise_across_traces": same_across_traces,
        }
        return noise, info

    if noise_type == "L1":
        # Here sigma_eff is treated as the Laplace scale used by the inversion-side
        # whitening (not the standard deviation of the Laplace distribution).
        # Therefore std(noise) = sqrt(2) * sigma_eff.
        if same_across_traces:
            base_shape = (shape[0], shape[2]) if len(shape) == 3 else (shape[0],)
            base_noise = rng.laplace(loc=0.0, scale=sigma_eff, size=base_shape)
            noise = _broadcast_same_noise(base_noise, shape)
        else:
            noise = rng.laplace(loc=0.0, scale=sigma_eff, size=shape)

        info = {
            "noise_type": "L1",
            "distribution": "Laplace",
            "base_sigma_for_inversion": base_sigma,
            "true_loge": true_loge,
            "laplace_scale_added": sigma_eff,
            "laplace_std_added": np.sqrt(2.0) * sigma_eff,
            "same_noise_across_traces": same_across_traces,
        }
        return noise, info

    raise ValueError(f"Unsupported noise_type: {noise_type}")



def save_common_files(out_dir, time, stf_time, stf, station_metadata, station_metadata_db):
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, "time.csv"), time, delimiter=",")
    np.savetxt(
        os.path.join(out_dir, "stf.csv"),
        np.column_stack([stf_time, stf]),
        delimiter=",",
        header="time,stf",
        comments="",
    )
    np.savetxt(
        os.path.join(out_dir, "station_metadata.csv"),
        station_metadata,
        delimiter=",",
        header="lat,lon",
        comments="",
    )
    np.savetxt(
        os.path.join(out_dir, "station_metadata_db.csv"),
        station_metadata_db,
        delimiter=",",
        header="dist_deg,baz",
        comments="",
    )



def save_waveforms(out_dir, U, suffix=""):
    if is3c:
        np.savetxt(os.path.join(out_dir, f"UZ{suffix}.csv"), U[:, :, 0], delimiter=",")
        np.savetxt(os.path.join(out_dir, f"UR{suffix}.csv"), U[:, :, 1], delimiter=",")
        np.savetxt(os.path.join(out_dir, f"UT{suffix}.csv"), U[:, :, 2], delimiter=",")
    else:
        np.savetxt(os.path.join(out_dir, f"U{suffix}.csv"), U, delimiter=",")



def write_truth_summary(out_dir, model, base_sigma, true_loge, noise_info=None):
    sigma_eff = effective_sigma(base_sigma, true_loge)
    with open(os.path.join(out_dir, "truth_summary.txt"), "w") as f:
        f.write("=== Synthetic Truth Summary ===\n")
        f.write(f"Number of phases: {model.Nphase}\n")
        f.write(f"Number of traces: {model.Ntrace}\n")
        f.write(f"3-component data: {is3c}\n")
        f.write(f"Sampling interval dt: {dt}\n")
        f.write(f"Time range: 0 to {tmax} s\n")
        f.write(f"Source-time function: Gaussian derivative, f0 = {f0} Hz\n")
        f.write(f"base_sigma_for_inversion: {base_sigma}\n")
        f.write(f"true_loge: {true_loge}\n")
        f.write(f"effective_sigma = base_sigma * exp(0.5 * true_loge): {sigma_eff}\n")
        if noise_info is not None:
            for key, val in noise_info.items():
                f.write(f"{key}: {val}\n")
        f.write("\n--- Arrival Times ---\n")
        f.write(np.array2string(model.arr, separator=", ") + "\n\n")
        f.write("--- Slowness ---\n")
        f.write(np.array2string(model.slw, separator=", ") + "\n\n")
        f.write("--- Amplitudes ---\n")
        f.write(np.array2string(model.amp, separator=", ") + "\n\n")
        if hasattr(model, "azi"):
            f.write("--- S polarization beta ---\n")
            f.write(np.array2string(model.azi, separator=", ") + "\n\n")
        if hasattr(model, "ph_hh"):
            f.write("--- Phase Difference: HH ---\n")
            f.write(np.array2string(model.ph_hh, separator=", ") + "\n\n")
        if hasattr(model, "ph_vh"):
            f.write("--- Phase Difference: VH ---\n")
            f.write(np.array2string(model.ph_vh, separator=", ") + "\n\n")
        if hasattr(model, "atts"):
            f.write("--- Attenuation (t* for S) ---\n")
            f.write(np.array2string(model.atts, separator=", ") + "\n\n")
        if hasattr(model, "wvtype"):
            f.write("--- Wave Type (P = 1, S = 0) ---\n")
            f.write(np.array2string(model.wvtype, separator=", ") + "\n\n")
        if locDiff:
            f.write("--- Distance perturbation (deg) ---\n")
            f.write(np.array2string(model.distDiff, separator=", ") + "\n\n")
            f.write("--- BAZ perturbation (deg) ---\n")
            f.write(np.array2string(model.bazDiff, separator=", ") + "\n\n")



def plot_waveforms(U, time, station_metadata_db=None, title="Synthetic Seismograms"):
    if is3c:
        components = ["Z", "R", "T"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 10), sharex=True)
        offset = 1.2 * np.max(np.abs(U)) if np.max(np.abs(U)) > 0 else 1.0
        n_traces = U.shape[1]
        for i, ax in enumerate(axes):
            for j in range(n_traces):
                ax.plot(time, U[:, j, i] + j * offset, color="black")
            ax.set_ylabel(f"{components[i]} amplitude (offset)")
            ax.set_title(f"{components[i]} component")
            ax.grid(True)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title)
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 6))
        n_traces = U.shape[1]
        offset = 1.2 * np.max(np.abs(U)) if np.max(np.abs(U)) > 0 else 1.0
        for i in range(n_traces):
            plt.plot(time, U[:, i] + i * offset, color="black")
            if station_metadata_db is not None:
                dist, baz = station_metadata_db[i, :]
                plt.text(time[-1] + 0.5, i * offset, f"{dist:.1f}°, {baz:.0f}°", va="center", fontsize=8)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (offset by trace index)")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()


# ============================================================
# Main
# ============================================================
def main():
    time = np.arange(0, tmax, dt)
    stf_time, stf = build_stf(f0, dt)
    station_metadata, station_metadata_db = generate_station_metadata(
        srcLat, srcLon, base_dist, base_baz, Ntrace
    )
    model, _ = make_model(time)

    synDir = os.path.join(filedir, "SynData", modname)

    # Clean synthetic data
    U_clean = forward_model(model, station_metadata, time, stf_time, stf)

    if make_clean_dataset:
        save_common_files(synDir, time, stf_time, stf, station_metadata, station_metadata_db)
        save_waveforms(synDir, U_clean)
        write_truth_summary(synDir, model, base_sigma, true_loge)
        with open(os.path.join(synDir, "Model.pkl"), "wb") as f:
            pickle.dump(model, f)

    if show_plot:
        plot_waveforms(U_clean, time, station_metadata_db=station_metadata_db, title=f"Clean synthetic: {modname}")
        if save_plot:
            plt.savefig(os.path.join(synDir, "synthetics_clean.png"), dpi=200, bbox_inches="tight")
        plt.show()

    # Noisy synthetic data
    if make_noisy_datasets:
        rng = np.random.default_rng(noise_seed)
        for noise_type in noise_types:
            noise_type = noise_type.upper()
            out_dir = os.path.join(filedir, "SynData", f"{modname}_noisy_{noise_type}")
            noise, noise_info = generate_noise(
                U_clean.shape,
                noise_type,
                base_sigma,
                true_loge,
                rng,
                same_across_traces=same_noise_across_traces,
            )
            U_noisy = U_clean + noise

            save_common_files(out_dir, time, stf_time, stf, station_metadata, station_metadata_db)
            save_waveforms(out_dir, U_noisy)
            save_waveforms(out_dir, noise, suffix="_noise")
            write_truth_summary(out_dir, model, base_sigma, true_loge, noise_info=noise_info)
            with open(os.path.join(out_dir, "Model.pkl"), "wb") as f:
                pickle.dump(model, f)

            print(f"[INFO] Saved noisy dataset: {out_dir}")
            print(f"        noise_type   = {noise_type}")
            print(f"        base_sigma   = {base_sigma}")
            print(f"        true_loge    = {true_loge}")
            print(f"        sigma_eff    = {effective_sigma(base_sigma, true_loge):.6g}")
            print(f"        same_across_traces = {same_noise_across_traces}")


if __name__ == "__main__":
    main()
