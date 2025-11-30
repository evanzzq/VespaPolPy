import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from vespainv.model import VespaModel, Prior, VespaModel3c, Prior3c, Bookkeeping
from vespainv.waveformBuilder import create_U_from_model_freqdomain, create_U_from_model_3c_freqdomain
from vespainv.utils import dest_point

# Parameter setup
# filedir = "H:/My Drive/Research/VespaPolPy"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

modname = "model12"
Nphase = 5
maxN = 10 # will be written into Prior
is3c = False
ampRange = (0, 1)
slwRange = (-2., 2.)

# Parameter setup: stf
f0 = 0.2
dt = 0.1

# Parameter setup: time vector
tmax = 100

# Parameter setup: array
srcLat = 0.0
srcLon = 0.0
base_dist = 110.0
base_baz = 30.0
base_az = (base_baz + 180)%360
Ntrace = 10
refLat, refLon = dest_point(srcLat, srcLon, base_az, base_dist)

# Parameter setup: location perturbation
locDiff = False
distDiff = np.random.uniform(-3.0, 3.0, Ntrace)
bazDiff  = np.random.uniform(-3.0, 3.0, Ntrace)

# Parameter setup: arrival times
defAll = True
arr = np.array([10,42,50,58,90])
slw = np.array([0., 0, 0.2, -0.2, 1])
amp = np.array([0.8, 0.8, 1.0, 0.7, 0.5])
azi = np.array([0,20,0,0,0]) # N/A for P type; for S, 0 means pure SV and 90 means pure SH
ph_hh = np.array([0,30,0,0,0]) # N/A for P and pure SV?
ph_vh = np.array([20,60,0,0,10]) # N/A for pure SH 
atts = np.array([1,1,1,1,1])
wvtype = np.array([1, 0, 1, 0, 1])

synDir = os.path.join(filedir, "SynData", modname)
os.makedirs(synDir, exist_ok=True)

# Create stf
stf_time_0 = np.arange(-4 / f0, 4 / f0 + dt, dt)
stf_0 = np.exp(-stf_time_0 ** 2 / (2 * (1 / (2 * np.pi * f0)) ** 2))
stf_time = stf_time_0[:-1]
stf = np.diff(stf_0) / np.diff(stf_time_0)
stf = stf / np.max(np.abs(stf))

# Pack and save stf
stf_array = np.column_stack([stf_time, stf])
np.savetxt(os.path.join(synDir, "stf.csv"), stf_array, delimiter=",", header="time,stf", comments="")

# Generate station metadata and save
time = np.arange(0, tmax, dt)
# np.random.seed(0)
dists = base_dist + np.random.uniform(-2.0, 2.0, Ntrace)
bazs = base_baz + np.random.uniform(-2.0, 2.0, Ntrace)
stlas, stlos = np.zeros_like(dists), np.zeros_like(dists)
for ista in range(Ntrace):
    stlas[ista], stlos[ista] = dest_point(srcLat, srcLon, (bazs[ista]+180)%360, dists[ista])
idx = np.argsort(dists); dists, bazs, stlas, stlos = dists[idx], bazs[idx], stlas[idx], stlos[idx] # sort by dist
station_metadata_db = np.column_stack((dists, bazs))
station_metadata = np.column_stack((stlas, stlos))

np.savetxt(os.path.join(synDir, "station_metadata_db.csv"), station_metadata_db, delimiter=",", header="dist_deg,baz", comments="")
np.savetxt(os.path.join(synDir, "station_metadata.csv"), station_metadata, delimiter=",", header="lat,lon", comments="")

# Define prior and model, and save
if is3c:
    prior = Prior3c(maxN=maxN, timeRange=(time[0],time[-1]), ampRange=ampRange, slwRange=slwRange)
    model = VespaModel3c.create_random(
        Nphase=Nphase, Ntrace=Ntrace, time=time, prior=prior, arr=arr
        ) if not defAll else VespaModel3c(
            Nphase=Nphase, Ntrace=Ntrace, arr=arr, slw=slw, amp=amp, azi=azi, ph_hh=ph_hh, ph_vh=ph_vh, atts=atts, wvtype=wvtype
        )
else:
    prior = Prior(maxN=maxN, timeRange=(time[0],time[-1]), ampRange=ampRange, slwRange=slwRange)
    model = VespaModel.create_random(
        Nphase=Nphase, Ntrace=Ntrace, time=time, prior=prior, arr=arr
        ) if not defAll else VespaModel(
            Nphase=Nphase, Ntrace=Ntrace, arr=arr, slw=slw, amp=amp, atts=atts
        )
    if locDiff:
        model.distDiff = distDiff
        model.bazDiff  = bazDiff

# Save model details as a human-readable text file
with open(os.path.join(synDir, "model_details.txt"), "w") as ftxt:
    ftxt.write("=== Synthetic Model Details ===\n")
    ftxt.write(f"Number of phases: {Nphase}\n")
    ftxt.write(f"Number of traces: {Ntrace}\n")
    ftxt.write(f"Base distance: {base_dist} km\n")
    ftxt.write(f"Base backazimuth: {base_baz} deg\n")
    ftxt.write(f"Time range: {time[0]} to {time[-1]} s\n")
    ftxt.write(f"Sampling interval: {dt} s\n")
    ftxt.write(f"Source-time function: Gaussian derivative, f0 = {f0} Hz\n")
    ftxt.write(f"3-component data: {is3c}\n\n")

    ftxt.write("--- Arrival Times ---\n")
    ftxt.write(np.array2string(model.arr, separator=", ") + "\n\n")
    ftxt.write("--- Slowness ---\n")
    ftxt.write(np.array2string(model.slw, separator=", ") + "\n\n")
    ftxt.write("--- Amplitudes ---\n")
    ftxt.write(np.array2string(model.amp, separator=", ") + "\n\n")
    if hasattr(model, 'ph_hh'):
        ftxt.write("--- Phase Difference: HH ---\n")
        ftxt.write(np.array2string(model.ph_hh, separator=", ") + "\n\n")
    if hasattr(model, 'ph_vh'):
        ftxt.write("--- Phase Difference: VH ---\n")
        ftxt.write(np.array2string(model.ph_vh, separator=", ") + "\n\n")
    if hasattr(model, 'atts'):
        ftxt.write("--- Attenuation (t* for S) ---\n")
        ftxt.write(np.array2string(model.atts, separator=", ") + "\n\n")
    if hasattr(model, 'wvtype'):
        ftxt.write("--- Wave Type (P = True) ---\n")
        ftxt.write(np.array2string(model.wvtype, separator=", ") + "\n\n")
    if locDiff:
        ftxt.write("--- Distance perturbation (deg) ---\n")
        ftxt.write(np.array2string(model.distDiff, separator=", ") + "\n\n")
        ftxt.write("--- BAZ perturbation (deg) ---\n")
        ftxt.write(np.array2string(model.bazDiff, separator=", ") + "\n\n")

with open(os.path.join(synDir, "Model.pkl"), "wb") as f1:
    pickle.dump(model, f1)
with open(os.path.join(synDir, "Prior.pkl"), "wb") as f2:
    pickle.dump(prior, f2)

# Generate U, plot, and save
bookkeeping = Bookkeeping(refLat=refLat, refLon=refLon, refBaz=base_baz, srcLat=srcLat, srcLon=srcLon, fitAtts=False, fitPhase=True, isMars=False)
with open(os.path.join(synDir, "Bookkeeping_0.pkl"), "wb") as f:
    pickle.dump(bookkeeping, f)
if is3c:
    U = create_U_from_model_3c_freqdomain(model, station_metadata, time, stf_time, stf, bookkeeping)
else:
    U = create_U_from_model_freqdomain(model, station_metadata, time, stf_time, stf, bookkeeping)

if is3c:
    components = ['Z', 'R', 'T']
    fig, axes = plt.subplots(1, 3, figsize=(12, 10), sharex=True)
    offset = 1.2 * np.max(np.abs(U))  # spacing between traces
    n_traces = U.shape[1]
    for i, ax in enumerate(axes):
        for j in range(n_traces):
            ax.plot(time, U[:, j, i] + j * offset, color='black')
        ax.set_ylabel(f"{components[i]} Amplitude (offset)")
        ax.set_title(f"{components[i]} Component")
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
else:
    plt.figure(figsize=(10, 6))
    n_traces = U.shape[1]
    offset = 1.2 * np.max(np.abs(U))  # spacing between traces
    for i in range(n_traces):
        plt.plot(time, U[:, i] + i * offset, color="black")
        dist, baz = station_metadata_db[i,:]
        plt.text(time[-1] + 0.5, i * offset, f"{dist:.1f}°, {baz:.0f}°", 
             va='center', fontsize=8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (offset by trace index)")
    plt.title("Synthetic Seismograms")
    plt.grid(True)
    plt.tight_layout()

plt.show()

np.savetxt(os.path.join(synDir, "time.csv"), time, delimiter=",")

if is3c:
    Z = U[:, :, 0]
    R = U[:, :, 1]
    T = U[:, :, 2]
    np.savetxt(os.path.join(synDir, "UZ.csv"), Z, delimiter=",")
    np.savetxt(os.path.join(synDir, "UR.csv"), R, delimiter=",")
    np.savetxt(os.path.join(synDir, "UT.csv"), T, delimiter=",")
else:
    np.savetxt(os.path.join(synDir, "U.csv"), U, delimiter=",")

# sys.exit(0)

# === Optional: Add noise to synthetic seismograms ===
add_noise = False
sigma = 0.05  # target standard deviation (std) for noise
np.random.seed(42)

if add_noise:
    print("[INFO] Generating noisy datasets (L2 = Gaussian, L1 = Laplace)...")

    # --- Prepare output directories ---
    synDir_L2 = os.path.join(filedir, "SynData", modname + "_noisy_L2")
    synDir_L1 = os.path.join(filedir, "SynData", modname + "_noisy_L1")
    os.makedirs(synDir_L2, exist_ok=True)
    os.makedirs(synDir_L1, exist_ok=True)

    # --- Save shared metadata ---
    np.savetxt(os.path.join(synDir_L2, "station_metadata.csv"), station_metadata, delimiter=",", header="distance,baz", comments="")
    np.savetxt(os.path.join(synDir_L1, "station_metadata.csv"), station_metadata, delimiter=",", header="distance,baz", comments="")
    np.savetxt(os.path.join(synDir_L2, "time.csv"), time, delimiter=",")
    np.savetxt(os.path.join(synDir_L1, "time.csv"), time, delimiter=",")

    # --- Generate noise and noisy datasets ---
    # U has shape (T, N, 3) if is3c else (T, N)
    if is3c:
        # Gaussian noise (L2)
        noise_L2 = np.random.normal(loc=0.0, scale=sigma, size=U.shape)
        U_noisy_L2 = U + noise_L2

        # Laplace noise (L1). Choose scale b so Laplace std = sigma:
        # Var(Laplace) = 2*b^2 => b = sigma / sqrt(2)
        laplace_b = sigma / np.sqrt(2.0)
        noise_L1 = np.random.laplace(loc=0.0, scale=laplace_b, size=U.shape)
        U_noisy_L1 = U + noise_L1

        # Save noisy data and noise traces for each component
        comps = [("UZ", 0), ("UR", 1), ("UT", 2)]
        for name, ic in comps:
            np.savetxt(os.path.join(synDir_L2, f"{name}.csv"), U_noisy_L2[:, :, ic], delimiter=",")
            np.savetxt(os.path.join(synDir_L2, f"{name}_noise.csv"), noise_L2[:, :, ic], delimiter=",")

            np.savetxt(os.path.join(synDir_L1, f"{name}.csv"), U_noisy_L1[:, :, ic], delimiter=",")
            np.savetxt(os.path.join(synDir_L1, f"{name}_noise.csv"), noise_L1[:, :, ic], delimiter=",")
    else:
        # 1-component case (T, N)
        noise_L2 = np.random.normal(loc=0.0, scale=sigma, size=U.shape)
        U_noisy_L2 = U + noise_L2

        laplace_b = sigma / np.sqrt(2.0)
        noise_L1 = np.random.laplace(loc=0.0, scale=laplace_b, size=U.shape)
        U_noisy_L1 = U + noise_L1

        np.savetxt(os.path.join(synDir_L2, "U.csv"), U_noisy_L2, delimiter=",")
        np.savetxt(os.path.join(synDir_L2, "U_noise.csv"), noise_L2, delimiter=",")
        np.savetxt(os.path.join(synDir_L1, "U.csv"), U_noisy_L1, delimiter=",")
        np.savetxt(os.path.join(synDir_L1, "U_noise.csv"), noise_L1, delimiter=",")

    print(f"[INFO] Noisy datasets saved to:\n  {synDir_L2}\n  {synDir_L1}")

    # === Fit and save noise covariance matrices (Kolb & Lekic style) ===
    from scipy.signal import correlate
    from scipy.linalg import toeplitz
    from scipy.optimize import curve_fit

    def fit_and_save_CD(noise_array, out_dir, comps_names):
        """
        noise_array: either noise_L2 or noise_L1
          - 3C: shape (T, N, 3)
          - 1C: shape (T, N)
        out_dir: directory to save CD files
        comps_names: list of tuples (name, ic) for 3C or [('U', 0)] for 1C
        """
        if is3c:
            for name, ic in comps_names:
                noise_stack = noise_array[:, :, ic]  # (T, N)
                n_samples, n_traces = noise_stack.shape
                dt_local = dt if 'dt' not in globals() else dt  # prefer local dt variable
                # fallback to prior.dt if available
                if 'prior' in globals() and hasattr(prior, 'dt'):
                    dt_local = prior.dt

                max_lag_seconds = 50.0
                max_lag = int(max_lag_seconds / dt_local)
                if max_lag < 1:
                    max_lag = n_samples

                # Zero-mean each trace
                noise_stack = noise_stack - np.mean(noise_stack, axis=0)

                # Compute autocovariance (non-negative lags) per trace
                acovs = []
                for i in range(n_traces):
                    trace = noise_stack[:, i]
                    acorr = correlate(trace, trace, mode="full")
                    acorr = acorr[n_samples - 1:]  # keep non-negative lags
                    acorr = acorr[:max_lag]
                    acorr /= n_samples
                    acovs.append(acorr)

                # Average autocovariance across traces
                avg_autocov = np.mean(acovs, axis=0)
                lags = np.arange(len(avg_autocov)) * dt_local

                # Normalized autocov for stable fit
                if avg_autocov[0] == 0:
                    print(f"[WARN] zero variance for {name}; skipping CD fit.")
                    continue
                avg_autocov_norm = avg_autocov / avg_autocov[0]

                # Model: a * exp(-lambda * tau) * cos(lambda * omega0 * tau)
                def akl_model(tau, a, lambd, omega0):
                    return a * np.exp(-lambd * tau) * np.cos(lambd * omega0 * tau)

                try:
                    popt, _ = curve_fit(
                        akl_model,
                        lags,
                        avg_autocov_norm,
                        p0=(1.0, 0.1, 2 * np.pi * 0.2),
                        maxfev=10000
                    )
                    a_fit_norm, lambda_fit, omega0_fit = popt
                    a_fit = a_fit_norm * avg_autocov[0]
                except Exception as e:
                    print(f"[WARN] Fit failed for {name}: {e}")
                    # Fallback: use empirical autocov extended to full length
                    full_lags = np.arange(n_samples) * dt_local
                    acov_fit = np.zeros(n_samples)
                    acov_fit[:len(avg_autocov)] = avg_autocov
                    CD_fit = toeplitz(acov_fit)
                    np.savetxt(os.path.join(out_dir, f"CD_{name}_empirical.csv"), CD_fit, delimiter=",")
                    continue

                # Build full autocovariance and Toeplitz covariance matrix
                full_lags = np.arange(n_samples) * dt_local
                acov_fit = a_fit * np.exp(-lambda_fit * full_lags) * np.cos(lambda_fit * omega0_fit * full_lags)
                CD_fit = toeplitz(acov_fit)

                # Save
                np.savetxt(os.path.join(out_dir, f"CD_{name}_fit.csv"), CD_fit, delimiter=",")

        else:
            # 1-component case
            noise_stack = noise_array  # (T, N)
            n_samples, n_traces = noise_stack.shape
            dt_local = dt if 'dt' not in globals() else dt
            if 'prior' in globals() and hasattr(prior, 'dt'):
                dt_local = prior.dt
            max_lag_seconds = 50.0
            max_lag = int(max_lag_seconds / dt_local)
            if max_lag < 1:
                max_lag = n_samples

            noise_stack = noise_stack - np.mean(noise_stack, axis=0)

            acovs = []
            for i in range(n_traces):
                trace = noise_stack[:, i]
                acorr = correlate(trace, trace, mode="full")
                acorr = acorr[n_samples - 1:]
                acorr = acorr[:max_lag]
                acorr /= n_samples
                acovs.append(acorr)

            avg_autocov = np.mean(acovs, axis=0)
            lags = np.arange(len(avg_autocov)) * dt_local
            if avg_autocov[0] == 0:
                print("[WARN] zero variance for U; skipping CD fit.")
                return
            avg_autocov_norm = avg_autocov / avg_autocov[0]

            def akl_model(tau, a, lambd, omega0):
                return a * np.exp(-lambd * tau) * np.cos(lambd * omega0 * tau)

            try:
                popt, _ = curve_fit(
                    akl_model,
                    lags,
                    avg_autocov_norm,
                    p0=(1.0, 0.1, 2 * np.pi * 0.2),
                    maxfev=10000
                )
                a_fit_norm, lambda_fit, omega0_fit = popt
                a_fit = a_fit_norm * avg_autocov[0]
            except Exception as e:
                print(f"[WARN] Fit failed for U: {e}")
                full_lags = np.arange(n_samples) * dt_local
                acov_fit = np.zeros(n_samples)
                acov_fit[:len(avg_autocov)] = avg_autocov
                CD_fit = toeplitz(acov_fit)
                np.savetxt(os.path.join(out_dir, f"CD_U_empirical.csv"), CD_fit, delimiter=",")
                return

            full_lags = np.arange(n_samples) * dt_local
            acov_fit = a_fit * np.exp(-lambda_fit * full_lags) * np.cos(lambda_fit * omega0_fit * full_lags)
            CD_fit = toeplitz(acov_fit)
            np.savetxt(os.path.join(out_dir, "CD_U_fit.csv"), CD_fit, delimiter=",")

    # Run fit & save for L2 and L1 separately
    if is3c:
        comps_names = [("UZ", 0), ("UR", 1), ("UT", 2)]
    else:
        comps_names = [("U", 0)]

    fit_and_save_CD(noise_L2, synDir_L2, comps_names)
    fit_and_save_CD(noise_L1, synDir_L1, comps_names)

    print("[INFO] Noise covariance matrices (and empirical fallbacks) saved for both L2 & L1.")
else:
    print("[INFO] Noise addition skipped.")
