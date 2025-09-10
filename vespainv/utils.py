import numpy as np

def dest_point(la1, lo1, az, delta):
    d2r = np.pi / 180
    la1 *= d2r
    lo1 *= d2r
    az *= d2r
    delta *= d2r
    lad = np.arcsin(np.sin(la1)*np.cos(delta)+np.cos(la1)*np.sin(delta)*np.cos(az))
    lod = lo1 + np.arctan2(np.sin(az)*np.sin(delta)*np.cos(la1), np.cos(delta)-np.sin(la1)*np.sin(lad))
    return lad/d2r, lod/d2r

def generate_arr(timeRange: np.ndarray, existing_arr: np.ndarray, min_space: float) -> float:
    """
    Generate a random arrival time within the time range,
    avoiding a buffer `min_space` around existing arrivals.
    
    Parameters:
    - timeRange: np.ndarray, the global time vector (only first and last used)
    - existing_arr: np.ndarray, current list of arrival times
    - min_space: float, minimum spacing required between arrivals

    Returns:
    - float, the new valid arrival time
    """
    tmin, tmax = timeRange[0], timeRange[-1]

    for _ in range(500):
        candidate = np.random.uniform(tmin, tmax)
        if np.all(np.abs(existing_arr - candidate) >= min_space):
            return candidate
    
    raise ValueError("Cannot place a new arrival without violating minimum spacing rule!")

def apply_constant_phase_shift(W: np.ndarray, phase_rad: float) -> np.ndarray:
    """
    Apply a constant phase shift in the frequency domain.

    Parameters:
    - W: complex FFT of the signal (1D array)
    - freqs: frequency array corresponding to W (from fftfreq)
    - phase_rad: phase shift in radians (e.g., np.pi/2 for 90°)

    Returns:
    - W_shifted: FFT of the signal with phase shift applied
    """
    N = len(W)
    phase_shift = np.ones(N, dtype=complex)

    # Sort out which indices are positive/negative freqs
    if N % 2 == 0:
        # Even length
        phase_shift[1:N//2] = np.exp(-1j * phase_rad)
        phase_shift[N//2+1:] = np.exp(1j * phase_rad)
        phase_shift[N//2] = 1  # Nyquist
    else:
        # Odd length
        phase_shift[1:(N+1)//2] = np.exp(-1j * phase_rad)
        phase_shift[(N+1)//2:] = np.exp(1j * phase_rad)

    return W * phase_shift

def prepare_inputs_from_sac(data_dir, isbp=False, isds=False, freqs=None, noise_dir=None, output_dir=None, snr_component='UZ', snr_threshold=None, outliers_manual=None, twin=None):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from obspy import read
    from obspy.geodetics import gps2dist_azimuth
    from glob import glob
    from sklearn.covariance import MinCovDet
    from scipy.signal import correlate
    from scipy.linalg import toeplitz
    from scipy.optimize import curve_fit

    os.makedirs(output_dir, exist_ok=True)

    sac_files = sorted(glob(os.path.join(data_dir, "*.sac")))
    traces = {"UZ": [], "UR": [], "UT": []}
    traces_noise = {"UZ": [], "UR": [], "UT": []} if noise_dir else None
    dists, bazs, stlas, stlos = [], [], [], []

    stations = {}
    evla = evlo = None

    for f in sac_files:
        tr = read(f)[0]
        if twin is not None:
            tr.trim(tr.stats.starttime + twin[0], tr.stats.starttime + twin[1], pad=True, fill_value=0)
        ch = tr.stats.channel[-1]  # Z/R/T
        net, sta = tr.stats.network, tr.stats.station
        key = f"{net}.{sta}"

        if key not in stations:
            stations[key] = {"Z": None, "R": None, "T": None, "norm": None}

        if isbp and freqs:
            tr.filter("bandpass", freqmin=freqs[0], freqmax=freqs[1], corners=2, zerophase=True)
        stations[key][ch] = tr

        # Load matching noise file if provided
        if noise_dir:
            fbase, fext = os.path.splitext(os.path.basename(f))
            fnoise = os.path.join(noise_dir, fbase + ".noise" + fext)
            if os.path.exists(fnoise):
                tr_noise = read(fnoise)[0]
                if twin is not None:
                    tr_noise.trim(tr_noise.stats.starttime + twin[0], tr_noise.stats.starttime + twin[1], pad=True, fill_value=0)
                if isbp and freqs:
                    tr_noise.filter("bandpass", freqmin=freqs[0], freqmax=freqs[1], corners=2, zerophase=True)
                stations[key][f"{ch}_noise"] = tr_noise
            else:
                print(f"Missing noise file for {f}")

    for key, comps in stations.items():
        trZ, trR, trT = comps["Z"], comps["R"], comps["T"]
        if None in (trZ, trR, trT):
            print(f"Skipping incomplete station {key}")
            continue

        # Check consistency
        if not (len(trZ.data) == len(trR.data) == len(trT.data)):
            print(f"Skipping inconsistent trace lengths for {key}")
            continue

        # Normalize traces
        norm = max(np.max(np.abs(trZ.data)), np.max(np.abs(trR.data)), np.max(np.abs(trT.data)))
        for tr in [trZ, trR, trT]:
            tr.data /= norm
        comps["norm"] = norm

        if len(traces["UZ"]) == 0:
            # Initialize time vector and event info
            npts = len(trZ.data)
            dt = trZ.stats.delta
            time = np.arange(0, npts * dt, dt)
            evla, evlo = trZ.stats.sac.evla, trZ.stats.sac.evlo

            # Downsample time axis
            if isds:
                factor = int(round((1 / dt) / isds))
            if isds and factor > 1:
                time = time[::factor]
                dt = time[1] - time[0]
            np.savetxt(os.path.join(output_dir, "time.csv"), time, delimiter=",")

        # Downsample data
        if isds:
            trZ.data = trZ.data[::factor]
            trR.data = trR.data[::factor]
            trT.data = trT.data[::factor]
        # Store traces
        traces["UZ"].append(trZ.data)
        traces["UR"].append(trR.data)
        traces["UT"].append(trT.data)

        # Store metadata
        stla = trZ.stats.sac.stla
        stlo = trZ.stats.sac.stlo
        dist_deg = trZ.stats.sac.gcarc
        _, baz, _ = gps2dist_azimuth(evla, evlo, stla, stlo)
        stlas.append(stla)
        stlos.append(stlo)
        dists.append(dist_deg)
        bazs.append(baz)

        if noise_dir:
            for ch, comp in zip(["Z", "R", "T"], ["UZ", "UR", "UT"]):
                tr_noise = comps.get(f"{ch}_noise")
                if tr_noise is None:
                    raise ValueError(f"Missing noise for {key} component {ch}")
                tr_noise.data /= norm
                # Downsample noise
                if isds and factor > 1:
                    tr_noise.data = tr_noise.data[::factor]
                traces_noise[comp].append(tr_noise.data)

    # Sort by distance
    idx = np.argsort(dists)
    for comp in ["UZ", "UR", "UT"]:
        traces[comp] = [traces[comp][i] for i in idx]
    dists = [dists[i] for i in idx]
    bazs = [bazs[i] for i in idx]
    stlas = [stlas[i] for i in idx]
    stlos = [stlos[i] for i in idx]
    if noise_dir:
        for comp in ["UZ", "UR", "UT"]:
            traces_noise[comp] = [traces_noise[comp][i] for i in idx]

    # === REMOVE HIGH AMPLITUDE NOISE TRACES ===
    if noise_dir:
        outlier_indices = set()
        for comp in ["UZ", "UR", "UT"]:
            noise_matrix = np.column_stack(traces_noise[comp])
            max_amps = np.max(np.abs(noise_matrix), axis=0)
            threshold = np.median(max_amps) + 5 * np.std(max_amps)
            bad = np.where(max_amps > threshold)[0]
            print(f"[{comp}] High-amplitude noise traces: {bad}")
            outlier_indices.update(bad.tolist())

        if outliers_manual: outlier_indices.update(outliers_manual)
        print(f"\n>>> Removing {len(outlier_indices)} outlier trace(s):", outlier_indices)

        # --- SNR-based outlier detection ---
        print(f"\n>>> Applying SNR threshold: {snr_threshold} (component: {snr_component})")

        n_traces = len(traces["UZ"])
        for i in range(n_traces):
            snr_vals = {}
            for comp in ["UZ", "UR", "UT"]:
                signal = traces[comp][i]
                noise = traces_noise[comp][i]
                signal_rms = np.sqrt(np.mean(signal ** 2))
                noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-10  # avoid div-by-zero
                snr = signal_rms / noise_rms
                snr_vals[comp] = snr

            # Determine the SNR to threshold on
            if snr_component.lower() == "min":
                snr_check = min(snr_vals.values())
            else:
                snr_check = snr_vals[snr_component.upper()]

            if snr_check < snr_threshold:
                outlier_indices.add(i)
                print(f"Trace {i}: SNRs={snr_vals} → flagged (used {snr_check:.2f})")
        
        # Remove bad traces
        for comp in ["UZ", "UR", "UT"]:
            traces[comp] = [tr for i, tr in enumerate(traces[comp]) if i not in outlier_indices]
            traces_noise[comp] = [tr for i, tr in enumerate(traces_noise[comp]) if i not in outlier_indices]

        dists = [d for i, d in enumerate(dists) if i not in outlier_indices]
        bazs = [b for i, b in enumerate(bazs) if i not in outlier_indices]
        stlas = [s for i, s in enumerate(stlas) if i not in outlier_indices]
        stlos = [s for i, s in enumerate(stlos) if i not in outlier_indices]


    # Save output
    for comp in ["UZ", "UR", "UT"]:
        np.savetxt(os.path.join(output_dir, f"{comp}.csv"), np.column_stack(traces[comp]), delimiter=",")
        if noise_dir:
            noise_stack = np.column_stack(traces_noise[comp])
            np.savetxt(os.path.join(output_dir, f"{comp}_noise.csv"), np.column_stack(traces_noise[comp]), delimiter=",")
            np.savetxt(os.path.join(output_dir, f"CD_{comp}.csv"), np.cov(noise_stack), delimiter=",")

            ### Robust covariance matrix: truncated version
            # RCM from MinCovDet
            n_samples = len(traces[comp][0])
            n_seconds_noise_for_cov = 50
            n_samples_noise_for_cov = int(n_seconds_noise_for_cov / dt)
            noise_stack_for_cov = noise_stack[:n_samples_noise_for_cov, :].T
            cov_mcd = MinCovDet(support_fraction=0.75).fit(noise_stack_for_cov).covariance_
            # Estimate autocovariance from diagonals
            avg_autocov = [np.mean(np.diag(cov_mcd, k=lag)) for lag in range(cov_mcd.shape[0])]
            pad_width = n_samples - len(avg_autocov)
            avg_autocov_padded = np.pad(avg_autocov, (0, pad_width), mode='constant')
            # Build using toeplitz and save
            CD_robust = toeplitz(avg_autocov_padded)
            np.savetxt(os.path.join(output_dir, f"CD_{comp}_robust.csv"), CD_robust, delimiter=",")

            ### Noise parameterization 3 from Kolb and Lekic (2014)
            # Parameters
            n_samples, n_traces = noise_stack.shape
            max_lag_seconds = 50
            max_lag = int(max_lag_seconds / dt)
            # Zero-mean each trace
            noise_stack = noise_stack - np.mean(noise_stack, axis=0)
            # Compute autocorrelation per trace (non-negative lags only)
            acovs = []
            for i in range(n_traces):
                trace = noise_stack[:, i]
                acorr = correlate(trace, trace, mode="full")
                acorr = acorr[n_samples - 1:]  # non-negative lags
                acorr = acorr[:max_lag]
                acorr /= n_samples
                acovs.append(acorr)
            # Average autocorrelations
            avg_autocov = np.mean(acovs, axis=0)
            lags = np.arange(len(avg_autocov)) * dt
            # Normalize for stable fitting
            avg_autocov_norm = avg_autocov / avg_autocov[0]
            # Model: a * exp(-λτ) * cos(λω₀τ)
            def model(tau, a, lambd, omega0):
                return a * np.exp(-lambd * tau) * np.cos(lambd * omega0 * tau)
            # Fit normalized data (keep original guess)
            try:
                popt, _ = curve_fit(
                    model,
                    lags,
                    avg_autocov_norm,
                    p0=(1.0, 0.1, 2 * np.pi * 0.2),
                    maxfev=10000
                )
                a_fit_norm, lambda_fit, omega0_fit = popt
                a_fit = a_fit_norm * avg_autocov[0]  # rescale amplitude to original scale
            except RuntimeError as e:
                print(f"[WARN] Fit failed for {comp}: {e}")
                continue
            # Generate full fitted autocovariance
            full_lags = np.arange(n_samples) * dt
            acov_fit = a_fit * np.exp(-lambda_fit * full_lags) * np.cos(lambda_fit * omega0_fit * full_lags)
            # Toeplitz covariance matrix
            CD_fit = toeplitz(acov_fit)
            np.savetxt(os.path.join(output_dir, f"CD_{comp}_fit.csv"), CD_fit, delimiter=",")

    np.savetxt(os.path.join(output_dir, "station_metadata.csv"),
               np.column_stack([dists, bazs]), delimiter=",", header="dist_deg,baz", comments='')
    np.savetxt(os.path.join(output_dir, "station_metadata_lalo.csv"),
               np.column_stack([stlas, stlos]), delimiter=",", header="lat,lon", comments='')
    np.savetxt(os.path.join(output_dir, "eventinfo.csv"),
               np.column_stack([evla, evlo]), delimiter=",", header="evla,evlo", comments='')


def make_vespagram(
    U: np.ndarray,                   # shape (n_time, n_traces)
    time: np.ndarray,               # shape (n_time,)
    metadata: np.ndarray,           # shape (n_traces, 2) = [dist, baz]
    refLat: float,
    refLon: float,
    srcLat: float,
    srcLon: float,
    slow_grid: np.ndarray,
    refBaz: float = None,
    clim: tuple = None
) -> np.ndarray:

    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    n_time, n_traces = U.shape
    vespa = np.zeros((len(slow_grid), n_time))

    for i, slow in enumerate(slow_grid):
        stack = np.zeros(n_time)

        for itrace in range(n_traces):
            trDist, trBaz = metadata[itrace]

            # Compute station lat/lon
            trLat, trLon = dest_point(srcLat, srcLon, trBaz, trDist)

            # Local dx, dy (same convention as forward modeling)
            dx = (trLon - refLon) * np.cos(np.radians(refLat))
            dy = trLat - refLat

            # Slowness vector
            if refBaz is not None:
                trBaz = refBaz
            slow_x = slow * np.cos(np.radians(90 - trBaz))
            slow_y = slow * np.sin(np.radians(90 - trBaz))

            # Time shift
            tshift = (slow_x * dx + slow_y * dy)

            # Interpolate and stack
            trace = U[:, itrace]
            trace /= np.max(np.abs(trace))  # normalize
            shifted = interp1d(
                time,
                trace,
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )(time+tshift)
            stack += shifted

        vespa[i, :] = stack / n_traces

    # Plot
    plt.figure(figsize=(10, 6))
    extent = [time[0], time[-1], slow_grid[0], slow_grid[-1]]

    if clim is None:
        vmax = np.max(np.abs(vespa))
        vmin = -vmax
    else:
        vmin, vmax = clim
        
    plt.imshow(vespa, aspect='auto', extent=extent, origin='lower',
               cmap='seismic', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Amplitude')
    plt.xlabel("Time (s)")
    plt.ylabel("Slowness (s/deg)")
    plt.title("Vespagram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return vespa

def bandpass(data, fs, fmin, fmax, corners=4, zerophase=True):
    """
    Vectorized bandpass filter using ObsPy's Stream.

    Parameters:
    - data: numpy array of shape
        - (n_samples,)
        - (n_samples, n_traces)
        - (n_samples, n_traces, n_components)
    - fs: Sampling frequency (Hz)
    - fmin: Low corner frequency (Hz)
    - fmax: High corner frequency (Hz)
    - corners: Filter order
    - zerophase: Apply filter forward and backward to avoid phase shift

    Returns:
    - Filtered data (same shape as input)
    """
    import obspy

    data = np.asarray(data)
    original_shape = data.shape

    if data.ndim == 1:
        data_reshaped = data[:, np.newaxis]
    elif data.ndim == 2:
        data_reshaped = data
    elif data.ndim == 3:
        n_samples, n_traces, n_comp = data.shape
        data_reshaped = data.reshape(n_samples, n_traces * n_comp)
    else:
        raise ValueError("Input data must be 1D, 2D, or 3D.")

    n_samples, n_series = data_reshaped.shape

    # Create list of Trace objects
    traces = []
    for i in range(n_series):
        tr = obspy.Trace()
        tr.data = data_reshaped[:, i].copy()
        tr.stats.sampling_rate = fs
        traces.append(tr)

    # Create Stream and filter
    st = obspy.Stream(traces)
    st.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=corners, zerophase=zerophase)

    # Collect filtered data
    filtered = np.stack([tr.data for tr in st], axis=-1)

    # Reshape back to original
    if data.ndim == 1:
        filtered = filtered.squeeze()
    elif data.ndim == 2:
        filtered = filtered
    elif data.ndim == 3:
        filtered = filtered.reshape(n_samples, n_traces, n_comp)

    return filtered


def calc_array_center(station_metadata, srcLat, srcLon):
    """
    Calculate approximate center of an array.

    Inputs:
      station_metadata: np.ndarray of shape (n_station, 2) [distance (deg), back-azimuth (deg)]
      srcLat, srcLon: event source latitude and longitude (degrees)

    Returns:
      centerLat, centerLon, centerBaz
    """

    n_station = station_metadata.shape[0]
    
    latitudes = []
    longitudes = []
    for i in range(n_station):
        dist, baz = station_metadata[i]
        lat, lon = dest_point(srcLat, srcLon, baz, dist)
        latitudes.append(lat)
        longitudes.append(lon)
    
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    
    centerLat = np.mean(latitudes)
    centerLon = np.mean(longitudes)
    
    # Compute centerBaz from src -> array center
    d2r = np.pi / 180
    r2d = 180 / np.pi
    dlon = (centerLon - srcLon) * d2r
    y = np.sin(dlon) * np.cos(centerLat * d2r)
    x = np.cos(srcLat * d2r) * np.sin(centerLat * d2r) - np.sin(srcLat * d2r) * np.cos(centerLat * d2r) * np.cos(dlon)
    centerBaz = (np.arctan2(y, x) * r2d) % 360

    return centerLat, centerLon, centerBaz

def create_stf(f0, dt):
    stf_time_0 = np.arange(-4 / f0, 4 / f0 + dt, dt)
    stf_0 = np.exp(-stf_time_0 ** 2 / (2 * (1 / (2 * np.pi * f0)) ** 2))
    stf_time = stf_time_0[:-1]
    stf = np.diff(stf_0) / np.diff(stf_time_0)
    stf = stf / np.max(np.abs(stf))
    return np.column_stack([stf_time, stf])

def est_stf_wid(stf, threshold=0.01):
    stf_time = stf[:,0]
    stf_data = stf[:,1]
    inds = np.where(np.abs(stf_data) >= threshold)[0]
    return (stf_time[inds[-1]] - stf_time[inds[0]]) if inds.size else 1.0

def est_dom_freq(data, fs):
    """
    Estimate the dominant frequency of a seismogram.

    Parameters:
    - data: 1D or 2D numpy array (single trace or multiple traces)
             shape = (n_samples,) or (n_samples, n_traces)
    - fs: Sampling frequency (Hz)

    Returns:
    - f0: Estimated dominant frequency (Hz) 
          (scalar)
    """
    import numpy as np

    n = data.shape[0]
    freqs = np.fft.rfftfreq(n, d=1/fs)

    if data.ndim == 1:
        fft_amp = np.abs(np.fft.rfft(data))
        fft_pwr = fft_amp**2
        f0 = np.sum(freqs * fft_pwr) / np.sum(fft_pwr)
    elif data.ndim == 2:
        def _single_trace_f0(trace):
            fft_amp = np.abs(np.fft.rfft(trace))
            fft_pwr = fft_amp**2
            return np.sum(freqs * fft_pwr) / np.sum(fft_pwr)

        f0_all = np.apply_along_axis(_single_trace_f0, axis=0, arr=data)
        f0 = np.nanmean(f0_all)
    else:
        raise ValueError("Input data must be 1D or 2D numpy array.")

    print(f"Dominant frequency: {f0: .2f} Hz")
    return f0

import numpy as np
from scipy.linalg import toeplitz

def compute_toeplitz_CDinv(CD, eps=1e-6):
    """
    From a full empirical covariance matrix CD, compute the inverse of the
    nearest PSD Toeplitz matrix formed by averaging diagonals and zeroing
    negative eigenvalues.
    
    Parameters:
        CD (ndarray): (N x N) covariance matrix (empirical or robust).
        eps (float): Minimum eigenvalue after clipping (default: 0).
    
    Returns:
        CDinv (ndarray): Inverse of the PSD Toeplitz matrix.
    """
    # Step 1: Average along diagonals
    n = CD.shape[0]
    diag_avg = [np.mean(np.diag(CD, k=i)) for i in range(n)]
    
    # Step 2: Construct symmetric Toeplitz matrix
    CD_toep = toeplitz(diag_avg)
    CD_toep = (CD_toep + CD_toep.T) / 2  # enforce symmetry just in case

    # Step 3: Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(CD_toep)
    eigvals_clipped = np.clip(eigvals, a_min=eps, a_max=None)

    # Step 3: Compute total energy and sort eigenvalues in descending order of energy contribution
    total_energy = np.sum(eigvals_clipped ** 2)
    sorted_indices = np.argsort(eigvals_clipped)[::-1]
    eigvals_sorted = eigvals_clipped[sorted_indices]
    eigvecs_sorted = eigvecs[:, sorted_indices]

    # Step 5: Cumulative energy ratio
    cumulative_energy = np.cumsum(eigvals_sorted ** 2)
    energy_ratio = cumulative_energy / total_energy

    # Step 6: Find n such that 99.9% of energy is preserved
    n = np.searchsorted(energy_ratio, 0.999) + 1

    # Step 7: Build selective inverse of eigenvalues
    eigvals_inv = np.zeros_like(eigvals_sorted)
    eigvals_inv[:n] = 1.0 / eigvals_sorted[:n]

    # Step 8: Reconstruct inverse matrix using truncated eigenvalues
    D_inv_trunc = np.diag(eigvals_inv)
    CDinv = eigvecs_sorted @ D_inv_trunc @ eigvecs_sorted.T

    # # ======= Plotting =======
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    # im0 = axs[0].imshow(CD_toep, cmap='viridis')
    # axs[0].set_title("Toeplitz Covariance")
    # plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    # im1 = axs[1].imshow(CDinv, cmap='viridis')
    # axs[1].set_title("Inverse Covariance")
    # plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # plt.tight_layout()
    # plt.show()
    # # =========================

    return CDinv

def inv_sqrt(C):
    from scipy.linalg import eigh
    # eigendecomposition for symmetric positive-definite matrix
    eigvals, eigvecs = eigh(C)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    return eigvecs @ D_inv_sqrt @ eigvecs.T

def prep_data(datadir, modname, is3c, comp, CDopt, isbp, freqs, isds=False, isnorm=False):
    import os
    from scipy.linalg import fractional_matrix_power, cholesky, inv, eigh
    if os.path.isfile(os.path.join(datadir, modname, "U.csv")):
        if is3c:
            response = input("U.csv in data directory, changing to 1c. Proceed? [y/n]").strip().lower()
            if response == "y":
                is3c = False
            else:
                print("Aborted.")
                return
        U_obs = np.loadtxt(os.path.join(datadir, modname, "U.csv"), delimiter=",")  # columns: data
    else:
        if is3c:
            Z_obs = np.loadtxt(os.path.join(datadir, modname, "UZ.csv"), delimiter=",")  # columns: data
            R_obs = np.loadtxt(os.path.join(datadir, modname, "UR.csv"), delimiter=",")  # columns: data
            T_obs = np.loadtxt(os.path.join(datadir, modname, "UT.csv"), delimiter=",")  # columns: data
            U_obs = np.stack([Z_obs, R_obs, T_obs], axis=-1)
        else:
            Uname = "U"+comp+".csv"
            U_obs = np.loadtxt(os.path.join(datadir, modname, Uname), delimiter=",")  # columns: data
    
    CDinv = None # CDopt: 0 - False, 1 - CD, 2- CD_robust, 3- CD_fit
    if CDopt:
        if CDopt == 1: robust_handle = ''
        elif CDopt == 2: robust_handle = '_robust'
        elif CDopt == 3: robust_handle = "_fit"
        else: raise ValueError(f"Invalid CDopt value: {CDopt}. Must be 0 (False), 1 (Empirical), or 2 (Robust).")
        if is3c:
            CD_Z = np.loadtxt(os.path.join(datadir, modname, "CD_UZ"+robust_handle+".csv"), delimiter=",")  # columns: data
            CD_R = np.loadtxt(os.path.join(datadir, modname, "CD_UR"+robust_handle+".csv"), delimiter=",")  # columns: data
            CD_T = np.loadtxt(os.path.join(datadir, modname, "CD_UT"+robust_handle+".csv"), delimiter=",")  # columns: data
            CDinv = [compute_toeplitz_CDinv(CD_Z), compute_toeplitz_CDinv(CD_R), compute_toeplitz_CDinv(CD_T)]
            # CD_sqrt_inv = [fractional_matrix_power(CD_Z, -0.5), fractional_matrix_power(CD_R, -0.5), fractional_matrix_power(CD_T, -0.5)]
            # CD_sqrt_inv2 = [inv(cholesky(CD_Z)), inv(cholesky(CD_R)), inv(cholesky(CD_T))]
            CD_sqrt_inv = [inv_sqrt(CD_Z), inv_sqrt(CD_R), inv_sqrt(CD_T)]
            # # tmp save for debug
            # savename = os.path.join(datadir, modname, 'CD_inv_debug.npz')
            # np.savez(savename, CD_Z=CD_Z, CDZ_inv=CDinv[0], CDZ_sqrt_inv=CD_sqrt_inv[0], CDZ_sqrt_inv2=CD_sqrt_inv2[0], CDZ_sqrt_inv3=CD_sqrt_inv3[0])
        else:
            CDname = "CD_U"+comp+robust_handle+".csv"
            CD = np.loadtxt(os.path.join(datadir, modname, CDname), delimiter=",")  # columns: data
            CDinv = compute_toeplitz_CDinv(CD)
            CD_sqrt_inv = fractional_matrix_power(CD, -0.5)
            # CD_sqrt_inv = inv(cholesky(CD))

    Utime  = np.loadtxt(os.path.join(datadir, modname, "time.csv"), delimiter=",")  # columns: time
    metadata = np.loadtxt(os.path.join(datadir, modname, "station_metadata.csv"), delimiter=",", skiprows=1)  # columns: distance, baz
    dt = Utime[1] - Utime[0]

    if isbp:
        U_obs = bandpass(U_obs, 1/dt, freqs[0], freqs[1])

    if isds:
        factor = int((1/dt) / isds)
        U_obs = U_obs[::factor]
        Utime = Utime[::factor]
        dt = Utime[1] - Utime[0]
    
    if isnorm: U_obs /= np.max(np.abs(U_obs)) # normalize
    
    return U_obs, Utime, CDinv, CD_sqrt_inv, metadata, is3c