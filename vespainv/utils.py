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

def prepare_inputs_from_sac(data_dir, isbp=False, freqs=None, noise_dir=None, output_dir=None):
    import os
    import numpy as np
    from obspy import read
    from obspy.geodetics import gps2dist_azimuth
    from glob import glob

    os.makedirs(output_dir, exist_ok=True)

    sac_files = sorted(glob(os.path.join(data_dir, "*.sac")))
    traces = {"UZ": [], "UR": [], "UT": []}
    traces_noise = {"UZ": [], "UR": [], "UT": []} if noise_dir else None
    dists, bazs, stlas, stlos = [], [], [], []

    stations = {}
    evla = evlo = None

    for f in sac_files:
        tr = read(f)[0]
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
            np.savetxt(os.path.join(output_dir, "time.csv"), time, delimiter=",")

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

    # Save output
    for comp in ["UZ", "UR", "UT"]:
        np.savetxt(os.path.join(output_dir, f"{comp}.csv"), np.column_stack(traces[comp]), delimiter=",")
        if noise_dir:
            noise_stack = np.column_stack(traces_noise[comp])
            np.savetxt(os.path.join(output_dir, f"CD_{comp}.csv"), np.cov(noise_stack, rowvar=False), delimiter=",")

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
        f0 = np.mean(f0_all)
    else:
        raise ValueError("Input data must be 1D or 2D numpy array.")

    print(f"Dominant frequency: {f0: .2f} Hz")
    return f0

def prep_data(datadir, modname, is3c, comp, isbp, freqs, isds=None, isnorm=False):
    import os
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
    
    CDinv = None
    if os.path.isfile(os.path.join(datadir, modname, "CD_Z.csv")):
        if is3c:
            CD_Z = np.loadtxt(os.path.join(datadir, modname, "CD_Z.csv"), delimiter=",")  # columns: data
            CD_R = np.loadtxt(os.path.join(datadir, modname, "CD_R.csv"), delimiter=",")  # columns: data
            CD_T = np.loadtxt(os.path.join(datadir, modname, "CD_T.csv"), delimiter=",")  # columns: data
            CDinv = [np.linalg.inv(CD_Z), np.linalg.inv(CD_R), np.linalg.inv(CD_T)]
        else:
            CDname = "CD_"+comp+".csv"
            CD = np.loadtxt(os.path.join(datadir, modname, CDname), delimiter=",")  # columns: data
            CDinv = np.linalg.inv(CD)

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
    
    return U_obs, Utime, CDinv, metadata, is3c