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

    # If no existing arrivals, the full range is valid
    if len(existing_arr) == 0:
        return np.random.uniform(tmin, tmax)

    # Start with full interval
    valid_intervals = [(tmin+min_space, tmax-min_space)]

    # Remove ±min_space around each existing arrival
    for arr in existing_arr:
        new_intervals = []
        for start, end in valid_intervals:
            # Exclude [arr - min_space, arr + min_space]
            exclude_start = max(tmin, arr - min_space)
            exclude_end = min(tmax, arr + min_space)

            # Left piece
            if exclude_start > start:
                new_intervals.append((start, exclude_start))
            # Right piece
            if exclude_end < end:
                new_intervals.append((exclude_end, end))

        valid_intervals = new_intervals

    # If no valid intervals remain
    if not valid_intervals:
        raise ValueError("No valid time intervals available.")

    # Choose one interval based on its length
    lengths = np.array([end - start for start, end in valid_intervals])
    probs = lengths / lengths.sum()
    idx = np.random.choice(len(valid_intervals), p=probs)
    start, end = valid_intervals[idx]
    return np.random.uniform(start, end)


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

def prepare_inputs_from_sac(data_dir, output_dir):

    import os
    from obspy import read
    from obspy.geodetics import gps2dist_azimuth
    from glob import glob

    os.makedirs(output_dir, exist_ok=True)

    # Match files
    sac_files = sorted(glob(os.path.join(data_dir, "*.sac")))
    stations = {}
    traces = {"UZ": [], "UR": [], "UT": []}
    dists = []
    bazs = []

    for f in sac_files:
        tr = read(f)[0]
        ch = tr.stats.channel[-1]  # Z, R, or T
        net = tr.stats.network
        sta = tr.stats.station
        key = f"{net}.{sta}"

        if key not in stations:
            stations[key] = {"Z": None, "R": None, "T": None}
        stations[key][ch] = tr

    for sta, comps in stations.items():
        trZ, trR, trT = comps["Z"], comps["R"], comps["T"]
        if trZ is None or trR is None or trT is None:
            print(f"Skipping incomplete station {sta}")
            continue

        # Check consistency
        if len(trZ.data) != len(trR.data) or len(trZ.data) != len(trT.data):
            print(f"Skipping inconsistent trace lengths for {sta}")
            continue

        # Time vector (just grab from one trace)
        if len(traces["UZ"]) == 0:
            npts = len(trZ.data)
            dt = trZ.stats.delta
            time = np.arange(0, npts * dt, dt)
            evla = trZ.stats.sac.evla
            evlo = trZ.stats.sac.evlo
            np.savetxt(os.path.join(output_dir, "time.csv"), time, delimiter=",")

        traces["UZ"].append(trZ.data)
        traces["UR"].append(trR.data)
        traces["UT"].append(trT.data)

        # Metadata
        stla = trZ.stats.sac.stla
        stlo = trZ.stats.sac.stlo
        dist_deg = trZ.stats.sac.gcarc
        _, baz, _ = gps2dist_azimuth(evla, evlo, stla, stlo)
        dists.append(dist_deg)
        bazs.append(baz)
    
    # Sort all data by increasing distance
    idx = np.argsort(dists)
    for comp in ["UZ", "UR", "UT"]:
        traces[comp] = [traces[comp][i] for i in idx]
    dists = [dists[i] for i in idx]
    bazs = [bazs[i] for i in idx]

    # Save component matrices
    for comp in ["UZ", "UR", "UT"]:
        arr = np.column_stack(traces[comp])
        np.savetxt(os.path.join(output_dir, f"{comp}.csv"), arr, delimiter=",")

    # Save station metadata
    metadata = np.column_stack([dists, bazs])
    np.savetxt(os.path.join(output_dir, "station_metadata.csv"), metadata, delimiter=",", header="dist_deg,baz", comments='')
    evinfo = np.column_stack([evla, evlo])
    np.savetxt(os.path.join(output_dir, "eventinfo.csv"), evinfo, delimiter=",", header="evla,evlo", comments='')

    print(f"Saved data to: {output_dir}")

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

def prep_data(datadir, modname, is3c, comp, isbp, freqs):
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

    U_obs /= np.max(np.abs(U_obs)) # normalize

    Utime  = np.loadtxt(os.path.join(datadir, modname, "time.csv"), delimiter=",")  # columns: time
    metadata = np.loadtxt(os.path.join(datadir, modname, "station_metadata.csv"), delimiter=",", skiprows=1)  # columns: distance, baz
    dt = Utime[1] - Utime[0]

    if isbp:
        U_obs = bandpass(U_obs, 1/dt, freqs[0], freqs[1])
    
    return U_obs, Utime, metadata, is3c