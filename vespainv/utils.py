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
    ensuring it is at least `min_space` away from all existing arrivals.

    Parameters:
    - time: np.ndarray, the global time vector
    - existing_arr: np.ndarray, current list of arrival times
    - min_space: float, minimum spacing required between arrivals

    Returns:
    - float, the new valid arrival time
    """
    tmin, tmax = timeRange[0], timeRange[-1]
    max_attempts = 1000

    for _ in range(max_attempts):
        candidate = np.random.uniform(tmin, tmax)
        if np.all(np.abs(existing_arr - candidate) >= min_space):
            return candidate

    raise ValueError("Could not generate a valid arrival time after many attempts.")

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

def prepare__inputs_from_sac(data_dir, output_dir):

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
    refBaz: float = None
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
            )(time-tshift)
            stack += shifted

        vespa[i, :] = stack / n_traces

    # Plot
    plt.figure(figsize=(10, 6))
    extent = [time[0], time[-1], slow_grid[0], slow_grid[-1]]
    plt.imshow(vespa, aspect='auto', extent=extent, origin='lower',
               cmap='seismic', vmin=-np.max(np.abs(vespa)), vmax=np.max(np.abs(vespa)))
    plt.colorbar(label='Amplitude')
    plt.xlabel("Time (s)")
    plt.ylabel("Slowness (s/deg)")
    plt.title("Vespagram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return vespa
