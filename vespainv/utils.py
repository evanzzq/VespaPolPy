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