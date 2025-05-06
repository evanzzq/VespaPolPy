import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from vespainv.model import VespaModel, Prior
from vespainv.utils import dest_point, apply_constant_phase_shift


def create_U_from_model(
    model: VespaModel,
    prior: Prior,
    metadata: np.ndarray,  # shape (n_traces, 2): [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray
) -> np.ndarray:
    """
    Forward model a synthetic seismogram from the VespaModel.

    Parameters:
    - model: VespaModel with arr, slw, amp, distDiff, bazDiff
    - prior: Prior object with refLat, refLon, refBaz
    - metadata: np.ndarray of shape (n_traces, 2), where each row is [dist, baz]
    - time: np.ndarray, time vector for synthetic seismograms
    - stf_time: np.ndarray, time vector for the source time function
    - stf: np.ndarray, source time function values

    Returns:
    - U_model: np.ndarray of shape (n_traces, len(time)), synthetic seismograms
    """

    n_traces = metadata.shape[0]
    U_model = np.zeros((len(time), n_traces))

    if model.Nphase == 0:
        return U_model

    refLat = prior.refLat
    refLon = prior.refLon
    refBaz = prior.refBaz
    srcLat = prior.srcLat
    srcLon = prior.srcLon

    for itrace in range(n_traces):
        
        trDist, trBaz = metadata[itrace]
        trDist += model.distDiff[itrace]
        trBaz += model.bazDiff[itrace]

        trLat, trLon = dest_point(srcLat, srcLon, trBaz, trDist)

        dx = (trLon - refLon) * np.cos(np.radians(refLat))
        dy = trLat - refLat

        trace = np.zeros(len(time))

        for iph in range(model.Nphase):
            
            slow = model.slw[iph]
            slow_x = slow * np.cos(np.radians(90-trBaz)) # refBaz
            slow_y = slow * np.sin(np.radians(90-trBaz)) # refBaz

            tshift = model.arr[iph] + (slow_x * dx + slow_y * dy)

            shifted = interp1d(
                stf_time + tshift,
                stf,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )(time)

            trace += model.amp[iph] * shifted

        U_model[:, itrace] = trace

    return U_model

def create_U_from_model_3c(
    model: VespaModel,
    prior: Prior,
    metadata: np.ndarray,  # shape (n_traces, 2): [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray,
    P_wvlt: np.ndarray = None,
    S_wvlt: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward model a synthetic seismogram from the VespaModel.

    Parameters:
    - model: VespaModel3c with arr, slw, amp, dip, azi, ph_hh, ph_vh, atts, svfac, wvtype, distDiff, bazDiff
    - prior: Prior object with refLat, refLon, refBaz
    - metadata: np.ndarray of shape (n_traces, 2), where each row is [dist, baz]
    - time: np.ndarray, time vector for synthetic seismograms
    - stf_time: np.ndarray, time vector for the source time function
    - stf: np.ndarray, source time function values

    Returns:
    - U_model: np.ndarray of shape (n_traces, len(time), 3), synthetic seismograms
    """

    from scipy.signal import hilbert

    n_traces = metadata.shape[0]
    U_model = np.zeros((len(time), n_traces, 3))

    if model.Nphase == 0:
        return U_model

    refLat = prior.refLat
    refLon = prior.refLon
    refBaz = prior.refBaz
    srcLat = prior.srcLat
    srcLon = prior.srcLon

    for itrace in range(n_traces):
        
        trDist, trBaz = metadata[itrace]
        trDist += model.distDiff[itrace]
        trBaz += model.bazDiff[itrace]

        trLat, trLon = dest_point(srcLat, srcLon, trBaz, trDist)

        dx = (trLon - refLon) * np.cos(np.radians(refLat))
        dy = trLat - refLat

        traceZ = np.zeros(len(time))
        traceR = np.zeros(len(time))
        traceT = np.zeros(len(time))

        for iph in range(model.Nphase):
            
            slow = model.slw[iph]
            slow_x = slow * np.cos(np.radians(90-trBaz)) # refBaz
            slow_y = slow * np.sin(np.radians(90-trBaz)) # refBaz

            tshift = model.arr[iph] + (slow_x * dx + slow_y * dy)

            P_wvlt = tstar_conv(stf, stf_time, model.atts[iph]*0.25)
            S_wvlt = tstar_conv(stf, stf_time, model.atts[iph])

            P_shifted = interp1d(
                stf_time + tshift,
                P_wvlt,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )(time)

            S_shifted = interp1d(
                stf_time + tshift,
                S_wvlt,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )(time)

            if model.wvtype[iph] == 1:
                P = model.amp[iph] * P_shifted
                SV = np.zeros_like(P)
                SH = np.zeros_like(P)
            else:
                SV = model.amp[iph] * model.svfac[iph] * S_shifted
                SH = model.amp[iph] * (1 - model.svfac[iph]) * S_shifted
                P = np.zeros_like(SV)
            
            Z, R, T = PVH_to_ZRT(P, SV, SH, model.slw[iph])

            Z *= np.cos(np.radians(model.dip[iph]))
            
            sin_inc = np.sin(np.radians(model.dip[iph]))
            sin_azi = np.sin(np.radians(model.azi[iph]))
            cos_azi = np.cos(np.radians(model.azi[iph]))
            exp_ph_vh = np.exp(-1j * np.radians(model.ph_vh[iph]))
            exp_ph_hh_vh = np.exp(-1j * (np.radians(model.ph_hh[iph]) + np.radians(model.ph_vh[iph])))

            R = np.real(hilbert(R) * sin_inc * cos_azi * exp_ph_vh)
            T = np.real(hilbert(T) * sin_inc * sin_azi * exp_ph_hh_vh)


            traceZ += Z
            traceR += R
            traceT += T

        U_model[:, itrace, 0] = traceZ
        U_model[:, itrace, 1] = traceR
        U_model[:, itrace, 2] = traceT

    return U_model, P_wvlt, S_wvlt

def create_U_from_model_3c_freqdomain(
    model: VespaModel,
    prior: Prior,
    metadata: np.ndarray,  # shape (n_traces, 2): [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray,
    P_wvlt_W: np.ndarray = None,
    S_wvlt_W: np.ndarray = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward model a synthetic seismogram from the VespaModel.

    Parameters:
    - model: VespaModel3c with arr, slw, amp, dip, azi, ph_hh, ph_vh, atts, svfac, wvtype, distDiff, bazDiff
    - prior: Prior object with refLat, refLon, refBaz
    - metadata: np.ndarray of shape (n_traces, 2), where each row is [dist, baz]
    - time: np.ndarray, time vector for synthetic seismograms
    - stf_time: np.ndarray, time vector for the source time function
    - stf: np.ndarray, source time function values

    Returns:
    - U_model: np.ndarray of shape (n_traces, len(time), 3), synthetic seismograms
    """
    n_traces = metadata.shape[0]
    U_model = np.zeros((len(time), n_traces, 3))

    if model.Nphase == 0:
        return U_model

    refLat = prior.refLat
    refLon = prior.refLon
    refBaz = prior.refBaz
    srcLat = prior.srcLat
    srcLon = prior.srcLon

    stf_shift = stf_time[-1]
    stf = np.pad(stf, (0, len(time)-len(stf)), mode='constant')
    stf_W = fft(stf)
    stf_freq = fftfreq(len(stf), stf_time[1]-stf_time[0])

    for itrace in range(n_traces):
        
        trDist, trBaz = metadata[itrace]
        trDist += model.distDiff[itrace]
        trBaz += model.bazDiff[itrace]

        trLat, trLon = dest_point(srcLat, srcLon, trBaz, trDist)

        dx = (trLon - refLon) * np.cos(np.radians(refLat))
        dy = trLat - refLat

        traceZ = np.zeros(len(time))
        traceR = np.zeros(len(time))
        traceT = np.zeros(len(time))

        for iph in range(model.Nphase):
            
            slow = model.slw[iph]
            slow_x = slow * np.cos(np.radians(90-trBaz)) # refBaz
            slow_y = slow * np.sin(np.radians(90-trBaz)) # refBaz

            tshift = model.arr[iph] + (slow_x * dx + slow_y * dy)

            P_wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph]*0.25)
            S_wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph])

            P_shifted_W = P_wvlt_W * np.exp(-2j * np.pi * stf_freq * (tshift-stf_shift))
            S_shifted_W = S_wvlt_W * np.exp(-2j * np.pi * stf_freq * (tshift-stf_shift))

            if model.wvtype[iph] == 1:
                P_W = model.amp[iph] * P_shifted_W
                SV_W = np.zeros_like(P_W)
                SH_W = np.zeros_like(P_W)
            else:
                SV_W = model.amp[iph] * model.svfac[iph] * S_shifted_W
                SH_W = model.amp[iph] * (1 - model.svfac[iph]) * S_shifted_W
                P_W = np.zeros_like(SV_W)
            
            Z_W, R_W, T_W = PVH_to_ZRT(P_W, SV_W, SH_W, model.slw[iph])

            Z_W *= np.cos(np.radians(model.dip[iph]))
            
            sin_inc = np.sin(np.radians(model.dip[iph]))
            sin_azi = np.sin(np.radians(model.azi[iph]))
            cos_azi = np.cos(np.radians(model.azi[iph]))

            R_W = apply_constant_phase_shift(R_W, np.radians(model.ph_vh[iph]))
            T_W = apply_constant_phase_shift(T_W, (np.radians(model.ph_hh[iph]) + np.radians(model.ph_vh[iph])))

            R_W *= sin_inc * cos_azi
            T_W *= sin_inc * sin_azi

            Z = np.real(ifft(Z_W))
            R = np.real(ifft(R_W))
            T = np.real(ifft(T_W))

            traceZ += Z
            traceR += R
            traceT += T

        U_model[:, itrace, 0] = traceZ
        U_model[:, itrace, 1] = traceR
        U_model[:, itrace, 2] = traceT

    return U_model, P_wvlt_W, S_wvlt_W

def tstar_conv(wvfm, time, t_star):
    dt = time[1] - time[0]
    N = len(wvfm)
    
    # Frequency array scaled by t*
    f = fftfreq(N, dt) * t_star
    f0 = 1.0 * t_star  # Reference frequency (1 Hz scaled by t*)
    f_f0 = f / f0

    # Fourier transform
    W = fft(wvfm)
    
    # Attenuation operator (careful with log and divide-by-zero)
    W_attenuated = W * np.exp(-np.pi * f) * np.power(
        f_f0, 1j * 2 * f, where=f_f0 > 0, out=np.zeros_like(f_f0, dtype=complex)
        )
    W_attenuated[0] = 0  # Zero the DC component

    # Inverse FFT
    wvfm_attenuated = ifft(W_attenuated).real

    return wvfm_attenuated

def tstar_conv_freqdomain(W: np.ndarray, freqs: np.ndarray, t_star: float) -> np.ndarray:
    """
    Apply t* attenuation in the frequency domain.

    Parameters:
    - W: Fourier-transformed waveform (1D complex array)
    - freqs: frequency array corresponding to W
    - t_star: t* value

    Returns:
    - W_attenuated: Attenuated waveform in frequency domain
    """
    f = freqs * t_star
    f0 = 1.0 * t_star  # reference frequency
    f_f0 = f / f0

    # Create attenuation operator
    attenuation = np.exp(-np.pi * f) * np.power(
        f_f0, 1j * 2 * f, where=f_f0 > 0, out=np.zeros_like(f_f0, dtype=complex)
    )
    W_attenuated = W * attenuation
    W_attenuated[0] = 0  # Zero the DC component

    return W_attenuated


def PVH_to_ZRT(P, SV, SH, slw, a0=6.3, b0=3.6, radius=6371.):
    """
    Transform PVH components to ZRT components.
    
    Parameters:
    - P, SV, SH: 1D numpy arrays of the same length
    - a0, b0: float (P and S wave velocities)
    - slw: slowness (float)

    Returns:
    - Z, R, T: numpy arrays
    """
    from obspy.geodetics import degrees2kilometers
    slw /= degrees2kilometers(1, radius) # s/deg to s/km
    qa0 = np.sqrt(a0**(-2) - slw**2)
    qb0 = np.sqrt(b0**(-2) - slw**2)

    denom = (b0**(-2) - slw**2)**2 + 4 * slw**2 * qa0 * qb0
    C1 = 2 * b0**(-2) * (b0**(-2) - slw**2) / denom
    C2 = 4 * b0**(-2) * qa0 * qb0 / denom

    # Ensure inputs are arrays
    P = np.asarray(P)
    SV = np.asarray(SV)
    SH = np.asarray(SH)

    # Stack inputs into (3, N) shape
    din = np.vstack([P, SV, SH])

    # Construct transformation matrix
    RMat = np.array([
        [-a0 * qa0 * C1, b0 * slw * C2, 0],
        [ a0 * slw * C2, b0 * qb0 * C1, 0],
        [0, 0, 2]
    ])

    # Apply transformation
    dout = RMat @ din

    Z, R, T = dout[0], dout[1], dout[2]
    return Z, R, T