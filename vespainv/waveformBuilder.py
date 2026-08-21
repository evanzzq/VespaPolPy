import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from vespainv.model import VespaModel, Prior, Bookkeeping
from vespainv.utils import dest_point, apply_constant_phase_shift


def _reference_slowness_components(slow, src_array, ref_baz):
    if src_array:
        return slow * np.sin(np.radians(ref_baz)), slow * np.cos(np.radians(ref_baz))
    ref_az = (ref_baz + 180) % 360
    return slow * np.sin(np.radians(ref_az)), slow * np.cos(np.radians(ref_az))


def _metadata_to_latlon(metadata_row, src_lat, src_lon, use_distbaz):
    if use_distbaz:
        tr_dist, tr_baz = metadata_row
        return dest_point(src_lat, src_lon, tr_baz, tr_dist)
    return metadata_row


def create_U_from_model_freqdomain(
    model: VespaModel,
    metadata: np.ndarray,  # shape (n_traces, 2): [lat, lon] per row; if isMars, [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray,
    bookkeeping: Bookkeeping
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

    fitAtts = bookkeeping.fitAtts
    isMars = bookkeeping.isMars # if isMars, ref location should be S0794a (CF impact)
    locDiff = bookkeeping.locDiff
    srcArray = bookkeeping.srcArray
    n_traces = metadata.shape[0]
    U_model = np.zeros((len(time), n_traces))

    if model.Nphase == 0:
        return U_model

    refLat = bookkeeping.refLat
    refLon = bookkeeping.refLon
    
    refBaz = bookkeeping.refBaz
    srcLat = bookkeeping.srcLat
    srcLon = bookkeeping.srcLon

    stf_shift = stf_time[-1]
    stf = np.pad(stf, (0, len(time)-len(stf)), mode='constant')
    stf_W = fft(stf)
    stf_freq = fftfreq(len(stf), stf_time[1]-stf_time[0])

    for itrace in range(n_traces):
        trLat, trLon = _metadata_to_latlon(metadata[itrace], srcLat, srcLon, isMars or srcArray)

        # it doesn't make sense to do locDiff in receiver array
        # in source array setting, srcLat/srcLon is actually station coordinates
        if (isMars or srcArray) and locDiff:
            from obspy.geodetics.base import locations2degrees, gps2dist_azimuth

            # radius of Mars w/ zero flattening
            # in Mars & source array case, trLat/Lon refers to source coordinates, srcLat/Lon refers to station (InSight)
            # coordinates; therefore trBaz is the azimuth of station-->source, i.e., back azimuth
            _, _, trBaz = gps2dist_azimuth(trLat, trLon, srcLat, srcLon, a=3389500, f=0)
            trDist = locations2degrees(trLat, trLon, srcLat, srcLon)
            trDist += model.distDiff[itrace]
            trBaz  += model.bazDiff[itrace]
            trLat, trLon = dest_point(srcLat, srcLon, trBaz, trDist) # trBaz used here because the geometry is reversed

        dx = (((trLon - refLon + 180.0) % 360.0) - 180.0) * np.cos(np.radians(refLat)) # lon wrapping needed
        dy = (trLat - refLat)

        # initialize
        trace_W = np.zeros(len(time), dtype=complex)

        for iph in range(model.Nphase):
            slow_x, slow_y = _reference_slowness_components(model.slw[iph], srcArray, refBaz)
            tshift = model.arr[iph] + (slow_x * dx + slow_y * dy)

            wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph]) if fitAtts else stf_W

            shifted_W = wvlt_W * np.exp(-2j * np.pi * stf_freq * (tshift-stf_shift))

            trace_W += model.amp[iph] * shifted_W

        trace = np.real(ifft(trace_W))
        U_model[:, itrace] = trace

    return U_model


import numpy as np
from scipy.fft import fft, ifft, fftfreq
def create_U_from_model_3c_freqdomain(
    model: VespaModel,
    metadata: np.ndarray,  # shape (n_traces, 2): [lat, lon] per row; if isMars, [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray,
    bookkeeping: Bookkeeping
):
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
    - U_model: np.ndarray of shape (len(time), n_traces, 3), synthetic seismograms
    """

    fitAtts = bookkeeping.fitAtts
    fitPhase = bookkeeping.fitPhase # ph_hh and ph_vh
    isMars = bookkeeping.isMars # if isMars, ref location should be S0794a (CF impact)
    locDiff = bookkeeping.locDiff
    srcArray = bookkeeping.srcArray
    pref = bookkeeping.pref
    fst_vp = bookkeeping.fstVp
    fst_vs = bookkeeping.fstVs

    n_traces = metadata.shape[0]
    U_model = np.zeros((len(time), n_traces, 3))

    if model.Nphase == 0:
        return U_model

    refLat = bookkeeping.refLat
    refLon = bookkeeping.refLon
    
    refBaz = bookkeeping.refBaz
    srcLat = bookkeeping.srcLat
    srcLon = bookkeeping.srcLon

    stf_shift = stf_time[-1]
    stf = np.pad(stf, (0, len(time)-len(stf)), mode='constant')
    stf_W = fft(stf)
    stf_freq = fftfreq(len(stf), stf_time[1]-stf_time[0])

    for itrace in range(n_traces):
        trLat, trLon = _metadata_to_latlon(metadata[itrace], srcLat, srcLon, isMars or srcArray)

        # it doesn't make sense to do locDiff in receiver array
        # in source array setting, srcLat/srcLon is actually station coordinates
        if (isMars or srcArray) and locDiff:
            from obspy.geodetics.base import gps2dist_azimuth, locations2degrees

            # radius of Mars w/ zero flattening
            # in Mars & source array case, trLat/Lon refers to source coordinates, srcLat/Lon refers to station (InSight)
            # coordinates; therefore trBaz is the azimuth of station-->source, i.e., back azimuth
            _, _, trBaz = gps2dist_azimuth(trLat, trLon, srcLat, srcLon, a=3389500, f=0)
            trDist = locations2degrees(trLat, trLon, srcLat, srcLon)
            trDist += model.distDiff[itrace]
            trBaz  += model.bazDiff[itrace]
            trLat, trLon = dest_point(srcLat, srcLon, trBaz, trDist) # trBaz used here because the geometry is reversed

        dx = (((trLon - refLon + 180.0) % 360.0) - 180.0) * np.cos(np.radians(refLat)) # lon wrapping needed
        dy = (trLat - refLat)

        # initialize
        traceZ_W = np.zeros(len(time), dtype=complex)
        traceR_W = np.zeros(len(time), dtype=complex)
        traceT_W = np.zeros(len(time), dtype=complex)

        for iph in range(model.Nphase):
            rel_slow = model.slw[iph]
            abs_slow = rel_slow + pref
            slow_x, slow_y = _reference_slowness_components(rel_slow, srcArray, refBaz)
            tshift = model.arr[iph] + (slow_x * dx + slow_y * dy)

            P_wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph]*0.25) if fitAtts else stf_W
            S_wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph]) if fitAtts else stf_W

            P_shifted_W = P_wvlt_W * np.exp(-2j * np.pi * stf_freq * (tshift-stf_shift))
            S_shifted_W = S_wvlt_W * np.exp(-2j * np.pi * stf_freq * (tshift-stf_shift))

            if model.wvtype[iph] == 1:
                P_W = model.amp[iph] * P_shifted_W
                SV_W = np.zeros_like(P_W)
                SH_W = np.zeros_like(P_W)
            else:
                SV_W = model.amp[iph] * S_shifted_W
                SH_W = model.amp[iph] * S_shifted_W
                P_W = np.zeros_like(SV_W)
            
            sin_azi = np.sin(np.radians(model.azi[iph]))
            cos_azi = np.cos(np.radians(model.azi[iph]))

            SV_W *= cos_azi
            SH_W *= sin_azi
            
            # use absolute slowness in FST
            if isMars:
                Z_W, R_W, T_W = PVH_to_ZRT(
                    P_W, SV_W, SH_W, abs_slow,
                    a0=fst_vp, b0=fst_vs, radius=3389.5
                )
            else:
                Z_W, R_W, T_W = PVH_to_ZRT(
                    P_W, SV_W, SH_W, abs_slow,
                    a0=fst_vp, b0=fst_vs
                )

            if fitPhase:
                R_W = apply_constant_phase_shift(R_W, np.radians(model.ph_vh[iph]))
                T_W = apply_constant_phase_shift(T_W, (np.radians(model.ph_hh[iph]) + np.radians(model.ph_vh[iph])))

            traceZ_W += Z_W
            traceR_W += R_W
            traceT_W += T_W

        traceZ = np.real(ifft(traceZ_W))
        traceR = np.real(ifft(traceR_W))
        traceT = np.real(ifft(traceT_W))

        U_model[:, itrace, 0] = traceZ
        U_model[:, itrace, 1] = traceR
        U_model[:, itrace, 2] = traceT

    return U_model

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


def PVH_to_ZRT(P, SV, SH, slw, a0=6.571, b0=4.1, radius=6371.):
    """
    Transform PVH components to ZRT components.
    
    Parameters:
    - P, SV, SH: 1D numpy arrays of the same length
    - a0, b0: float (P and S wave velocities)
    - slw: slowness (float)

    Returns:
    - Z, R, T: numpy arrays
    """
    kilometers_per_degree = 2.0 * np.pi * radius / 360.0
    slw /= kilometers_per_degree  # s/deg to s/km
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
        [a0 * qa0 * C1, -b0 * slw * C2, 0], # reverse Z polarity
        [ a0 * slw * C2, b0 * qb0 * C1, 0],
        [0, 0, 2]
    ])

    # Apply transformation
    dout = RMat @ din

    Z, R, T = dout[0], dout[1], dout[2]
    return Z, R, T

############################
#          Archive         #
############################

def create_U_from_model(
    model: VespaModel,
    prior: Prior,
    metadata: np.ndarray,  # shape (n_traces, 2): [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray,
    bookkeeping: Bookkeeping
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

    phaseBaz = bookkeeping.phaseBaz
    fitAtts = bookkeeping.fitAtts

    n_traces = metadata.shape[0]

    # build synthetic on padded time axis to avoid edge problem
    dt = time[1] - time[0]
    tbuf_left = abs(stf_time[0])
    tbuf_right = abs(stf_time[-1])
    time_pad = np.arange(time[0] - tbuf_left, time[-1] + tbuf_right + dt, dt)
    U_model_pad = np.zeros((len(time_pad), n_traces))

    if model.Nphase == 0:
        return U_model

    refLat = prior.refLat
    refLon = prior.refLon
    srcLat = prior.srcLat
    srcLon = prior.srcLon

    refDist = locations2degrees(srcLat, srcLon, refLat, refLon)
    _, refBaz, _ = gps2dist_azimuth(srcLat, srcLon, refLat, refLon)

    for itrace in range(n_traces):
        
        trDist, trBaz = metadata[itrace]
        trDist += model.distDiff[itrace]
        trBaz += model.bazDiff[itrace]

        if not phaseBaz:
            dx = (trDist - refDist) * np.sin(np.radians(trBaz))
            dy = (trDist - refDist) * np.cos(np.radians(trBaz))

        trace = np.zeros(len(time_pad))

        for iph in range(model.Nphase):
            
            slow = model.slw[iph]
            slow_x = slow * np.cos(np.radians(90-trBaz)) # refBaz
            slow_y = slow * np.sin(np.radians(90-trBaz)) # refBaz

            if phaseBaz:
                dx = (trDist - refDist) * np.sin(np.radians(trBaz+model.baz[iph]))
                dy = (trDist - refDist) * np.cos(np.radians(trBaz+model.baz[iph]))

            tshift = model.arr[iph] + (slow_x * dx + slow_y * dy)

            stf_use = stf.copy()
            if fitAtts: stf_use = tstar_conv(stf_use, stf_time, model.atts[iph])

            shifted = interp1d(
                stf_time + tshift,
                stf_use,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )(time_pad)

            trace += model.amp[iph] * shifted

        U_model_pad[:, itrace] = trace
    
    # then crop back
    mask = (time_pad >= time[0]) & (time_pad <= time[-1])
    U_model = U_model_pad[mask, :]

    return U_model

def create_U_from_model_3c(
    model: VespaModel,
    prior: Prior,
    metadata: np.ndarray,  # shape (n_traces, 2): [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray
):
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

    # build synthetic on padded time axis to avoid edge problem
    dt = time[1] - time[0]
    tbuf_left = abs(stf_time[0])
    tbuf_right = abs(stf_time[-1])
    time_pad = np.arange(time[0] - tbuf_left, time[-1] + tbuf_right + dt, dt)
    U_model_pad = np.zeros((len(time_pad), n_traces, 3))

    if model.Nphase == 0:
        return U_model

    refLat = prior.refLat
    refLon = prior.refLon
    srcLat = prior.srcLat
    srcLon = prior.srcLon

    refDist = locations2degrees(srcLat, srcLon, refLat, refLon)
    _, refBaz, _ = gps2dist_azimuth(srcLat, srcLon, refLat, refLon)

    for itrace in range(n_traces):
        
        trDist, trBaz = metadata[itrace]
        trDist += model.distDiff[itrace]
        trBaz += model.bazDiff[itrace]

        dx = (trDist - refDist) * np.sin(np.radians(trBaz))
        dy = (trDist - refDist) * np.cos(np.radians(trBaz))

        traceZ = np.zeros(len(time_pad))
        traceR = np.zeros(len(time_pad))
        traceT = np.zeros(len(time_pad))

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
            )(time_pad)

            S_shifted = interp1d(
                stf_time + tshift,
                S_wvlt,
                kind="linear",
                bounds_error=False,
                fill_value=0.0
            )(time_pad)

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

        U_model_pad[:, itrace, 0] = traceZ
        U_model_pad[:, itrace, 1] = traceR
        U_model_pad[:, itrace, 2] = traceT
    
    # then crop back
    mask = (time_pad >= time[0]) & (time_pad <= time[-1])
    U_model = U_model_pad[mask, :, :]

    return U_model

def create_U_from_model_3c_freqdomain_new(
    model: VespaModel,
    prior: Prior,
    U_4D: np.ndarray, # shape: (len(time), n_traces, n_phases, 3 comp)
    metadata: np.ndarray,  # shape (n_traces, 2): [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray,
    fitAtts: bool,
    idx_all: list = None, # for phase
    idx_loc_all: list = None # for locDiff
):
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
    if idx_all is None:
        idx_all = list(range(model.Nphase))

    n_traces = metadata.shape[0]

    refLat = prior.refLat
    refLon = prior.refLon
    srcLat = prior.srcLat
    srcLon = prior.srcLon

    refDist = locations2degrees(srcLat, srcLon, refLat, refLon)
    _, refBaz, _ = gps2dist_azimuth(srcLat, srcLon, refLat, refLon)

    stf_shift = stf_time[-1]
    stf = np.pad(stf, (0, len(time)-len(stf)), mode='constant')
    stf_W = fft(stf)
    stf_freq = fftfreq(len(stf), stf_time[1]-stf_time[0])

    for itrace in range(n_traces):
        
        trDist, trBaz = metadata[itrace]
        trDist += model.distDiff[itrace]
        trBaz += model.bazDiff[itrace]

        dx = (trDist - refDist) * np.sin(np.radians(trBaz))
        dy = (trDist - refDist) * np.cos(np.radians(trBaz))

        traceZ = np.zeros(len(time))
        traceR = np.zeros(len(time))
        traceT = np.zeros(len(time))

        for iph in range(model.Nphase):

            # Only do the ones that have changed
            if iph not in idx_all:
                continue
            
            slow = model.slw[iph]
            slow_x = slow * np.cos(np.radians(90-trBaz)) # refBaz
            slow_y = slow * np.sin(np.radians(90-trBaz)) # refBaz

            tshift = model.arr[iph] + (slow_x * dx + slow_y * dy)

            P_wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph]*0.25) if fitAtts else stf_W
            S_wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph]) if fitAtts else stf_W

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

        U_4D[:, itrace, iph, 0] = traceZ
        U_4D[:, itrace, iph, 1] = traceR
        U_4D[:, itrace, iph, 2] = traceT

    return U_4D

def create_U_from_model_3c_freqdomain_old_pol_angle_def(
    model: VespaModel,
    prior: Prior,
    metadata: np.ndarray,  # shape (n_traces, 2): [dist, baz] per row
    time: np.ndarray,
    stf_time: np.ndarray,
    stf: np.ndarray,
    bookkeeping: Bookkeeping
):
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

    fitAtts = bookkeeping.fitAtts
    phaseBaz = bookkeeping.phaseBaz
    fitPhase = bookkeeping.fitPhase

    n_traces = metadata.shape[0]
    U_model = np.zeros((len(time), n_traces, 3))

    if model.Nphase == 0:
        return U_model

    refLat = prior.refLat
    refLon = prior.refLon
    srcLat = prior.srcLat
    srcLon = prior.srcLon

    refDist = locations2degrees(srcLat, srcLon, refLat, refLon)
    _, refBaz, _ = gps2dist_azimuth(srcLat, srcLon, refLat, refLon)

    stf_shift = stf_time[-1]
    stf = np.pad(stf, (0, len(time)-len(stf)), mode='constant')
    stf_W = fft(stf)
    stf_freq = fftfreq(len(stf), stf_time[1]-stf_time[0])

    for itrace in range(n_traces):
        
        trDist, trBaz = metadata[itrace]
        trDist += model.distDiff[itrace]
        trBaz += model.bazDiff[itrace]

        if not phaseBaz:
            dx = (trDist - refDist) * np.sin(np.radians(trBaz))
            dy = (trDist - refDist) * np.cos(np.radians(trBaz))

        traceZ_W = np.zeros(len(time), dtype=complex)
        traceR_W = np.zeros(len(time), dtype=complex)
        traceT_W = np.zeros(len(time), dtype=complex)

        for iph in range(model.Nphase):
            
            slow = model.slw[iph]
            slow_x = slow * np.cos(np.radians(90-trBaz)) # refBaz
            slow_y = slow * np.sin(np.radians(90-trBaz)) # refBaz

            if phaseBaz:
                dx = (trDist - refDist) * np.sin(np.radians(trBaz+model.baz[iph]))
                dy = (trDist - refDist) * np.cos(np.radians(trBaz+model.baz[iph]))

            tshift = model.arr[iph] + (slow_x * dx + slow_y * dy)

            P_wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph]*0.25) if fitAtts else stf_W
            S_wvlt_W = tstar_conv_freqdomain(stf_W, stf_freq, model.atts[iph]) if fitAtts else stf_W

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

            if fitPhase:
                R_W = apply_constant_phase_shift(R_W, np.radians(model.ph_vh[iph]))
                T_W = apply_constant_phase_shift(T_W, (np.radians(model.ph_hh[iph]) + np.radians(model.ph_vh[iph])))

            R_W *= sin_inc * cos_azi
            T_W *= sin_inc * sin_azi

            traceZ_W += Z_W
            traceR_W += R_W
            traceT_W += T_W

        traceZ = np.real(ifft(traceZ_W))
        traceR = np.real(ifft(traceR_W))
        traceT = np.real(ifft(traceT_W))

        U_model[:, itrace, 0] = traceZ
        U_model[:, itrace, 1] = traceR
        U_model[:, itrace, 2] = traceT

    return U_model
