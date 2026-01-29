import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from vespainv.utils import generate_arr

import numpy as np

def compute_log_likelihood(U_obs, U_model, CDinv=None, sigma=0.04):
    """
    Compute log-likelihood for 1- or 3-component seismic data.
    
    Parameters:
        U_obs : ndarray
            Observed data. Shape (T, N) for 1-comp, (T, N, 3) for 3-comp.
        U_model : ndarray
            Modeled data. Same shape as U_obs.
        sigma : float
            Noise standard deviation for diagonal covariance case.
        CDinv : None or ndarray or list of ndarray
            - If None: use diagonal covariance with sigma.
            - If 1-comp: 2D ndarray (N, N)
            - If 3-comp: list of 3 ndarrays, each (N, N)
    
    Returns:
        log_likelihood : float
    """
    residual = U_obs - U_model

    if CDinv is None:
        return -0.5 * np.sum((residual / sigma)**2)

    # One-component case
    if residual.ndim == 2:
        # residual: (T, N), CDinv: (T, T)
        term = residual.T @ CDinv @ residual  # shape: (N, N)
        return -0.5 * np.trace(term)

    # Three-component case
    elif residual.ndim == 3:
        log_like = 0.0
        for i in range(3):  # loop over components
            r_i = residual[:, :, i]  # shape: (T, N)
            CDinv_i = CDinv[i]       # shape: (T, T)
            term = r_i.T @ CDinv_i @ r_i  # shape: (N, N)
            log_like += -0.5 * np.trace(term)
        return log_like

    else:
        raise ValueError("U_obs must be 2D or 3D array.")

def compute_log_likelihood_L1(U_obs, U_model, CD_sqrt_inv=None):
    """
    Compute L1 log-likelihood for 1- or 3-component seismic data with whitening.

    Parameters
    ----------
    U_obs : ndarray
        Observed data. Shape (T, N) for 1-comp, (T, N, 3) for 3-comp.
    U_model : ndarray
        Modeled data. Same shape as U_obs.
    CD_sqrt_inv : None, ndarray, or list of ndarray
        - If None: assumes identity whitening (no correlation).
        - 1-comp: (T, T) array
        - 3-comp: list of (T, T) arrays, one per component

    Returns
    -------
    log_likelihood : float
        L1 log-likelihood (up to additive constant).
    """
    residual = U_obs - U_model

    if CD_sqrt_inv is None:
        whitened = residual
    elif residual.ndim == 2:
        whitened = CD_sqrt_inv @ residual  # whiten time dimension
    elif residual.ndim == 3:
        whitened = np.empty_like(residual)
        for i in range(3):
            whitened[:, :, i] = CD_sqrt_inv[i] @ residual[:, :, i]
    else:
        raise ValueError("U_obs must be 2D or 3D array.")

    return -np.sum(np.abs(whitened))

def birth(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase < prior.maxN:
        model_new.Nphase += 1
        model_new.arr = np.append(
            model_new.arr, generate_arr(prior.timeRange, model_new.arr, prior.minSpace)
            )
        model_new.slw = np.append(model_new.slw, np.random.uniform(prior.slwRange[0], prior.slwRange[1]))
        model_new.amp = np.append(model_new.amp, np.random.uniform(prior.ampRange[0], prior.ampRange[1]))
        model_new.atts = np.append(model_new.atts, np.random.uniform(prior.attsRange[0], prior.attsRange[1]))
        success = True
    else:
        success = False
    return model_new, success

def birth3c(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase < prior.maxN:
        model_new.Nphase += 1
        model_new.arr = np.append(
            model_new.arr, generate_arr(prior.timeRange, model_new.arr, prior.minSpace)
            )
        model_new.slw = np.append(model_new.slw, np.random.uniform(prior.slwRange[0], prior.slwRange[1]))
        model_new.amp = np.append(model_new.amp, np.random.uniform(prior.ampRange[0], prior.ampRange[1]))
        model_new.azi = np.append(model_new.azi, np.random.uniform(prior.aziRange[0], prior.aziRange[1]))
        model_new.ph_hh = np.append(model_new.ph_hh, np.random.uniform(prior.ph_hhRange[0], prior.ph_hhRange[1]))
        model_new.ph_vh = np.append(model_new.ph_vh, np.random.uniform(prior.ph_vhRange[0], prior.ph_vhRange[1]))
        model_new.atts = np.append(model_new.atts, np.random.uniform(prior.attsRange[0], prior.attsRange[1]))
        model_new.wvtype = np.append(model_new.wvtype, np.random.randint(2))
        return model_new, True
    else:
        return model_new, False


def death(model):
    model_new = copy.deepcopy(model)
    if model_new.Nphase > 0:
        idx = np.random.randint(model_new.Nphase) if model_new.Nphase > 0 else 0
        model_new.arr = np.delete(model_new.arr, idx)
        model_new.slw = np.delete(model_new.slw, idx)
        model_new.amp = np.delete(model_new.amp, idx)
        model_new.atts = np.delete(model_new.atts, idx)
        model_new.Nphase -= 1
        success = True
    else:
        success = False
    return model_new, success

def death3c(model):
    model_new = copy.deepcopy(model)
    success = False
    if model_new.Nphase > 0:
        idx = np.random.randint(model_new.Nphase) if model_new.Nphase > 0 else 0
        model_new.arr = np.delete(model_new.arr, idx)
        model_new.slw = np.delete(model_new.slw, idx)
        model_new.amp = np.delete(model_new.amp, idx)
        model_new.azi = np.delete(model_new.azi, idx)
        model_new.ph_hh = np.delete(model_new.ph_hh, idx)
        model_new.ph_vh = np.delete(model_new.ph_vh, idx)
        model_new.atts = np.delete(model_new.atts, idx)
        model_new.wvtype = np.delete(model_new.wvtype, idx)
        model_new.Nphase -= 1
        success = True
    return model_new, success


def update_arr(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase and update
    idx = np.random.randint(model_new.Nphase)
    model_new.arr[idx] += prior.arrStd * np.random.randn()
    # Check range
    if not (prior.timeRange[0] <= model_new.arr[idx] <= prior.timeRange[1]):
        return model, False
    # Check spacing with all other phases
    arr_others = np.delete(model_new.arr, idx)
    if np.any(np.abs(arr_others - model_new.arr[idx]) < prior.minSpace):
        return model, False
    # Success, return
    return model_new, True


def update_slw(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase and update
    idx = np.random.randint(model_new.Nphase)
    model_new.slw[idx] += prior.slwStd * np.random.randn()
    # Check range
    if not (prior.slwRange[0] <= model_new.slw[idx] <= prior.slwRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_amp(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a phase and update
    idx = np.random.randint(model_new.Nphase)
    model_new.amp[idx] += prior.ampStd * np.random.randn()
    # Check range
    if not (prior.ampRange[0] <= model_new.amp[idx] <= prior.ampRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_distDiff(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Ntrace)
    model_new.distDiff[idx] += prior.distDiffStd * np.random.randn()
    # Check range
    if not (prior.distDiffRange[0] <= model_new.distDiff[idx] <= prior.distDiffRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_bazDiff(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Ntrace)
    model_new.bazDiff[idx] += prior.bazDiffStd * np.random.randn()
    # Check range
    if not (prior.bazDiffRange[0] <= model_new.bazDiff[idx] <= prior.bazDiffRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_azi(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Nphase)
    model_new.azi[idx] += prior.aziStd * np.random.randn()
    # Check range
    if not (prior.aziRange[0] <= model_new.azi[idx] <= prior.aziRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_ph_hh(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Nphase)
    model_new.ph_hh[idx] += prior.ph_hhStd * np.random.randn()
    # Check range
    if not (prior.ph_hhRange[0] <= model_new.ph_hh[idx] <= prior.ph_hhRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_ph_vh(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Nphase)
    model_new.ph_vh[idx] += prior.ph_vhStd * np.random.randn()
    # Check range
    if not (prior.ph_vhRange[0] <= model_new.ph_vh[idx] <= prior.ph_vhRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_atts(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Nphase)
    model_new.atts[idx] += prior.attsStd * np.random.randn()
    # Check range
    if not (prior.attsRange[0] <= model_new.atts[idx] <= prior.attsRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_loge(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    model_new.loge += prior.logeStd * np.random.randn()
    # Check range
    if not (prior.logeRange[0] <= model_new.loge <= prior.logeRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_wvtype(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Nphase)
    model_new.wvtype[idx] = np.abs(model_new.wvtype[idx] - 1)
    # Success, return
    return model_new, True

def rjmcmc_run(U_obs, CDinv, CD_sqrt_inv, metadata, Utime, stf, prior, bookkeeping, saveDir):

    from vespainv.model import VespaModel
    from vespainv.waveformBuilder import create_U_from_model_freqdomain
    
    if CD_sqrt_inv is not None: CD_sqrt_inv = np.asarray(CD_sqrt_inv)
    if CDinv is not None: CDinv = np.asarray(CDinv)

    trace_len = U_obs.shape[0]
    n_traces = U_obs.shape[1]

    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep
    fitAtts = bookkeeping.fitAtts
    fitLoge = bookkeeping.fitLoge
    locDiff = bookkeeping.locDiff
    normOpt = bookkeeping.normOpt

    # Extract stf and its time vectors
    stf_time = stf[:, 0]
    stf_data = stf[:, 1]

    # # Start from a random model
    # model = VespaModel.create_random(
    #     Nphase=np.random.randint(1, Nmax + 1), Ntrace=n_traces, time=Utime, prior=prior
    #     )
    
    # Start from an empty model
    model = VespaModel.create_empty(Ntrace=n_traces, prior=prior)

    trace_shape = (trace_len, n_traces)
    samples = []
    logL_trace = []

    U_model = np.zeros(trace_shape)
    if normOpt == 1: 
        if CD_sqrt_inv is not None:      
            CD_sqrt_inv_c = CD_sqrt_inv * np.exp(-0.5 * model.loge)
        logL = compute_log_likelihood_L1(U_obs, U_model, CD_sqrt_inv_c)
    if normOpt == 2:
        if CDinv is not None:
            CDinv_c = CDinv * np.exp(-model.loge)
        logL = compute_log_likelihood(U_obs, U_model, CDinv_c)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    maxN = prior.maxN
    Nphase = []

    # --- Sliding window setup ---
    window_size = 1000
    n_actions = 8
    # Track recent attempts and successes
    action_counts = {i: deque(maxlen=window_size) for i in range(n_actions)}
    action_success = {i: deque(maxlen=window_size) for i in range(n_actions)}
    # Track time-series of acceptance ratios
    acceptance_ratios = {i: [] for i in range(n_actions)}

    for iStep in range(totalSteps):

        # # dynamically change allowed max phase number
        # prior.maxN = int(min(iStep / burnInSteps * maxN + 1, maxN))

        if model.Nphase == 0:
            actions = [0]
        else:
            actionPool = np.arange(4)
            if fitAtts: actionPool = np.append(actionPool, [5])
            if fitLoge: actionPool = np.append(actionPool, [6])
            if locDiff: actionPool = np.append(actionPool, [7, 8])
            actions = np.random.choice(actionPool, size=actionsPerStep, replace=False)

        model_new = model
        applied_actions = []  # Track successful actions (not yet accepted)

        for iAction in actions:

            if model_new.Nphase == 0:
                iAction = 0  # force birth if no phases
            success = False

            if iAction == 0:
                model_new, success = birth(model_new, prior)
            elif iAction == 1:
                model_new, success = death(model_new)
            elif iAction == 2:
                model_new, success = update_arr(model_new, prior)
            elif iAction == 3:
                model_new, success = update_slw(model_new, prior)
            elif iAction == 4:
                model_new, success = update_amp(model_new, prior)
            elif iAction == 5:
                model_new, success = update_atts(model_new, prior)
            elif iAction == 6:
                model_new, success = update_loge(model_new, prior)
            elif iAction == 7:
                model_new, success = update_distDiff(model_new, prior)
            elif iAction == 8:
                model_new, success = update_bazDiff(model_new, prior)
            
            if success:
                applied_actions.append(iAction)
                action_counts[iAction].append(1)  # always count attempt

        U_model_new = create_U_from_model_freqdomain(model_new, metadata, Utime, stf_time, stf_data, bookkeeping)
        
        if normOpt == 1: 
            if CD_sqrt_inv is not None:      
                CD_sqrt_inv_c = CD_sqrt_inv * np.exp(-0.5 * model_new.loge)
            new_logL = compute_log_likelihood_L1(U_obs, U_model_new, CD_sqrt_inv_c)
        if normOpt == 2:
            if CDinv is not None:
                CDinv_c = CDinv * np.exp(-model_new.loge)
            new_logL = compute_log_likelihood(U_obs, U_model_new, CDinv_c)

        log_accept_ratio = (new_logL - logL) + np.log((model.Nphase + 1) / (model_new.Nphase + 1)) + trace_len * (model.loge - model_new.loge)
        
        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            U_model = U_model_new
            logL = new_logL

        logL_trace.append(logL)
        Nphase.append(model.Nphase)

        # Compute sliding-window acceptance ratios
        if iStep >= window_size:
            for i in range(n_actions):
                attempts = sum(action_counts[i])
                successes = sum(action_success[i])
                ratio = successes / attempts if attempts > 0 else 0.0
                acceptance_ratios[i].append(ratio)

        # Save only selected models after burn-in
        if iStep >= burnInSteps and (iStep - burnInSteps) % save_interval == 0:
            samples.append(model)
        
        # Checkpoint log/plot every 1%
        if (iStep + 1) % checkpoint_interval == 0:
            # Save (overwrite) log-likelihood plot
            fig, ax1 = plt.subplots()
            # Plot log-likelihood on left y-axis
            ax1.plot(logL_trace, 'k-', label='logL')
            ax1.set_xlabel("Step")
            ax1.set_ylabel("log Likelihood", color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            # Create second y-axis for Nphase
            ax2 = ax1.twinx()
            ax2.plot(Nphase, 'b--', label='Nphase')
            ax2.set_ylabel("Nphase", color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            # Optional: combined legend
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            fig.tight_layout()
            fig.savefig(os.path.join(saveDir, "logL_nphase.png"))
            plt.close(fig)

            # Plot acceptance ratio
            ncols = 2
            nrows = int(np.ceil(n_actions / ncols))

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 2.5 * nrows), sharex=True)

            for i in range(n_actions):
                row = i // ncols
                col = i % ncols
                ax = axes[row, col] if nrows > 1 else axes[col]  # handle 1-row case
                if acceptance_ratios[i]:  # avoid empty
                    ax.plot(acceptance_ratios[i], color='tab:blue')
                # ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"Action {i}", fontsize=10)
                ax.set_ylabel("Acc. Ratio", fontsize=9)
                ax.grid(True)

            # Set common x-label
            for ax in axes[-1, :] if nrows > 1 else [axes[-1]]:
                ax.set_xlabel("Step index", fontsize=10)

            fig.suptitle("Sliding-window Acceptance Ratios (Window = 1000 steps)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
            fig.savefig(os.path.join(saveDir, "acceptance_ratios.png"))
            plt.close(fig)

            # Overwrite progress log
            elapsed = time.time() - start_time
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(saveDir, "progress_log.txt"), "a") as f:
                f.write(f"[{now}] Step {iStep+1}/{totalSteps}, Elapsed: {elapsed:.2f} sec\n")

    return samples, logL_trace, Nphase

def rjmcmc_run3c(U_obs, CDinv, CD_sqrt_inv, metadata, Utime, stf, prior, bookkeeping, saveDir):

    from vespainv.model import VespaModel3c
    from vespainv.waveformBuilder import create_U_from_model_3c_freqdomain

    if CD_sqrt_inv is not None: CD_sqrt_inv = np.asarray(CD_sqrt_inv)
    if CDinv is not None: CDinv = np.asarray(CDinv)
    
    trace_len = U_obs.shape[0]
    n_traces = U_obs.shape[1]

    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep
    locDiff = bookkeeping.locDiff
    fitAtts = bookkeeping.fitAtts
    fitLoge = bookkeeping.fitLoge
    fitPhase = bookkeeping.fitPhase
    normOpt = bookkeeping.normOpt

    # Extract stf and its time vectors
    stf_time = stf[:, 0]
    stf_data = stf[:, 1]

    # Start from a random model with one phase
    model = VespaModel3c.create_random(
        Nphase=1, Ntrace=n_traces, time=Utime, prior=prior
        )

    samples = []
    logL_trace = []

    U_model = create_U_from_model_3c_freqdomain(model, metadata, Utime, stf_time, stf_data, bookkeeping)
    
    if normOpt == 1: 
        if CD_sqrt_inv is not None:      
            CD_sqrt_inv_c = CD_sqrt_inv * np.exp(-0.5 * model.loge)
            logL = compute_log_likelihood_L1(U_obs, U_model, CD_sqrt_inv_c)
        else:
            logL = compute_log_likelihood_L1(U_obs, U_model, CD_sqrt_inv)
    if normOpt == 2:
        if CDinv is not None:
            CDinv_c = CDinv * np.exp(-model.loge)
            logL = compute_log_likelihood(U_obs, U_model, CDinv_c)
        else:
            logL = compute_log_likelihood(U_obs, U_model, CDinv)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100
    maxN = prior.maxN
    Nphase = []

    # --- Sliding window setup ---
    window_size = 1000
    n_actions = 12

    # Track recent attempts and successes
    action_counts = {i: deque(maxlen=window_size) for i in range(n_actions)}
    action_success = {i: deque(maxlen=window_size) for i in range(n_actions)}

    # Track time-series of acceptance ratios
    acceptance_ratios = {i: [] for i in range(n_actions)}

    for iStep in range(totalSteps):

        # dynamically change allowed max phase number
        # prior.maxN = int(min(iStep / burnInSteps * maxN + 1, maxN))

        if model.Nphase == 0:
            actions = [0]
        else:
            actionPool = np.arange(7)
            if fitPhase: actionPool = np.append(actionPool, [7, 8])
            if fitAtts: actionPool = np.append(actionPool, [9])
            if fitLoge: actionPool = np.append(actionPool, [10])
            if locDiff: actionPool = np.append(actionPool, [11, 12])
            actions = np.random.choice(actionPool, size=actionsPerStep, replace=False)

        model_new = model
        applied_actions = []  # Track successful actions (not yet accepted)

        for iAction in actions:
            if model_new.Nphase == 0:
                iAction = 0  # force birth if no phases

            success = False

            if iAction == 0:
                model_new, success = birth3c(model_new, prior)
            elif iAction == 1:
                model_new, success = death3c(model_new)
            elif iAction == 2:
                model_new, success = update_arr(model_new, prior)
            elif iAction == 3:
                model_new, success = update_slw(model_new, prior)
            elif iAction == 4:
                model_new, success = update_amp(model_new, prior)
            elif iAction == 5:
                model_new, success = update_azi(model_new, prior)
            elif iAction == 6:
                model_new, success = update_wvtype(model_new, prior)
            elif iAction == 7:
                model_new, success = update_ph_hh(model_new, prior)
            elif iAction == 8:
                model_new, success = update_ph_vh(model_new, prior)
            elif iAction == 9:
                model_new, success = update_atts(model_new, prior)
            elif iAction == 10:
                model_new, success = update_loge(model_new, prior)
            elif iAction == 11:
                model_new, success = update_distDiff(model_new, prior)
            elif iAction == 12:
                model_new, success = update_bazDiff(model_new, prior)

            if success:
                applied_actions.append(iAction)
                action_counts[iAction].append(1)  # always count attempt

        # Evaluate proposed model
        U_model_new = create_U_from_model_3c_freqdomain(model_new, metadata, Utime, stf_time, stf_data, bookkeeping)
        
        if normOpt == 1: 
            if CD_sqrt_inv is not None:      
                CD_sqrt_inv_c = CD_sqrt_inv * np.exp(-0.5 * model_new.loge)
                new_logL = compute_log_likelihood_L1(U_obs, U_model_new, CD_sqrt_inv_c)
            else:
                new_logL = compute_log_likelihood_L1(U_obs, U_model_new, CD_sqrt_inv)
        if normOpt == 2:
            if CDinv is not None:
                CDinv_c = CDinv * np.exp(-model_new.loge)
                new_logL = compute_log_likelihood(U_obs, U_model_new, CDinv_c)
            else:
                new_logL = compute_log_likelihood(U_obs, U_model_new, CDinv)

        log_accept_ratio = (new_logL - logL) + np.log((model.Nphase + 1) / (model_new.Nphase + 1)) + trace_len * (model.loge - model_new.loge)
        
        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            U_model = U_model_new
            logL = new_logL
            for iAction in applied_actions:
                action_success[iAction].append(1)
        else:
            for iAction in applied_actions:
                action_success[iAction].append(0)

        logL_trace.append(logL)
        Nphase.append(model.Nphase)

        # Compute sliding-window acceptance ratios
        if iStep >= window_size:
            for i in range(n_actions):
                attempts = sum(action_counts[i])
                successes = sum(action_success[i])
                ratio = successes / attempts if attempts > 0 else 0.0
                acceptance_ratios[i].append(ratio)

        # Save only selected models after burn-in
        if iStep >= burnInSteps and (iStep - burnInSteps) % save_interval == 0:
            samples.append(model)
        
        # Checkpoint log/plot every 1%
        if (iStep + 1) % checkpoint_interval == 0:
            # Save (overwrite) log-likelihood plot
            fig, ax1 = plt.subplots()
            # Plot log-likelihood on left y-axis
            ax1.plot(logL_trace, 'k-', label='logL')
            ax1.set_xlabel("Step")
            ax1.set_ylabel("log Likelihood", color='k')
            ax1.tick_params(axis='y', labelcolor='k')
            # Create second y-axis for Nphase
            ax2 = ax1.twinx()
            ax2.plot(Nphase, 'b--', label='Nphase')
            ax2.set_ylabel("Nphase", color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            # Optional: combined legend
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
            fig.tight_layout()
            fig.savefig(os.path.join(saveDir, "logL_nphase.png"))
            plt.close(fig)

            # Setup: number of rows/columns
            ncols = 2
            nrows = int(np.ceil(n_actions / ncols))

            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 2.5 * nrows), sharex=True)

            for i in range(n_actions):
                row = i // ncols
                col = i % ncols
                ax = axes[row, col] if nrows > 1 else axes[col]  # handle 1-row case
                if acceptance_ratios[i]:  # avoid empty
                    ax.plot(acceptance_ratios[i], color='tab:blue')
                # ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"Action {i}", fontsize=10)
                ax.set_ylabel("Acc. Ratio", fontsize=9)
                ax.grid(True)

            # Set common x-label
            for ax in axes[-1, :] if nrows > 1 else [axes[-1]]:
                ax.set_xlabel("Step index", fontsize=10)

            fig.suptitle("Sliding-window Acceptance Ratios (Window = 1000 steps)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
            fig.savefig(os.path.join(saveDir, "acceptance_ratios.png"))
            plt.close(fig)

            # Overwrite progress log
            elapsed = time.time() - start_time
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(saveDir, "progress_log.txt"), "a") as f:
                f.write(f"[{now}] Step {iStep+1}/{totalSteps}, Elapsed: {elapsed:.2f} sec\n")

    return samples, logL_trace, Nphase