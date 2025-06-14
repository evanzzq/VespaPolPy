import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from vespainv.utils import generate_arr

import numpy as np

def compute_log_likelihood(U_obs, U_model, sigma=0.08, CDinv=None):
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
        # residual: (T, N), CDinv: (N, N)
        term = residual @ CDinv @ residual.T  # shape: (T, T)
        return -0.5 * np.trace(term)

    # Three-component case
    elif residual.ndim == 3:
        log_like = 0.0
        for i in range(3):  # loop over components
            r_i = residual[:, :, i]  # shape: (T, N)
            CDinv_i = CDinv[i]       # shape: (N, N)
            term = r_i @ CDinv_i @ r_i.T  # shape: (T, T)
            log_like += -0.5 * np.trace(term)
        return log_like

    else:
        raise ValueError("U_obs must be 2D or 3D array.")

def birth(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase < prior.maxN:
        model_new.Nphase += 1
        model_new.arr = np.append(
            model_new.arr, generate_arr(prior.timeRange, model_new.arr, prior.minSpace)
            )
        model_new.slw = np.append(model_new.slw, np.random.uniform(prior.slwRange[0], prior.slwRange[1]))
        model_new.amp = np.append(model_new.amp, np.random.uniform(prior.ampRange[0], prior.ampRange[1]))
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
        model_new.dip = np.append(model_new.dip, np.random.uniform(prior.dipRange[0], prior.dipRange[1]))
        model_new.azi = np.append(model_new.azi, np.random.uniform(prior.aziRange[0], prior.aziRange[1]))
        model_new.ph_hh = np.append(model_new.ph_hh, np.random.uniform(prior.ph_hhRange[0], prior.ph_hhRange[1]))
        model_new.ph_vh = np.append(model_new.ph_vh, np.random.uniform(prior.ph_vhRange[0], prior.ph_vhRange[1]))
        model_new.atts = np.append(model_new.atts, np.random.uniform(prior.attsRange[0], prior.attsRange[1]))
        model_new.svfac = np.append(model_new.svfac, np.random.uniform(prior.svfacRange[0], prior.svfacRange[1]))
        model_new.wvtype = np.append(model_new.wvtype, np.random.randint(2))
        return model_new, True
    else:
        return model_new, False


def death(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase > 0:
        model_new.Nphase -= 1
        idx = np.random.randint(model_new.Nphase) if model_new.Nphase > 0 else 0
        model_new.arr = np.delete(model_new.arr, idx)
        model_new.slw = np.delete(model_new.slw, idx)
        model_new.amp = np.delete(model_new.amp, idx)
        success = True
    else:
        success = False
    return model_new, success

def death3c(model, prior):
    model_new = copy.deepcopy(model)
    success = False
    if model_new.Nphase > 0:
        model_new.Nphase -= 1
        success = True
        idx = np.random.randint(model_new.Nphase) if model_new.Nphase > 0 else 0
        model_new.arr = np.delete(model_new.arr, idx)
        model_new.slw = np.delete(model_new.slw, idx)
        model_new.amp = np.delete(model_new.amp, idx)
        model_new.dip = np.delete(model_new.dip, idx)
        model_new.azi = np.delete(model_new.azi, idx)
        model_new.ph_hh = np.delete(model_new.ph_hh, idx)
        model_new.ph_vh = np.delete(model_new.ph_vh, idx)
        model_new.atts = np.delete(model_new.atts, idx)
        model_new.svfac = np.delete(model_new.svfac, idx)
        model_new.wvtype = np.delete(model_new.wvtype, idx)
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

def update_nc(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    eps = np.finfo(float).eps
    model_new.nc1 += prior.nc1Std * np.random.randn()
    model_new.nc1 = np.clip(model_new.nc1, eps, 1)
    model_new.nc2 += prior.nc2Std * np.random.randn()
    # Return
    return model_new, True

def update_sig(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Update
    eps = np.finfo(float).eps
    model_new.sig = np.maximum(eps, model_new.sig + prior.sigStd * np.random.randn() * prior.stdU)
    # Return
    return model_new, True

def update_dist(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Ntrace)
    model_new.distDiff[idx] += prior.distStd * np.random.randn()
    # Check range
    if not (prior.distRange[0] <= model_new.distDiff[idx] <= prior.distRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_baz(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Ntrace)
    model_new.bazDiff[idx] += prior.bazStd * np.random.randn()
    # Check range
    if not (prior.bazRange[0] <= model_new.bazDiff[idx] <= prior.bazRange[1]):
        return model, False
    # Success, return
    return model_new, True

def update_dip(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Nphase)
    model_new.dip[idx] += prior.dipStd * np.random.randn()
    # Check range
    if not (prior.dipRange[0] <= model_new.dip[idx] <= prior.dipRange[1]):
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

def update_svfac(model, prior):
    # Copy model
    model_new = copy.deepcopy(model)
    # Select a trace and update
    idx = np.random.randint(model_new.Nphase)
    model_new.svfac[idx] += prior.svfacStd * np.random.randn()
    # Check range and wave type
    if not (prior.svfacRange[0] <= model_new.svfac[idx] <= prior.svfacRange[1]) or model_new.wvtype[idx] == 1:
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

def choose_actions(locDiff, fitNoise, actionsPerStep):
    actionPool = [0, 1, 2, 3, 4]
    if locDiff:
        actionPool.extend([7, 8])
    if fitNoise:
        actionPool.extend([5, 6])

    actionPool = np.array(actionPool)

    if 5 in actionPool:
        base_actions = actionPool[actionPool != 5]
        base_weight = 0.99 / len(base_actions)
        weights = np.array([0.01 if a == 5 else base_weight for a in actionPool])
    else:
        weights = np.full(len(actionPool), 1.0 / len(actionPool))

    return np.random.choice(actionPool, size=actionsPerStep, replace=True, p=weights)

def rjmcmc_run(U_obs, CDinv, metadata, Utime, stf, prior, bookkeeping, saveDir):

    from vespainv.model import VespaModel, Prior
    from vespainv.waveformBuilder import create_U_from_model

    trace_len = U_obs.shape[0]
    n_traces = U_obs.shape[1]

    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep
    locDiff = bookkeeping.locDiff
    fitNoise = bookkeeping.fitNoise

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
    logL = compute_log_likelihood(U_obs, U_model, CDinv=CDinv)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    maxN = prior.maxN

    s2 = s3 = s4 = a2 = a3 = a4 = 0 # for logging success rates: s(uccess) and a(ll)

    for iStep in range(totalSteps):

        # dynamically change allowed max phase number
        prior.maxN = int(min(iStep / burnInSteps * maxN + 1, maxN))

        if model.Nphase == 0:
            actions = [0]
        else:
            actions = choose_actions(locDiff, fitNoise, actionsPerStep)
        
        model_new = model

        for i in range(len(actions)):
            iAction = actions[i] if not model_new.Nphase > 0 else 0
            if iAction == 0:
                model_new, _ = birth(model_new, prior)
            elif iAction == 1:
                model_new, _ = death(model_new, prior)
                if model_new.Nphase == 0 and i+1 < len(actions):
                    if actions[i+1] in [2, 3, 4]:
                        actions[i+1] = 0
            elif iAction == 2:
                model_new, sucess = update_arr(model_new, prior)
                s2 += 1 if sucess else 0
                a2 += 1
            elif iAction == 3:
                model_new, sucess = update_slw(model_new, prior)
                s3 += 1 if sucess else 0
                a3 += 1
            elif iAction == 4:
                model_new, sucess = update_amp(model_new, prior)
                s4 += 1 if sucess else 0
                a4 += 1
            elif iAction == 5:
                model_new, change_corr = update_nc(model_new, prior)
            elif iAction == 6:
                model_new, _ = update_sig(model_new, prior)
            elif iAction == 7:
                model_new, _ = update_dist(model_new, prior)
            elif iAction == 8:
                model_new, _ = update_baz(model_new, prior)

        U_model_new = create_U_from_model(model_new, prior, metadata, Utime, stf_time, stf_data)
        new_logL = compute_log_likelihood(U_obs, U_model_new, CDinv=CDinv)

        log_accept_ratio = ((new_logL - logL) + np.log((model.Nphase + 1) / model_new.Nphase)) if model_new.Nphase > 0 else (new_logL - logL)
        if np.log(np.random.rand()) < log_accept_ratio:
            model = model_new
            U_model = U_model_new
            logL = new_logL

        logL_trace.append(logL)

        # Save only selected models after burn-in
        if iStep >= burnInSteps and (iStep - burnInSteps) % save_interval == 0:
            samples.append(model)
        
        # Checkpoint log/plot every 1%
        if (iStep + 1) % checkpoint_interval == 0:
            # Save (overwrite) log-likelihood plot
            fig, ax = plt.subplots()
            ax.plot(logL_trace, 'k-')
            ax.set_xlabel("Step")
            ax.set_ylabel("log Likelihood")
            fig.tight_layout()
            fig.savefig(os.path.join(saveDir, "logL.png"))  # overwrites each time
            plt.close(fig)

            # Overwrite progress log
            elapsed = time.time() - start_time
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(saveDir, "progress_log.txt"), "a") as f:
                f.write(f"[{now}] Step {iStep+1}/{totalSteps}, Elapsed: {elapsed:.2f} sec\n")

    with open(os.path.join(saveDir, "progress_log.txt"), "a") as f:
        f.write(f"Acceptance rates: arr {s2/a2*100:.2f}%, slw {s3/a3*100:.2f}%, amp {s4/a4*100:.2f}%\n")

    return samples, logL_trace

def rjmcmc_run3c(U_obs, CDinv, metadata, Utime, stf, prior, bookkeeping, saveDir):

    from vespainv.model import VespaModel3c, Prior3c
    from vespainv.waveformBuilder import create_U_from_model_3c_freqdomain, create_U_from_model_3c_freqdomain_new

    trace_len = U_obs.shape[0]
    n_traces = U_obs.shape[1]

    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep
    locDiff = bookkeeping.locDiff
    fitAtts = bookkeeping.fitAtts

    # Extract stf and its time vectors
    stf_time = stf[:, 0]
    stf_data = stf[:, 1]

    # Start from a random model with one phase
    model = VespaModel3c.create_random(
        Nphase=1, Ntrace=n_traces, time=Utime, prior=prior
        )

    samples = []
    logL_trace = []

    U_model = create_U_from_model_3c_freqdomain(model, prior, metadata, Utime, stf_time, stf_data, fitAtts)
    logL = compute_log_likelihood(U_obs, U_model, CDinv=CDinv)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100
    maxN = prior.maxN
    Nphase = []

    # --- Sliding window setup ---
    window_size = 1000
    n_actions = 14

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
            actionPool = np.arange(11)
            if fitAtts: actionPool = np.append(actionPool, [11])
            if locDiff: actionPool = np.append(actionPool, [12, 13])
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
                model_new, success = death3c(model_new, prior)
            elif iAction == 2:
                model_new, success = update_arr(model_new, prior)
            elif iAction == 3:
                model_new, success = update_slw(model_new, prior)
            elif iAction == 4:
                model_new, success = update_amp(model_new, prior)
            elif iAction == 5:
                model_new, success = update_dip(model_new, prior)
            elif iAction == 6:
                model_new, success = update_azi(model_new, prior)
            elif iAction == 7:
                model_new, success = update_ph_hh(model_new, prior)
            elif iAction == 8:
                model_new, success = update_ph_vh(model_new, prior)
            elif iAction == 9:
                model_new, success = update_svfac(model_new, prior)
            elif iAction == 10:
                model_new, success = update_wvtype(model_new, prior)
            elif iAction == 11:
                model_new, success = update_atts(model_new, prior)
            elif iAction == 12:
                model_new, success = update_dist(model_new, prior)
            elif iAction == 13:
                model_new, success = update_baz(model_new, prior)

            if success:
                applied_actions.append(iAction)
                action_counts[iAction].append(1)  # always count attempt

        # Evaluate proposed model
        U_model_new = create_U_from_model_3c_freqdomain(model_new, prior, metadata, Utime, stf_time, stf_data, fitAtts)
        new_logL = compute_log_likelihood(U_obs, U_model_new, CDinv=CDinv)

        log_accept_ratio = (new_logL - logL)
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

    return samples, logL_trace