import copy, time, os, datetime
import numpy as np
import matplotlib.pyplot as plt
from vespainv.utils import generate_arr

def compute_log_likelihood(U_obs, U_model, sigma=0.08):
    residual = U_obs - U_model
    return -0.5 * np.sum((residual / sigma)**2)

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
        success = True
    else:
        success = False
    return model_new, success


def death(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase > 1:
        model_new.Nphase -= 1
        idx = np.random.randint(model_new.Nphase)
        model_new.arr = np.delete(model_new.arr, idx)
        model_new.slw = np.delete(model_new.slw, idx)
        model_new.amp = np.delete(model_new.amp, idx)
        success = True
    else:
        success = False
    return model_new, success

def death3c(model, prior):
    model_new = copy.deepcopy(model)
    if model_new.Nphase > 1:
        model_new.Nphase -= 1
        idx = np.random.randint(model_new.Nphase)
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
        success = True
    else:
        success = False
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
    # Check range
    if not (prior.svfacRange[0] <= model_new.svfac[idx] <= prior.svfacRange[1]):
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

def rjmcmc_run(U_obs, metadata, Utime, stf, prior, bookkeeping, saveDir):

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
    Temp = bookkeeping.Temp

    # Extract stf and its time vectors
    stf_time = stf[:, 0]
    stf_data = stf[:, 1]

    # # Start from a random model
    # model = VespaModel.create_random(
    #     Nphase=np.random.randint(1, Nmax + 1), Ntrace=n_traces, time=Utime, prior=prior
    #     )
    
    # Start from an empty model
    model = VespaModel.create_empty(Ntrace=n_traces)

    trace_shape = (trace_len, n_traces)
    samples = []
    logL_trace = []

    U_model = np.zeros(trace_shape)
    logL = compute_log_likelihood(U_obs, U_model)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    for iStep in range(totalSteps):

        if model.Nphase == 0:
            actions = [0]
        else:
            actions = np.random.choice(7 if locDiff else 5, size=actionsPerStep, replace=False)
        
        model_new = model

        for iAction in actions:
            if iAction == 0:
                model_new, _ = birth(model_new, prior)
            elif iAction == 1:
                model_new, _ = death(model_new, prior)
            elif iAction == 2:
                model_new, _ = update_arr(model_new, prior)
            elif iAction == 3:
                model_new, _ = update_slw(model_new, prior)
            elif iAction == 4:
                model_new, _ = update_amp(model_new, prior)
            elif iAction == 5:
                model_new, _ = update_dist(model_new, prior)
            elif iAction == 6:
                model_new, _ = update_baz(model_new, prior)

        U_model_new = create_U_from_model(model_new, prior, metadata, Utime, stf_time, stf_data)
        new_logL = compute_log_likelihood(U_obs, U_model_new)

        log_accept_ratio = ((new_logL - logL) + np.log((model.Nphase + 1) / model_new.Nphase)) / Temp
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

    return samples, logL_trace

def rjmcmc_run3c(U_obs, metadata, Utime, stf, prior, bookkeeping, saveDir):

    from vespainv.model import VespaModel3c, Prior3c
    from vespainv.waveformBuilder import create_U_from_model_3c_freqdomain

    trace_len = U_obs.shape[0]
    n_traces = U_obs.shape[1]

    totalSteps = bookkeeping.totalSteps
    burnInSteps = bookkeeping.burnInSteps
    nSaveModels = bookkeeping.nSaveModels
    save_interval = (totalSteps - burnInSteps) // nSaveModels
    actionsPerStep = bookkeeping.actionsPerStep
    locDiff = bookkeeping.locDiff
    Temp = bookkeeping.Temp

    # Extract stf and its time vectors
    stf_time = stf[:, 0]
    stf_data = stf[:, 1]

    # Start from a random model with one phase
    model = VespaModel3c.create_random(
        Nphase=1, Ntrace=n_traces, time=Utime, prior=prior
        )
    
    # # Start from an empty model
    # model = VespaModel3c.create_empty(Ntrace=n_traces)

    trace_shape = (trace_len, n_traces, 3)
    samples = []
    logL_trace = []

    # U_model = np.zeros(trace_shape)
    U_model, _, _ = create_U_from_model_3c_freqdomain(model, prior, metadata, Utime, stf_time, stf_data)
    logL = compute_log_likelihood(U_obs, U_model)

    start_time = time.time()
    checkpoint_interval = totalSteps // 100

    P_wvlt, S_wvlt = None, None

    for iStep in range(totalSteps):

        if model.Nphase == 0:
            actions = [0]
        else:
            actions = np.random.choice(14 if locDiff else 12, size=actionsPerStep, replace=False)
        
        model_new = model
        successAll = False
        attChange = False

        for iAction in actions:
            if iAction == 0:
                model_new, success = birth3c(model_new, prior)
                successAll |= success
            elif iAction == 1:
                model_new, success = death3c(model_new, prior)
                successAll |= success
            elif iAction == 2:
                model_new, success = update_arr(model_new, prior)
                successAll |= success
            elif iAction == 3:
                model_new, success = update_slw(model_new, prior)
                successAll |= success
            elif iAction == 4:
                model_new, success = update_amp(model_new, prior)
                successAll |= success
            elif iAction == 5:
                model_new, success = update_dip(model_new, prior)
                successAll |= success
            elif iAction == 6:
                model_new, success = update_azi(model_new, prior)
                successAll |= success
            elif iAction == 7:
                model_new, success = update_ph_hh(model_new, prior)
                successAll |= success
            elif iAction == 8:
                model_new, success = update_ph_vh(model_new, prior)
                successAll |= success
            elif iAction == 9:
                model_new, success = update_atts(model_new, prior)
                successAll |= success
                attChange = success
            elif iAction == 10:
                model_new, success = update_svfac(model_new, prior)
                successAll |= success
            elif iAction == 11:
                model_new, success = update_wvtype(model_new, prior)
                successAll |= success
            elif iAction == 12:
                model_new, success = update_dist(model_new, prior)
                successAll |= success
            elif iAction == 13:
                model_new, success = update_baz(model_new, prior)
                successAll |= success

        if attChange:
            U_model_new, P_wvlt, S_wvlt = create_U_from_model_3c_freqdomain(model_new, prior, metadata, Utime, stf_time, stf_data)
        else:
            U_model_new, P_wvlt, S_wvlt = create_U_from_model_3c_freqdomain(model_new, prior, metadata, Utime, stf_time, stf_data, P_wvlt, S_wvlt)
        
        new_logL = compute_log_likelihood(U_obs, U_model_new)

        log_accept_ratio = (new_logL - logL)/Temp  # assume log_alpha is 0
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

    return samples, logL_trace