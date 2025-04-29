import numpy as np
import matplotlib.pyplot as plt
from vespainv.model import VespaModel, Prior

def plot_ensemble_vespagram(ensemble, Utime, prior, amp_weighted=False, true_model=None):
    arrSave, slwSave, ampSave = [], [], []

    for model in ensemble:
        arrSave.append(model.arr)
        slwSave.append(model.slw)
        if amp_weighted:
            ampSave.append(model.amp)

    arrAll = np.concatenate(arrSave)
    slwAll = np.concatenate(slwSave)

    if amp_weighted:
        ampAll = np.concatenate(ampSave)
        valid = ~np.isnan(arrAll) & ~np.isnan(slwAll) & ~np.isnan(ampAll)
        arrAll, slwAll, ampAll = arrAll[valid], slwAll[valid], ampAll[valid]
    else:
        valid = ~np.isnan(arrAll) & ~np.isnan(slwAll)
        arrAll, slwAll = arrAll[valid], slwAll[valid]

    # Define bins
    xRange = [np.min(Utime), np.max(Utime)]
    yRange = prior.slwRange
    nBins = 50
    xEdges = np.linspace(xRange[0], xRange[1], nBins)
    yEdges = np.linspace(yRange[0], yRange[1], nBins)

    if amp_weighted:
        histCounts = np.zeros((nBins - 1, nBins - 1), dtype=np.float32)
        for i in range(len(arrAll)):
            xIdx = np.searchsorted(xEdges, arrAll[i]) - 1
            yIdx = np.searchsorted(yEdges, slwAll[i]) - 1
            if 0 <= xIdx < nBins - 1 and 0 <= yIdx < nBins - 1:
                histCounts[xIdx, yIdx] += ampAll[i]
    else:
        histCounts, _, _ = np.histogram2d(arrAll, slwAll, bins=[xEdges, yEdges])
        histCounts = histCounts.astype(np.float32)

    alphaData = np.where(histCounts > 0, 1.0, 0.0)

    plt.figure(figsize=(8, 6))
    h = plt.imshow(histCounts.T, extent=[xEdges[0], xEdges[-1], yEdges[0], yEdges[-1]],
                   origin='lower', aspect='auto', cmap='hot_r')
    h.set_alpha(alphaData.T)  # Transparency map must match the shape and dtype
    plt.colorbar(label="Amplitude Weighted Counts" if amp_weighted else "Counts")
    plt.xlabel("Arrival Time (s)")
    plt.ylabel("Slowness (s/deg)")
    plt.title("Ensemble Vespagram" + (" (Amp-Weighted)" if amp_weighted else ""))
    plt.grid(True)

    if true_model is not None:
        for i in range(true_model.Nphase):
            plt.arrow(true_model.arr[i], true_model.slw[i],
                      dx=0.0, dy=0.002,  # upward arrow
                      head_width=0.2, head_length=0.002,
                      fc='cyan', ec='cyan')
        plt.plot(true_model.arr, true_model.slw, 'cx', label="True Arrivals")
        plt.legend()

    plt.tight_layout()

def plot_seismogram_compare(U, time, offset=1.5, ensemble=None, prior=None, metadata=None, stf=None):

    from vespainv.waveformBuilder import create_U_from_model, create_U_from_model_3c_freqdomain

    is3c = True if U.ndim == 3 else False
    n_traces = U.shape[1]

    if ensemble is not None:
        U_model = np.zeros_like(U)
        for model in ensemble:
            U_model += (
                create_U_from_model_3c_freqdomain(model, prior, metadata, time, stf[:, 0], stf[:, 1]) 
                if is3c 
                else create_U_from_model(model, prior, metadata, time, stf[:, 0], stf[:, 1])
                )
        U_model /= len(ensemble)

    if is3c:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        comp_labels = ['Z', 'R', 'T']
        for comp in range(3):
            ax = axs[comp]
            for i in range(n_traces):
                trace = U[:, i, comp]
                trace /= np.max(np.abs(trace))
                ax.plot(time, trace + i * offset, color='black')
                if U_model is not None:
                    trace_model = U_model[:, i, comp]
                    trace_model /= np.max(np.abs(trace_model))
                    ax.plot(time, trace_model + i * offset, color='red')
            ax.set_title(f"Component {comp_labels[comp]}")
            ax.set_xlabel("Time (s)")
        axs[0].set_ylabel("Trace Index")
    else:
        plt.figure(figsize=(10, 8))
        for i in range(n_traces):
            trace = U[:, i]
            trace /= np.max(np.abs(trace))
            plt.plot(time, trace + i * offset, color='black')
            if U_model is not None:
                trace_model = U_model[:, i]
                trace_model /= np.max(np.abs(trace_model))
                plt.plot(time, trace_model + i * offset, color='red')
        plt.xlabel("Time (s)")
        plt.ylabel("Trace Index")
        plt.title("Input Seismogram")

        plt.grid(True)
        plt.tight_layout()