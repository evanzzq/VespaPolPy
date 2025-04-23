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
