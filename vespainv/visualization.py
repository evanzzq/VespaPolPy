import numpy as np
import matplotlib.pyplot as plt
from vespainv.model import VespaModel, Prior
from scipy.stats import gaussian_kde

def plot_ensemble_vespagram(ensemble, Utime, prior, amp_weighted=False, true_model=None, is3c=False):
    arrSave, slwSave, ampSave = [], [], []

    arrAll = np.concatenate([m.arr for m in ensemble])
    slwAll = np.concatenate([m.slw for m in ensemble])
    ampAll = np.concatenate([m.amp for m in ensemble])
    if is3c:
        aziAll = np.concatenate([m.azi for m in ensemble])
        dipAll = np.concatenate([m.dip for m in ensemble])
        ph_hhAll = np.concatenate([m.ph_hh for m in ensemble])
        ph_vhAll = np.concatenate([m.ph_vh for m in ensemble])
        attsAll = np.concatenate([m.atts for m in ensemble])
        SVfacAll = np.concatenate([m.svfac for m in ensemble])
        isP_All = np.concatenate([m.wvtype for m in ensemble])

    valid = ~np.isnan(arrAll) & ~np.isnan(slwAll) & ~np.isnan(ampAll)
    arrAll, slwAll, ampAll, aziAll, dipAll, ph_hhAll, ph_vhAll, attsAll, SVfacAll,isP_All = (
        arrAll[valid], slwAll[valid], ampAll[valid], aziAll[valid], dipAll[valid], ph_hhAll[valid], ph_vhAll[valid], attsAll[valid], SVfacAll[valid], isP_All[valid])
    
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

    if is3c:
        print("Click to define a box: first lower-left, then upper-right")
        pts = plt.ginput(2)
        plt.close()
        (tmin, pmin), (tmax, pmax) = sorted(pts)

        # Get indices inside the selected box
        mask_box = (arrAll >= tmin) & (arrAll <= tmax) & (slwAll >= pmin) & (slwAll <= pmax)
        if np.sum(mask_box) == 0:
            print("No data points selected.")
            return
        
        # True model phases within click range
        if true_model:
            idx = np.where((true_model.arr >= tmin) & (true_model.arr <= tmax))[0]
            arrTrue = true_model.arr[idx]
            slwTrue = true_model.slw[idx]
            ampTrue = true_model.amp[idx]
            aziTrue = true_model.azi[idx]
            dipTrue = true_model.dip[idx]
            ph_hhTrue = true_model.ph_hh[idx]
            ph_vhTrue = true_model.ph_vh[idx]
            attsTrue = true_model.atts[idx]
            svfacTrue = true_model.svfac[idx]

        # Plot KDEs
        def plot_kde(ax, data, label, range_, true_value=None):
            data = data[mask_box]
            data = data[~np.isnan(data)]
            ax.set_title(label)

            if true_value is not None:
                for val in np.atleast_1d(true_value):
                    ax.axvline(val, color='red', linestyle='--', linewidth=1.5)

            if len(data) < 2 or np.std(data) < 1e-6:
                ax.text(0.5, 0.5, 'Insufficient or constant data', ha='center', va='center')
                return

            try:
                kde = gaussian_kde(data)
                x = np.linspace(*range_, 100)
                ax.plot(x, kde(x), label='KDE')
            except np.linalg.LinAlgError:
                ax.hist(data, bins=30, range=range_, density=True, color='gray', alpha=0.7, label='Histogram')

            ax.set_xlim(range_)
            ax.legend()

        fig, axs = plt.subplots(2, 5, figsize=(16, 6))
        axs = axs.flatten()

        plot_kde(axs[0], arrAll, 'Arrival Time (s)', [tmin, tmax], true_value=arrTrue if true_model else None)
        plot_kde(axs[1], slwAll, 'Rel. Slowness (s/deg)', [pmin, pmax], true_value=slwTrue if true_model else None)
        plot_kde(axs[2], ampAll, 'Amplitude', prior.ampRange, true_value=ampTrue if true_model else None)
        plot_kde(axs[3], aziAll, 'Pol. Az.', prior.aziRange, true_value=aziTrue if true_model else None)
        plot_kde(axs[4], dipAll, 'Pol. Dip.', prior.dipRange, true_value=dipTrue if true_model else None)
        plot_kde(axs[5], ph_hhAll, r'$\phi_{HH}$', prior.ph_hhRange, true_value=ph_hhTrue if true_model else None)
        plot_kde(axs[6], ph_vhAll, r'$\phi_{VH}$', prior.ph_vhRange, true_value=ph_vhTrue if true_model else None)
        plot_kde(axs[7], attsAll, 't* (s)', prior.attsRange, true_value=attsTrue if true_model else None)
        plot_kde(axs[8], SVfacAll, 'SV/SH Ratio', prior.svfacRange, true_value=svfacTrue if true_model else None)

        # P/S histogram
        ps_vals = isP_All[mask_box]
        axs[9].hist(ps_vals, bins=[-0.5, 0.5, 1.5])
        axs[9].set_xticks([0, 1])
        axs[9].set_xticklabels(['S', 'P'])
        axs[9].set_title('P or S')

        # SV/SH Ratio
        

        plt.tight_layout()
        plt.show()

def plot_seismogram_compare(U, time, offset=1.5, ensemble=None, prior=None, metadata=None, stf=None):

    from vespainv.waveformBuilder import create_U_from_model, create_U_from_model_3c_freqdomain

    is3c = True if U.ndim == 3 else False
    n_traces = U.shape[1]

    if ensemble is not None:
        U_model = np.zeros_like(U)
        for model in ensemble:
            U_model += (
                create_U_from_model_3c_freqdomain(model, prior, metadata, time, stf[:, 0], stf[:, 1], False) # tmp fix!!! 
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
            dist, baz = metadata[i,:]
            trace = U[:, i]
            trace /= np.max(np.abs(trace))
            plt.plot(time, trace + i * offset, color='black')
            if U_model is not None:
                trace_model = U_model[:, i]
                trace_model /= np.max(np.abs(trace_model))
                plt.plot(time, trace_model + i * offset, color='red')
            plt.text(time[-1] + 0.5, i * offset, f"{dist:.2f}°, {baz:.2f}°", va='center', fontsize=8)
        plt.xlabel("Time (s)")
        plt.ylabel("Trace Index")
        plt.title("Input Seismogram")

        plt.grid(True)
        plt.tight_layout()