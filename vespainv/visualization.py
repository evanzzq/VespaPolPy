import numpy as np
import matplotlib.pyplot as plt
from vespainv.model import Bookkeeping
from scipy.stats import gaussian_kde

def plot_ensemble_vespagram(ensemble, Utime, prior, amp_weighted=False, true_model=None, is3c=False, third_click=False):

    # Initialize as None; if third_click == True, this will be update
    selected_pt = None

    arrAll = np.concatenate([m.arr for m in ensemble])
    slwAll = np.concatenate([m.slw for m in ensemble])
    ampAll = np.concatenate([m.amp for m in ensemble])
    attsAll = np.concatenate([m.atts for m in ensemble])
    valid = ~np.isnan(arrAll) & ~np.isnan(slwAll) & ~np.isnan(ampAll)

    if is3c:
        aziAll = np.concatenate([m.azi for m in ensemble])
        # dipAll = np.concatenate([m.dip for m in ensemble])
        ph_hhAll = np.concatenate([m.ph_hh for m in ensemble])
        ph_vhAll = np.concatenate([m.ph_vh for m in ensemble])
        isP_All = np.concatenate([m.wvtype for m in ensemble])  # Assume 1=P, 0=S

        arrAll, slwAll, ampAll, aziAll, ph_hhAll, ph_vhAll, attsAll, isP_All = (
            arrAll[valid], slwAll[valid], ampAll[valid],
            aziAll[valid], ph_hhAll[valid], ph_vhAll[valid], 
            attsAll[valid], isP_All[valid]
        )

        # === NEW: filter by user input ===
        wave_type = input("Select wave type to plot (P or S, otherwise all): ").strip().upper()
        if wave_type == "P":
            mask_wave = (isP_All == 1)
        elif wave_type == "S":
            mask_wave = (isP_All == 0)
        else:
            mask_wave = np.ones_like(isP_All, dtype=bool)

        arrAll, slwAll, ampAll, aziAll, ph_hhAll, ph_vhAll, attsAll, isP_All = (
            arrAll[mask_wave], slwAll[mask_wave], ampAll[mask_wave], 
            aziAll[mask_wave], ph_hhAll[mask_wave], ph_vhAll[mask_wave], 
            attsAll[mask_wave], isP_All[mask_wave]
        )

    else:  # Non-3c case
        arrAll, slwAll, ampAll = (
            arrAll[valid], slwAll[valid], ampAll[valid]
        )
    
    # Kernel density estimation

    # Define bins
    xRange = [np.min(Utime), np.max(Utime)]
    yRange = prior.slwRange

    xy = np.vstack([arrAll, slwAll])
    if amp_weighted:
        weights = np.abs(ampAll)
        total_weight = np.sum(weights)
    else:
        weights = None
        total_weight = len(arrAll)

    # Check rank / condition number of covariance
    C = np.cov(xy)
    cond = np.linalg.cond(C)
    print(cond)

    if cond > 1e3:  # 1e3 euristic threshold: covariance nearly singular
        print("Covariance nearly singular – using 2D histogram instead of KDE.")
        # --- your existing 2D hist code here ---
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

        plt.figure(figsize=(8, 6))
        vmax = np.nanmax(np.abs(histCounts))
        h = plt.imshow(
            histCounts.T,
            extent=[xEdges[0], xEdges[-1], yEdges[0], yEdges[-1]],
            origin='lower',
            aspect='auto',
            cmap='seismic',
            vmin=-vmax, vmax=vmax
        )
        plt.colorbar(label="Amplitude Weighted Counts" if amp_weighted else "Counts")
        plt.xlabel("Arrival Time (s)")
        plt.ylabel("Slowness (s/deg)")
        plt.title("Ensemble Vespagram (Hist)")
        plt.grid(True)
        if true_model is not None:
            plt.scatter(true_model.arr, true_model.slw, c='k', marker='x', s=80, label='True model')
            plt.legend()
        plt.show(block=False)
    else:
        # --- safe to do KDE ---
        kde = gaussian_kde(xy, weights=weights)
        xx, yy = np.meshgrid(
            np.linspace(xRange[0], xRange[1], 200),
            np.linspace(yRange[0], yRange[1], 200)
        )
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        zz *= total_weight

        plt.figure(figsize=(8, 6))
        h = plt.contourf(xx, yy, zz, levels=100, cmap='hot')
        plt.colorbar(h, label="Density" if not amp_weighted else "Amp-weighted density")
        plt.xlabel("Arrival Time (s)")
        plt.ylabel("Slowness (s/deg)")
        plt.title("Ensemble Vespagram (KDE)")
        plt.grid(True)
        if true_model is not None:
            plt.scatter(true_model.arr, true_model.slw, c='k', marker='x', s=80, label='True model')
            plt.legend()
        plt.show(block=False)

    # Get posterior plot range from two clicks
    print("Click twice to define a box (first lower-left, then upper-right):")
    pts = plt.ginput(2)
    (tmin, pmin), (tmax, pmax) = sorted(pts)
    print(f"Selected range: arrival time {tmin:.2f} to {tmax:.2f} s, slowness {pmin:.2f} to {pmax:.2f} s/deg.\n")

    # Get moveout correction point from one click
    if third_click:
        print("Click once to select arrival time - slowness pair to apply moveout correction:")
        selected_pt_tmp = plt.ginput(1)
        selected_pt = selected_pt_tmp[0]
        print(f"Selected point: arrival time {selected_pt[0]:.2f} s, slowness {selected_pt[1]:.2f} s/deg.")

    # Get indices inside the selected box
    mask_box = (arrAll >= tmin) & (arrAll <= tmax) & (slwAll >= pmin) & (slwAll <= pmax)
    if np.sum(mask_box) == 0:
        print("No data points selected.")
        return

    def plot_kde(ax, data, label, range_, true_value=None, circular=False):
        """
        Plot a KDE or histogram of posterior samples.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to plot on.
        data : array-like
            Data samples.
        label : str
            Plot title.
        range_ : tuple
            (min, max) plotting range.
        true_value : float or list, optional
            True/reference value(s) to mark.
        circular : bool, optional
            Whether to treat the variable as circular (e.g., angles).
            If True, data are wrapped and duplicated to ensure smooth KDE near boundaries.
        """
        data = data[mask_box]
        data = data[~np.isnan(data)]
        ax.set_title(label)

        if true_value is not None:
            for val in np.atleast_1d(true_value):
                ax.axvline(val, color='red', linestyle='--', linewidth=0.5)

        num_unique = len(np.unique(data))
        if  len(data) < 5 or num_unique < 10:
            ax.hist(data, bins=30, range=range_, color='gray', alpha=0.7)
            ax.text(0.5, 0.9, 'Insufficient variance\n(showing histogram)', ha='center',
                    va='top', transform=ax.transAxes, fontsize=9, color='darkred')
            ax.set_xlim(range_)
            return

        try:
            if circular:
                # --- Circular wrapping ---
                low, high = range_
                width = high - low
                # wrap into [low, high)
                data_wrapped = ((data - low) % width) + low
                # duplicate shifted versions to remove boundary edge effects
                data_aug = np.concatenate([data_wrapped, data_wrapped - width, data_wrapped + width])
            else:
                data_aug = data

            kde = gaussian_kde(data_aug)
            x = np.linspace(*range_, 200)
            ax.plot(x, kde(x), label='KDE')
            ax.set_xlim(range_)
            ax.legend()

        except np.linalg.LinAlgError:
            ax.hist(data, bins=30, range=range_, color='gray', alpha=0.7)
            ax.text(0.5, 0.9, 'KDE failed\n(showing histogram)', ha='center',
                    va='top', transform=ax.transAxes, fontsize=9, color='darkred')
            ax.set_xlim(range_)

    if is3c:    
        # True model phases within click range
        if true_model:
            idx = np.where((true_model.arr >= tmin) & (true_model.arr <= tmax))[0]
            arrTrue = true_model.arr[idx]
            slwTrue = true_model.slw[idx]
            ampTrue = true_model.amp[idx]
            bazTrue = true_model.baz[idx]
            aziTrue = true_model.azi[idx]
            ph_hhTrue = true_model.ph_hh[idx]
            ph_vhTrue = true_model.ph_vh[idx]
            attsTrue = true_model.atts[idx]

        # Plot KDEs
        fig, axs = plt.subplots(3, 3, figsize=(8, 8))
        axs = axs.flatten()

        plot_kde(axs[0], arrAll, 'Arrival Time (s)', [tmin, tmax], true_value=arrTrue if true_model else None)
        plot_kde(axs[1], slwAll, 'Rel. Slowness (s/deg)', [pmin, pmax], true_value=slwTrue if true_model else None)
        plot_kde(axs[2], ampAll, 'Amplitude', prior.ampRange, true_value=ampTrue if true_model else None)
        # plot_kde(axs[3], bazAll, 'Phase BAZ', prior.bazRange, true_value=bazTrue if true_model else None)
        plot_kde(axs[4], aziAll, 'Pol. Az.', prior.aziRange, true_value=aziTrue if true_model else None, circular=True)
        plot_kde(axs[5], ph_hhAll, r'$\phi_{HH}$', prior.ph_hhRange, true_value=ph_hhTrue if true_model else None, circular=True)
        plot_kde(axs[6], ph_vhAll, r'$\phi_{VH}$', prior.ph_vhRange, true_value=ph_vhTrue if true_model else None, circular=True)
        plot_kde(axs[7], attsAll, 't* (s)', prior.attsRange, true_value=attsTrue if true_model else None)

        # P/S histogram
        ps_vals = isP_All[mask_box]
        axs[8].hist(ps_vals, bins=[-0.5, 0.5, 1.5])
        axs[8].set_xticks([0, 1])
        axs[8].set_xticklabels(['S', 'P'])
        axs[8].set_title('P or S')
        
        plt.tight_layout()
        plt.show()
    else:
        # True model phases within click range
        if true_model:
            idx = np.where((true_model.arr >= tmin) & (true_model.arr <= tmax))[0]
            arrTrue = true_model.arr[idx]
            slwTrue = true_model.slw[idx]
            ampTrue = true_model.amp[idx]
            attsTrue = true_model.atts[idx]

        # Plot KDEs
        fig, axs = plt.subplots(2, 3, figsize=(16, 6))
        axs = axs.flatten()

        plot_kde(axs[0], arrAll, 'Arrival Time (s)', [tmin, tmax], true_value=arrTrue if true_model else None)
        plot_kde(axs[1], slwAll, 'Rel. Slowness (s/deg)', [pmin, pmax], true_value=slwTrue if true_model else None)
        plot_kde(axs[2], ampAll, 'Amplitude', prior.ampRange, true_value=ampTrue if true_model else None)
        plot_kde(axs[3], attsAll, 't* (s)', prior.attsRange, true_value=attsTrue if true_model else None) ############ tmp fix!!!!!!!!!
        
        plt.tight_layout()
        plt.show()
    
    return selected_pt

def plot_seismogram_compare(U, time, offset=1.5, ensemble=None, prior=None, metadata=None, stf=None, bookkeeping=None, moveout_pt=None):

    from vespainv.waveformBuilder import create_U_from_model_freqdomain, create_U_from_model_3c_freqdomain

    is3c = True if U.ndim == 3 else False
    n_traces = U.shape[1]

    if ensemble is not None:
        U_model = np.zeros_like(U)
        for model in ensemble:
            U_model += (
                create_U_from_model_3c_freqdomain(model, metadata, time, stf[:, 0], stf[:, 1], bookkeeping)
                if is3c 
                else create_U_from_model_freqdomain(model, metadata, time, stf[:, 0], stf[:, 1], bookkeeping)
                )
        U_model /= len(ensemble)

    # Apply moveout correction if moveout_pt is provided
    if moveout_pt:

        from obspy.geodetics.base import gps2dist_azimuth, locations2degrees

        arr, slow = moveout_pt
        U_shifted = np.zeros_like(U)
        U_model_shifted = np.zeros_like(U)

        refLat = bookkeeping.refLat
        refLon = bookkeeping.refLon
        srcLat = bookkeeping.srcLat
        srcLon = bookkeeping.srcLon
        refDist = locations2degrees(srcLat, srcLon, refLat, refLon)
        
        for itrace in range(n_traces):
            trDist, trBaz = metadata[itrace]
            trDist += model.distDiff[itrace]
            trBaz += model.bazDiff[itrace]
            # Get slowness on x and y directions
            slow_x = slow * np.cos(np.radians(90-trBaz)) # refBaz
            slow_y = slow * np.sin(np.radians(90-trBaz)) # refBaz
            # Get dx and dy
            dx = (trDist - refDist) * np.sin(np.radians(trBaz))
            dy = (trDist - refDist) * np.cos(np.radians(trBaz))
            # Get tshift
            tshift = slow_x * dx + slow_y * dy
            # Shift traces
            if is3c:
                for ic in range(3):
                    U_shifted[:, itrace, ic] = np.interp(time, time - tshift, U[:, itrace, ic], left=0.0, right=0.0)
                    U_model_shifted[:, itrace, ic] = np.interp(time, time - tshift, U_model[:, itrace, ic], left=0.0, right=0.0)
            else:
                U_shifted[:, itrace] = np.interp(time, time - tshift, U[:, itrace], left=0.0, right=0.0)
                U_model_shifted[:, itrace] = np.interp(time, time - tshift, U_model[:, itrace], left=0.0, right=0.0)
        
        # Overwrite
        U = U_shifted
        U_model = U_model_shifted

    # Plot seismograms
    if is3c:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        comp_labels = ['Z', 'R', 'T']
        for comp in range(3):
            ax = axs[comp]
            for i in range(n_traces):
                trace = U[:, i, comp]
                ax.plot(time, trace + i * offset, color='black')
                if U_model is not None:
                    trace_model = U_model[:, i, comp]
                    ax.plot(time, trace_model + i * offset, color='red')
                if moveout_pt:
                    ax.axvline(x=arr, color='r', linestyle='--')
            ax.set_title(f"Component {comp_labels[comp]}")
            ax.set_xlabel("Time (s)")
        axs[0].set_ylabel("Trace Index")
    else:
        plt.figure(figsize=(10, 8))
        for i in range(n_traces):
            dist, baz = metadata[i,:]
            trace = U[:, i]
            plt.plot(time, trace + i * offset, color='black')
            if U_model is not None:
                trace_model = U_model[:, i]
                plt.plot(time, trace_model + i * offset, color='red')
            if moveout_pt:
                    ax.axvline(x=arr, color='r', linestyle='--')
            plt.text(time[-1] + 0.5, i * offset, f"{dist:.2f}°, {baz:.2f}°", va='center', fontsize=8)
        plt.xlabel("Time (s)")
        plt.ylabel("Trace Index")
        plt.title("Input Seismogram")

        plt.grid(True)
        plt.tight_layout()