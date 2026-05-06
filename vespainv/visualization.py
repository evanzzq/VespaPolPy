import numpy as np
import matplotlib.pyplot as plt
import re
from cmap import Colormap
from vespainv.model import Bookkeeping
from vespainv.utils import dest_point
from scipy.stats import gaussian_kde

def plot_ensemble_vespagram(ensemble, Utime, prior, amp_weighted=False, true_model=None, is3c=False, third_click=False):

    cmap = Colormap('matlab:hot')

    # Initialize as None; if third_click == True, this will be update
    selected_pt = None

    arrAll = np.concatenate([m.arr for m in ensemble])
    slwAll = np.concatenate([m.slw for m in ensemble])
    ampAll = np.concatenate([m.amp for m in ensemble])
    attsAll = np.concatenate([m.atts for m in ensemble])
    valid = ~np.isnan(arrAll) & ~np.isnan(slwAll) & ~np.isnan(ampAll)

    # --- Histogram of loge across ALL models in the ensemble ---
    logeAll = np.array([m.loge for m in ensemble], dtype=float)
    logeAll = logeAll[~np.isnan(logeAll)]

    plt.figure(figsize=(6, 4))
    plt.hist(logeAll, bins=40, range=prior.logeRange, color="gray", alpha=0.7, edgecolor="white")
    plt.xlabel("loge")
    plt.ylabel("Count")
    plt.title(f"loge histogram (all models), N={logeAll.size}")
    plt.tight_layout()
    plt.show(block=False)

    if is3c:
        aziAll = np.concatenate([m.azi for m in ensemble])
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

    fig, ax = plt.subplots(figsize=(8, 6))

    if cond > 1:
        print("Covariance nearly singular – using 2D histogram instead of KDE.")
        from scipy.ndimage import gaussian_filter

        nBins = 150
        xEdges = np.linspace(xRange[0], xRange[1], nBins)
        yEdges = np.linspace(yRange[0], yRange[1], nBins)

        if amp_weighted:
            histCounts = np.zeros((nBins - 1, nBins - 1), dtype=float)
            for i in range(len(arrAll)):
                xIdx = np.searchsorted(xEdges, arrAll[i]) - 1
                yIdx = np.searchsorted(yEdges, slwAll[i]) - 1
                if 0 <= xIdx < nBins - 1 and 0 <= yIdx < nBins - 1:
                    histCounts[xIdx, yIdx] += ampAll[i]
        else:
            histCounts, _, _ = np.histogram2d(arrAll, slwAll, bins=[xEdges, yEdges])
            histCounts = histCounts.astype(float)

        sigma = 2.0
        histSmooth = gaussian_filter(histCounts, sigma=sigma)

        from matplotlib.colors import TwoSlopeNorm

        gamma = 0.8
        histPlot = np.sign(histSmooth) * (np.abs(histSmooth) ** gamma)

        vmax = np.nanmax(np.abs(histPlot))

        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

        h = ax.pcolormesh(
            xEdges,
            yEdges,
            histPlot.T,
            shading="auto",
            cmap="RdBu_r",
            norm=norm
        )

        fig.colorbar(
            h, ax=ax,
            label="Smoothed amplitude-weighted counts" if amp_weighted else "Smoothed counts"
        )
        ax.set_title("Ensemble Vespagram (Smoothed Histogram)")

    else:
        kde = gaussian_kde(xy, weights=weights)
        xx, yy = np.meshgrid(
            np.linspace(xRange[0], xRange[1], 200),
            np.linspace(yRange[0], yRange[1], 200)
        )
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        zz *= total_weight

        h = ax.contourf(xx, yy, zz, levels=100, cmap=cmap.to_mpl())
        fig.colorbar(h, ax=ax, label="Density" if not amp_weighted else "Amp-weighted density")
        ax.set_title("Ensemble Vespagram (KDE)")

    if true_model is not None:
        ax.scatter(true_model.arr, true_model.slw, c='k', marker='x', s=80, label='True model')
        ax.legend()

    ax.set_xlabel("Arrival Time (s)")
    ax.set_ylabel("Slowness (s/deg)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    print("Click twice to define a box (first lower-left, then upper-right):")
    pts = plt.ginput(2, timeout=-1)

    (t1, p1), (t2, p2) = pts
    tmin, tmax = sorted([t1, t2])
    pmin, pmax = sorted([p1, p2])

    print(f"Selected range: arrival time {tmin:.2f} to {tmax:.2f} s, slowness {pmin:.2f} to {pmax:.2f} s/deg.\n")

    plt.show()
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
    
    # Print no. of phases across ensembles
    n_points = int(np.sum(mask_box))
    print(f"{n_points} phase-samples in the box (across the ensemble).")

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
        if len(data) < 5e5 or num_unique < 10:
            ax.hist(data, bins=50, range=range_, color='gray', alpha=0.7)
            ax.set_xlim(range_)
            return

        try:
            if circular:
                low, high = range_
                width = high - low
                data_wrapped = ((data - low) % width) + low
                data_aug = np.concatenate([data_wrapped, data_wrapped - width, data_wrapped + width])
            else:
                data_aug = data

            kde = gaussian_kde(data_aug)
            x = np.linspace(*range_, 200)
            ax.plot(x, kde(x), label='KDE')
            ax.set_xlim(range_)
            ax.legend()

        except np.linalg.LinAlgError:
            ax.hist(data, bins=90, range=range_, color='gray', alpha=0.7)
            ax.text(0.5, 0.9, 'KDE failed\n(showing histogram)', ha='center',
                    va='top', transform=ax.transAxes, fontsize=9, color='darkred')
            ax.set_xlim(range_)


    def plot_2d_kde(ax, xdata, ydata, xlabel, ylabel, xrange_, yrange_,
                    true_x=None, true_y=None, circular_x=False, circular_y=False,
                    nbins=80):
        """
        Plot a 2D KDE or 2D histogram of posterior samples.
        """
        x = xdata[mask_box]
        y = ydata[mask_box]

        valid = (~np.isnan(x)) & (~np.isnan(y))
        x = x[valid]
        y = y[valid]

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xrange_)
        ax.set_ylim(yrange_)

        # plot true/reference values
        if true_x is not None and true_y is not None:
            ax.scatter(np.atleast_1d(true_x), np.atleast_1d(true_y),
                    marker='x', c='red', s=40, linewidths=1.0, zorder=5)

        num_unique_x = len(np.unique(x))
        num_unique_y = len(np.unique(y))

        # if too few samples / too few unique values, use 2D histogram directly
        if len(x) < 200 or num_unique_x < 10 or num_unique_y < 10:
            H, xedges, yedges = np.histogram2d(
                x, y, bins=nbins, range=[xrange_, yrange_]
            )
            ax.imshow(
                H.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )
            return

        try:
            # ---- wrap circular variables ----
            def wrap_and_shifts(data, range_, circular):
                if not circular:
                    return [data]
                low, high = range_
                width = high - low
                data_wrapped = ((data - low) % width) + low
                return [data_wrapped - width, data_wrapped, data_wrapped + width]

            x_versions = wrap_and_shifts(x, xrange_, circular_x)
            y_versions = wrap_and_shifts(y, yrange_, circular_y)

            # create augmented pairs
            x_aug = []
            y_aug = []
            for xv in x_versions:
                for yv in y_versions:
                    x_aug.append(xv)
                    y_aug.append(yv)
            x_aug = np.concatenate(x_aug)
            y_aug = np.concatenate(y_aug)

            values = np.vstack([x_aug, y_aug])
            kde = gaussian_kde(values)

            nx, ny = 200, 200
            xx, yy = np.meshgrid(
                np.linspace(*xrange_, nx),
                np.linspace(*yrange_, ny)
            )
            coords = np.vstack([xx.ravel(), yy.ravel()])
            zz = kde(coords).reshape(xx.shape)

            h = ax.contourf(xx, yy, zz, levels=100, cmap='plasma')  # or 'magma', 'inferno'
            ax.contour(xx, yy, zz, colors='k', linewidths=0.3)
            plt.colorbar(h, ax=ax, label='Density')

        except np.linalg.LinAlgError:
            H, xedges, yedges = np.histogram2d(
                x, y, bins=nbins, range=[xrange_, yrange_]
            )
            ax.imshow(
                H.T,
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                origin='lower',
                aspect='auto',
                cmap='viridis'
            )
            ax.text(0.5, 0.95, 'KDE failed\n(showing 2D histogram)',
                    ha='center', va='top', transform=ax.transAxes,
                    fontsize=9, color='white')
    
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

        fig, axs = plt.subplots(3, 3, figsize=(8, 8))
        axs = axs.flatten()

        plot_kde(axs[0], arrAll, 'Arrival Time (s)', [tmin, tmax], true_value=arrTrue if true_model else None)
        plot_kde(axs[1], slwAll, 'Rel. Slowness (s/deg)', [pmin, pmax], true_value=slwTrue if true_model else None)
        plot_kde(axs[2], ampAll, 'Amplitude', prior.ampRange, true_value=ampTrue if true_model else None)
        plot_kde(axs[3], aziAll, 'Pol. Az.', prior.aziRange, true_value=aziTrue if true_model else None, circular=True)

        plot_2d_kde(
            axs[4],
            aziAll, ph_hhAll,
            xlabel='Pol. Az.',
            ylabel=r'$\phi_{HH}$',
            xrange_=prior.aziRange,
            yrange_=prior.ph_hhRange,
            true_x=aziTrue if true_model else None,
            true_y=ph_hhTrue if true_model else None,
            circular_x=True,
            circular_y=True
        )
        axs[4].set_title(r'Pol. Az. vs $\phi_{HH}$')

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

def plot_seismogram_compare(U, time, offset=1.5, ensemble=None, prior=None, metadata=None, stf=None, bookkeeping=None, moveout_pt=None, mode="All"):

    from vespainv.waveformBuilder import create_U_from_model_freqdomain, create_U_from_model_3c_freqdomain

    isMars = bookkeeping.isMars # if isMars, ref location should be S0794a (CF impact)
    srcArray = bookkeeping.srcArray

    is3c = True if U.ndim == 3 else False
    n_traces = U.shape[1]

    if ensemble is not None:
        U_model = np.zeros_like(U)
        for model in ensemble:
            # filter P or S phases based on mode
            if is3c:
                if mode == "P":
                    iph = 0
                    while True:
                        if iph >= model.Nphase: break
                        if model.wvtype[iph] == 0:
                            model.Nphase -= 1
                            model.arr = np.delete(model.arr, iph)
                            model.slw = np.delete(model.slw, iph)
                            model.amp = np.delete(model.amp, iph)
                            model.atts = np.delete(model.atts, iph)
                            model.azi = np.delete(model.azi, iph)
                            model.ph_hh = np.delete(model.ph_hh, iph)
                            model.ph_vh = np.delete(model.ph_vh, iph)
                            model.wvtype = np.delete(model.wvtype, iph)
                        else:
                            iph += 1
                elif mode == "S":
                    iph = 0
                    while True:
                        if iph >= model.Nphase: break
                        if model.wvtype[iph] == 1:
                            model.Nphase -= 1
                            model.arr = np.delete(model.arr, iph)
                            model.slw = np.delete(model.slw, iph)
                            model.amp = np.delete(model.amp, iph)
                            model.atts = np.delete(model.atts, iph)
                            model.azi = np.delete(model.azi, iph)
                            model.ph_hh = np.delete(model.ph_hh, iph)
                            model.ph_vh = np.delete(model.ph_vh, iph)
                            model.wvtype = np.delete(model.wvtype, iph)
                        else:
                            iph += 1
            # aggreagate U
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
        
        refBaz = bookkeeping.refBaz
        refAz = (refBaz + 180)%360
        
        srcLat = bookkeeping.srcLat
        srcLon = bookkeeping.srcLon
        
        for itrace in range(n_traces):
            if isMars:
                trDist, trBaz = metadata[itrace]
                trLat, trLon = dest_point(srcLat, srcLon, trBaz, trDist) # trBaz used here because the geometry is reversed
            # else, metadata is in (lat, lon)
            else:
                trLat, trLon = metadata[itrace]
                # Get slowness on x and y directions
            if srcArray:
                slow_x = slow * np.sin(np.radians(refBaz))
                slow_y = slow * np.cos(np.radians(refBaz))
            else:
                slow_x = slow * np.sin(np.radians(refAz))
                slow_y = slow * np.cos(np.radians(refAz))
            # Get dx and dy
            dx = (((trLon - refLon + 180.0) % 360.0) - 180.0) * np.cos(np.radians(refLat)) # lon wrapping needed
            dy = (trLat - refLat)
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

        # tmp: partical motion plot
        plt.figure(figsize=(8, 8))
        trace_R = U[:, 0, 1]
        trace_T = U[:, 0, 2]
        trace_R_model = U_model[:, 0, 1]
        trace_T_model = U_model[:, 0, 2]
        plt.plot(trace_R, trace_T, color='k')
        plt.plot(trace_R_model, trace_T_model, color='r')
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
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

def phase_count_distribution_by_model(
    ensemble,
    tmin, tmax,
    pmin, pmax,
    is3c=False,
    wave_type=None,          # None / "P" / "S"
    make_plots=True,
    bins=30,
):
    """
    Compute per-model phase-count statistics inside a (t, slowness) box.

    Statistics are computed for:
      (1) raw phase counts per model
      (2) fraction of phases per model in the box
      (3) inverse fraction (total phases per in-box phase)

    Parameters
    ----------
    ensemble : list
        List of posterior models.
    tmin, tmax : float
        Arrival-time bounds (s).
    pmin, pmax : float
        Slowness bounds (s/deg).
    is3c : bool
        If True, enable wave-type filtering using model.wvtype (1=P, 0=S).
    wave_type : None | "P" | "S"
        Optional wave-type filter (used only if is3c=True).
    make_plots : bool
        If True, plot histograms.
    bins : int
        Histogram bin count.

    Returns
    -------
    summary : dict
        Dictionary of per-model statistics.
    """

    wave_type_norm = None if wave_type is None else wave_type.upper()

    n_models = len(ensemble)
    counts_per_model = np.zeros(n_models, dtype=int)
    totals_per_model = np.zeros(n_models, dtype=int)

    # ---- per-model counting ----
    for i, m in enumerate(ensemble):
        arr = np.asarray(m.arr)
        slw = np.asarray(m.slw)

        valid = ~np.isnan(arr) & ~np.isnan(slw)

        if is3c and wave_type_norm is not None:
            wv = np.asarray(m.wvtype)
            valid &= ~np.isnan(wv)
            if wave_type_norm == "P":
                valid &= (wv == 1)
            elif wave_type_norm == "S":
                valid &= (wv == 0)

        totals_per_model[i] = np.sum(valid)

        mask_box = (
            (arr >= tmin) & (arr <= tmax) &
            (slw >= pmin) & (slw <= pmax) &
            valid
        )

        counts_per_model[i] = np.sum(mask_box)

    # ---- fractions ----
    frac_per_model = counts_per_model / totals_per_model
    inv_frac_per_model = 1.0 / frac_per_model

    # ---- summary statistics ----
    summary = {
        "n_models": n_models,
        "box": {"tmin": tmin, "tmax": tmax, "pmin": pmin, "pmax": pmax},
        "wave_type": wave_type_norm,

        # raw counts
        "sum_counts": int(np.sum(counts_per_model)),
        "mean_counts": float(np.mean(counts_per_model)),
        "median_counts": float(np.median(counts_per_model)),
        "std_counts": float(np.std(counts_per_model)),

        # fraction stats
        "mean_frac": float(np.mean(frac_per_model)),
        "median_frac": float(np.median(frac_per_model)),
        "std_frac": float(np.std(frac_per_model)),
        "iqr_frac": float(np.percentile(frac_per_model, 75) - np.percentile(frac_per_model, 25)),
        "p05_frac": float(np.percentile(frac_per_model, 5)),
        "p95_frac": float(np.percentile(frac_per_model, 95)),
        "max_frac": float(np.max(frac_per_model)),

        # inverse-fraction stats
        "mean_inv_frac": float(np.mean(inv_frac_per_model)),
        "median_inv_frac": float(np.median(inv_frac_per_model)),
        "std_inv_frac": float(np.std(inv_frac_per_model)),
        "iqr_inv_frac": float(np.percentile(inv_frac_per_model, 75) - np.percentile(inv_frac_per_model, 25)),
        "p05_inv_frac": float(np.percentile(inv_frac_per_model, 5)),
        "p95_inv_frac": float(np.percentile(inv_frac_per_model, 95)),
        "max_inv_frac": float(np.max(inv_frac_per_model)),
    }

    # ---- plots ----
    if make_plots:
        suffix = f" ({wave_type_norm})" if (is3c and wave_type_norm) else ""

        plt.figure(figsize=(7, 4))
        plt.hist(counts_per_model, bins=bins)
        plt.xlabel("Phases in box (per model)")
        plt.ylabel("Number of models")
        plt.title(f"Per-model phase counts{suffix}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.hist(frac_per_model, bins=bins)
        plt.xlabel("Fraction of phases in box")
        plt.ylabel("Number of models")
        plt.title(f"Per-model phase fraction{suffix}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(7, 4))
        plt.hist(inv_frac_per_model, bins=bins)
        plt.xlabel("1 / fraction  (total phases per in-box phase)")
        plt.ylabel("Number of models")
        plt.title(f"Per-model inverse fraction{suffix}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return summary


def _region_bounds(region, prior=None):
    tmin = region.get("tmin")
    tmax = region.get("tmax")
    pmin = region.get("pmin")
    pmax = region.get("pmax")

    if (tmin is None or tmax is None) and prior is not None:
        tmin = prior.timeRange[0] if tmin is None else tmin
        tmax = prior.timeRange[1] if tmax is None else tmax
    if (pmin is None or pmax is None) and prior is not None:
        pmin = prior.slwRange[0] if pmin is None else pmin
        pmax = prior.slwRange[1] if pmax is None else pmax

    if tmin is None or tmax is None or pmin is None or pmax is None:
        raise ValueError(
            "Each convergence region needs tmin, tmax, pmin, and pmax. "
            "Set pmin/pmax to None only when passing prior."
        )

    tmin, tmax = sorted([float(tmin), float(tmax)])
    pmin, pmax = sorted([float(pmin), float(pmax)])
    return tmin, tmax, pmin, pmax


def _extract_region_arr_slw(ensemble, region, is3c=False):
    tmin, tmax, pmin, pmax = region["_bounds"]
    wave_type = region.get("wave_type")
    wave_type = None if wave_type is None else str(wave_type).upper()

    arr_samples = []
    slw_samples = []

    for m in ensemble:
        arr = np.asarray(m.arr, dtype=float)
        slw = np.asarray(m.slw, dtype=float)
        valid = ~np.isnan(arr) & ~np.isnan(slw)

        if is3c and wave_type in ("P", "S"):
            wvtype = np.asarray(m.wvtype)
            valid &= ~np.isnan(wvtype)
            if wave_type == "P":
                valid &= (wvtype == 1)
            else:
                valid &= (wvtype == 0)

        in_region = (
            valid &
            (arr >= tmin) & (arr <= tmax) &
            (slw >= pmin) & (slw <= pmax)
        )
        if np.any(in_region):
            arr_samples.append(arr[in_region])
            slw_samples.append(slw[in_region])

    if not arr_samples:
        return np.array([]), np.array([])

    return np.concatenate(arr_samples), np.concatenate(slw_samples)


def plot_chain_convergence_by_region(
    ensembles,
    chain_labels,
    regions,
    prior=None,
    is3c=False,
    bins=40,
    save_dir=None,
):
    """
    Plot chain-by-chain posterior samples inside code-defined arrival boxes.

    For each region, this makes overlaid 1D density histograms for arrival
    time and slowness. Individual chains are thin and semi-transparent; the
    pooled all-chain distribution is shown as a thicker black reference.

    Parameters
    ----------
    ensembles : list of list
        One posterior-model ensemble per chain.
    chain_labels : list
        Labels for each chain, e.g. [0, 1, 2] or ["single"].
    regions : list of dict
        Each dict should have name, tmin, tmax, pmin, pmax, and optionally
        wave_type ("P" or "S" for 3C inversions).
    prior : Prior or Prior3c, optional
        Used only to fill missing bounds when a region value is None.
    is3c : bool
        If True, region wave_type filters model.wvtype.
    bins : int
        Histogram bin count for the 1D plots.
    save_dir : str, optional
        If provided, save figures as PNGs in this directory.

    Returns
    -------
    summaries : list of dict
        Per-region, per-chain sample counts.
    """
    if len(ensembles) != len(chain_labels):
        raise ValueError("ensembles and chain_labels must have the same length.")

    if not regions:
        print("No convergence regions defined; skipping convergence plots.")
        return []

    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)

    cmap = plt.get_cmap("tab20", max(len(ensembles), 1))
    summaries = []

    for iregion, raw_region in enumerate(regions):
        region = dict(raw_region)
        region["_bounds"] = _region_bounds(region, prior=prior)
        tmin, tmax, pmin, pmax = region["_bounds"]
        name = region.get("name", f"region_{iregion + 1}")
        wave_type = region.get("wave_type")
        suffix = f", {wave_type}" if wave_type else ""

        chain_samples = []
        counts = {}
        for ensemble, label in zip(ensembles, chain_labels):
            arr, slw = _extract_region_arr_slw(ensemble, region, is3c=is3c)
            chain_samples.append((arr, slw))
            counts[label] = int(arr.size)

        summaries.append({
            "name": name,
            "box": {"tmin": tmin, "tmax": tmax, "pmin": pmin, "pmax": pmax},
            "wave_type": wave_type,
            "counts": counts,
        })

        print(f"\nConvergence region '{name}' ({tmin:.2f}-{tmax:.2f} s, "
              f"{pmin:.2f}-{pmax:.2f} s/deg{suffix})")
        for label, count in counts.items():
            print(f"  Chain {label}: {count} samples")

        arr_pooled = []
        slw_pooled = []
        fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        for ichain, (label, (arr, slw)) in enumerate(zip(chain_labels, chain_samples)):
            if arr.size == 0:
                continue
            color = cmap(ichain)
            arr_pooled.append(arr)
            slw_pooled.append(slw)
            axs[0].hist(
                arr, bins=bins, range=(tmin, tmax), density=True,
                histtype="step", linewidth=1.2, color=color,
                linestyle="--", alpha=0.65,
            )
            axs[1].hist(
                slw, bins=bins, range=(pmin, pmax), density=True,
                histtype="step", linewidth=1.2, color=color,
                linestyle="--", alpha=0.65,
            )

        if arr_pooled:
            arr_pooled = np.concatenate(arr_pooled)
            slw_pooled = np.concatenate(slw_pooled)
            axs[0].hist(
                arr_pooled, bins=bins, range=(tmin, tmax), density=True,
                histtype="step", linewidth=2.2, color="black",
            )
            axs[1].hist(
                slw_pooled, bins=bins, range=(pmin, pmax), density=True,
                histtype="step", linewidth=2.2, color="black",
            )

        axs[0].set_xlabel("Arrival Time (s)")
        axs[0].set_ylabel("Density")
        axs[0].set_title(f"{name}: Arrival Time")
        axs[1].set_xlabel("Slowness (s/deg)")
        axs[1].set_ylabel("Density")
        axs[1].set_title(f"{name}: Slowness")
        for ax in axs:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_dir is not None:
            import os
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
            fig.savefig(os.path.join(save_dir, f"{safe_name}_1d_hist.png"), dpi=200)

    return summaries
