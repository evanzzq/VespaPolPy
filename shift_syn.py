import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ================== USER PARAMETERS ==================

filedir = r"H:/My Drive/Research/VespaPolPy"
# filedir = r"/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

# Source and destination model names (folders under SynData/)
src_modname = "model11"
dst_modname = "model11_shifted"

# 1c or 3c?
is3c = False

# Time-shift settings:
#   keys   = trace indices (0-based)
#   values = shift in seconds (+ => later, - => earlier)
SHIFT_MAP = {
    2: 1.3,    # example: shift trace 2 by +0.3 s
    5: -1.9,   # example: shift trace 5 by -0.2 s
    # add more if you want
}

# ====================================================


def shift_traces_interp(U, time, shift_map, is3c=False):
    """
    Shift selected traces by arbitrary (possibly non-integer) times using interpolation.
    Positive shift -> later arrival (shift to the right).

    For output at time t, we sample U_original(t - shift).
    """
    U_shifted = U.copy()
    for idx, tshift in shift_map.items():
        t_src = time - tshift  # sample original at t - tshift

        if is3c:
            for ic in range(3):
                U_shifted[:, idx, ic] = np.interp(
                    t_src, time, U[:, idx, ic], left=0.0, right=0.0
                )
        else:
            U_shifted[:, idx] = np.interp(
                t_src, time, U[:, idx], left=0.0, right=0.0
            )

    return U_shifted


def plot_before_after_1c(time, U_orig, U_shifted, station_metadata_db, outpath):
    """
    Overlay original (black) and shifted (red) seismograms for 1C.
    """
    n_traces = U_orig.shape[1]
    amp_max = max(np.max(np.abs(U_orig)), np.max(np.abs(U_shifted)))
    offset = 1.2 * amp_max

    plt.figure(figsize=(10, 6))

    for i in range(n_traces):
        # original in black
        plt.plot(time, U_orig[:, i] + i * offset,
                 color="black", linewidth=1.0)

        # shifted in red (overlay)
        plt.plot(time, U_shifted[:, i] + i * offset,
                 color="red", linewidth=0.8)

        # station labels
        if station_metadata_db is not None and station_metadata_db.shape[0] == n_traces:
            dist, baz = station_metadata_db[i, :]
            plt.text(time[-1] + 0.5, i * offset,
                     f"{dist:.1f}°, {baz:.0f}°",
                     va="center", fontsize=7)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (offset by trace index)")
    plt.title("1C Seismograms (Original = Black, Shifted = Red)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_before_after_3c(time, U_orig, U_shifted, station_metadata_db, outpath):
    """
    Overlay original (black) and shifted (red) seismograms for 3C (Z, R, T).
    """
    components = ["Z", "R", "T"]
    Tn, N, _ = U_orig.shape
    amp_max = max(np.max(np.abs(U_orig)), np.max(np.abs(U_shifted)))
    offset = 1.2 * amp_max

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ic, comp in enumerate(components):
        ax = axes[ic]
        for i in range(N):
            # orig: black
            ax.plot(time, U_orig[:, i, ic] + i * offset,
                    color="black", linewidth=1.0)

            # shifted: red overlay
            ax.plot(time, U_shifted[:, i, ic] + i * offset,
                    color="red", linewidth=0.8)

            # label on right margin
            if station_metadata_db is not None and station_metadata_db.shape[0] == N:
                dist, baz = station_metadata_db[i, :]
                ax.text(time[-1] + 0.5, i * offset,
                        f"{dist:.1f}°, {baz:.0f}°",
                        va="center", fontsize=7)

        ax.set_ylabel(f"{comp} Amplitude")
        ax.set_title(f"{comp}-component (Black: Original, Red: Shifted)")
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

def main():
    syn_base = os.path.join(filedir, "SynData")
    src_dir = os.path.join(syn_base, src_modname)
    dst_dir = os.path.join(syn_base, dst_modname)

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)
    print(f"[INFO] Source:      {src_dir}")
    print(f"[INFO] Destination: {dst_dir}")

    # --- Load time vector ---
    time_path = os.path.join(src_dir, "time.csv")
    time = np.loadtxt(time_path, delimiter=",")
    dt = time[1] - time[0]
    print(f"[INFO] Loaded time vector ({len(time)} samples, dt={dt:.4f} s)")

    # --- Load station metadata for labeling (if present) ---
    station_metadata_db = None
    station_metadata_db_path = os.path.join(src_dir, "station_metadata_db.csv")
    if os.path.exists(station_metadata_db_path):
        station_metadata_db = np.loadtxt(station_metadata_db_path, delimiter=",", skiprows=1)
        print(f"[INFO] Loaded station_metadata_db.csv with shape {station_metadata_db.shape}")
    else:
        print("[WARN] station_metadata_db.csv not found; plots will omit dist/baz labels.")

    # --- Copy metadata & model files to destination ---
    files_to_copy = [
        "station_metadata.csv",
        "station_metadata_db.csv",
        "stf.csv",
        "Model.pkl",
        "Prior.pkl",
        "model_details.txt",
    ]

    for fname in files_to_copy:
        src_f = os.path.join(src_dir, fname)
        if os.path.exists(src_f):
            dst_f = os.path.join(dst_dir, fname)
            shutil.copyfile(src_f, dst_f)
            print(f"[INFO] Copied {fname}")
        else:
            print(f"[WARN] {fname} not found in source; skipping.")

    # --- Load synthetic data U (original) ---
    if is3c:
        uz_path = os.path.join(src_dir, "UZ.csv")
        ur_path = os.path.join(src_dir, "UR.csv")
        ut_path = os.path.join(src_dir, "UT.csv")

        if not (os.path.exists(uz_path) and os.path.exists(ur_path) and os.path.exists(ut_path)):
            raise FileNotFoundError("UZ.csv / UR.csv / UT.csv not found in source directory.")

        Z = np.loadtxt(uz_path, delimiter=",")
        R = np.loadtxt(ur_path, delimiter=",")
        Tcomp = np.loadtxt(ut_path, delimiter=",")

        if Z.ndim == 1:
            Z = Z[:, np.newaxis]
            R = R[:, np.newaxis]
            Tcomp = Tcomp[:, np.newaxis]

        U_orig = np.stack([Z, R, Tcomp], axis=-1)  # (T, N, 3)
        print(f"[INFO] Loaded 3C data: U.shape = {U_orig.shape}")
    else:
        u_path = os.path.join(src_dir, "U.csv")
        if not os.path.exists(u_path):
            raise FileNotFoundError("U.csv not found in source directory.")

        U_orig = np.loadtxt(u_path, delimiter=",")
        if U_orig.ndim == 1:
            U_orig = U_orig[:, np.newaxis]
        print(f"[INFO] Loaded 1C data: U.shape = {U_orig.shape}")

    # --- Apply time shifts ---
    if SHIFT_MAP:
        print(f"[INFO] Applying time shifts (seconds): {SHIFT_MAP}")
        U_shifted = shift_traces_interp(U_orig, time, SHIFT_MAP, is3c=is3c)
    else:
        print("[INFO] SHIFT_MAP is empty; no shifts applied.")
        U_shifted = U_orig.copy()

    # --- Save shifted data in destination folder ---
    if is3c:
        Zs = U_shifted[:, :, 0]
        Rs = U_shifted[:, :, 1]
        Ts = U_shifted[:, :, 2]

        np.savetxt(os.path.join(dst_dir, "UZ.csv"), Zs, delimiter=",")
        np.savetxt(os.path.join(dst_dir, "UR.csv"), Rs, delimiter=",")
        np.savetxt(os.path.join(dst_dir, "UT.csv"), Ts, delimiter=",")
        print("[INFO] Saved shifted 3C data (UZ/UR/UT.csv)")
    else:
        np.savetxt(os.path.join(dst_dir, "U.csv"), U_shifted, delimiter=",")
        print("[INFO] Saved shifted 1C data (U.csv)")

    # Copy time.csv
    shutil.copyfile(time_path, os.path.join(dst_dir, "time.csv"))

    # --- Plot before vs after and save figure ---
    fig_path = os.path.join(dst_dir, "synthetics_before_after.png")
    if is3c:
        plot_before_after_3c(time, U_orig, U_shifted, station_metadata_db, fig_path)
    else:
        plot_before_after_1c(time, U_orig, U_shifted, station_metadata_db, fig_path)

    print(f"[INFO] Saved comparison figure to: {fig_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
