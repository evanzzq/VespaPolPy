import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from vespainv.model import VespaModel3c, Prior3c, Bookkeeping
from vespainv.waveformBuilder import create_U_from_model_3c_freqdomain
from vespainv.utils import dest_point

# ==========================================================
# USER CONFIGURATION
# ==========================================================
# filedir = "H:/My Drive/Research/VespaPolPy"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"
modname = "model_vary_param"
param_to_vary = "azi"               # choose from ['azi', 'ph_hh', 'ph_vh']
param_values = np.array([0,45])  # degrees
f0 = 0.2
dt = 0.1
tmax = 100
Ntrace = 1
Nphase = 1
ampRange = (0, 1)
slwRange = (0., 10.)
base_dist = 75.0
base_baz = 30.0
base_az = (base_baz + 180)%360
srcLat, srcLon = 0.0, 0.0
refLat, refLon = dest_point(srcLat, srcLon, base_az, base_dist)
amplim = 2

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================
synDir = os.path.join(filedir, "SynData", modname)
os.makedirs(synDir, exist_ok=True)

# ==========================================================
# SOURCE-TIME FUNCTION (stf)
# ==========================================================
stf_time_0 = np.arange(-4 / f0, 4 / f0 + dt, dt)
stf_0 = np.exp(-stf_time_0 ** 2 / (2 * (1 / (2 * np.pi * f0)) ** 2))
stf_time = stf_time_0[:-1]
stf = np.diff(stf_0) / np.diff(stf_time_0)
stf = stf / np.max(np.abs(stf))

# ==========================================================
# STATION METADATA
# ==========================================================
time = np.arange(0, tmax, dt)
dists = base_dist + np.random.uniform(0, 0, Ntrace)
bazs = base_baz + np.random.uniform(0, 0, Ntrace)
stlas, stlos = np.zeros_like(dists), np.zeros_like(dists)
for ista in range(Ntrace):
    stlas[ista], stlos[ista] = dest_point(srcLat, srcLon, (bazs[ista]+180)%360, dists[ista])
idx = np.argsort(dists); dists, bazs, stlas, stlos = dists[idx], bazs[idx], stlas[idx], stlos[idx] # sort by dist
station_metadata_db = np.column_stack((dists, bazs))
station_metadata = np.column_stack((stlas, stlos))

# ==========================================================
# MODEL AND PRIOR
# ==========================================================
arr = np.array([50])
slw = np.array([4.])
amp = np.array([1.])
azi = np.array([45.])
ph_hh = np.array([0.])
ph_vh = np.array([0.])
atts = np.array([1])
wvtype = np.array([0])

prior = Prior3c(
    maxN=1,
    timeRange=(time[0], time[-1]), ampRange=ampRange, slwRange=slwRange
)

model = VespaModel3c(
    Nphase=Nphase, Ntrace=Ntrace, arr=arr, slw=slw, amp=amp,
    azi=azi, ph_hh=ph_hh, ph_vh=ph_vh,
    atts=atts, wvtype=wvtype
)

bk = Bookkeeping(refLat=refLat, refLon=refLon, refBaz=base_baz, srcLat=srcLat, srcLon=srcLon, fitAtts=False, fitPhase=True, isMars=False)

# ==========================================================
# SYNTHETIC GENERATION LOOP
# ==========================================================
colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
Us = []

for val in param_values:
    # Copy model
    m = VespaModel3c(**vars(model))

    # Assign varied parameter
    if param_to_vary == 'azi':
        m.azi = np.array([val])
    elif param_to_vary == 'ph_hh':
        m.ph_hh = np.array([val])
    elif param_to_vary == 'ph_vh':
        m.ph_vh = np.array([val])
    else:
        raise ValueError(f"Unsupported parameter: {param_to_vary}")

    # Generate synthetics
    U = create_U_from_model_3c_freqdomain(m, station_metadata, time, stf_time, stf, bk)
    Us.append(U)

# ==========================================================
# PLOT 1: SEISMOGRAMS (VERTICAL STACK, MID TRACE)
# ==========================================================
components = ['Z', 'R', 'T']
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
n_traces = Us[0].shape[1]
mid = n_traces // 2

for i, ax in enumerate(axes):
    for U, val, color in zip(Us, param_values, colors):
        ax.plot(time, U[:, mid, i], color=color, lw=1.5, label=f"{val:+.0f}°")
    ax.set_ylabel(f"{components[i]}")
    ax.set_xlim([40, 60])
    ax.set_ylim([-amplim, amplim])
    ax.grid(True)

axes[-1].set_xlabel("Time (s)")
axes[0].legend(title=f"{param_to_vary} (°)", loc='upper right')
plt.suptitle(f"Synthetic Seismograms vs {param_to_vary}")
plt.tight_layout()
plt.show()

# ==========================================================
# PLOT 2: PARTICLE MOTION (R–Z, T–Z, R–T)
# ==========================================================
pairs = [('R', 'Z', 1, 0), ('T', 'Z', 2, 0), ('R', 'T', 1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, (xlab, ylab, xi, yi) in zip(axes, pairs):
    for U, val, color in zip(Us, param_values, colors):
        x = U[:, mid, xi]
        y = U[:, mid, yi]
        ax.plot(x, y, color=color, lw=1.5, label=f"{val:+.0f}°")
    ax.set_xlim([-amplim, amplim])
    ax.set_ylim([-amplim, amplim])
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(True)
    ax.set_aspect('equal', 'box')

axes[0].legend(title=f"{param_to_vary} (°)")
plt.suptitle(f"Particle Motion (Mid Trace) vs {param_to_vary}")
plt.tight_layout()
plt.show()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ==========================================================
# PLOT 3: 3D PARTICLE MOTION (T–R–Z)
#   - X: T (to the right)
#   - Y: R (towards viewer)
#   - Z: Z (up)
# ==========================================================
fig = plt.figure(figsize=(8, 6))
ax3d = fig.add_subplot(111, projection='3d')

for U, val, color in zip(Us, param_values, colors):
    T = U[:, mid, 2]   # T component
    R = U[:, mid, 1]   # R component
    Zc = U[:, mid, 0]  # Z component
    ax3d.plot(T, R, Zc, color=color, lw=1.5, label=f"{val:+.0f}°")

# Axis labels
ax3d.set_xlabel("T (right)")
ax3d.set_ylabel("R (towards viewer)")
ax3d.set_zlabel("Z (up)")

# Limits
max_range = amplim
ax3d.set_xlim(-max_range, max_range)
ax3d.set_ylim(-max_range, max_range)
ax3d.set_zlim(-max_range, max_range)

# View angle so R roughly "comes out"
ax3d.view_init(elev=20, azim=-60)

# --- Remove grids + ticks ---
ax3d.grid(False)
ax3d.set_xticks([])
ax3d.set_yticks([])
ax3d.set_zticks([])

plt.title(f"3D Particle Motion (Mid Trace) vs {param_to_vary}")
plt.tight_layout()
plt.show()
