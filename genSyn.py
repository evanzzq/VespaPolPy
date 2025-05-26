import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from vespainv.model import VespaModel, Prior, VespaModel3c, Prior3c
from vespainv.waveformBuilder import create_U_from_model, create_U_from_model_3c, create_U_from_model_3c_freqdomain
from vespainv.utils import dest_point
from parameter_setup import filedir

# Parameter setup
modname = "model4"
Nphase = 3
is3c = False
ampRange = (-1, 1)
slwRange = (0, 1)

# Parameter setup: stf
f0 = 0.5
dt = 0.05

# Parameter setup: time vector
tmax = 100

# Parameter setup: array
srcLat = 0.0
srcLon = 0.0
base_dist = 35.0
base_baz = 30.0
Ntrace = 25
refLat, refLon = dest_point(srcLat, srcLon, base_baz, base_dist)

# Parameter setup: location perturbation
locDiff = False
distDiff = np.random.uniform(-5.0, 5.0, Ntrace)
bazDiff  = np.random.uniform(-5.0, 5.0, Ntrace)

# Parameter setup: arrival times
defAll = True
arr = np.array([25, 50, 75])
slw = np.array([0.2, 0.4, 0.6])
amp = np.array([1, 0.8, 0.6])
dip = np.array([20, 45, 90, 50])
azi = np.array([0, 0, 90, 45])
ph_hh = np.array([10, 20, 30, 40])
ph_vh = np.array([40, 30, 20, 10])
atts = np.array([1, 1, 1, 1])
svfac = np.array([0, 1, 0, 0.5])
wvtype = np.array([1, 0, 0, 0])

synDir = os.path.join(filedir, "SynData", modname)
os.makedirs(synDir, exist_ok=True)

# Create stf
stf_time_0 = np.arange(-4 / f0, 4 / f0 + dt, dt)
stf_0 = np.exp(-stf_time_0 ** 2 / (2 * (1 / (2 * np.pi * f0)) ** 2))
stf_time = stf_time_0[:-1]
stf = np.diff(stf_0) / np.diff(stf_time_0)
stf = stf / np.max(np.abs(stf))

# Pack and save stf
stf_array = np.column_stack([stf_time, stf])
np.savetxt(os.path.join(synDir, "stf.csv"), stf_array, delimiter=",", header="time,stf", comments="")

# Generate station metadata and save
time = np.arange(0, tmax, dt)
# np.random.seed(0)
dists = base_dist + np.random.uniform(-5.0, 5.0, Ntrace)
bazs = base_baz + np.random.uniform(-5.0, 5.0, Ntrace)
idx = np.argsort(dists); dists, bazs = dists[idx], bazs[idx] # sort by dist
station_metadata = np.column_stack((dists, bazs))

np.savetxt(os.path.join(synDir, "station_metadata.csv"), station_metadata, delimiter=",", header="distance,baz", comments="")

# Define prior and model, and save
if is3c:
    prior = Prior3c(refLat=refLat, refLon=refLon, refBaz=base_baz, srcLat=srcLat, srcLon=srcLon, timeRange=(time[0],time[-1]), ampRange=ampRange, slwRange=slwRange)
    model = VespaModel3c.create_random(
        Nphase=Nphase, Ntrace=Ntrace, time=time, prior=prior, arr=arr
        ) if not defAll else VespaModel3c(
            Nphase=Nphase, Ntrace=Ntrace, arr=arr, slw=slw, amp=amp, dip=dip, azi=azi, ph_hh=ph_hh, ph_vh=ph_vh, atts=atts, svfac=svfac, wvtype=wvtype
        )
else:
    prior = Prior(refLat=refLat, refLon=refLon, refBaz=base_baz, srcLat=srcLat, srcLon=srcLon, timeRange=(time[0],time[-1]), ampRange=ampRange, slwRange=slwRange)
    model = VespaModel.create_random(
        Nphase=Nphase, Ntrace=Ntrace, time=time, prior=prior, arr=arr
        ) if not defAll else VespaModel(
            Nphase=Nphase, Ntrace=Ntrace, arr=arr, slw=slw, amp=amp
        )
    if locDiff:
        model.distDiff = distDiff
        model.bazDiff  = bazDiff

# Save model details as a human-readable text file
with open(os.path.join(synDir, "model_details.txt"), "w") as ftxt:
    ftxt.write("=== Synthetic Model Details ===\n")
    ftxt.write(f"Number of phases: {Nphase}\n")
    ftxt.write(f"Number of traces: {Ntrace}\n")
    ftxt.write(f"Base distance: {base_dist} km\n")
    ftxt.write(f"Base backazimuth: {base_baz} deg\n")
    ftxt.write(f"Time range: {time[0]} to {time[-1]} s\n")
    ftxt.write(f"Sampling interval: {dt} s\n")
    ftxt.write(f"Source-time function: Gaussian derivative, f0 = {f0} Hz\n")
    ftxt.write(f"3-component data: {is3c}\n\n")

    ftxt.write("--- Arrival Times ---\n")
    ftxt.write(np.array2string(model.arr, separator=", ") + "\n\n")
    ftxt.write("--- Slowness ---\n")
    ftxt.write(np.array2string(model.slw, separator=", ") + "\n\n")
    ftxt.write("--- Amplitudes ---\n")
    ftxt.write(np.array2string(model.amp, separator=", ") + "\n\n")
    if hasattr(model, 'dip'):
        ftxt.write("--- Polarization Dip Angle ---\n")
        ftxt.write(np.array2string(model.dip, separator=", ") + "\n\n")
    if hasattr(model, 'azi'):
        ftxt.write("--- Polarization Azimuth Angle ---\n")
        ftxt.write(np.array2string(model.azi, separator=", ") + "\n\n")
    if hasattr(model, 'ph_hh'):
        ftxt.write("--- Phase Difference: HH ---\n")
        ftxt.write(np.array2string(model.ph_hh, separator=", ") + "\n\n")
    if hasattr(model, 'ph_vh'):
        ftxt.write("--- Phase Difference: VH ---\n")
        ftxt.write(np.array2string(model.ph_vh, separator=", ") + "\n\n")
    if hasattr(model, 'atts'):
        ftxt.write("--- Attenuation (t* for S) ---\n")
        ftxt.write(np.array2string(model.atts, separator=", ") + "\n\n")
    if hasattr(model, 'svfac'):
        ftxt.write("--- SV Amplitude Ratio ---\n")
        ftxt.write(np.array2string(model.svfac, separator=", ") + "\n\n")
    if hasattr(model, 'wvtype'):
        ftxt.write("--- Wave Type (P = True) ---\n")
        ftxt.write(np.array2string(model.wvtype, separator=", ") + "\n\n")
    if locDiff:
        ftxt.write("--- Distance perturbation (deg) ---\n")
        ftxt.write(np.array2string(model.distDiff, separator=", ") + "\n\n")
        ftxt.write("--- BAZ perturbation (deg) ---\n")
        ftxt.write(np.array2string(model.bazDiff, separator=", ") + "\n\n")

with open(os.path.join(synDir, "Model.pkl"), "wb") as f1:
    pickle.dump(model, f1)
with open(os.path.join(synDir, "Prior.pkl"), "wb") as f2:
    pickle.dump(prior, f2)

# Generate U, plot, and save
if is3c:
    U, _, _ = create_U_from_model_3c_freqdomain(model, prior, station_metadata, time, stf_time, stf)
else:
    U = create_U_from_model(model, prior, station_metadata, time, stf_time, stf)

if is3c:
    components = ['Z', 'R', 'T']
    fig, axes = plt.subplots(1, 3, figsize=(12, 10), sharex=True)
    offset = 1.2 * np.max(np.abs(U))  # spacing between traces
    n_traces = U.shape[1]
    for i, ax in enumerate(axes):
        for j in range(n_traces):
            ax.plot(time, U[:, j, i] + j * offset, color='black')
        ax.set_ylabel(f"{components[i]} Amplitude (offset)")
        ax.set_title(f"{components[i]} Component")
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
else:
    plt.figure(figsize=(10, 6))
    n_traces = U.shape[1]
    offset = 1.2 * np.max(np.abs(U))  # spacing between traces
    for i in range(n_traces):
        plt.plot(time, U[:, i] + i * offset, color="black")
        dist, baz = station_metadata[i,:]
        plt.text(time[-1] + 0.5, i * offset, f"{dist:.1f}°, {baz:.0f}°", 
             va='center', fontsize=8)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (offset by trace index)")
    plt.title("Synthetic Seismograms")
    plt.grid(True)
    plt.tight_layout()

plt.show()

np.savetxt(os.path.join(synDir, "time.csv"), time, delimiter=",")

if is3c:
    Z = U[:, :, 0]
    R = U[:, :, 1]
    T = U[:, :, 2]
    np.savetxt(os.path.join(synDir, "UZ.csv"), Z, delimiter=",")
    np.savetxt(os.path.join(synDir, "UR.csv"), R, delimiter=",")
    np.savetxt(os.path.join(synDir, "UT.csv"), T, delimiter=",")
else:
    np.savetxt(os.path.join(synDir, "U.csv"), U, delimiter=",")

sys.exit(0)