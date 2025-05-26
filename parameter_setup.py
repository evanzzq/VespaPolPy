# ---- Parameter setup ----
filedir = "H:\My Drive\Research\VespaPolPy"
# filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

isSyn = True
is3c = True # for synthetic this will be overriden
comp = "Z" # only applies to real data

modname = "model5"
runname = "run2"
totalSteps = int(1e5)

burnInSteps = int(6e4)
nSaveModels = 100
actionsPerStep = 2

maxN = 3

ampRange = (-1., 1.) # only applies to real data
slwRange = (0., 8.) # only applies to real data
minSpace = 1.0

isbp = True
freqs = (0.02, 1.0)

locDiff = False
distRange = (-5., -5.)
bazRange = (-5., -5.)

fitNoise = False
fitAtts = False