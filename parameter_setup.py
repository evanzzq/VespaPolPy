# ---- Parameter setup ----
filedir = "H:\My Drive\Research\VespaPolPy"
# filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

isSyn = False
is3c = True # for synthetic this will be overriden
comp = "Z" # only applies to real data

modname = "201111221848"
runname = "run24_3c"
totalSteps = int(1e6)

burnInSteps = int(8e5)
nSaveModels = 1000
actionsPerStep = 2

maxN = 10

ampRange = (-1., 1.) # only applies to real data
slwRange = (0., 8.) # only applies to real data
minSpace = 1.0

isbp = False
freqs = (0.02, 1.0)
isds = False

locDiff = False
distRange = (-5., -5.)
bazRange = (-5., -5.)

fitNoise = False
fitAtts = False