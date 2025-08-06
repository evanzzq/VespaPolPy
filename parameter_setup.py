# ---- Parameter setup ----
# filedir = "H:\My Drive\Research\VespaPolPy"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

isSyn = False
is3c = True # for synthetic this will be overriden
comp = "Z" # only applies to real data

modname = "201111221848"
runname = "run31_3c_CD_fit"
totalSteps = int(5e4)

burnInSteps = int(4e4)
nSaveModels = 200
actionsPerStep = 2

maxN = 5

ampRange = (-1., 1.) # only applies to real data
slwRange = (2., 8.) # only applies to real data
minSpace = 1.0

CDopt = 3 # 0 - False (single Sigma value), 1 - Empirical, 2 - Robust, 3 - Fit

isbp = False
freqs = (0.02, 1.0)
isds = False

bazRange = (-50., 0.)

locDiff = False
distDiffRange = (-5., -5.)
bazDiffRange = (-5., -5.)

phaseBaz = False
fitAtts = False