# ---- Parameter setup ----
filedir = "H:\My Drive\Research\VespaPolPy"
# filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

isSyn = True
is3c = True # for synthetic this will be overriden
comp = "Z" # only applies to real data

modname = "model6"
runname = "run1_3c_CD_fit_multichain_L2"
num_chains = 1
totalSteps = int(1e5)
burnInSteps = int(6e4)
nSaveModels = 200
actionsPerStep = 2

maxN = 10

ampRange = (-1., 1.) # only applies to real data
slwRange = (0., 3.) # only applies to real data
minSpace = 3.0

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
fitPhase = True
normOpt = 2