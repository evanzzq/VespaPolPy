# ---- Parameter setup ----
# filedir = "H:\My Drive\Research\VespaPolPy"
filedir = "/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/VespaPolPy"

isSyn = False
is3c = False # for synthetic this will be overriden
comp = "Z" # only applies to real data

modname = "200812021231"
runname = "run2_Z_CD_robust"
totalSteps = int(1e5)

burnInSteps = int(5e4)
nSaveModels = 500
actionsPerStep = 2

maxN = 5

ampRange = (-1., 1.) # only applies to real data
slwRange = (3., 10.) # only applies to real data
minSpace = 1.0

CDopt = 2 # 0 - False, 1 - Empirical, 2 - Robust

isbp = False
freqs = (0.02, 1.0)
isds = False

bazRange = (-50., 0.)

locDiff = False
distDiffRange = (-5., -5.)
bazDiffRange = (-5., -5.)

phaseBaz = True
fitAtts = False