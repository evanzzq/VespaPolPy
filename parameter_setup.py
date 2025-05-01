# ---- Parameter setup ----
isSyn = False
is3c = False # for synthetic this will be overriden
comp = "Z" # only applies to real data

modname = "201111221848"
runname = "run2_Z"
totalSteps = int(1e6)

burnInSteps = int(8e5)
nSaveModels = 100
actionsPerStep = 2

ampRange = (-1., 1.) # only applies to real data
slwRange = (0., 8.) # only applies to real data

isbp = True
freqs = (0.02, 1.0)

locDiff = False
distRange = (-5., -5.)
bazRange = (-5., -5.)

Temp = 1 # acceptance rate for worse models lower if smaller