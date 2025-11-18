from dataclasses import dataclass, field
import numpy as np
from vespainv.utils import generate_arr

@dataclass
class Bookkeeping:
    # array geometry
    refLat: float
    refLon: float
    refBaz: float # always lit. baz, i.e., station --> source
    # src Lat/Lon refers to source in receiver array, station in source array
    srcLat: float
    srcLon: float
    # inversion setup
    totalSteps:     int = 1e6
    burnInSteps:    int = None
    nSaveModels:    int = 100
    actionsPerStep: int = 2
    locDiff:        bool = False # perturb array location in (dist, baz)
    fitAtts:        bool = False # attenuation
    fitPhase:       bool = False # ph_hh and ph_vh
    srcArray:       bool = False # receiver or source array
    normOpt:        int = 2
    isMars:         bool = False
    # pref and tref for phase-aligned case (relative slowness), default to zero
    pref:           float = 0.0 # also serves as flag
    tref:           float = 0.0

    def __post_init__(self):
        if self.burnInSteps is None:
            self.burnInSteps = self.totalSteps // 2

@dataclass
class Prior:

    timeRange: tuple

    maxN: int = 5
    minSpace: float = 1.0
    slwRange: tuple = (-0.2, 0.2)
    ampRange: tuple = (-1, 1)
    attsRange: tuple = (0, 4)
    distDiffRange: tuple = (-5, 5)
    bazDiffRange: tuple = (-5, 5)

    arrStd: float = 0.15
    slwStd: float = None
    ampStd: float = None
    attsStd: float = None
    distDiffStd: float = None
    bazDiffStd: float = None

    sourceArray: bool = False

    def __post_init__(self):
        if self.slwStd is None:
            self.slwStd = 0.2 * (self.slwRange[1] - self.slwRange[0])
        if self.ampStd is None:
            self.ampStd = 0.2 * (self.ampRange[1] - self.ampRange[0])
        if self.attsStd is None:
            self.attsStd = 0.1 * (self.attsRange[1] - self.attsRange[0])
        if self.distDiffStd is None:
            self.distDiffStd = 0.2 * (self.distDiffRange[1] - self.distDiffRange[0])
        if self.bazDiffStd is None:
            self.bazDiffStd = 0.2 * (self.bazDiffRange[1] - self.bazDiffRange[0])
    
    @classmethod
    def example(cls, **kwargs):
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        params = {k: v for k, v in kwargs.items() if k in fields}
        return cls(**params)

@dataclass
class VespaModel:
    # Core parameters
    Nphase: int
    Ntrace: int
    arr: np.ndarray
    slw: np.ndarray
    amp: np.ndarray
    atts: np.ndarray
    distDiff: np.ndarray = None
    bazDiff: np.ndarray = None

    def __post_init__(self):
        if self.distDiff is None:
            self.distDiff = np.zeros(self.Ntrace)
        if self.bazDiff is None:
            self.bazDiff = np.zeros(self.Ntrace)

    @classmethod
    def create_empty(cls, Ntrace: int, prior: Prior):
        return cls(
            Nphase=0,
            Ntrace=Ntrace,
            arr=np.array([]),
            slw=np.array([]),
            amp=np.array([]),
            atts=np.array([]),
            distDiff=np.zeros(Ntrace),
            bazDiff=np.zeros(Ntrace)
        )
    
    @classmethod
    def create_random(cls, Nphase: int, Ntrace: int, time: np.ndarray, prior: Prior, arr: np.ndarray = None):
        if arr is None:
            arr=np.empty(Nphase)
            for iph in range(Nphase):
                arr[iph] = generate_arr(time, arr[:iph], prior.minSpace)
        return cls(
            Nphase=Nphase,
            Ntrace=Ntrace,
            arr=arr,
            slw=np.random.uniform(prior.slwRange[0], prior.slwRange[1], Nphase),
            amp=np.random.uniform(prior.ampRange[0], prior.ampRange[1], Nphase),
            atts=np.random.uniform(prior.attsRange[0], prior.attsRange[1], Nphase),
            distDiff=np.zeros(Ntrace),
            bazDiff=np.zeros(Ntrace)
        )

@dataclass
class Prior3c:
    
    timeRange: tuple

    maxN: int = 5
    minSpace: float = 1.0
    slwRange: tuple = (-0.2, 0.2)
    ampRange: tuple = (-1, 1)
    aziRange: tuple = (-180, 180)
    ph_hhRange: tuple = (-180, 180)
    ph_vhRange: tuple = (-90, 90)
    attsRange: tuple = (0, 4)
    distDiffRange: tuple = (-1, 1)
    bazDiffRange: tuple = (-10, 10)

    arrStd: float = 1.0
    slwStd: float = None
    ampStd: float = None
    aziStd: float = None
    ph_hhStd: float = None
    ph_vhStd: float = None
    attsStd: float = None
    distDiffStd: float = None
    bazDiffStd: float = None

    def __post_init__(self):
        if self.slwStd is None:
            self.slwStd = 0.2 * (self.slwRange[1] - self.slwRange[0])
        if self.ampStd is None:
            self.ampStd = 0.2 * (self.ampRange[1] - self.ampRange[0])
        if self.aziStd is None:
            self.aziStd = 0.2 * (self.aziRange[1] - self.aziRange[0])
        if self.ph_hhStd is None:
            self.ph_hhStd = 0.2 * (self.ph_hhRange[1] - self.ph_hhRange[0])
        if self.ph_vhStd is None:
            self.ph_vhStd = 0.2 * (self.ph_vhRange[1] - self.ph_vhRange[0])
        if self.attsStd is None:
            self.attsStd = 0.2 * (self.attsRange[1] - self.attsRange[0])
        if self.distDiffStd is None:
            self.distDiffStd = 0.2 * (self.distDiffRange[1] - self.distDiffRange[0])
        if self.bazDiffStd is None:
            self.bazDiffStd = 0.2 * (self.bazDiffRange[1] - self.bazDiffRange[0])
    
    @classmethod
    def example(cls, **kwargs):
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        params = {k: v for k, v in kwargs.items() if k in fields}
        return cls(**params)

@dataclass
class VespaModel3c:
    # Core parameters
    Nphase: int
    Ntrace: int
    # Below are for each phase
    arr: np.ndarray
    slw: np.ndarray
    amp: np.ndarray
    dip: np.ndarray
    azi: np.ndarray
    ph_hh: np.ndarray
    ph_vh: np.ndarray
    atts: np.ndarray
    wvtype: np.ndarray
    # Below are for each trace/station/marsquake
    distDiff: np.ndarray = None
    bazDiff: np.ndarray = None
    def __post_init__(self):
        if self.distDiff is None:
            self.distDiff = np.zeros(self.Ntrace)
        if self.bazDiff is None:
            self.bazDiff = np.zeros(self.Ntrace)

    @classmethod
    def create_empty(cls, Ntrace: int):
        return cls(
            Nphase=0,
            Ntrace=Ntrace,
            arr=np.array([]),
            slw=np.array([]),
            amp=np.array([]),
            baz=np.array([]),
            dip=np.array([]),
            azi=np.array([]),
            ph_hh=np.array([]),
            ph_vh=np.array([]),
            atts=np.array([]),
            wvtype=np.array([]),
            distDiff=np.zeros(Ntrace),
            bazDiff=np.zeros(Ntrace)
        )
    
    @classmethod
    def create_random(cls, Nphase: int, Ntrace: int, time: np.ndarray, prior: Prior, arr: np.ndarray = None):
        if arr is None:
            arr=np.empty(Nphase)
            for iph in range(Nphase):
                arr[iph] = generate_arr(time, arr[:iph], prior.minSpace)
        return cls(
            Nphase=Nphase,
            Ntrace=Ntrace,
            arr=arr,
            slw=np.random.uniform(prior.slwRange[0], prior.slwRange[1], Nphase),
            amp=np.random.uniform(prior.ampRange[0], prior.ampRange[1], Nphase),
            baz=np.zeros(Nphase),
            # dip=np.random.uniform(prior.dipRange[0], prior.dipRange[1], Nphase),
            dip=np.zeros(Nphase),
            azi=np.random.uniform(prior.aziRange[0], prior.aziRange[1], Nphase),
            ph_hh=np.random.uniform(prior.ph_hhRange[0], prior.ph_hhRange[1], Nphase),
            ph_vh=np.random.uniform(prior.ph_vhRange[0], prior.ph_vhRange[1], Nphase),
            atts=np.random.uniform(prior.attsRange[0], prior.attsRange[1], Nphase),
            wvtype=np.random.randint(2, size=Nphase),
            distDiff=np.zeros(Ntrace),
            bazDiff=np.zeros(Ntrace)
        )
