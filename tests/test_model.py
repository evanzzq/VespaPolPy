from vespainv.model import Bookkeeping, Prior


def test_bookkeeping_defaults_burnin_to_half_total_steps():
    bookkeeping = Bookkeeping(
        refLat=0.0,
        refLon=0.0,
        refBaz=0.0,
        srcLat=0.0,
        srcLon=0.0,
        totalSteps=100,
    )
    assert bookkeeping.burnInSteps == 50


def test_prior_populates_default_step_sizes():
    prior = Prior(timeRange=(0.0, 10.0), slwRange=(-2.0, 2.0), ampRange=(-1.0, 1.0))
    assert prior.slwStd == 0.8
    assert prior.ampStd == 0.4
