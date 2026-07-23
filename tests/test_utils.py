import numpy as np

from vespainv.rjmcmc import compute_log_likelihood_L1
from vespainv.utils import _source_event_id_from_trace_path, generate_arr, inv_sqrt


def test_generate_arr_respects_minimum_spacing():
    np.random.seed(0)
    existing = np.array([2.0, 5.0, 8.0])
    candidate = generate_arr(np.array([0.0, 10.0]), existing, 0.5)
    assert np.all(np.abs(existing - candidate) >= 0.5)


def test_source_event_id_from_trace_path_uses_prefix_before_underscore():
    path = "/data/19940713114522_G.CRZF.00.BHZ.sac"

    assert _source_event_id_from_trace_path(path) == "19940713114522"


def test_covariance_whitening_is_finite_with_tiny_negative_eigenvalue():
    covariance = np.array([[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]])
    whitening = inv_sqrt(covariance)

    assert np.isrealobj(whitening)
    assert np.all(np.isfinite(whitening))
    log_likelihood = compute_log_likelihood_L1(
        np.ones((2, 1)), np.zeros((2, 1)), whitening
    )
    assert np.isfinite(log_likelihood)


def test_nonfinite_model_is_rejected_instead_of_poisoning_chain():
    observed = np.zeros((2, 1))
    modeled = np.array([[np.nan], [0.0]])

    assert compute_log_likelihood_L1(observed, modeled, sigma=1.0) == -np.inf
