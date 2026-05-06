import numpy as np

from vespainv.utils import generate_arr


def test_generate_arr_respects_minimum_spacing():
    np.random.seed(0)
    existing = np.array([2.0, 5.0, 8.0])
    candidate = generate_arr(np.array([0.0, 10.0]), existing, 0.5)
    assert np.all(np.abs(existing - candidate) >= 0.5)
