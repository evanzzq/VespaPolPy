import numpy as np

from vespainv.utils import _source_event_id_from_trace_path, generate_arr


def test_generate_arr_respects_minimum_spacing():
    np.random.seed(0)
    existing = np.array([2.0, 5.0, 8.0])
    candidate = generate_arr(np.array([0.0, 10.0]), existing, 0.5)
    assert np.all(np.abs(existing - candidate) >= 0.5)


def test_source_event_id_from_trace_path_uses_prefix_before_underscore():
    path = "/data/19940713114522_G.CRZF.00.BHZ.sac"

    assert _source_event_id_from_trace_path(path) == "19940713114522"
