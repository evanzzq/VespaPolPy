import matplotlib
import numpy as np

from vespainv.utils import make_tapir_dataset_vespagram, make_vespagram


matplotlib.use("Agg")


def test_make_vespagram_does_not_modify_waveforms():
    time = np.linspace(0.0, 2.0, 21)
    waveforms = np.column_stack([np.sin(time), np.cos(time)])
    original = waveforms.copy()
    metadata_db = np.array([[10.0, 180.0], [11.0, 180.0]])

    result = make_vespagram(
        waveforms,
        time,
        metadata_db,
        refLat=10.5,
        refLon=0.0,
        srcLat=0.0,
        srcLon=0.0,
        slow_grid=np.array([0.0, 1.0]),
        refBaz=0.0,
        srcArray=False,
        show=False,
    )

    assert result.shape == (2, 21)
    np.testing.assert_array_equal(waveforms, original)


def test_make_vespagram_supports_fourth_root_stack():
    time = np.linspace(0.0, 2.0, 21)
    waveforms = np.column_stack([np.sin(time), np.cos(time)])
    result = make_vespagram(
        waveforms,
        time,
        np.array([[10.0, 180.0], [11.0, 180.0]]),
        refLat=10.5,
        refLon=0.0,
        srcLat=0.0,
        srcLon=0.0,
        slow_grid=np.array([0.0, 1.0]),
        refBaz=0.0,
        srcArray=False,
        root_order=4,
        show=False,
    )

    assert result.shape == (2, 21)
    assert np.all(np.isfinite(result))


def test_make_tapir_dataset_vespagram_loads_standard_dataset(tmp_path):
    dataset = tmp_path / "prepared_event"
    dataset.mkdir()
    time = np.linspace(0.0, 2.0, 21)
    waveforms = np.column_stack([np.sin(time), np.cos(time)])
    np.savetxt(dataset / "time.csv", time, delimiter=",")
    np.savetxt(dataset / "UZ.csv", waveforms, delimiter=",")
    np.savetxt(
        dataset / "station_metadata_db.csv",
        np.array([[10.0, 180.0], [11.0, 180.0]]),
        delimiter=",",
        header="dist_deg,baz",
        comments="",
    )
    np.savetxt(
        dataset / "station_metadata.csv",
        np.array([[10.0, 0.0], [11.0, 0.0]]),
        delimiter=",",
        header="lat,lon",
        comments="",
    )
    np.savetxt(
        dataset / "eventinfo.csv",
        np.array([[0.0, 0.0]]),
        delimiter=",",
        header="evla,evlo",
        comments="",
    )

    vespa, loaded_time, slow_grid = make_tapir_dataset_vespagram(
        dataset,
        component="Z",
        slow_grid=np.array([0.0, 1.0]),
        show=False,
    )

    assert vespa.shape == (2, 21)
    np.testing.assert_array_equal(loaded_time, time)
    np.testing.assert_array_equal(slow_grid, np.array([0.0, 1.0]))
