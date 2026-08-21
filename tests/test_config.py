from pathlib import Path

from vespainv.config import load_config, load_dataset_manifest, write_dataset_manifest


def test_load_config_reads_example_config():
    config = load_config(Path("configs/example_parameter_setup.yaml"))
    assert "defaults" in config
    assert len(config["experiments"]) == 2
    assert config["experiments"][0]["dataset"] == "earth_receiver_minimal"
    assert config["experiments"][1]["dataset"] == "mars_minimal"


def test_dataset_manifest_round_trip(tmp_path: Path):
    write_dataset_manifest(
        tmp_path,
        body="earth",
        array_type="source",
        bandpass=(0.02, 0.4),
        downsample_hz=2.0,
        time_window=(10.0, 30.0),
    )

    manifest = load_dataset_manifest(tmp_path)

    assert manifest["body"] == "earth"
    assert manifest["array_type"] == "source"
    assert manifest["processing"]["bandpass"] == [0.02, 0.4]


def test_load_config_resolves_workspace_placeholders(tmp_path: Path):
    workspace_path = tmp_path / "workspace.yaml"
    workspace_path.write_text(
        "\n".join(
            [
                "paths:",
                f"  real_data_root: {tmp_path / 'RealData'}",
                f"  runs_root: {tmp_path / 'runs' / 'data'}",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"workspace: {workspace_path.name}",
                "defaults:",
                "  ref_manual: false",
                "  is3c: true",
                "  comp: Z",
                "  sigma: 0.02",
                "  num_chains: 1",
                "  totalSteps: 10",
                "  burnInSteps: 5",
                "  nSaveModels: 1",
                "  actionsPerStep: 1",
                "  maxN: 1",
                "  man_stf: false",
                "  ampRange: [-1.0, 1.0]",
                "  slwRange: [2.0, 10.0]",
                "  minSpace: 1.0",
                "  CDopt: 0",
                "  locDiff: false",
                "  distDiffRange: [-1.0, 1.0]",
                "  bazDiffRange: [-1.0, 1.0]",
                "  fitAtts: false",
                "  fitLoge: true",
                "  fitPhase: false",
                "  normOpt: 1",
                "  isMars: false",
                "  srcArray: false",
                "  pref: 0.0",
                "  refLat: 0.0",
                "  refLon: 0.0",
                "  refBaz: 0.0",
                "experiments:",
                "  - dataset: example_event",
                "    runname: example_run",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["paths"]["real_data_root"] == str((tmp_path / "RealData").resolve())
    assert config["paths"]["runs_root"] == str((tmp_path / "runs" / "data").resolve())
