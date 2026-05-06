from pathlib import Path

from vespainv.config import load_config


def test_load_config_reads_example_config():
    config = load_config(Path("configs/example_parameter_setup.yaml"))
    assert "defaults" in config
    assert len(config["experiments"]) == 1
