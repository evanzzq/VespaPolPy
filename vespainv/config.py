from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_mapping(path: str | Path) -> dict:
    """Load a generic YAML mapping."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")
    return config


def load_config(path: str | Path) -> dict:
    """Load a TAPIR YAML config file."""
    config_path = Path(path)
    config = load_yaml_mapping(config_path)
    if "experiments" not in config:
        raise ValueError(f"Config at {config_path} is missing an 'experiments' section.")
    if "defaults" not in config:
        config["defaults"] = {}

    return config
