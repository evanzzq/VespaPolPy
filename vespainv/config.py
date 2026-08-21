from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def load_yaml_mapping(path: str | Path) -> dict:
    """Load a generic YAML mapping."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")
    return config


def _merge_mappings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")
_PATHLIKE_KEYS = {"filedir", "workspace"}
_PATHLIKE_SUFFIXES = ("_dir", "_root", "_path", "_file")


def _is_pathlike_key(key: str) -> bool:
    return key in _PATHLIKE_KEYS or key.endswith(_PATHLIKE_SUFFIXES)


def _resolve_relative_path(value: str, base_dir: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_placeholders(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        def replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            current = context
            for part in key.split("."):
                if not isinstance(current, dict) or part not in current:
                    raise ValueError(f"Unknown config placeholder '{key}'.")
                current = current[part]
            return str(current)

        return _PLACEHOLDER_PATTERN.sub(replacer, value)
    if isinstance(value, list):
        return [_resolve_placeholders(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_placeholders(item, context) for key, item in value.items()}
    return value


def _resolve_path_fields(value: Any, base_dir: Path, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_path_fields(item, base_dir, parent_key=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_path_fields(item, base_dir, parent_key=parent_key) for item in value]
    if isinstance(value, str) and parent_key and _is_pathlike_key(parent_key):
        return _resolve_relative_path(value, base_dir)
    return value


def load_workspace(path: str | Path) -> dict:
    workspace_path = Path(path)
    workspace = load_yaml_mapping(workspace_path)
    local_override_path = workspace_path.with_name(f"{workspace_path.stem}.local{workspace_path.suffix}")
    if local_override_path.exists():
        local_override = load_yaml_mapping(local_override_path)
        workspace = _merge_mappings(workspace, local_override)
    paths = workspace.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"Workspace config at {workspace_path} must contain a 'paths' mapping.")

    resolved_paths = {
        key: _resolve_relative_path(str(value), workspace_path.parent)
        for key, value in paths.items()
    }
    workspace["paths"] = resolved_paths
    return workspace


def load_yaml_with_workspace(path: str | Path) -> dict:
    config_path = Path(path)
    config = load_yaml_mapping(config_path)

    workspace_ref = config.get("workspace")
    if workspace_ref:
        workspace_path = Path(workspace_ref)
        if not workspace_path.is_absolute():
            workspace_path = (config_path.parent / workspace_path).resolve()
        workspace = load_workspace(workspace_path)
        config = _resolve_placeholders(config, workspace)
        config["paths"] = workspace["paths"]

    config = _resolve_path_fields(config, config_path.parent)
    return config


def load_config(path: str | Path) -> dict:
    """Load a TAPIR YAML config file."""
    config_path = Path(path)
    config = load_yaml_with_workspace(config_path)
    if "experiments" not in config:
        raise ValueError(f"Config at {config_path} is missing an 'experiments' section.")
    if "defaults" not in config:
        config["defaults"] = {}

    return config


def write_dataset_manifest(
    output_dir: str | Path,
    *,
    body: str,
    array_type: str,
    bandpass,
    downsample_hz,
    time_window,
) -> Path:
    """Record preparation choices needed by later inversion steps."""
    output_path = Path(output_dir) / "dataset.yaml"
    manifest = {
        "schema_version": 1,
        "body": body,
        "array_type": array_type,
        "processing": {
            "bandpass": list(bandpass) if bandpass is not None else None,
            "downsample_hz": downsample_hz,
            "time_window": list(time_window) if time_window is not None else None,
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)
    return output_path


def load_dataset_manifest(dataset_dir: str | Path) -> dict:
    path = Path(dataset_dir) / "dataset.yaml"
    if not path.is_file():
        return {}
    return load_yaml_mapping(path)
