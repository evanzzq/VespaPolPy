# TAPIR

![TAPIR LOGO](docs/images/tapir_logo.png)

TAPIR is a Python toolbox for transdimensional array-based phase inversion and related synthetic/visualization workflows. This repository is being cleaned up from a research-code layout into a more reusable open-source package while preserving the current inversion core.

## Current Status

The core inversion modules live in [`vespainv/`](/Users/evanzhang/Documents/Research/VespaPolPy/vespainv). Legacy exploratory notebooks and helper scripts are kept in [`notebooks/`](/Users/evanzhang/Documents/Research/VespaPolPy/notebooks) and [`scripts/`](/Users/evanzhang/Documents/Research/VespaPolPy/scripts) so they do not define the package surface.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## Quick Start

Prepare Earth SAC files into TAPIR inputs:

```bash
tapir prep-earth --config configs/example_prep_earth.yaml
```

Then run an inversion from a YAML config:

```bash
tapir run --config configs/example_parameter_setup.yaml
```

The compatibility wrapper still works too:

```bash
python main.py --config configs/example_parameter_setup.yaml
```

## Repository Layout

- [`vespainv/`](/Users/evanzhang/Documents/Research/VespaPolPy/vespainv): importable package code
- [`configs/`](/Users/evanzhang/Documents/Research/VespaPolPy/configs): shareable example configs
- [`scripts/`](/Users/evanzhang/Documents/Research/VespaPolPy/scripts): legacy and exploratory helper scripts
- [`notebooks/`](/Users/evanzhang/Documents/Research/VespaPolPy/notebooks): research notebooks not required for package use
- [`tests/`](/Users/evanzhang/Documents/Research/VespaPolPy/tests): lightweight regression tests

## Configuration

The runner currently expects a config with:

- `defaults`: shared settings
- `experiments`: one or more experiment overrides

The `filedir` field should point to a directory containing data folders such as `SynData/`, `RealData/`, and `runs/`.

See [`configs/example_parameter_setup.yaml`](/Users/evanzhang/Documents/Research/VespaPolPy/configs/example_parameter_setup.yaml).

## Workflow

### Earth workflow

1. Start from SAC files for one event.
2. Run `tapir prep-earth` to generate TAPIR-ready inputs.
   Example:
   `tapir prep-earth --config configs/example_prep_earth.yaml`
3. Confirm the event directory contains:
   - `UZ.csv`, `UR.csv`, `UT.csv` for 3C, or `U?.csv` for 1C
   - `time.csv`
   - `station_metadata.csv` in `lat,lon`
   - `station_metadata_db.csv` in `dist_deg,baz`
   - `eventinfo.csv`
   - optional fitted covariance files such as `CD_UZ_fit.csv`
4. Create a config YAML pointing `filedir` at the workspace containing `RealData/`.
5. Run `tapir run --config your_config.yaml`.

`prepare_inputs_from_sac()` is the underlying Python function used by `tapir prep-earth`. It is intended for Earth data preprocessing from SAC and can still be called directly in Python, but the CLI is now the preferred entrypoint.

See [`configs/example_prep_earth.yaml`](/Users/evanzhang/Documents/Research/VespaPolPy/configs/example_prep_earth.yaml) for the preprocessing config format.

### Mars workflow

Mars preprocessing is not wrapped by TAPIR at the moment. Prepare the files externally, then place them in the expected event directory yourself:

- `UZ.csv`, `UR.csv`, `UT.csv` for 3C, or `U.csv` for 1C
- `time.csv`
- `station_metadata_db.csv` in `dist_deg,baz`
- `eventinfo.csv`
- optional fitted covariance files if `CDopt: 3`
- `stf.csv` if `man_stf: true`

Then set `isMars: true` in the config and run `tapir run --config your_config.yaml`.

When `isMars: true`, TAPIR now treats the metadata as `dist/baz`, enforces source-array geometry, and uses Mars-specific geometry constants in the 3C transform.

## Internal Loaders

`prep_data()` is now an internal loader used by the inversion runner. In normal use you should not need to call it directly. The intended flow is:

- `tapir prep-earth` for Earth SAC preprocessing
- manual file preparation for Mars
- `tapir run` for inversion

## Data and Outputs

This repository does not package example seismic datasets. Expected runtime data directories are ignored by Git:

- `RealData/`
- `SynData/`
- `runs/`

## Legacy Material

Several notebooks and scripts are intentionally retained for personal experiments and reproducibility, but they should be treated as auxiliary materials rather than the TAPIR public API.

## License

No license file has been added in this pass because license choice has real downstream consequences. Pick one explicitly before public release.

## README Figure

Put your README image in [`docs/images/`](/Users/evanzhang/Documents/Research/VespaPolPy/docs/images), for example as `tapir-overview.png`, and then replace the placeholder comment at the top of this README with:

```md
![TAPIR overview](docs/images/tapir-overview.png)
```
