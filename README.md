# TAPIR

![TAPIR LOGO](docs/images/tapir_logo.png)

TAPIR is a Python toolbox for transdimensional array-based phase inversion, synthetic waveform generation, and ensemble-based analysis of array data.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

## Quick Start

Prepare Earth SAC files into TAPIR-ready inputs:

```bash
tapir prep-earth --config configs/example_prep_earth.yaml
```

Run an inversion:

```bash
tapir run --config configs/example_parameter_setup.yaml
```

The compatibility entrypoint also works:

```bash
python main.py --config configs/example_parameter_setup.yaml
```

## Workflows

### Earth Data

Use `tapir prep-earth` to convert SAC files into the CSV-based input format expected by the inversion workflow. The preprocessing step can apply optional bandpass filtering, downsampling, noise-based covariance estimation, and trace rejection.

The typical Earth workflow is:

1. Gather SAC files for one event.
2. Prepare the event directory with `tapir prep-earth`.
3. Review or adjust the inversion YAML config.
4. Run `tapir run --config your_config.yaml`.

Example preprocessing settings are provided in `configs/example_prep_earth.yaml`.

### Mars Data

Mars preprocessing is expected to be handled externally. TAPIR can then run inversion on manually prepared files placed in an event directory.

For Mars runs, provide:

- `UZ.csv`, `UR.csv`, `UT.csv` for 3C data, or `U.csv` for 1C data
- `time.csv`
- `station_metadata_db.csv` in `dist_deg,baz`
- `eventinfo.csv`
- optional fitted covariance files when `CDopt: 3`
- `stf.csv` when `man_stf: true`

Set `isMars: true` in the inversion config. In this mode, TAPIR interprets metadata as `dist/baz`, enforces source-array geometry, and uses Mars-specific geometry constants in the 3C transform.

## Configuration

Inversion runs use YAML configuration files with:

- `defaults`: shared settings
- `experiments`: one or more experiment-specific overrides

Example inversion settings are provided in `configs/example_parameter_setup.yaml`.

For Earth preprocessing, `tapir prep-earth` can also read a separate YAML file such as `configs/example_prep_earth.yaml`.

## Expected Data Layout

TAPIR expects runtime data to live in workspace directories such as:

- `RealData/`
- `SynData/`
- `runs/`

An Earth event prepared for inversion typically contains:

- waveform CSVs
- `time.csv`
- `station_metadata.csv`
- `station_metadata_db.csv`
- `eventinfo.csv`
- optional fitted covariance files

## Repository Layout

- `vespainv/`: core package code
- `configs/`: example configuration files
- `scripts/`: helper and legacy scripts
- `notebooks/`: exploratory notebooks
- `tests/`: lightweight regression tests

## Notes

The `prep_data()` loader is intended for internal use by the inversion runner. In normal use, the public entrypoints are `tapir prep-earth` and `tapir run`.

## License

A project license has not been added yet.
