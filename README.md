# TAPIR

<p align="center">
  <img src="docs/images/tapir_logo.png" alt="TAPIR logo" width="500">
</p>


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

Prepare Earth source-array SAC files, where one station records many events:

```bash
tapir prep-source-earth --config configs/example_prep_source_earth.yaml
```

Regenerate a QC summary figure from an existing prepared event directory:

```bash
tapir plot-prep /path/to/RealData/my_event
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

Use `tapir prep-earth` to convert SAC files into the CSV-based input format expected by the inversion workflow. The preprocessing step can apply optional bandpass filtering, optional downsampling, optional time-window trimming, noise-based covariance estimation, and automatic trace rejection based on noise statistics and SNR. It can also optionally save a quick-look QC PDF into the prepared event directory.

`prep-earth` reads all SAC files in the event directory and uses the SAC channel header to identify components, but in practice the directory should ideally contain only the Z/R/T files intended for preparation. When a noise directory is provided, the prep step now writes only the fitted covariance outputs (`CD_UZ_fit.csv`, `CD_UR_fit.csv`, `CD_UT_fit.csv`) for downstream use.

The typical Earth workflow is:

1. Gather SAC files for one event.
2. Prepare the event directory with `tapir prep-earth`.
3. Review or adjust the inversion YAML config.
4. Run `tapir run --config your_config.yaml`.

Example preprocessing settings are provided in `configs/example_prep_earth.yaml`.

For source-array Earth data, use `tapir prep-source-earth` on SAC files named with
the event id before the first underscore, such as
`19940713114522_G.CRZF.00.BHZ.sac`. This mode groups Z/R/T traces by event,
writes one waveform column per event, stores source/event coordinates in
`station_metadata.csv`, and writes the single receiver station location to
`eventinfo.csv`. Use an inversion config with `srcArray: true`.

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

Relative `filedir` values are resolved relative to the YAML config file. For
configs stored in `configs/`, use `filedir: ".."` to point at the repository
root containing `RealData/`, `SynData/`, and `runs/`.

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
