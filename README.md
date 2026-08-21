# TAPIR

<p align="center">
  <img src="docs/images/tapir_logo.png" alt="TAPIR logo" width="500">
</p>


TAPIR is a Python toolbox for transdimensional array-based phase inversion, data preparation, and ensemble-based analysis of array data.

## Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

For reproducible experiment environments, install the tested dependency set:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
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

Validate an Earth receiver-array dataset before running it:

```bash
tapir validate-data /path/to/RealData/my_event --cdopt 3
```

Validate externally prepared Mars data:

```bash
tapir validate-data /path/to/RealData/mars_event --mars --cdopt 3 --manual-stf
```

Run an inversion:

```bash
tapir run --config configs/example_parameter_setup.yaml
```

Run the small, self-contained smoke examples:

```bash
tapir run --config configs/example_run_earth.yaml
tapir run --config configs/example_run_mars.yaml
```

Summarize completed or in-progress chain traces:

```bash
tapir summarize /path/to/runs/dataset/runname
```

The compatibility entrypoint also works:

```bash
python main.py --config configs/example_parameter_setup.yaml
```

## Workflows

### Earth Data

Use `tapir prep-earth` to convert SAC files into the CSV-based input format expected by the inversion workflow. The preprocessing step can apply optional bandpass filtering, optional downsampling, optional time-window trimming, noise-based covariance estimation, and automatic trace rejection based on noise statistics and SNR. It can also optionally save a quick-look QC PDF into the prepared event directory.

Earth preparation writes these choices to `dataset.yaml` in the prepared data
directory. During inversion, TAPIR automatically applies the recorded passband
to the source-time function using the same two-corner, zero-phase Butterworth
filter used for the waveform and noise traces.

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

Set `isMars: true` in the inversion config. In this mode, TAPIR interprets metadata as `dist/baz`, enforces source-array geometry, and uses Mars-specific geometry constants in the 3C transform. The effective half-space velocities used by the free-surface transform can be set with `fstVp` and `fstVs` (in km/s). If omitted, they default to 6.571/4.1 for Earth and 5.0/3.0 for Mars.

Because Mars data are prepared externally, set the data passband in the Mars
run experiment, for example `bandpass: [0.2, 0.6]`. TAPIR does not refilter the
Mars waveform CSVs; it applies this passband to the STF and saves the result as
`stf_used.csv` in the run directory.

`tapir validate-data` is the supported handoff between external Mars processing
and TAPIR. It checks file presence, dimensions, sampling, coordinates, optional
source-time functions, and fitted covariance matrices without modifying data.
See [the data-format reference](docs/data-format.md) for the CSV contract.

## Configuration

Inversion runs use YAML configuration files with:

- `workspace`: shared path roots for the local machine or project clone
- `defaults`: shared settings
- `experiments`: one or more experiment-specific overrides

The recommended layout is:

- one workspace YAML for local path roots
- one prep YAML per preparation job
- one run YAML per inversion campaign

Example inversion settings are provided in `configs/example_parameter_setup.yaml`.
Example workspace settings are provided in `configs/workspace.yaml`.
Keep machine-specific paths in `configs/workspace.local.yaml`; it is ignored by
Git and automatically overrides the shared workspace file.

For Earth preprocessing, `tapir prep-earth` can also read a separate YAML file such as `configs/example_prep_earth.yaml`.

## Expected Data Layout

TAPIR expects runtime data to live in workspace directories such as:

- `RealData/`
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
- `examples/data/`: small constructed datasets for runnable examples and tests
- `docs/`: data-format and workflow documentation
- `tests/`: lightweight regression tests

## Run Progress

Each chain writes `log_likelihood.txt`, `loge.txt`, `Nphase.txt`,
`progress.txt`, and `likelihood_phase_count_progress.png` while sampling.
For multi-chain runs these files are kept in each `chain_N/` directory. Final
ensemble files are written when the chain completes.

`tapir summarize RUN_DIR` reads these traces, prints per-chain statistics, and
writes `run_summary.png`. It works while chains are running as soon as their
trace files contain data.

## Notes

The `prep_data()` loader is intended for internal use by the inversion runner. In normal use, the public entrypoints are `tapir prep-earth` and `tapir run`.

Contributions are described in [CONTRIBUTING.md](CONTRIBUTING.md). Citation
metadata are provided in [CITATION.cff](CITATION.cff), and release changes are
tracked in [CHANGELOG.md](CHANGELOG.md).

## License

TAPIR is released under the [MIT License](LICENSE).
