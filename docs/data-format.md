# TAPIR-compatible data

TAPIR waveform CSVs contain samples in rows and traces in columns. Files have
no header unless noted below, and all values must be finite.

## Waveforms and time

- `time.csv`: one strictly increasing, uniformly sampled time vector in seconds.
- `UZ.csv`, `UR.csv`, `UT.csv`: three-component data.
- `U.csv`: optional generic one-component data. Otherwise a one-component run
  reads `UZ.csv`, `UR.csv`, or `UT.csv` according to the selected component.

All selected waveform files must have the same number of rows as `time.csv`
and the same number of trace columns.

## Geometry

`eventinfo.csv` has the header `lat,lon` and one coordinate row.

Earth receiver arrays use `station_metadata.csv`, with header `lat,lon` and one
receiver row per waveform column.

Earth source arrays and Mars use `station_metadata_db.csv`, with header
`dist_deg,baz` and one source row per waveform column. Distance is in degrees.
`baz` is the azimuth from the single receiver toward the source, in degrees
clockwise from north. In these workflows `eventinfo.csv` identifies the single
receiver location.

## Covariance and source-time function

With `CDopt: 3`, TAPIR reads `CD_UZ_fit.csv`, `CD_UR_fit.csv`, and
`CD_UT_fit.csv` for 3C data. Each is a symmetric square matrix with one row and
column per time sample. One-component runs read only their selected component.

With `man_stf: true`, `stf.csv` must have the header `time,stf`, followed by an
increasing time coordinate and source-time-function amplitude.

Earth preparation writes `dataset.yaml`, including `processing.bandpass`. The
runner reads this value and applies it to the STF. For externally prepared Mars
data, declare the same `bandpass: [fmin, fmax]` in the run YAML. If both the run
config and an Earth dataset manifest supply passbands, they must match.

## Validation examples

```bash
tapir validate-data DATASET
tapir validate-data DATASET --source-array --cdopt 3
tapir validate-data DATASET --mars --cdopt 3 --manual-stf
tapir validate-data DATASET --no-is-3c --component Z
```
