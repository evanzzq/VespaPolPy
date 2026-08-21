# Minimal example data

These small, constructed numerical fixtures demonstrate TAPIR's file layout.
They are not scientific observations and are not intended for interpretation.

Validate them with:

```bash
tapir validate-data examples/data/earth_receiver_minimal
tapir validate-data examples/data/mars_minimal --mars --manual-stf
```

The Earth fixture includes the `dataset.yaml` manifest normally written by
`prep-earth`. Mars passbands belong in the run configuration because Mars data
preparation is external to TAPIR.
