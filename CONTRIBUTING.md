# Contributing to TAPIR

Contributions are welcome through GitHub issues and pull requests.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .[dev]
pytest
```

Before opening a pull request, run the test suite and keep generated data,
experiment outputs, machine-specific paths, and unpublished research material
out of the commit. Add focused tests for behavioral changes. Scientific changes
should describe assumptions, units, coordinate conventions, and their effect on
existing results.

Public command names use hyphens, such as `validate-data`; Python functions and
configuration implementation use standard Python underscore naming where
appropriate.

## Reporting problems

Include the TAPIR version or commit, Python version, operating system, relevant
configuration, and the smallest data-layout description needed to reproduce the
problem. Do not upload data that you are not permitted to redistribute.
