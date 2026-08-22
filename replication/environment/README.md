# Replication environment

The replication runner supports Python 3.11, 3.12, and 3.13. Run it from the
repository root in a clean virtual environment:

```bash
python -m venv .venv-replication
. .venv-replication/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,replication]"
```

The `replication` extra supplies Torch for fitting the CPU MDN, SciPy for the
conventional Rust comparator, and Matplotlib for optional figures. The saved-estimator
and support-warning checks themselves use the NumPy inference path and do not
need Torch. On systems that require a platform-specific CPU Torch wheel,
install that wheel according to the official Torch instructions before the
editable package install.

`requirements-base.txt` and `requirements-replication.txt` spell out the
minimum dependency floors without pretending to be a cross-platform lockfile.
The runner records the versions actually used in every report.

All computational stages set the usual BLAS/OpenMP thread counts to one by
default. Override that only when intentionally measuring another threading
configuration, and retain the resulting machine metadata with the report.
