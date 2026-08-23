# Replication code

The repository maintains two public-data comparisons:

- [`eight_schools`](eight_schools/) compares the neural posterior with
  quadrature and conditional Gaussian calculations for the canonical Eight
  Schools data.
- [`rust_1987`](rust_1987/) reproduces the conventional group-4 bus
  replacement estimate, constructs a dense Bayesian grid reference, evaluates
  a passing model-specific simulation classifier, and preserves the failed
  generic diagonal-MDN comparison with that same reference.

Install the optional dependencies with:

```bash
python -m pip install ".[replication]"
```

The headline numbers and interpretation are in [`BENCHMARKS.md`](../BENCHMARKS.md).
Each benchmark directory contains its own commands, configuration, and compact
machine-readable result. Raw Rust bus histories and generated estimators are
not committed.
