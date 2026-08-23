# Changelog

## 0.1.0b1 - Unreleased

- Added the `StructuralModel`/`fit`/`infer` workflow for learning approximate
  joint posteriors from prior-predictive simulations.
- Added named posterior draws, summaries, transformations, batch inference,
  support diagnostics, and representation-level predictive checks.
- Added checksummed, pickle-free estimator and result artifacts with model and
  data-representation compatibility checks.
- Added NumPy-only CPU inference from trained neural estimators; Torch is
  required for training.
- Added public-data comparisons for Eight Schools and Rust's bus replacement
  model. Eight Schools passes its bounded smoke thresholds; the empirical Rust
  neural posterior fails its frozen accuracy gates.
- Added tests and CI for Python 3.11--3.13.

## 0.1.0-alpha

- Initial simulator, estimation, result, and command-line interfaces.
