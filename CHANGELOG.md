# Changelog

## 0.1.0b2

- Added experimental `fit(..., backend="spline")`, adapting the recent
  stockpiling inference architecture to the public simulator interface.
  The validated MDN remains the default: the new flow passes conjugate
  recovery checks but misses some Eight Schools accuracy limits.
- Preserved train-only preprocessing, bounded parameter transforms, seeded
  sampling, joint posterior outputs, and checksummed numeric save/load files.
- Added optional Torch/nflows dependencies for spline training and inference;
  existing MDN artifacts still load with NumPy-only CPU inference.
- Added conjugate-posterior recovery and artifact round-trip checks for the
  spline backend, and a separate reproducible Eight Schools comparison.
- Included a context-only connection in every spline conditioner so its first
  coordinate can respond to observed data for every permutation seed.

## 0.1.0b1 - Unreleased

- Reworked the README around the posterior target, a runnable workflow, and
  the public-API Eight Schools accuracy comparison. Added `THEORY.md` with
  assumptions, proofs, references, and the boundary between population
  properties and implementation evidence; include it in source distributions.
- Added the `StructuralModel`/`fit`/`infer` workflow for learning approximate
  joint posteriors from prior-predictive simulations.
- Added named posterior draws, summaries, transformations, batch inference,
  support diagnostics, and representation-level predictive checks.
- Added checksummed, pickle-free estimator and result artifacts with model and
  data-representation compatibility checks.
- Added NumPy-only CPU inference from trained neural estimators; Torch is
  required for training.
- Added public-data comparisons for Eight Schools and Rust's bus replacement
  model. Five public-API fits of the marginalized Eight Schools
  hyperposterior pass quadrature-based distributional checks. A Rust-specific
  simulation-trained grid classifier matches the dense posterior reference
  across five seeds, while the preserved generic diagonal-MDN run fails its
  frozen empirical accuracy gates.
- Replaced the earlier centered ten-dimensional Eight Schools smoke table: it
  was not an accuracy result and its “exact” entries were finite reference-draw
  means rather than deterministic quadrature moments.
- Added tests and CI for Python 3.11--3.13.

## 0.1.0-alpha

- Initial simulator, estimation, result, and command-line interfaces.
