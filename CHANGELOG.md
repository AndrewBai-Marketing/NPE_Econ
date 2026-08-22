# Changelog

## 0.1.0b1 - Unreleased

- Prepared a standalone source-only public-beta repository with MIT licensing,
  citation metadata, contribution/security guidance, issue templates, and
  least-privilege GitHub Actions for Python 3.11--3.13.
- Added installed-wheel quickstart, full-posterior, custom-simulator, batch,
  and exact structural-grid examples plus checked snippets generated from the
  executable scripts.
- Added a small checksummed, pickle-free demonstration estimator trained only
  on synthetic replacement-model panels; base-wheel quickstart inference does
  not import Torch.
- Added a frozen 11-stage beta-evidence runner covering tests, wheel/sdist
  installation, exact comparisons, tiny SBC, support warnings, persistence,
  CPU neural execution, and the component benchmark. The larger profile is
  explicitly unrun future work.
- Added an Eight Schools smoke comparison against quadrature and conditional
  Gaussian calculations, retaining visible MDN approximation errors.
- Added a pinned public-data Rust bus-replacement preprocessing path, an
  independently implemented exact Bellman/NFXP comparator, a dense Bayesian
  grid reference, and guarded Iskhakov-style Monte Carlo infrastructure.
- Completed the frozen Rust 20,000-simulation MDN and 200-case calibration
  profiles. Calibration and policy-functional gates passed, but the empirical
  parameter-mean, marginal-CDF, and joint-posterior gates failed; the release
  therefore makes no successful empirical Rust NPE claim.
- Completed a three-panel, one-discount-factor Iskhakov-style smoke and retained
  its bias, coverage, and break-even outputs as descriptive only; the full
  six-factor by 250-panel campaign remains unrun.
- Added release-asset and sdist-metadata normalization tools. Public source
  archives use neutral ownership and conventional file modes; no PyPI
  publication path is configured.
- Added the high-level `StructuralModel`, `ParameterSpec`, observation-adapter,
  `fit`, `TrainedEstimator`, and `InferenceResult` public API.
- Added the primary five-component diagonal-Gaussian Torch MDN with seeded
  simulation/training, validation splitting, early stopping, best-weight
  restoration, CPU/device metadata, and compatible weight resume.
- Added NumPy-only CPU inference from saved neural weights, explicit batch
  semantics, named posterior summaries, covariance/correlation, posterior
  predictive checks, and visible distribution-support warnings.
- Added versioned, checksummed, pickle-free estimator and result artifacts plus
  model/adapter/parameter fingerprints and strict compatibility checks.
- Bounded artifact manifests, payload sizes, NPZ decompression, headers, and
  array dimensions; reject duplicate, traversal, symlink, special-file, and
  unsupported archive content before numeric loading.
- Allowlist built-in adapter reconstruction. Custom adapters now require a
  compatible caller-supplied model or an explicit trusted-code import opt-in.
- Require explicit identifiers and structured configuration for stateful
  prior/simulator callables, and fit stateful adapters on the complete training
  split rather than an implicit prefix.
- Retain ten-bin SBC rank histograms with randomized tie ranks, and harden
  predictive checks for fitted-adapter state, exact widths, and discrete ties.
- Added analytic conjugate-normal and exact-grid dynamic replacement validation
  workflows, SBC summaries, predeclared thresholds, and a component benchmark.
- Added standalone exact-toy, custom-simulator, and fresh-process save/reload
  examples, focused beta tests, release audits, and user documentation.
- Prepared the `structnpe` distribution metadata for a narrow public beta.
- Restricted Python package discovery to the `structnpe` namespace.
- Declared NumPy and pandas as runtime dependencies, retained Torch as an
  optional neural dependency, and added separate build, development, and
  documentation extras.
- Added a Python 3.11--3.13 CI matrix with distribution-content and clean-wheel
  import checks.

## 0.1.0-alpha

- Added research-alpha `structnpe` package layer.
- Added simulator and model-index simulator APIs.
- Added simulation-bank storage.
- Added finite-grid, model-index, Gaussian, and alpha MDN estimator interfaces.
- Added CLI commands for project init, simulation, training, inference,
  validation, counterfactuals, and report discovery.
- Added small private-data-free templates and examples.
- Added theorem-guided package documentation and validation warnings.
