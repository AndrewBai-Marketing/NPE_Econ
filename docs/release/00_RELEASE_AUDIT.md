# Release audit for `structnpe` 0.1.0b1

Audit date: 2026-08-22  
Scope: standalone public-beta source snapshot  
Distribution boundary: the `structnpe` namespace only

## Release boundary

This directory is a source-only public export. It contains the reviewed
package, tests, documentation, executable examples, benchmarks, synthetic
validation records, canonical replication scaffolding, and GitHub
configuration needed for the beta. It contains no other research project,
mechanism-identification material, private reference packet, empirical outcome,
Slurm output, local environment, or inherited repository history.

The source export was assembled rather than copied as a whole directory. The
setuptools package finder independently restricts wheels and source
distributions to `structnpe` and its public package data. The detailed source
and publication audit is in
[`07_GITHUB_PUBLICATION_AUDIT.md`](07_GITHUB_PUBLICATION_AUDIT.md).

## Supported public path

The primary beta workflow is:

```text
StructuralModel + ParameterSpec + observation adapter
  -> fit on seeded simulations
  -> TrainedEstimator
  -> infer one or many observations
  -> named aligned joint draws in InferenceResult
  -> summaries, transformations, validation, and predictive checks
```

Training uses the optional Torch dependency. A saved five-component
diagonal-Gaussian MDN can be loaded and sampled on CPU using only NumPy and
pandas. Estimator and result artifacts are versioned, checksummed,
resource-bounded JSON/NPZ directory formats; the loaders do not use pickle.

Legacy finite-grid, ridge-Gaussian, model-index, project-file, and CLI paths
remain for compatibility but are not the front-page workflow.

## Public evidence included

- analytic conjugate-normal posterior comparison;
- descriptive 32-case SBC pipeline smoke;
- exact 63-cell structural-grid posterior comparison with 12 declared checks;
- training-support warning and save/reload determinism checks;
- component-level CPU timing with simulation, training, loading, and inference
  reported separately;
- a tiny safe-format synthetic demonstration estimator;
- Eight Schools quadrature validation;
- a pinned, checksum-verified Rust bus-data preprocessing, conventional NFXP
  comparator, dense grid, and completed full MDN/calibration evidence record;
- a guarded Iskhakov-style Monte Carlo with a completed three-panel smoke and
  an explicitly unrun full campaign.

The complete bounded beta runner is documented in
[`replication/README.md`](../../replication/README.md). Larger validation
profiles are frozen but unrun and are labeled `UNRUN_FUTURE_WORK`.

## Evidence that remains limited

- The tiny SBC has too few cases to establish calibration.
- The structural-grid result is one fixed synthetic panel. Its policy and
  predictive checks were close, while nearest-grid joint total variation was
  0.2521.
- The Eight Schools smoke exposes nontrivial approximation error for the
  declared MDN rather than tuning it away.
- The full empirical Rust MDN posterior failed frozen parameter-mean,
  marginal-CDF, and joint-distribution gates even though conventional,
  calibration, and policy-functional checks passed; no successful empirical
  NPE replication is claimed.
- The Iskhakov-style run has only three panels at one discount factor and
  supplies no substantive bias, coverage, or break-even claim.
- Timing is machine-dependent and supplies no universal speed claim.

## Explicit nonclaims

The beta does not establish structural identification, simulator correctness,
general posterior fidelity, frequentist coverage, universal amortization, or
superiority over conventional methods. It does not supply automatic model
choice, MAP estimates, arbitrary code-safe reconstruction, or automatic PyPI
publication.

## Publication state

The clean source export may be initialized as a new repository. No historical
repository is imported merely to create commit count. A tag, remote push,
GitHub release, or package publication requires separate authorization and is
not performed by the local preparation task.
