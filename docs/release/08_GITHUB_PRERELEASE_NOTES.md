# `structnpe` 0.1.0b1 public beta

`structnpe` trains an amortized estimator on data simulated from a structural
model and returns named, aligned draws from an approximate joint posterior for
new observations. This first public beta is intended for pilot use and
model-specific validation, not unqualified production or empirical use.

## Included

- a public `StructuralModel -> fit -> infer` workflow with named posterior
  draws, conventional summaries, transformations, batch inference, and
  posterior predictive checks;
- a NumPy-only CPU path for loading and sampling saved neural estimators;
- versioned, checksummed, pickle-free estimator and result artifacts;
- executable installed-wheel examples for quick inference, full Bayesian use,
  a custom simulator, repeated inference, and exact-grid validation;
- an eleven-stage reproducibility runner covering package/distribution tests,
  exact comparisons, tiny SBC, support warnings, persistence, CPU training,
  and component timing;
- analytic conjugate-normal and structural exact-grid smoke comparisons;
- an Eight Schools quadrature comparison;
- an independent Rust bus-replacement NFXP comparator, dense Bayesian grid,
  and a bounded Iskhakov-style Monte Carlo package;
- Python 3.11--3.13 CI and a manually authorized GitHub prerelease workflow.

## Validation results to read before use

The fixed synthetic structural smoke passed all twelve declared checks, but
its nearest-grid joint total variation was `0.2521`. Policy and predictive
functionals were closer than the full joint distribution.

The completed Rust exercise is a negative empirical NPE result. The pinned
data preparation, NFXP estimate, dense-grid posterior, 200-case
prior-predictive calibration, and policy-functional gates passed. The
20,000-simulation MDN nevertheless failed the frozen empirical parameter-mean,
marginal-CDF, and joint-distribution gates. It is not presented as a successful
empirical posterior replication.

The Eight Schools and Iskhakov-style results are bounded smoke exercises. The
larger declared profiles are unrun.

## Important limits

- Accuracy is conditional on the simulator, prior, observation representation,
  training distribution, and fitted estimator.
- The beta does not prove structural identification or correct specification.
- The primary posterior family has five diagonal-Gaussian mixture components
  and can miss dependence, tails, boundaries, or modes.
- The 32-case SBC result is a pipeline smoke, not a calibration campaign.
- Training is an up-front amortized cost; the package claims no universal
  one-off speed advantage.
- MPEC and MAP estimation are not implemented.

See the [validation record](02_EXACT_POSTERIOR_VALIDATION.md),
[known limitations](05_KNOWN_LIMITATIONS.md), and
[replication guide](../../replication/README.md) for the full scope.

## Installation

Install the wheel for saved-estimator CPU inference:

```bash
python -m pip install structnpe-0.1.0b1-py3-none-any.whl
```

Install the optional neural dependency before fitting:

```bash
python -m pip install "structnpe[neural] @ file:///path/to/structnpe-0.1.0b1-py3-none-any.whl"
```

This GitHub prerelease does not publish to PyPI.
