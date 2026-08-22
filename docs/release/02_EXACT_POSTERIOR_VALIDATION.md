# Exact-posterior validation record

Status: **both final smoke profiles passed**  
Run date: 2026-08-21  
Release target: `0.1.0b1`

This record reports the final synthetic smoke runs against an analytic
conjugate posterior and an exactly enumerated finite-state structural
posterior. Every acceptance metric was compared with a threshold declared
before the final run. The machine-readable results and short generated reports
are the authoritative records:

- [conjugate-normal metrics](../../validation/exact_example/output/validation_metrics.json),
  [report](../../validation/exact_example/output/report.md), and
  [thresholds](../../validation/exact_example/thresholds.json);
- [structural metrics](../../validation/structural_example/output/validation_metrics.json),
  [report](../../validation/structural_example/output/report.md), and
  [thresholds](../../validation/structural_example/thresholds.json).

## Profiles and commands

The recorded results are the `smoke` rows only. The `full` rows are frozen,
larger manual configurations; they were **not run for this record**, and no
full-profile pass is claimed.

| benchmark/profile | status | simulations | evaluation | posterior draws | validation split | network | epochs | batch / patience |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| conjugate smoke | **PASS** | 2,500 | 32 prior-predictive cases | 1,500 per case | 0.20 | hidden 32, depth 2, 5 components | 35 | 128 / 7 |
| conjugate full | not run | 20,000 | 200 prior-predictive cases | 5,000 per case | 0.15 | hidden 64, depth 2, 5 components | 100 | 256 / 15 |
| structural smoke | **PASS** | 4,000 | one fixed synthetic panel | 3,000 | 0.20 | hidden 48, depth 2, 5 components | 50 | 128 / 8 |
| structural full | not run | 30,000 | one fixed synthetic panel | 10,000 | 0.15 | hidden 64, depth 3, 5 components | 120 | 256 / 18 |

From the repository root, the recorded smoke workflows are:

```bash
python validation/exact_example/run_validation.py --profile smoke
python validation/structural_example/run_validation.py --profile smoke
```

The larger profiles should use separate output directories so they do not
replace the recorded smoke artifacts:

```bash
python validation/exact_example/run_validation.py \
  --profile full --output-dir validation/exact_example/output_full
python validation/structural_example/run_validation.py \
  --profile full --output-dir validation/structural_example/output_full
```

The exact smoke seeds were 41001 (training), 41002 (evaluation), and 41003
(inference). The structural smoke seeds were 51001 (training), 51002
(observed panel), and 51003 (inference). Both runs used CPU training and
inference on Python 3.13.11 and NumPy 2.4.2. The runners recorded the package
environment as `source-checkout`; the saved estimators' training metadata
records package version `0.1.0b1`.

Recorded evidence identities:

The hashes below identify the public, path-sanitized JSON copies. Sanitizing
the checkpoint-location metadata did not change any configuration, draw,
metric, check, or estimator payload.

| benchmark | validation-metrics SHA-256 | estimator model fingerprint |
| --- | --- | --- |
| conjugate-normal smoke | `0200b62aa36e065e74ec8bf676f596ac8bd603f2d7e2cdb652f95bd210156f95` | `75d88578087c66f82587ba5c1753116f19a1957dab6bfa8986565b11dbc23497` |
| structural smoke | `9a22c97e3ad6ae4b516bf0b010e473006d286d6ffc4840c6c8073c7f46659704` | `c1371104ce7303d2a37688c3a9a98987f5f8f6cf059a897fe9dd680b2663e75d` |

## Analytic conjugate-normal comparison

The benchmark uses `theta ~ Normal(0, 1)` and 20 conditionally independent
`Normal(theta, 1)` observations. The sample mean is sufficient, and the exact
posterior is analytic. All seven predeclared checks passed:

| acceptance metric | observed | maximum | result |
| --- | ---: | ---: | --- |
| posterior mean MAE | 0.0114774 | 0.100 | pass |
| posterior SD MAE | 0.00691708 | 0.070 | pass |
| posterior variance MAE | 0.00307791 | 0.040 | pass |
| central 95% endpoint MAE | 0.0227796 | 0.150 | pass |
| marginal quantile-grid L1 mean | 0.0141603 | 0.100 | pass |
| absolute error of 95% coverage | 0.0187500 | 0.200 | pass |
| predictive SD MAE | 0.00487845 | 0.080 | pass |

The approximate posterior's aggregate 95% coverage was 0.96875. Its average
posterior mean, SD, and variance were -0.03673, 0.22491, and 0.05060; the exact
averages were -0.03203, 0.21822, and 0.04762. The approximate and exact average
predictive SDs were 0.31716 and 0.31244.

Threshold SHA-256:
`676aeab7d484fc09c40cba5dc0eb433558afd50284798c36233366b9103d6349`.

### Tiny SBC summary

The same 32 prior-predictive cases provide a bounded SBC-style diagnostic.
These coverage rates and rank summaries are descriptive; only 10 or 11 cases
fall in each coarse region.

| region | n | bias | RMSE | cov. 50 | cov. 80 | cov. 90 | cov. 95 | mean rank fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 32 | -0.02003 | 0.22578 | 0.4688 | 0.8125 | 0.8438 | 0.9688 | 0.5316 |
| lower | 11 | -0.00011 | 0.15788 | 0.5455 | 1.0000 | 1.0000 | 1.0000 | 0.4996 |
| middle | 10 | 0.01383 | 0.19663 | 0.5000 | 0.9000 | 0.9000 | 1.0000 | 0.4940 |
| upper | 11 | -0.07072 | 0.29703 | 0.3636 | 0.5455 | 0.6364 | 0.9091 | 0.5978 |

At this sample size, one case moves overall coverage by 3.125 percentage
points and a regional rate by roughly 9--10 points. The exact analytic
posterior itself covered 29 of 32 truths at 95% (0.90625), illustrating why
the regional differences and rank histogram should not be interpreted as
stable calibration estimates.

## Exact-grid structural comparison

The structural benchmark is a self-contained dynamic replacement model with
five states, keep/replace actions, discount factor 0.9, 12 agents, and 10
periods. Its prior is uniform on a 63-cell grid. The one predeclared observed
panel was generated at the midpoint parameter `(maintenance_cost,
replacement_cost) = (0.4, 3.5)`. State-action counts are sufficient under the
benchmark's fixed initial states and deterministic action-conditional
transitions, so the grid posterior is exactly enumerable.

All 12 predeclared checks passed:

| acceptance metric | observed | maximum | result |
| --- | ---: | ---: | --- |
| parameter mean normalized MAE | 0.0294578 | 0.100 | pass |
| parameter SD normalized MAE | 0.0125370 | 0.100 | pass |
| central 95% endpoint normalized MAE | 0.0273073 | 0.150 | pass |
| covariance normalized maximum absolute error | 0.00586685 | 0.050 | pass |
| correlation maximum absolute error | 0.0436754 | 0.250 | pass |
| marginal quantile-grid L1 normalized mean | 0.0446904 | 0.120 | pass |
| joint-grid Voronoi total variation | 0.252070 | 0.350 | pass |
| policy mean absolute error | 0.00102319 | 0.060 | pass |
| policy SD absolute error | 0.00202850 | 0.050 | pass |
| policy 95% endpoint MAE | 0.00451966 | 0.080 | pass |
| predictive cell-probability maximum absolute error | 0.00229757 | 0.060 | pass |
| parameter-box support violation rate | 0 | 0 | pass |

The joint-law check maps continuous approximate draws to their nearest cell on
the Cartesian parameter grid, then compares those 63 projected masses with
the exact grid posterior. Thus, 0.252070 is total variation for that declared
Voronoi projection; it is not total variation between an unprojected
continuous law and a discrete law.

The policy quantity is the expected replacement share after a 0.5 replacement
subsidy and a fresh solution of the dynamic problem:

| policy summary | exact | approximate |
| --- | ---: | ---: |
| posterior mean | 0.230300 | 0.229277 |
| posterior SD | 0.0210925 | 0.0231210 |
| 2.5% endpoint | 0.191986 | 0.192095 |
| 97.5% endpoint | 0.278113 | 0.287044 |

The posterior-predictive comparison uses expected state-action frequencies in
the declared ten cells. Its maximum absolute discrepancy was 0.00229757, at
state 4 under `keep` (exact 0.0601718; approximate 0.0578742).

Threshold SHA-256:
`96cf03a5f80f9d7849ca8842f6aed232367907c9b02d67a8c657191161b78e92`.

## Timings

| benchmark | training | posterior inference |
| --- | ---: | ---: |
| conjugate-normal smoke | 55.7327 s | 0.003759 s for 32 batched cases |
| structural smoke | 57.0521 s | 0.001055 s for one panel |

These are observed wall-clock timings from one CPU run. They
are included for reproducibility, not as a controlled performance benchmark or
a claim of universal speed.

## Scope and nonclaims

- A smoke pass shows that the fitted approximation met these predeclared
  tolerances for these synthetic configurations. It is not a full-profile
  result or substantive scientific validation.
- The conjugate comparison is conditional on its declared prior, simulator,
  sufficient representation, seeds, training budget, and posterior family. It
  does not establish structural identification or real-world specification.
- The structural comparison covers one predeclared midpoint-truth panel. It
  does not establish amortized accuracy across observations, transfer to other
  structural models, or identification.
- The policy calculation is a synthetic model functional, not an empirical
  managerial or welfare conclusion.
- Posterior-predictive agreement on selected state-action frequencies does not
  establish correct model specification.
- The SBC sample is deliberately tiny. Overall and especially region-specific
  coverage and ranks have substantial Monte Carlo uncertainty.
