# Public-data benchmarks

This page compares `structnpe` with independent posterior references on the
two public examples maintained in this repository. The comparisons evaluate
posterior accuracy, not just whether the software runs.

## Results at a glance

| Example | Parameter | Reference posterior mean | `structnpe` posterior mean | Difference (`structnpe` - reference) |
| --- | --- | ---: | ---: | ---: |
| Eight Schools | population mean `mu` | 6.5031 | 6.1148 | -0.3883 |
| Eight Schools | heterogeneity `tau` | 4.6855 | 6.0892 | +1.4037 |
| Rust bus replacement | replacement cost | 10.6095 | 12.8636 | +2.2541 |
| Rust bus replacement | maintenance slope | 2.5108 | 3.2528 | +0.7420 |

The Eight Schools smoke passed its predeclared bounded thresholds. The Rust
empirical neural posterior did not pass its predeclared marginal and joint
accuracy thresholds. These are mixed results, not evidence of uniform
accuracy.

## Eight Schools

The data are the canonical eight treatment-effect estimates

```text
y     = [28, 8, -3, 7, -1, 1, 18, 12]
sigma = [15, 10, 16, 11, 9, 11, 10, 18].
```

The benchmark estimates the joint posterior of the population mean `mu`, the
between-school standard deviation `tau`, and the eight school effects under:

```text
mu ~ Normal(0, 10)
tau ~ HalfNormal(10)
theta_j | mu, tau ~ Normal(mu, tau)
y_j | theta_j ~ Normal(theta_j, sigma_j).
```

The reference integrates `tau` by dense one-dimensional quadrature and uses
the conditional Gaussian distribution for `mu` and the school effects.

| Diagnostic | Result |
| --- | ---: |
| Standardized parameter-mean MAE | 0.187 |
| Standardized posterior-SD MAE | 0.147 |
| Maximum absolute correlation error | 0.366 |
| Absolute error in posterior mean of `tau` | 1.404 |

From a GitHub repository checkout, run the bounded comparison with:

```bash
python -m pip install ".[replication]"
python replication/eight_schools/run_validation.py --profile smoke --quiet
```

The full configuration has not been run. This result covers one hierarchical
model and one observed dataset. Code and the compact result are in the
[`replication/eight_schools` directory](https://github.com/AndrewBai-Marketing/NPE_Econ/tree/main/replication/eight_schools).

## Rust bus replacement

The benchmark uses the public group-4 bus panel associated with Rust's engine
replacement model: 37 buses, 117 periods per bus, and 4,292 choice
observations. The data downloader is pinned to an immutable source commit and
SHA-256 checksum; raw bus histories are not redistributed here.

The estimated parameters are replacement cost and the mileage-dependent
maintenance-cost slope. The transition probabilities and discount factor are
fixed at the declared benchmark values. The Bayesian comparison uses
independent uniform priors:

```text
replacement cost ~ Uniform(4, 18)
maintenance slope ~ Uniform(0.2, 6).
```

First, the independently implemented conventional estimator reproduces the
public NFXP result:

| Parameter | Public NFXP result | This repository |
| --- | ---: | ---: |
| Replacement cost | 10.07494 | 10.07494 |
| Maintenance slope | 2.29309 | 2.29309 |

That agreement validates the benchmark implementation. It is not a result
from `structnpe`.

The Bayesian reference is a completed `141 x 117` dense posterior grid. The
20,000-simulation neural estimator was then evaluated on the same empirical
panel:

| Diagnostic | `structnpe` result | Frozen limit | Status |
| --- | ---: | ---: | --- |
| Maximum marginal CDF discrepancy | 0.473 | 0.100 | fail |
| Coarsened joint total variation | 0.514 | 0.150 | fail |
| Maximum mean error in reference SDs | 1.565 | 0.250 | fail |
| Policy-mean maximum absolute error | 0.0172 | 0.0300 | pass |

The neural posterior therefore failed even though its prior-predictive
calibration and policy-mean gates passed. Average calibration over simulated
panels did not guarantee an accurate approximation for this empirical panel.

From a GitHub repository checkout, reproduce the data and conventional
references with:

```bash
python -m replication.rust_1987.download_data
python -m replication.rust_1987.preprocess
python -m replication.rust_1987.nfxp \
  replication/rust_1987/data/processed/group4.csv
python -m replication.rust_1987.grid_posterior \
  replication/rust_1987/data/processed/group4.csv \
  --output replication/rust_1987/results/dense_grid_141x117.npz
```

The longer train, infer, compare, and calibration commands are documented in
the [`replication/rust_1987` directory](https://github.com/AndrewBai-Marketing/NPE_Econ/tree/main/replication/rust_1987).
The frozen result is
[`expected/smoke_metrics.json`](https://github.com/AndrewBai-Marketing/NPE_Econ/blob/main/replication/rust_1987/expected/smoke_metrics.json).

## Interpretation

The public evidence supports three narrow conclusions:

1. the Eight Schools result is within its declared bounded smoke thresholds;
2. the independent Rust model and conventional estimator reproduce their
   public reference; and
3. the current neural posterior is not accurate enough for the empirical Rust
   panel under the frozen validation criteria.

It does not establish identification, correct structural specification,
universal posterior accuracy, or a speed advantage over a tractable
likelihood. A new model needs its own reference comparison, simulation-based
calibration, or other predeclared validation study.
