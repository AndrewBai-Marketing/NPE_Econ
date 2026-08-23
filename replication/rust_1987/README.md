# Rust (1987) bus-engine replacement

This directory is an independently written, self-contained NumPy/SciPy
replication of the canonical group-4 bus-engine replacement calculation, plus
a current-API `structnpe` posterior experiment. It does not copy the legacy
`ddc_npe` implementation or third-party source code.

The conventional NFXP, dense-grid reference, and a model-specific structured
classifier pass their numerical comparisons. The generic current-API MDN's
full simulator calibration also passes, but its empirical posterior fails the
frozen marginal and joint agreement gates in
[`validation_config.json`](validation_config.json). It is therefore not a
headline-eligible empirical MDN replication. The two compact committed records
are [`expected/structured_classifier_metrics.json`](expected/structured_classifier_metrics.json)
and [`expected/smoke_metrics.json`](expected/smoke_metrics.json).

## Frozen model and data convention

- Group 4: GMC A5308, model year 1975 (`a530875.asc`).
- 37 buses, 117 periods per bus, 4,329 rectangular panel rows.
- Choice likelihood: periods 1--116 for each bus, or 4,292 observations.
- 90 mileage states and 5,000-mile bins.
- Discount factor `0.9999`.
- Linear maintenance cost `0.001 * maintenance_slope * state`.
- Replacement cost `RC` and type-I extreme-value choice shocks.
- Independent uniform prior: `RC ~ U(4,18)` and slope `~ U(0.2,6)`.

The downloader uses the immutable OpenSourceEconomics `zurcher-data` v0.14
commit `43d37bd1d24a0a6c6e73e747545ad490d1f7f95b`. The exact raw URL is embedded
in [`constants.py`](constants.py); the expected SHA-256 is
`5e85a1c33c11632effbec3ffb213c8e4c92501a49dfe388ad28a203f8c732387`.
Raw data and individual processed rows are deliberately excluded from version
control. The scripts never load pickle.

The default `historical_ruspy` preprocessing uses floor-discretized states but
codes the first post-replacement increment as one. This reproduces transition
counts `(1682, 2555, 55)` and probabilities
`(0.39189189, 0.59529357, 0.01281454)`. The deterministic processed CSV has
SHA-256 `7f9e169e5cb2f18204c342935cadab13df188af4275c45fe07dfe0f94840376b`.

This convention is intentionally explicit. Ferrall's audit explains that the
historical ceiling treatment makes the first post-replacement increment at
least one; the corrected group-4 counts are `(1715, 2522, 55)`. Select
`--convention state_difference` to generate that corrected first stage. The
canonical result below uses the historical convention because that is the
declared target, not because it is asserted to be the uniquely correct data
treatment.

Sources:

- [Rust (1987), *Econometrica*](https://www.jstor.org/stable/1911259)
- [Pinned OpenSourceEconomics data repository](https://github.com/OpenSourceEconomics/zurcher-data/tree/43d37bd1d24a0a6c6e73e747545ad490d1f7f95b)
- [Public ruspy group-4 replication documentation](https://ruspy.readthedocs.io/en/latest/tutorials/replication/replication.html)
- [Ferrall (2023) data-convention audit](https://ferrall.github.io/OODP/Rust1987.html)

## Reproduce the data and conventional estimate

From the repository root, with `structnpe` installed in the environment:

```bash
python -m replication.rust_1987.download_data
python -m replication.rust_1987.preprocess
python -m replication.rust_1987.nfxp \
  replication/rust_1987/data/processed/group4.csv \
  --output replication/rust_1987/results/nfxp.json
```

The independent normalized-value NFXP comparator returns:

| Quantity | Result | Public reference |
|---|---:|---:|
| Replacement cost | 10.07494221 | 10.07494 |
| Maintenance slope | 2.29309298 | 2.29309 |
| Negative choice log likelihood | 163.58428366 | -- |
| Bellman residual | `3.02e-14` | -- |

The measured warm run took 0.0215 seconds, 9 optimizer iterations, and 11
objective/score evaluations on the development CPU. This is not a portable
performance guarantee.

MPEC is not implemented. The result file reports that absence explicitly; no
NFXP-versus-MPEC claim is made.

## Dense Bayesian reference

The declared-prior reference is a `141 x 117` Cartesian grid (16,497 points)
with trapezoidal quadrature. It evaluates the same exact choice likelihood and
Bellman model as NFXP:

```bash
python -m replication.rust_1987.grid_posterior \
  replication/rust_1987/data/processed/group4.csv \
  --output replication/rust_1987/results/dense_grid_141x117.npz
```

The completed grid took 12.51 seconds on the development CPU. Its posterior
summary was:

| Parameter | Mean | SD | Median | 95% interval |
|---|---:|---:|---:|---:|
| Replacement cost | 10.6095 | 1.4404 | 10.5000 | [8.2000, 13.8000] |
| Maintenance slope | 2.5108 | 0.5862 | 2.4500 | [1.5000, 3.8000] |

This is a posterior under the stated uniform prior. It is not "Rust's
posterior," and it need not be centered exactly on the maximum-likelihood
estimate.

## Passing model-specific structured classifier

The empirical failure below is specific to the generic diagonal-MDN
approximation, not to simulation of the Rust model. A small model-specific
classifier recovers the same posterior target. It is included as a transparent
positive benchmark and is clearly separate from `structnpe.fit`.

Let `n_s` be the observed number of choices made in mileage state `s`, `r_s`
the replacements among them, and `p_s(theta_g)` the exact Rust policy at grid
class `theta_g = (RC_g, slope_g)`. The target conditional choice posterior is

```text
pi(theta_g | r,n) proportional to
    q_g product_s p_s(theta_g)^r_s [1-p_s(theta_g)]^(n_s-r_s),
```

where `q_g` is the trapezoidal prior mass. The structured classifier does not
insert the exact policy into this expression. For each class and state it
simulates `M=500` fixed-exposure panels and estimates the policy using
Jeffreys smoothing:

```text
K_gs ~ Binomial(M n_s, p_s(theta_g)),
p_hat_gs = (K_gs + 1/2) / (M n_s + 1).
```

This aggregate draw is exactly simulation-equivalent to materializing the
same conditional panels, because the sum of `M` independent
`Binomial(n_s,p_s(theta_g))` counts is `Binomial(M n_s,p_s(theta_g))`. It avoids
writing 435,000 redundant panel rows; it does not replace simulation with the
exact likelihood. Conditioning on the observed exposure histogram means this
is not a reusable estimator for arbitrary panels.

Reproduce the five-seed benchmark with:

```bash
python -m replication.rust_1987.structured_classifier \
  replication/rust_1987/data/processed/group4.csv \
  --output replication/rust_1987/results/structured_classifier_metrics.json
```

By default, the CLI reads the prior, dense-grid definition, and numerical
comparison limits from [`validation_config.json`](validation_config.json). It
calls that configuration frozen only when both its path and SHA-256 match the
committed default. It also authenticates the processed CSV using its SHA-256,
transition counts, and transition probabilities before calling the result the
canonical group-4 benchmark. A custom `--thresholds` path is recorded as
non-frozen even when its bytes are copied from the default; a different CSV is
reported as noncanonical. The CLI reports the `141 x 117` dense posterior, an
exact-likelihood `29 x 30` coarse posterior, and simulation-estimated
classifier posteriors for seeds 1701--1705. The committed path-sanitized result is
[`expected/structured_classifier_metrics.json`](expected/structured_classifier_metrics.json).
The CLI applies the public `compact_benchmark_result` transformation, so the
default command reproduces the committed numeric record to its declared
precision. Compact evidence floats are rounded to 14 decimal places after all
calculations; byte-for-byte or unrounded floating-point identity across
platforms is not claimed. Python callers that need the expanded per-parameter
interval records and individual gate booleans can use `run_benchmark` directly.

The exact coarse posterior mean was `(10.60949, 2.51076)`, compared with the
dense mean `(10.60952, 2.51077)`. Across the five simulation seeds, classifier
means ranged from `(10.56581, 2.48028)` to `(10.64081, 2.52850)` coordinate by
coordinate. The worst seeded diagnostics were:

| Diagnostic | Worst seed | Frozen limit |
|---|---:|---:|
| Maximum marginal CDF supremum | 0.08645 | 0.10000 |
| Joint 12-by-10 TV | 0.14653 | 0.15000 |
| Maximum mean error in dense-grid SDs | 0.05201 | 0.25000 |
| Policy-mean maximum absolute error | 0.00121 | 0.03000 |
| Policy-interval endpoint maximum error | 0.00202 | 0.06000 |
| Expected-replacement mean absolute error | 0.00211 | 0.50000 |
| Expected-replacement interval endpoint error | 0.04226 | 1.00000 |

All five seeds satisfy the frozen numerical comparison limits, including the
support check by construction. This is not a claim that the generic MDN
passes, and it does not inherit the MDN-specific full-profile promotion or
all-prior calibration result. Its narrow lesson is that a likelihood-aligned,
finite-grid classifier can estimate this particular strongly correlated Rust
posterior reliably.

## Generic MDN: train, infer, compare, and calibrate

The simulator is genuinely batched and returns state-visit and replacement
counts for each simulated panel. Training uses the public `StructuralModel`,
`ParameterSpec`, `ArrayAdapter`, and `fit` API:

```bash
python -m replication.rust_1987.train \
  --output replication/rust_1987/results/full_estimator

python -m replication.rust_1987.infer \
  replication/rust_1987/results/full_estimator \
  --data replication/rust_1987/data/processed/group4.csv

python -m replication.rust_1987.compare \
  replication/rust_1987/results/full_estimator \
  replication/rust_1987/results/dense_grid_141x117.npz \
  replication/rust_1987/data/processed/group4.csv \
  --output replication/rust_1987/results/full_comparison.json

python -m replication.rust_1987.calibrate \
  replication/rust_1987/results/full_estimator \
  --output replication/rust_1987/results/full_calibration.json
```

`infer.py --batch-simulations K` exercises loaded batched inference. The
comparison reports both marginal CDF discrepancies and a joint 12-by-10
coarsened-grid total-variation distance, plus posterior means, covariance,
policy means and interval endpoints, and expected 12-month replacements. All
acceptance thresholds were frozen before the first empirical MDN comparison.

### Preserved first smoke result: failed

The first post-freeze smoke used 500 training simulations, five epochs, and
2,000 empirical posterior draws. Simulation took 4.14 seconds, training
excluding simulation took 55.08 seconds, and loaded inference took 0.122
seconds. It was deliberately evaluated despite being ineligible for promotion.

It failed every approximation gate except the empirical support check:

| Diagnostic | Smoke value | Frozen limit |
|---|---:|---:|
| RC marginal CDF supremum | 0.427 | 0.100 |
| Slope marginal CDF supremum | 0.398 | 0.100 |
| Joint 12-by-10 TV | 0.878 | 0.150 |
| Maximum mean error in grid SDs | 1.124 | 0.250 |
| Policy-mean maximum absolute error | 0.0333 | 0.0300 |
| Policy-interval endpoint maximum error | 0.181 | 0.060 |

The grid expected 0.0484 replacements over 12 months for 37 initially new
buses, while the smoke NPE mean was 0.568. A separate 20-case calibration
smoke was too small for acceptance and also failed its bias, coverage, and rank
gates. These results are retained in the path-sanitized
[`expected/smoke_metrics.json`](expected/smoke_metrics.json), without an
estimator or row-level data. They show that the runnable interface works;
they do **not** show that the empirical NPE approximation works. The 20,000
simulation/full-calibration profile was subsequently run and is reported next.

### Frozen full-profile result: calibration passed, empirical posterior failed

The unchanged full profile used 20,000 simulations and requested 100 epochs;
early stopping selected epoch 17. Simulation took 23.27 seconds and training
excluding simulation took 67.90 seconds with single-threaded CPU BLAS/Torch
execution. The empirical comparison used 20,000 posterior draws and 2,000
policy draws. It was promotion-eligible but failed overall:

| Diagnostic | Full value | Frozen limit | Result |
|---|---:|---:|---|
| Maximum marginal CDF supremum | 0.473 | 0.100 | fail |
| Joint 12-by-10 TV | 0.514 | 0.150 | fail |
| Maximum mean error in grid SDs | 1.565 | 0.250 | fail |
| Policy-mean maximum absolute error | 0.0172 | 0.0300 | pass |
| Policy-interval endpoint maximum error | 0.0427 | 0.0600 | pass |
| Expected-replacement mean absolute error | 0.0330 | 0.5000 | pass |

The NPE empirical posterior mean was `(12.864, 3.253)`, compared with the grid
mean `(10.610, 2.511)`. Thus good policy-functional agreement did not rescue
poor marginal and joint posterior agreement.

The separate 200-case, 1,000-draw calibration profile passed all frozen
all-prior gates: both 90% coverages were `0.91`; biases were `-0.249` and
`-0.0689`; rank-histogram L1 distances were `0.19` and `0.12`. This coexistence
is informative: average prior-predictive calibration does not guarantee a good
approximation at this particular empirical panel. Because every promotion gate
was required, the package still makes **no successful empirical NPE replication
claim**.

## Scope and nonclaims

- A conventional MLE is not a Bayesian posterior.
- A dense finite grid is a numerical posterior reference, not an analytic one.
- Passing one canonical model would not establish universal NPE accuracy,
  identification, or correct model specification.
- Policy bands condition on the model, prior, transition convention, and MDN.
- Training cost must be included in amortization. If NPE per-panel inference is
  not faster than NFXP, the break-even is reported as nonexistent.
- The scripts materialize simulation summaries in memory.
- Raw and row-level processed data must be regenerated from the pinned source.
