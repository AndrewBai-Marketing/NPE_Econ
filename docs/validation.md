# Validation

Validation assesses a fitted approximation under a named simulator, prior,
representation, and configuration. It does not validate the simulator as a
description of the real world.

```python
report = estimator.validate(
    model=model,
    simulations=1_000,
    draws=1_000,
    seed=2468,
)
```

The validation contract includes:

- per-parameter SBC ranks and rank histograms;
- bias and RMSE;
- empirical central-interval coverage at 50%, 80%, 90%, and 95%;
- coarse prior-support regions where sample sizes permit;
- posterior predictive diagnostics for declared representation statistics;
- the stored distribution-support diagnostic.

Generic SBC output stores ten-bin counts for every parameter rather than only
the mean rank. When a truth exactly equals one or more posterior draws, its
discrete rank is randomized uniformly over the admissible tied positions using
the validation seed; the tie rule is recorded in result metadata.

CI should use a tiny bounded smoke configuration. A fuller manually invoked
configuration is required for substantive reporting. Smoke success is not a
scientific validation result.

Where an exact or enumerated posterior is available, compare posterior means,
standard deviations, credible intervals, marginal distributions, covariance
or correlation, and a predictive quantity. Freeze thresholds in configuration
before the final run and emit them with machine-readable metrics. Do not tune a
threshold after inspecting the result.

The final `0.1.0b1` synthetic smoke runs are recorded in the
[exact-posterior validation record](release/02_EXACT_POSTERIOR_VALIDATION.md).
Both the analytic conjugate-normal comparison and the fixed-panel structural
exact-grid comparison passed their predeclared smoke thresholds. The larger
`full` profiles were not run, and the smoke results retain the scope and
small-sample qualifications stated in that record.

The canonical Rust extension is a deliberately visible counterexample to
overgeneralizing those passes. Its NFXP and dense-grid comparators succeeded,
and its 200-case prior-predictive calibration and policy gates passed, but the
full empirical MDN posterior failed the frozen parameter-mean, marginal-CDF,
and joint-distribution gates. The release therefore does not claim a
successful empirical Rust posterior replication. See
[`replication/rust_1987/README.md`](../replication/rust_1987/README.md).

Interpret failures directly. Poor SBC may reflect inadequate simulations,
optimization, or posterior family. A predictive failure may reflect either the
approximation or model specification. A support warning means the observation
is unlike the stored training representation; it is not a formal rejection
test.
