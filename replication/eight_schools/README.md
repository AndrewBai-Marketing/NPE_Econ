# Eight Schools Bayesian validation

This replication uses the canonical reported effects
`[28, 8, -3, 7, -1, 1, 18, 12]` and standard errors
`[15, 10, 16, 11, 9, 11, 10, 18]`. The maintained model is

```text
mu ~ Normal(0, 10)
tau ~ HalfNormal(10)
theta_j | mu, tau ~ Normal(mu, tau)
y_j | theta_j ~ Normal(theta_j, sigma_j)
```

The reference integrates `tau` on a dense log grid and integrates `mu`
analytically conditional on `tau`; joint draws of `mu`, `tau`, and all eight
school effects then use their conditional Gaussian laws. No MCMC convergence
claim is needed for this comparator.

Run the bounded smoke profile with the neural extra installed:

```bash
python replication/eight_schools/run_validation.py --profile smoke --quiet
```

The output includes marginal summaries, joint correlation error, uncertainty
over `tau`, school-level partial-pooling intervals, predictive-mean agreement,
and an optional forest plot when Matplotlib is installed. The five-component
diagonal-Gaussian MDN is intentionally not guaranteed to fit the hierarchical
funnel well; a failure is evidence about this beta estimator, not something to
tune away silently.

The larger profile is versioned but explicitly `UNRUN_FUTURE_WORK` for
0.1.0b1. This benchmark validates one stated Bayesian model only. It does not
establish structural identification or general posterior fidelity.

Data/source context: the values are the standard Eight Schools example as
documented by [TensorFlow Probability](https://www.tensorflow.org/probability/examples/Eight_Schools).

