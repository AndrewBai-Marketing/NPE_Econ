# Eight Schools hyperposterior validation

This replication estimates the posterior of the population mean `mu` and
between-school heterogeneity `tau` for the canonical reported effects
`[28, 8, -3, 7, -1, 1, 18, 12]` and standard errors
`[15, 10, 16, 11, 9, 11, 10, 18]`.

The hierarchical model is

```math
\mu\sim\mathcal N(0,10^2),\qquad
\tau\sim\mathrm{HalfNormal}(10),
```

```math
\theta_j\mid\mu,\tau\sim\mathcal N(\mu,\tau^2),\qquad
y_j\mid\theta_j\sim\mathcal N(\theta_j,\sigma_j^2).
```

Because the public object is $`p(\mu,\tau\mid y)`$, the simulator integrates
out each latent school effect:

```math
y_j\mid\mu,\tau
\sim\mathcal N\!\left(\mu,\tau^2+\sigma_j^2\right).
```

This is an exact marginalization, not an approximation or a different prior.
It avoids asking a diagonal Gaussian mixture to learn eight nuisance latent
coordinates and the centered hierarchical funnel.

Run the complete five-seed comparison with:

```bash
python replication/eight_schools/run_validation.py --quiet
```

The script trains `structnpe.fit` under the fixed configuration in
[`config.json`](config.json), compares posterior means, standard deviations,
95% intervals, marginal CDFs, correlation, and a coarsened two-dimensional
total-variation diagnostic with dense quadrature, and exits nonzero unless
every seed passes every committed limit. Compact canonical evidence is
committed in [`expected_metrics.json`](expected_metrics.json).

Deterministic quadrature gives

```math
\mathbb E[(\mu,\tau)\mid y]=(6.4720,4.7531),\qquad
\mathrm{SD}[(\mu,\tau)\mid y]=(4.1912,3.6838).
```

All five fixed seeds pass. The worst marginal-CDF error is `0.04997`; the
worst coarsened joint total-variation error is `0.06350`; and the worst
posterior-mean error is `0.11701` exact posterior standard deviations. The
result validates this dataset, prior, estimand, and representation only. It is
not a guarantee for arbitrary hierarchical models.

Data/source context: these are the standard Eight Schools values documented
by [TensorFlow Probability](https://www.tensorflow.org/probability/examples/Eight_Schools).
