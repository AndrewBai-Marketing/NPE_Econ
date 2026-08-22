# Parameter summaries

`InferenceResult.summary()` returns a conventional posterior table without
renaming Bayesian quantities as frequentist ones.

```python
table = result.summary(point="mean")
```

The columns are:

| column | meaning |
| --- | --- |
| `parameter` | declared human-readable name |
| `estimate` | selected posterior mean or median |
| `posterior_mean` | mean of returned draws |
| `posterior_sd` | standard deviation of returned posterior draws |
| `median` | posterior median |
| `q025`, `q975` | central 95% Bayesian credible interval |
| `ess` | Monte Carlo effective sample size |
| `warning` | parameter or result-level diagnostic |

`point` accepts `"mean"` and `"median"`. `point="MAP"` raises
`NotImplementedError`; the beta never labels a histogram bin or sample mode as
a MAP estimate.

MDN draws are generated independently conditional on the fitted approximation,
so their Monte Carlo ESS equals the draw count. This says nothing about whether
the approximating posterior is accurate.

```python
cov = result.covariance()
corr = result.correlation()
draw_frame = result.to_dataframe()
```

Covariance and correlation use constrained user-facing units. Constant or
numerically degenerate parameters receive warnings rather than fabricated
correlations. `to_dataframe()` returns one row per draw with columns in the
declared parameter order.

`posterior_sd` is not a standard error, and the reported interval is not a
confidence interval. Frequentist covariance, confidence sets, and bootstrap
inference are outside this beta.
