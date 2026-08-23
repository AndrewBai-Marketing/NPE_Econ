# structnpe

## What it estimates

`structnpe` estimates a conditional distribution over structural parameters,
not a single coefficient and not the likelihood itself. You provide a prior
\(\pi\), a simulator \(P_\theta\), and a deterministic data representation
\(S\), fitted on the training split and then frozen where applicable:

$$
\Theta\sim\pi,\qquad
Y\mid\Theta=\theta\sim P_\theta,\qquad
X=S(Y).
$$

For observed data \(y_{\mathrm{obs}}\), when the relevant densities exist, the
target is

$$
p(\theta\mid S(Y)=s_{\mathrm{obs}})
=
\frac{p_S(s_{\mathrm{obs}}\mid\theta)\,\pi(\theta)}
     {\int p_S(s_{\mathrm{obs}}\mid\vartheta)
            \pi(\vartheta)\,d\vartheta},
\qquad s_{\mathrm{obs}}=S(y_{\mathrm{obs}}),
$$

where \(p_S(\cdot\mid\theta)\) is the distribution of the simulator output
after applying \(S\). The package never needs to evaluate that likelihood.
Instead, it draws prior-predictive training pairs

$$
\theta_i\sim\pi,\qquad y_i\sim P_{\theta_i}.
$$

With the declared coordinatewise parameter transform \(T\) and fitted
standardization constants,

$$
u=D_x^{-1}\!\left(S(y)-a_x\right),
\qquad
z=D_\theta^{-1}\!\left(T(\theta)-a_\theta\right).
$$

The estimator fits the neural conditional density

$$
q_\phi(z\mid u)
=
\sum_{k=1}^{K}\omega_k(u)\,
\mathcal N\!\left(z;\mu_k(u),
\operatorname{diag}\{\exp(\ell_k(u))\}\right),
$$

where \(u\) is the standardized \(S(y)\), \(z\) is the transformed and
standardized \(\theta\), the log variances \(\ell_k\) are clipped to
\([-7,5]\), and the current default is \(K=5\). Training approximately minimizes

$$
\widehat{\mathcal L}_n(\phi)
=-\frac{1}{n}\sum_{i=1}^{n}\log q_\phi(z_i\mid u_i).
$$

For fixed preprocessing, define the population criterion

$$
\mathcal L(\phi):=\mathbb E[-\log q_\phi(Z\mid U)].
$$

When the required expectations exist, it decomposes as

$$
\mathcal L(\phi)
=H(Z\mid U)
+\mathbb E_U\!\left[
D_{\mathrm{KL}}\!\left(
p(\,\cdot\mid U)\,\Vert\,q_\phi(\,\cdot\mid U)
\right)\right].
$$

Thus the ideal population solution is the forward-KL projection of the
representation posterior onto the chosen neural mixture family. In practice,
finite simulations, a finite network, and numerical training make it an
approximation. Inference samples and maps back to the declared parameter units:

$$
z^{(r)}\sim q_{\widehat\phi}(\,\cdot\mid u_{\mathrm{obs}}),
\qquad
\theta^{(r)}=T^{-1}\!\left(a_\theta+D_\theta z^{(r)}\right).
$$

The returned rows are aligned draws from the induced approximate joint
posterior. The representation target
\(p(\theta\mid S(Y)=s_{\mathrm{obs}})\) equals the full-data posterior
\(p(\theta\mid y_{\mathrm{obs}})\) only when \(S\) is sufficient (or otherwise
information-preserving) under the maintained model; neural approximation error
can remain even then.

Consequently, for any user-defined scalar or vector functional \(h\), the same
joint draws estimate posterior objects such as

$$
\mathbb E[h(\Theta)\mid S(Y)=s_{\mathrm{obs}}]
\approx \frac{1}{R}\sum_{r=1}^{R}h(\theta^{(r)}),
$$

including parameter means when \(h(\theta)=\theta\), probabilities when \(h\)
is an indicator, and model-defined profit, welfare, elasticity, or policy
effects when those functions are supplied by the user.

## Use a trained estimator

Loading and inference run on CPU and do not require Torch:

```bash
python -m pip install .
```

```python
import numpy as np
from structnpe import load_estimator

estimator = load_estimator("examples/assets/demo_estimator")
observed = np.array([32, 1, 27, 3, 18, 7, 12, 4, 6, 10], dtype=float)

posterior = estimator.infer(observed, draws=10_000, seed=123)
print(posterior.summary())
draws = posterior.to_dataframe()
```

The bundled estimator is a small synthetic replacement-model demonstration.
It is not trained on the Rust bus data and cannot be used with arbitrary
datasets. Run the example with:

```bash
python examples/quickstart.py
```

## Train on your model

Training requires Torch:

```bash
python -m pip install ".[neural]"
```

```python
from structnpe import ParameterSpec, StructuralModel, SummaryAdapter, fit

model = StructuralModel(
    prior=my_prior,
    simulator=my_simulator,
    parameters=[
        ParameterSpec("preference"),
        ParameterSpec("cost", lower=0),
    ],
    observation_adapter=SummaryAdapter(statistics=("mean", "std")),
    prior_id="my_model.prior.v1",
    simulator_id="my_model.simulator.v1",
)

estimator = fit(model, simulations=100_000, seed=1)
posterior = estimator.infer(observed_data, draws=20_000, seed=2)
```

The prior and simulator may be ordinary Python callables. The complete
[custom-model example](examples/custom_model.py) includes a batched simulator,
parameter bounds, saving, reloading, and fixed-seed inference.

## What you can estimate

From the joint posterior draws you can calculate:

- posterior means, medians, standard deviations, and credible intervals;
- posterior probabilities, covariance, and correlation;
- any user-written transformation of the parameters; and
- separate posteriors for many datasets generated under the same model,
  prior, and representation.

Elasticities, willingness to pay, profit, welfare, and policy effects are
available only when your structural model defines those quantities and you
apply the corresponding function to the joint draws. They are not automatic
outputs. `predictive_check()` provides diagnostics for the stored data
representation when the executable simulator is attached.

## Public-data comparisons

Accuracy comes before speed. The Rust example separates the conventional
maximum-likelihood estimate from the Bayesian posterior means, because they
are different estimands:

| Rust method and estimand | Replacement cost | Maintenance slope |
| --- | ---: | ---: |
| Public NFXP maximum likelihood | 10.0749 | 2.2931 |
| This repository's NFXP maximum likelihood | 10.0749 | 2.2931 |
| Dense-grid posterior mean | 10.6095 | 2.5108 |
| Simulation-trained grid posterior mean (seed 1701) | 10.5832 | 2.4972 |

The final row uses the same 37-bus panel and prior as the dense reference. All
five fixed simulation seeds passed the frozen posterior and policy
comparison limits; the worst marginal-CDF error was 0.08645 (limit 0.10) and
the worst joint total-variation error was 0.14653 (limit 0.15). This is a
Rust-specific finite-grid classifier, not the generic `structnpe.fit` MDN.

The generic API is exercised directly on the Eight Schools
hyperparameter posterior. Integrating out the latent school effects gives the
exact simulator

$$
y_j\mid\mu,\tau
\sim \mathcal N\!\left(\mu,\tau^2+\sigma_j^2\right),
$$

so `structnpe.fit` targets an approximation
$q_\phi(\mu,\tau\mid y)\approx p(\mu,\tau\mid y)$, not an unnecessarily
enlarged ten-dimensional parameterization.

| Eight Schools result | Population mean `mu` | Heterogeneity `tau` |
| --- | ---: | ---: |
| Deterministic quadrature posterior mean | 6.4720 | 4.7531 |
| `structnpe.fit`, average posterior mean over five seeds | 6.4178 | 4.7042 |
| Range of the five fitted posterior means | [6.0483, 6.9624] | [4.5063, 5.0521] |

The last two rows summarize variation across independent training runs; the
average is not an ensemble posterior and the range is not a credible interval.
The quadrature posterior standard deviations are `(4.1912, 3.6838)`; the
five-seed averages from `structnpe.fit` are `(4.1787, 3.7718)`. Every fixed
seed passes the numerical gates. Across the five runs, the worst marginal-CDF
error is `0.04997`, the worst coarsened joint total-variation error is
`0.06350`, and the worst posterior-mean error is `0.11701` exact posterior
standard deviations.

The generic diagonal-MDN Rust run remains a documented failure. It is not
used as evidence for generic accuracy.

See [BENCHMARKS.md](BENCHMARKS.md) for models, priors, diagnostics, commands,
and machine-readable results. These examples do not establish uniformly
accurate inference across structural models.

## When it is useful

`structnpe` is aimed at applications where:

- the model can generate realistic simulated datasets;
- evaluating or differentiating the likelihood is difficult;
- Bayesian uncertainty over a joint parameter vector matters; and
- the trained estimator will be reused across many comparable datasets.

It is usually a poor choice for a one-off problem with a cheap, reliable
likelihood or exact posterior. It does not establish model identification,
correct specification, or causal interpretation. Every application needs a
model-specific comparison or calibration study before substantive use.

## Project status

This is an MIT-licensed public beta for Python 3.11--3.13. The current neural
posterior is a configurable finite mixture of diagonal Gaussians (five
components by default); it can miss tails, boundaries, dependence, or
multimodality. Saved estimators are tied to the prior, simulator, parameter
ordering, and data representation used in training.

- [Benchmark evidence](BENCHMARKS.md)
- [Complete custom-model example](examples/custom_model.py)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Citation metadata](CITATION.cff)
