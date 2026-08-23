# structnpe

`structnpe` learns an approximate Bayesian posterior for the parameters of a
model that you can simulate.

You provide:

1. a prior for the parameter vector `theta`;
2. a simulator that generates data from `theta`; and
3. a fixed representation `S(data)` used by the estimator.

Training generates parameter--dataset pairs and fits

```text
q(theta | S(data))  ~=  p(theta | S(data)).
```

For observed data, the output is a set of aligned draws from this approximate
joint posterior. If `S(data)` is only a summary, the target is the posterior
conditional on that summary—not necessarily the full-data posterior.

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

Accuracy comes before speed. These are the checked-in results for the two
public examples in this repository:

| Example | Parameter | Reference posterior mean | `structnpe` posterior mean |
| --- | --- | ---: | ---: |
| Eight Schools | population mean `mu` | 6.5031 | 6.1148 |
| Eight Schools | heterogeneity `tau` | 4.6855 | 6.0892 |
| Rust bus replacement | replacement cost | 10.6095 | 12.8636 |
| Rust bus replacement | maintenance slope | 2.5108 | 3.2528 |

The Eight Schools bounded smoke comparison passed its declared thresholds,
with visible error in heterogeneity and joint dependence. The Rust neural
posterior failed its frozen marginal and joint accuracy gates. The independent
conventional Rust implementation does reproduce the public NFXP estimates,
but that validates the comparator—not the neural posterior.

See [BENCHMARKS.md](BENCHMARKS.md) for models, priors, diagnostics, commands,
and machine-readable results. The evidence is mixed; this beta does not claim
uniformly accurate inference across structural models.

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
posterior is a five-component diagonal-Gaussian mixture; it can miss tails,
boundaries, dependence, or multimodality. Saved estimators are tied to the
prior, simulator, parameter ordering, and data representation used in
training.

- [Benchmark evidence](BENCHMARKS.md)
- [Complete custom-model example](examples/custom_model.py)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Citation metadata](CITATION.cff)
