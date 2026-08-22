# Quickstart

This example trains a small estimator for a scalar normal-location simulator.
The budget is for learning the workflow, not for establishing posterior
quality.

```python
import numpy as np

from structnpe import ParameterSpec, StructuralModel, fit, load_estimator


def prior(n, rng):
    return rng.uniform(-2.99, 2.99, size=(n, 1))


def simulator(theta, rng):
    return rng.normal(loc=float(theta[0]), scale=1.0, size=20)


model = StructuralModel(
    prior=prior,
    simulator=simulator,
    parameters=[
        ParameterSpec(
            "location",
            lower=-3.0,
            upper=3.0,
            description="Normal location parameter",
        )
    ],
    prior_id="uniform-location-v1",
    simulator_id="normal-location-v1",
)

estimator = fit(
    model,
    simulations=5_000,
    seed=1234,
    epochs=20,
    device="cpu",
    output_dir="runs/location_demo",
)

observed = np.array([
    0.72, 1.03, 0.41, 1.26, 0.58,
    0.95, 0.77, 1.14, 0.32, 0.88,
    1.19, 0.63, 0.54, 1.35, 0.69,
    0.91, 1.08, 0.47, 0.83, 1.22,
])
result = estimator.infer(observed, draws=2_000, seed=5678)

print(result.summary())
print(result.covariance())
print(result.diagnostics())

estimator.save("saved/location_estimator")
result.save("saved/location_result")

loaded = load_estimator("saved/location_estimator", model=model)
reloaded_result = loaded.infer(observed, draws=2_000, seed=5678)
```

`result.draws` is in the declared user-facing parameter units. The interval in
`summary()` is a Bayesian credible interval, and `posterior_sd` is posterior
dispersion rather than a frequentist standard error.

Before substantive use, run validation at a separately declared scale and
inspect distribution-support warnings. Successful training alone does not
establish identification or correct specification.
