# API migration for 0.1.0b1

The beta adds an object-oriented façade without deleting the alpha project and
CLI workflow.

## Preferred beta path

```python
from structnpe import ParameterSpec, StructuralModel, fit

model = StructuralModel(
    prior=prior,
    simulator=simulator,
    parameters=[ParameterSpec("theta", lower=0.0)],
    prior_id="my-prior-v1",
    simulator_id="my-simulator-v1",
)
estimator = fit(model, simulations=50_000, seed=1234)
result = estimator.infer(observed_data, draws=20_000, seed=5678)
print(result.summary())
```

The new path returns `TrainedEstimator` and `InferenceResult` objects, uses
named constrained parameters, carries diagnostics, and saves checksummed beta
artifact directories.

## Retained alpha path

These remain available in 0.1.0b1:

- `SimulatorSpec` and `ModelIndexSimulatorSpec`;
- `run_training`;
- path-based `infer` and `validate`;
- existing project configuration files;
- the `structnpe` CLI and its output filenames.

They retain their file-oriented return values and trusted-local NPZ behavior.
They are compatibility APIs, not aliases that silently change estimator
semantics. Existing Gaussian defaults therefore remain legacy defaults; the new
`fit` façade documents the real five-component MDN.

## Output terminology

The beta result table uses `estimate`, `posterior_mean`, `posterior_sd`,
`median`, `q025`, `q975`, `ess`, and `warning`. Alpha CSV columns such as
`mean`, `sd`, `q05`, `q50`, and `q95` are not silently relabeled. Migrate code
by reading `result.summary()` rather than assuming the legacy CSV schema.

## Artifact migration

Alpha single-file NPZ models are not safe beta bundles and are not
automatically converted. Continue using them only as trusted-local artifacts,
or retrain through `fit` and save a versioned directory:

```python
estimator.save("saved/my_estimator")
```

No alpha API is scheduled for removal before 0.2. Any later removal requires a
separate deprecation cycle and migration note.
