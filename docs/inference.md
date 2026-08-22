# Inference

Completed beta estimators use NumPy for CPU forward evaluation and posterior
drawing, so the base installation is sufficient.

```python
from structnpe import load_estimator

estimator = load_estimator("saved/my_estimator", model=model)
result = estimator.infer(observed_data, draws=20_000, seed=5678)
```

For one dataset, `result.draws` has shape `(draws, parameters)`. To make batch
semantics explicit:

```python
batched = estimator.infer(observed_datasets, draws=5_000, seed=5678, batch=True)
```

The batched draw array has shape `(datasets, draws, parameters)`. The same
frozen adapter processes simulations and observations; incompatible fields,
shape, parameter order, or preprocessing state raise an error.

Inference records preprocessing, draw-generation, and total elapsed time. A
stored nearest-neighbor support diagnostic compares the observation with the
training representation. An out-of-support result remains available, but its
warning appears in `diagnostics()` and `summary()`. It is a conservative
distribution-support warning, not a hypothesis test.

Supplying `model` to `load_estimator` checks the model fingerprint. The expert
option `allow_incompatible=True` may bypass a fingerprint mismatch with a
recorded warning; it never bypasses checksum, completeness, architecture, or
shape failures.

The default loader reconstructs only built-in adapters. For a custom adapter,
the preferred path is the example above: pass a compatible model containing
adapter code that you imported and reviewed. Loading without a model requires
`allow_custom_adapter=True`; that explicit trust opt-in may import and
initialize code selected by the artifact and emits a warning. It does not
bypass checksum, schema, version, fingerprint, or shape checks.

```python
ppc = result.predictive_check(model=model)
```

Predictive checks require executable simulator code. Passing selected checks
does not prove correct specification.

Any timing claim must report hardware, observation shape, draw count, batch
size, loading convention, and separate training cost. Do not quote an
unconditional speedup.
