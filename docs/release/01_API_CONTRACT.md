# Frozen public-beta API contract

Contract version: `1`  
Package version: `0.1.0b1`  
Namespace: `structnpe`

This contract freezes the narrow documented path for the beta. Existing alpha
functions and the CLI remain available for compatibility, but are not silently
reinterpreted as this API.

## Public model objects

```python
ParameterSpec(
    name,
    lower=None,
    upper=None,
    transform="auto",
    description=None,
    unit=None,
)

StructuralModel(
    prior,
    simulator,
    parameter_names=None,
    parameters=None,
    observation_adapter=None,
    prior_id=None,
    simulator_id=None,
    prior_config=None,
    simulator_config=None,
    batched_simulator=False,
)
```

`prior(n, rng)` (or `prior.sample(n, rng)`) returns a finite `n x p` numeric
array in the declared order and support. `simulator(theta, rng)` (or
`simulator.simulate(theta, rng)`) receives an explicit NumPy generator. A
batched simulator is used only when explicitly declared. Hidden global random
state is outside the contract.

Stateful callable instances and callables without a stable importable
function/class identity require an explicit identifier and structured config
for the prior or simulator. This prevents class-only fingerprints from silently
colliding across different instance state; the package still does not inspect
or hash arbitrary source semantics.

`ParameterSpec` supports `identity`, `log`, `lower_log`, `upper_log`, `logit`,
and `auto`. `auto` is determined only from declared bounds: two finite bounds
use scaled logit, a finite lower bound uses lower-log, a finite upper bound uses
upper-log, and no finite bounds use identity. Training draws must be strictly
inside bounds required by a transform. User-facing draws are inverse
transformed to declared units.

The built-in observation adapters are `ArrayAdapter` and `SummaryAdapter`.
Raw simulations and observations always pass through the same adapter method.
The adapter freezes input expectations, output dimension, dtype, configuration,
and fitted state. Custom adapters must be importable classes with structured
configuration; lambdas, local functions, pickle payloads, and silent reshaping
are not portable artifact formats.

Loading reconstructs built-in adapters from a fixed allowlist. For a custom
adapter, passing a compatible `StructuralModel` uses its already-imported class
and configuration and applies the verified saved state. Model-free custom
reconstruction is disabled by default because it would import artifact-selected
code.

## Training

```python
estimator = fit(
    model,
    simulations=200_000,
    seed=1234,
    validation_fraction=0.1,
    hidden_dim=64,
    depth=2,
    components=5,
    epochs=100,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-4,
    patience=15,
    device="cpu",
    output_dir=None,
    resume_from=None,
    progress=True,
)
```

The sole documented primary estimator is a diagonal-Gaussian MDN with five
components by default. Torch is a training extra. Saved beta estimators perform
the forward pass and posterior drawing on CPU with NumPy, so accelerator
packages are not required for inference. `device="auto"` and explicit
accelerators are optional; accelerator determinism is not guaranteed.

The seed deterministically derives separate simulation, split, optimization,
and support-reference streams. Early stopping restores the best validation
checkpoint. `resume_from` must pass the same compatibility and fingerprint
checks as loading. Training metadata records the package and estimator version,
architecture, seeds, simulation count, component fingerprints, ordered
parameters/transforms, split, duration, hardware/device, best epoch/checkpoint,
loss history, and UTC creation time.

## Estimator and inference

```python
estimator.infer(observed_data, draws=20_000, seed=5678, batch=False, device="cpu")
estimator.validate(model=None, simulations=..., draws=..., seed=...)
estimator.save(path)
estimator.training_metadata
estimator.model_fingerprint

load_estimator(
    path,
    *,
    model=None,
    allow_incompatible=False,
    allow_custom_adapter=False,
)
```

Inference validates the frozen representation and returns constrained
user-unit draws. One dataset produces shape `(draws, parameters)`; explicit
batch mode produces `(datasets, draws, parameters)`. It records preprocessing,
drawing, and total elapsed times. Supplying a model on load checks its
fingerprint. `allow_incompatible=True` is an expert override: it never bypasses
corruption or shape checks and records a prominent compatibility warning.
`allow_custom_adapter=True` is a separate trust opt-in for model-free custom
adapter reconstruction. It may import and initialize code named by the artifact
and emits a prominent warning; it does not waive integrity, schema, version,
fingerprint, or shape checks. Package-version incompatibility is rejected before
adapter reconstruction unless the compatibility override was explicitly used.

The estimator stores a bounded standardized training-reference set and fixes a
nearest-neighbor threshold at training time from the held-out validation
representations. Observations beyond that threshold remain inferable but receive a visible
distribution-support warning. This is not a hypothesis test.

## Result

```python
result.draws
result.summary(point="mean")
result.covariance()
result.correlation()
result.predictive_check(model=None, ...)
result.to_dataframe()
result.save(path)
result.diagnostics()
```

`summary()` returns a pandas table with `parameter`, `estimate`,
`posterior_mean`, `posterior_sd`, `median`, `q025`, `q975`, `ess`, and
`warning`. `point` may be `mean` or `median`. `point="MAP"` raises
`NotImplementedError`; a sample mode is never mislabeled MAP. Draws sampled
directly and independently from the fitted mixture have ESS equal to their
count, which describes Monte Carlo dependence only—not posterior accuracy.
Intervals are Bayesian credible intervals and `posterior_sd` is not a standard
error.

Predictive checks require an attached or explicitly supplied executable model.
They compare declared observed and replicated representation statistics and
state that passing selected checks does not establish correct specification.

## Validation contract

`validate` reports per-parameter rank, bias, RMSE, and empirical 50%, 80%, 90%,
and 95% central-credible-interval coverage, plus coarse prior-support regions
where sample size permits. CI uses a tiny bounded configuration; fuller runs
are manual. Exact and structural validation thresholds are declared in their
configuration before the final run and are emitted with machine-readable
metrics. A failed threshold remains a failed validation result.

## Artifact contract

A beta estimator is a directory with a versioned `manifest.json`, safe numeric
weights/state, structured adapter and metadata, file checksums, and a final
completeness marker. Loading rejects missing files, checksum mismatch,
unsupported artifact/package versions, parameter-order changes, unreconstructable
adapters, architecture mismatch, or representation-shape mismatch. No beta
loader uses pickle. Result artifacts use the same structured/checksummed rule.

The pickle-free numeric format does not make artifact-selected Python code safe.
Built-in adapters avoid that import boundary; model-free custom adapter loading
crosses it only through the explicit trust opt-in described above.

The model fingerprint is SHA-256 over canonical JSON containing the package
major/minor compatibility line, ordered parameter definitions, adapter
configuration, architecture, prior identifier/config, and simulator
identifier/config. It prevents obvious cross-model use; it is not claimed to
hash arbitrary source semantics.

## Backward compatibility and nonclaims

`SimulatorSpec`, `ModelIndexSimulatorSpec`, `run_training`, path-based `infer`,
path-based `validate`, project configs, and the CLI are retained. Their alpha
NPZ format remains a legacy trusted-local workflow and is not the beta artifact
format.

The API returns an approximate posterior conditional on the maintained prior,
simulator, and observation representation. It does not prove structural
identification, correct specification, frequentist coverage, robustness under
shift, or universal speed superiority.
