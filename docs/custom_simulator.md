# Custom simulator contract

The beta intentionally uses an explicit contract. It does not inspect a
simulator's source code or infer parameter meanings automatically.

## Prior

Provide either `prior(n, rng)` or an object with `sample(n, rng)`. It must
return a finite numeric array of shape `(n, p)` in the same order as the
declared `ParameterSpec` objects.

```python
def prior(n, rng):
    price = rng.uniform(0.1, 4.0, size=n)
    persistence = rng.uniform(0.01, 0.99, size=n)
    return np.column_stack([price, persistence])
```

Every random draw must use the supplied `numpy.random.Generator`. Hidden use
of global `numpy.random` state is outside the contract.

## Simulator

By default, provide `simulator(theta, rng)` for one ordered parameter vector.
The package performs deterministic batching around that function. Declare
`batched_simulator=True` only when the function accepts an `(n, p)` parameter
array and returns exactly `n` datasets.

Outputs must have stable fields, shapes, and dtypes. Non-finite or inconsistent
outputs fail with an error instead of being silently reshaped.

## Observation adapter

Simulation output and observed data always go through the same adapter.
`ArrayAdapter` is the default for fixed-shape numeric observations.
`SummaryAdapter` is for an explicitly configured representation.

A custom adapter must be an importable class with structured configuration and
fitted state. It must freeze its expected input schema, output dimension, and
dtype. Lambdas, local functions, pickle payloads, silent missing-field filling,
and observation-specific reshaping are not portable beta artifacts.

On load, built-in adapters are reconstructed from a fixed allowlist. The
preferred custom-adapter path is to import and review the adapter yourself,
construct a compatible `StructuralModel`, and pass it to `load_estimator`.
Without a model, `allow_custom_adapter=True` explicitly permits the loader to
import and initialize the class named by the artifact and emits a warning. Use
that opt-in only when the artifact and installed adapter code are trusted.

## Parameters and identifiers

Declare either `parameter_names` for unconstrained values or, preferably,
ordered `ParameterSpec` objects. Supported transforms are `identity`, `log`,
`lower_log`, `upper_log`, `logit`, and `auto`. `auto` is derived only from the
declared bounds. Prior draws must lie strictly inside transform boundaries.

Supply stable `prior_id` and `simulator_id` values, plus structured versioned
configuration where relevant. A stateful callable instance, or another
callable without a stable importable function/class identity, requires both its
explicit identifier and its corresponding `prior_config` or
`simulator_config`; class name alone is not a fingerprint of instance state.
These declarations participate in compatibility fingerprints, but are not
claimed to hash arbitrary source semantics.

For predictive checks after loading in a fresh process, supply the same
executable model again. Posterior drawing itself does not execute the
simulator.
