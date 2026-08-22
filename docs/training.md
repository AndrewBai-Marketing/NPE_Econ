# Training

Install `.[neural]` before training. The sole documented beta estimator is a
five-component diagonal-Gaussian mixture-density network (MDN). Its target is
the posterior under the supplied prior, simulator, and frozen observation
representation.

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
    output_dir="runs/my_model",
    progress=True,
)
```

Parameters are transformed and standardized for training. Returned draws are
inverse transformed to declared units. Early stopping is based on validation
negative log likelihood and restores the best checkpoint.

The train/validation split is fixed before fitting a stateful observation
adapter. `adapter.fit(...)` receives the complete training split and never the
validation observations; the rule and count are stored in training metadata.
The beta currently materializes the simulated observation bank and its
representations in memory, so very large or variable-structure encoders need a
separate, reviewed data pipeline.

The root seed derives separate streams for simulation, splitting,
optimization, and the support reference. CPU is the default. `device="auto"`
may choose an available accelerator; an unavailable requested device falls
back to CPU and records a visible warning in training metadata.
Accelerator operations may be nondeterministic even with fixed seeds.

Training metadata records package and estimator versions, architecture,
seeds, simulation count, fingerprints, parameter order and transforms, split,
duration, hardware, best epoch/checkpoint, loss history, and creation time.

`resume_from` is accepted only when artifact schema, architecture, parameter
order, adapter representation, and model fingerprint are compatible. An
incomplete or corrupted checkpoint is never treated as resumable.

Simulation time and neural training time are distinct costs. Record them
separately; a large up-front cost can dominate a one-dataset application.
