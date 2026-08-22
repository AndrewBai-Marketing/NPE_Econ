# Troubleshooting

## Training says Torch is unavailable

Install the neural extra:

```bash
python -m pip install ".[neural]"
```

The base package supports NumPy inference from completed beta estimators, not
MDN training.

## The observation shape is incompatible

Use the same observation adapter and schema used during training. Do not flatten,
pad, reorder, or drop fields merely to satisfy the stored dimension. Correct
the upstream representation or retrain with the intended adapter.

## Loading reports a fingerprint mismatch

Check ordered parameter definitions, transforms, adapter configuration,
simulator/prior identifiers, and architecture. Retraining is usually the right
answer. `allow_incompatible=True` records a prominent expert warning and cannot
bypass corruption or shape checks.

## Loading reports a checksum or completeness error

Treat the artifact as incomplete or corrupted. Restore it from a verified copy
or rerun training. Do not edit the manifest to suppress the error.

## A custom adapter cannot be reconstructed

The default loader reconstructs only the built-in `ArrayAdapter` and
`SummaryAdapter`. For a custom adapter, import and review its code, construct a
compatible `StructuralModel`, and pass that model to `load_estimator`; the
loader uses the model's adapter and applies the verified saved state.

If no model is available, `allow_custom_adapter=True` permits artifact-directed
import and reconstruction and emits a prominent warning. This is a trust
opt-in, not a compatibility or integrity bypass. Use it only when both the
artifact and installed adapter module are trusted. Lambdas, local functions,
and pickle-only state are not portable beta adapters.

## The result has a distribution-support warning

Inference is still returned. Inspect the transformed observation, training
support, prior, and simulator before interpretation. The warning is not a
hypothesis test and should not be hidden.

## Training cannot use the requested accelerator

MDN training requires the neural extra even on CPU. `device="cpu"` is the
supported default. During training, `device="auto"` selects CUDA only when
available; an unavailable requested CUDA device falls back to CPU and records a
warning in training metadata. Accelerator results may not be bitwise
deterministic across devices or library versions.

## Inference requested an unavailable accelerator

Completed beta estimators use NumPy when inference resolves to CPU, and the
base installation is sufficient. `device="cpu"` and `device="auto"` use this
portable path. If explicit CUDA inference is requested but Torch or CUDA is
unavailable, inference falls back to NumPy CPU and records a visible result
warning. This inference fallback is separate from training device selection.

## An old `.npz` project will not load as a beta estimator

Alpha NPZ files are trusted-local legacy artifacts. Use the retained legacy CLI
with the original simulator code, or retrain and save in the beta directory
format. There is no automatic security upgrade for an arbitrary old file.
