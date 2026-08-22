"""Train, save, and reload a small bring-your-own-simulator model on CPU.

The defaults are a bounded demonstration configuration, not a validation
claim. Increase the simulation and training budgets only after declaring
model-specific validation targets.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

from structnpe import ParameterSpec, StructuralModel, SummaryAdapter, fit, load_estimator


N_OBSERVATIONS = 30


def prior(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack(
        [
            rng.uniform(-2.95, 2.95, size=int(n)),
            rng.uniform(0.15, 1.95, size=int(n)),
        ]
    )


def simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Accept either one parameter row or a complete training batch."""

    parameters = np.asarray(theta, dtype=float)
    if parameters.ndim == 1:
        beta, sigma = parameters
        return rng.normal(beta, sigma, size=N_OBSERVATIONS)
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("theta must have shape (2,) or (batch, 2)")
    beta = parameters[:, [0]]
    sigma = parameters[:, [1]]
    return rng.normal(beta, sigma, size=(len(parameters), N_OBSERVATIONS))


def build_model() -> StructuralModel:
    # README:custom:start
    model = StructuralModel(
        prior=prior,
        simulator=simulator,
        parameters=[
            ParameterSpec("beta", lower=-3.0, upper=3.0),
            ParameterSpec("sigma", lower=0.1, upper=2.0, unit="outcome units"),
        ],
        observation_adapter=SummaryAdapter(
            statistics=("mean", "std"),
            expected_shape=(N_OBSERVATIONS,),
        ),
        prior_id="structnpe.examples.normal_location_scale.prior.v1",
        simulator_id="structnpe.examples.normal_location_scale.simulator.v1",
        prior_config={"beta": [-2.95, 2.95], "sigma": [0.15, 1.95]},
        simulator_config={"observations": N_OBSERVATIONS, "likelihood": "normal"},
        batched_simulator=True,
    )
    # README:custom:end
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace, output_root: Path) -> None:
    model = build_model()
    fit_start = time.perf_counter()
    estimator = fit(
        model,
        simulations=args.simulations,
        seed=303,
        validation_fraction=0.2,
        hidden_dim=32,
        depth=2,
        components=5,
        epochs=args.epochs,
        batch_size=128,
        patience=6,
        device="cpu",
        progress=not args.quiet,
    )
    fit_seconds = time.perf_counter() - fit_start

    observed = np.array(
        [
            0.45, 0.91, 0.68, 1.04, 0.31, 0.72, 0.57, 0.88, 0.62, 0.79,
            0.53, 1.12, 0.40, 0.95, 0.67, 0.84, 0.59, 0.76, 0.48, 1.01,
            0.70, 0.64, 0.86, 0.55, 0.93, 0.44, 0.81, 0.60, 0.74, 0.98,
        ],
        dtype=float,
    )
    before = estimator.infer(observed, draws=args.draws, seed=404)
    artifact = output_root / "custom_estimator"
    estimator.save(artifact, overwrite=True)
    reloaded = load_estimator(artifact, model=model)
    after = reloaded.infer(observed, draws=args.draws, seed=404)
    if not np.array_equal(before.draws, after.draws):
        raise RuntimeError("save/reload did not preserve fixed-seed posterior draws")

    print(before.summary().to_string(index=False))
    print(f"training_seconds={fit_seconds:.3f}")
    print(f"saved_estimator={artifact}")
    print("CUSTOM_SIMULATOR_OK")


def main() -> int:
    args = parse_args()
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        run(args, args.output)
    else:
        with tempfile.TemporaryDirectory(prefix="structnpe-custom-") as directory:
            run(args, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
