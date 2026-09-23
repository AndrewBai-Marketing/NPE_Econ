"""Train, save, reload, and use a small custom simulator."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from structnpe import ParameterSpec, StructuralModel, SummaryAdapter, fit, load_estimator


N_OBSERVATIONS = 30


def prior(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack(
        [
            rng.uniform(-2.95, 2.95, size=n),
            rng.uniform(0.15, 1.95, size=n),
        ]
    )


def simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Accept one parameter vector or a complete training batch."""

    theta = np.asarray(theta, dtype=float)
    if theta.ndim == 1:
        return rng.normal(theta[0], theta[1], size=N_OBSERVATIONS)
    return rng.normal(
        theta[:, [0]],
        theta[:, [1]],
        size=(len(theta), N_OBSERVATIONS),
    )


def build_model() -> StructuralModel:
    return StructuralModel(
        prior=prior,
        simulator=simulator,
        parameters=[
            ParameterSpec("location", lower=-3.0, upper=3.0),
            ParameterSpec("scale", lower=0.1, upper=2.0),
        ],
        observation_adapter=SummaryAdapter(
            statistics=("mean", "std"),
            expected_shape=(N_OBSERVATIONS,),
        ),
        prior_id="structnpe.examples.normal.prior.v1",
        simulator_id="structnpe.examples.normal.simulator.v1",
        prior_config={"location": [-2.95, 2.95], "scale": [0.15, 1.95]},
        simulator_config={"observations": N_OBSERVATIONS},
        batched_simulator=True,
    )


def run(*, simulations: int, epochs: int, draws: int, output: Path) -> None:
    model = build_model()
    estimator = fit(
        model,
        backend="spline",
        simulations=simulations,
        epochs=epochs,
        seed=303,
        progress=False,
    )
    estimator.save(output, overwrite=True)

    observed = np.array(
        [
            0.45, 0.91, 0.68, 1.04, 0.31, 0.72, 0.57, 0.88, 0.62, 0.79,
            0.53, 1.12, 0.40, 0.95, 0.67, 0.84, 0.59, 0.76, 0.48, 1.01,
            0.70, 0.64, 0.86, 0.55, 0.93, 0.44, 0.81, 0.60, 0.74, 0.98,
        ],
        dtype=float,
    )
    reloaded = load_estimator(output, model=model)
    posterior = reloaded.infer(observed, draws=draws, seed=404)
    print(posterior.summary().to_string(index=False))
    print("CUSTOM_MODEL_OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--draws", type=int, default=2_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        run(
            simulations=args.simulations,
            epochs=args.epochs,
            draws=args.draws,
            output=args.output,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="structnpe-custom-") as directory:
            run(
                simulations=args.simulations,
                epochs=args.epochs,
                draws=args.draws,
                output=Path(directory) / "estimator",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
