"""Minimal bring-your-own-simulator example for ``structnpe``.

The example estimates a Normal location and scale from a fixed-length raw
sample.  It is an interface demonstration, not a claim that this small training
budget is adequate for a substantive application.
"""

from __future__ import annotations

import argparse

import numpy as np

from structnpe import ParameterSpec, StructuralModel, SummaryAdapter, fit


N_OBSERVATIONS = 30


def prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample parameters strictly inside their declared supports."""

    location = rng.uniform(-2.95, 2.95, size=int(n))
    scale = rng.uniform(0.15, 1.95, size=int(n))
    return np.column_stack([location, scale])


def simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate one dataset without consulting global random state."""

    location, scale = np.asarray(theta, dtype=float).reshape(-1)
    return rng.normal(float(location), float(scale), size=N_OBSERVATIONS)


def build_model() -> StructuralModel:
    return StructuralModel(
        prior=prior,
        simulator=simulator,
        parameters=[
            ParameterSpec(
                "location",
                lower=-3.0,
                upper=3.0,
                transform="auto",
                description="Outcome location",
                unit="outcome units",
            ),
            ParameterSpec(
                "scale",
                lower=0.1,
                upper=2.0,
                transform="auto",
                description="Outcome standard deviation",
                unit="outcome units",
            ),
        ],
        observation_adapter=SummaryAdapter(
            statistics=("mean", "std"),
            expected_shape=(N_OBSERVATIONS,),
        ),
        prior_id="custom_normal.prior.v1",
        simulator_id="custom_normal.simulator.v1",
        prior_config={"location": [-2.95, 2.95], "scale": [0.15, 1.95]},
        simulator_config={"family": "normal", "n_observations": N_OBSERVATIONS},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=5_000)
    parser.add_argument("--draws", type=int, default=4_000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = build_model()
    estimator = fit(
        model,
        simulations=args.simulations,
        seed=args.seed,
        validation_fraction=0.15,
        hidden_dim=48,
        depth=2,
        components=5,
        epochs=args.epochs,
        batch_size=128,
        patience=10,
        device="cpu",
        progress=not args.quiet,
    )

    observed_rng = np.random.default_rng(args.seed + 10_000)
    observed = simulator(np.array([0.6, 0.8]), observed_rng)
    result = estimator.infer(observed, draws=args.draws, seed=args.seed + 1)

    print(f"Posterior draws shape: {np.asarray(result.draws).shape}")
    print(result.summary(point="mean").to_string(index=False))
    print("Posterior covariance:")
    print(result.covariance())
    print("Posterior correlation:")
    print(result.correlation())
    print("Inference diagnostics:")
    print(result.diagnostics())
    print("Posterior predictive check:")
    print(result.predictive_check(model=model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
