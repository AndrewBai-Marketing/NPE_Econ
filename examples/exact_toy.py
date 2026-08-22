"""Exact conjugate-normal example for the public ``structnpe`` beta.

The maintained model is deliberately transparent:

* theta ~ Normal(0, 1);
* y_i | theta ~ Normal(theta, 1), for i = 1, ..., 20;
* the observation adapter retains the sample mean, which is sufficient here.

Consequently the exact posterior is available analytically.  This example is
small enough for a laptop smoke run; the more systematic comparison lives in
``validation/exact_example/run_validation.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import numpy as np

from structnpe import ParameterSpec, StructuralModel, SummaryAdapter, fit


N_OBSERVATIONS = 20
PRIOR_SD = 1.0
OBSERVATION_SD = 1.0


def normal_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw the scalar structural parameter in declared order."""

    return rng.normal(0.0, PRIOR_SD, size=(int(n), 1))


def normal_simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate one raw dataset using only the supplied generator."""

    value = float(np.asarray(theta, dtype=float).reshape(-1)[0])
    return rng.normal(value, OBSERVATION_SD, size=N_OBSERVATIONS)


def build_model() -> StructuralModel:
    """Construct the portable model used by this example and its validation."""

    return StructuralModel(
        prior=normal_prior,
        simulator=normal_simulator,
        parameters=[
            ParameterSpec(
                "theta",
                transform="identity",
                description="Normal location parameter",
                unit="outcome units",
            )
        ],
        observation_adapter=SummaryAdapter(
            statistics=("mean",),
            expected_shape=(N_OBSERVATIONS,),
        ),
        prior_id="structnpe.examples.conjugate_normal.prior.v1",
        simulator_id="structnpe.examples.conjugate_normal.simulator.v1",
        prior_config={"mean": 0.0, "sd": PRIOR_SD},
        simulator_config={"n_observations": N_OBSERVATIONS, "sd": OBSERVATION_SD},
    )


def make_observed(theta: float = 0.35, seed: int = 20260821) -> np.ndarray:
    """Generate one reproducible raw observed dataset."""

    return normal_simulator(np.array([float(theta)]), np.random.default_rng(seed))


def exact_posterior(observed: np.ndarray) -> tuple[float, float]:
    """Return the exact posterior mean and standard deviation."""

    y = np.asarray(observed, dtype=float)
    if y.shape != (N_OBSERVATIONS,):
        raise ValueError(f"observed must have shape {(N_OBSERVATIONS,)}, got {y.shape}")
    prior_precision = 1.0 / (PRIOR_SD**2)
    observation_precision = 1.0 / (OBSERVATION_SD**2)
    posterior_variance = 1.0 / (prior_precision + N_OBSERVATIONS * observation_precision)
    posterior_mean = posterior_variance * N_OBSERVATIONS * observation_precision * float(y.mean())
    return posterior_mean, float(np.sqrt(posterior_variance))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=4_000)
    parser.add_argument("--draws", type=int, default=4_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path)
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
        hidden_dim=32,
        depth=2,
        components=5,
        epochs=args.epochs,
        batch_size=128,
        patience=10,
        device="cpu",
        output_dir=args.output_dir,
        progress=not args.quiet,
    )

    observed = make_observed()
    result = estimator.infer(observed, draws=args.draws, seed=args.seed + 1)
    exact_mean, exact_sd = exact_posterior(observed)
    z975 = NormalDist().inv_cdf(0.975)

    print(f"Posterior draws shape: {np.asarray(result.draws).shape}")
    print(result.summary(point="mean").to_string(index=False))
    print(
        "Exact posterior: "
        f"mean={exact_mean:.6f}, sd={exact_sd:.6f}, "
        f"95% credible interval=({exact_mean - z975 * exact_sd:.6f}, "
        f"{exact_mean + z975 * exact_sd:.6f})"
    )
    print("Posterior covariance:")
    print(result.covariance())
    print("Posterior predictive check:")
    print(result.predictive_check(model=model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
