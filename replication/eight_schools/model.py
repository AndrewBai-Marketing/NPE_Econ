"""Hierarchical Eight Schools simulator used by the public replication."""

from __future__ import annotations

import numpy as np

from structnpe import ArrayAdapter, ParameterSpec, StructuralModel


EFFECTS = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
STANDARD_ERRORS = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
SCHOOL_NAMES = tuple(chr(ord("A") + index) for index in range(8))
MU_PRIOR_SD = 10.0
TAU_PRIOR_SD = 10.0


def hierarchical_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``(mu, tau, theta_1, ..., theta_8)`` from the stated prior."""

    size = int(n)
    mu = rng.normal(0.0, MU_PRIOR_SD, size=size)
    tau = np.maximum(np.abs(rng.normal(0.0, TAU_PRIOR_SD, size=size)), 1.0e-8)
    school_effects = rng.normal(mu[:, None], tau[:, None], size=(size, 8))
    return np.column_stack([mu, tau, school_effects])


def schools_simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate the eight reported effects, accepting one row or a batch."""

    parameters = np.asarray(theta, dtype=float)
    if parameters.ndim == 1:
        if parameters.shape != (10,):
            raise ValueError("one Eight Schools parameter row must have length ten")
        return rng.normal(parameters[2:], STANDARD_ERRORS)
    if parameters.ndim != 2 or parameters.shape[1] != 10:
        raise ValueError("Eight Schools parameters must have shape (batch, 10)")
    return rng.normal(parameters[:, 2:], STANDARD_ERRORS[None, :])


def build_model() -> StructuralModel:
    parameters = [
        ParameterSpec("mu", description="Population mean coaching effect", unit="SAT points"),
        ParameterSpec(
            "tau",
            lower=0.0,
            description="Between-school standard deviation",
            unit="SAT points",
        ),
    ]
    parameters.extend(
        ParameterSpec(
            f"theta_{name}",
            description=f"Latent coaching effect for school {name}",
            unit="SAT points",
        )
        for name in SCHOOL_NAMES
    )
    return StructuralModel(
        prior=hierarchical_prior,
        simulator=schools_simulator,
        parameters=parameters,
        observation_adapter=ArrayAdapter(expected_shape=(8,)),
        prior_id="structnpe.replication.eight_schools.hierarchical_prior.v1",
        simulator_id="structnpe.replication.eight_schools.normal_measurement.v1",
        prior_config={
            "mu": {"distribution": "normal", "mean": 0.0, "sd": MU_PRIOR_SD},
            "tau": {"distribution": "half_normal", "sd": TAU_PRIOR_SD},
            "theta_given_mu_tau": {"distribution": "normal"},
        },
        simulator_config={
            "reported_standard_errors": STANDARD_ERRORS.tolist(),
            "observation": "eight estimated school effects",
        },
        batched_simulator=True,
    )

