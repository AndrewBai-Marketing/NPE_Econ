"""Marginal Eight Schools simulator used by the public replication.

The public estimator targets only the population hyperparameters ``(mu, tau)``.
The eight latent school effects are integrated out exactly, so this simulator
induces the same marginal posterior for ``(mu, tau)`` as the usual centered
hierarchical model without asking the density estimator to learn eight
additional latent coordinates.
"""

from __future__ import annotations

import numpy as np

from structnpe import ArrayAdapter, ParameterSpec, StructuralModel


EFFECTS = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
STANDARD_ERRORS = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
MU_PRIOR_SD = 10.0
TAU_PRIOR_SD = 10.0


def hyperparameter_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``(mu, tau)`` from the stated hyperprior."""

    size = int(n)
    mu = rng.normal(0.0, MU_PRIOR_SD, size=size)
    tau = np.abs(rng.normal(0.0, TAU_PRIOR_SD, size=size))
    zero = tau == 0.0
    while np.any(zero):  # A lower-log transform requires strict positivity.
        tau[zero] = np.abs(rng.normal(0.0, TAU_PRIOR_SD, size=int(zero.sum())))
        zero = tau == 0.0
    return np.column_stack([mu, tau])


def marginalized_simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate reported effects after integrating out school-level effects.

    Since

    ``theta_j | mu, tau ~ Normal(mu, tau)`` and
    ``y_j | theta_j ~ Normal(theta_j, sigma_j)``, the marginalized observation
    law is ``y_j | mu, tau ~ Normal(mu, sqrt(tau**2 + sigma_j**2))``.
    """

    parameters = np.asarray(theta, dtype=float)
    if parameters.ndim == 1:
        if parameters.shape != (2,):
            raise ValueError("one Eight Schools parameter row must have length two")
        scale = np.sqrt(parameters[1] ** 2 + STANDARD_ERRORS**2)
        return rng.normal(parameters[0], scale)
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("Eight Schools parameters must have shape (batch, 2)")
    scale = np.sqrt(parameters[:, 1, None] ** 2 + STANDARD_ERRORS[None, :] ** 2)
    return rng.normal(parameters[:, 0, None], scale)


def build_model() -> StructuralModel:
    return StructuralModel(
        prior=hyperparameter_prior,
        simulator=marginalized_simulator,
        parameters=[
            ParameterSpec(
                "mu",
                description="Population mean coaching effect",
                unit="SAT points",
            ),
            ParameterSpec(
                "tau",
                lower=0.0,
                description="Between-school standard deviation",
                unit="SAT points",
            ),
        ],
        observation_adapter=ArrayAdapter(expected_shape=(8,)),
        prior_id="structnpe.replication.eight_schools.hyperprior.v2",
        simulator_id="structnpe.replication.eight_schools.marginal_measurement.v2",
        prior_config={
            "mu": {"distribution": "normal", "mean": 0.0, "sd": MU_PRIOR_SD},
            "tau": {"distribution": "half_normal", "sd": TAU_PRIOR_SD},
        },
        simulator_config={
            "reported_standard_errors": STANDARD_ERRORS.tolist(),
            "observation": "eight estimated school effects",
            "latent_school_effects": "integrated_out_exactly",
        },
        batched_simulator=True,
    )
