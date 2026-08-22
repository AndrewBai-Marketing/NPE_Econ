"""One-dimensional quadrature reference for the Eight Schools posterior."""

from __future__ import annotations

import numpy as np

try:
    from .model import EFFECTS, MU_PRIOR_SD, STANDARD_ERRORS, TAU_PRIOR_SD
except ImportError:  # pragma: no cover - direct script import path
    from model import EFFECTS, MU_PRIOR_SD, STANDARD_ERRORS, TAU_PRIOR_SD


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def tau_quadrature(
    *,
    points: int = 40_001,
    tau_min: float = 1.0e-5,
    tau_max: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return tau nodes, normalized masses, and conditional mu moments.

    The grid is uniform in ``log(tau)``. The Jacobian is included explicitly,
    and trapezoid endpoint weights are used. Conditional integration over
    ``mu`` is analytic.
    """

    if points < 101 or tau_min <= 0.0 or tau_max <= tau_min:
        raise ValueError("invalid quadrature grid")
    log_tau = np.linspace(np.log(tau_min), np.log(tau_max), int(points))
    tau = np.exp(log_tau)
    variances = STANDARD_ERRORS[None, :] ** 2 + tau[:, None] ** 2
    precisions = 1.0 / variances
    prior_precision = 1.0 / (MU_PRIOR_SD**2)
    posterior_precision = prior_precision + precisions.sum(axis=1)
    posterior_variance = 1.0 / posterior_precision
    posterior_mean = posterior_variance * (precisions @ EFFECTS)

    quadratic = np.sum(precisions * (EFFECTS[None, :] ** 2), axis=1)
    log_marginal = (
        -0.5 * np.sum(np.log(2.0 * np.pi * variances), axis=1)
        -0.5 * np.log(2.0 * np.pi * MU_PRIOR_SD**2)
        +0.5 * np.log(2.0 * np.pi * posterior_variance)
        -0.5 * (quadratic - posterior_mean**2 / posterior_variance)
    )
    log_half_normal = (
        np.log(2.0)
        -0.5 * np.log(2.0 * np.pi * TAU_PRIOR_SD**2)
        -0.5 * (tau / TAU_PRIOR_SD) ** 2
    )
    log_mass = log_marginal + log_half_normal + log_tau
    log_mass[[0, -1]] += np.log(0.5)
    log_mass -= _logsumexp(log_mass)
    mass = np.exp(log_mass)
    return tau, mass / mass.sum(), posterior_mean, posterior_variance


def sample_exact_posterior(
    draws: int,
    *,
    seed: int,
    quadrature_points: int = 40_001,
) -> np.ndarray:
    """Draw from the quadrature/conditional-Gaussian reference posterior."""

    if draws < 1:
        raise ValueError("draws must be positive")
    tau_nodes, mass, mu_mean, mu_variance = tau_quadrature(points=quadrature_points)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(tau_nodes), size=int(draws), p=mass)
    tau = tau_nodes[indices]
    mu = rng.normal(mu_mean[indices], np.sqrt(mu_variance[indices]))

    tau_variance = tau[:, None] ** 2
    sampling_variance = STANDARD_ERRORS[None, :] ** 2
    school_variance = 1.0 / (1.0 / tau_variance + 1.0 / sampling_variance)
    school_mean = school_variance * (
        mu[:, None] / tau_variance + EFFECTS[None, :] / sampling_variance
    )
    school_effects = rng.normal(school_mean, np.sqrt(school_variance))
    return np.column_stack([mu, tau, school_effects])

