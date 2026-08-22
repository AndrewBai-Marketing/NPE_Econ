from __future__ import annotations

import numpy as np

from replication.eight_schools.exact import sample_exact_posterior, tau_quadrature
from replication.eight_schools.model import EFFECTS, build_model


def test_tau_quadrature_is_normalized_and_finite() -> None:
    tau, mass, mu_mean, mu_variance = tau_quadrature(points=1001)
    assert tau.shape == mass.shape == mu_mean.shape == mu_variance.shape
    assert np.all(tau > 0.0)
    assert np.all(mass >= 0.0)
    assert np.all(mu_variance > 0.0)
    assert np.isclose(mass.sum(), 1.0)


def test_exact_joint_draws_and_model_contract() -> None:
    first = sample_exact_posterior(200, seed=71, quadrature_points=1001)
    second = sample_exact_posterior(200, seed=71, quadrature_points=1001)
    assert first.shape == (200, 10)
    assert np.array_equal(first, second)
    assert np.all(first[:, 1] > 0.0)
    model = build_model()
    simulated = model.simulate_one(first[0], np.random.default_rng(72))
    assert simulated.shape == EFFECTS.shape
    assert np.all(np.isfinite(simulated))

