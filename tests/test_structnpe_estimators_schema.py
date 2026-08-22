from pathlib import Path

import numpy as np
import pytest

from structnpe.estimators import FiniteGridClassifier, GaussianPosterior, MDNPosterior, ModelIndexPosterior
from structnpe.posterior import load_posterior
from structnpe.simulation_bank import SimulationBank


def test_gaussian_estimator_schema_and_save_load(tmp_path: Path) -> None:
    x = np.linspace(-1, 1, 30).reshape(-1, 1)
    bank = SimulationBank(summaries=x, theta=2 * x)
    est = GaussianPosterior().fit(bank)
    pred = est.predict(np.array([0.1]))
    assert pred["mean"].shape == (1,)
    draws = est.sample(np.array([0.1]), 5, seed=1)
    assert draws.shape == (5, 1)
    path = est.save(tmp_path / "gaussian.npz")
    loaded = load_posterior(path)
    assert loaded.sample(np.array([0.1]), 5, seed=1).shape == (5, 1)


def test_finite_grid_estimator_schema() -> None:
    theta = np.array([[-1.0], [0.0], [1.0]] * 10)
    summaries = theta + 0.01
    bank = SimulationBank(summaries=summaries, theta=theta)
    est = FiniteGridClassifier().fit(bank)
    pred = est.predict(np.array([1.0]))
    assert pred["weights"].sum() == pytest.approx(1.0)
    assert est.sample(np.array([1.0]), 3, seed=2).shape == (3, 1)


def test_model_index_estimator_schema() -> None:
    model = np.array([0, 1] * 15)
    theta = model.reshape(-1, 1).astype(float)
    summaries = theta + 0.05
    bank = SimulationBank(summaries=summaries, theta=theta, model_index=model)
    est = ModelIndexPosterior().fit(bank)
    pred = est.predict(np.array([1.0]))
    assert pred["model_probs"].sum() == pytest.approx(1.0)
    assert est.sample(np.array([1.0]), 4, seed=3).shape == (4, 2)


def test_mdn_alpha_uses_common_interface() -> None:
    x = np.linspace(-1, 1, 20).reshape(-1, 1)
    bank = SimulationBank(summaries=x, theta=x)
    est = MDNPosterior().fit(bank)
    assert est.sample(np.array([0.0]), 2, seed=4).shape == (2, 1)
