from pathlib import Path

import numpy as np

from structnpe.simulator import ModelIndexSimulatorSpec, SimulatorSpec
from structnpe.simulation_bank import load_bank, simulate_bank


class ToySpec(SimulatorSpec):
    def sample_prior(self, n, rng):
        return rng.normal(size=(n, 1))

    def simulate(self, theta, rng, **kwargs):
        return np.array([theta[0] + rng.normal(scale=0.1)])

    def summarize(self, data):
        return np.asarray(data, dtype=float)


class ToyModelIndex(ModelIndexSimulatorSpec):
    def sample_model(self, n, rng):
        return rng.integers(0, 2, size=n)

    def sample_prior_given_model(self, m, rng):
        return np.array([float(m)])

    def simulate_given_model(self, m, theta_m, rng, **kwargs):
        return np.array([float(m) + theta_m[0]])

    def summarize(self, data):
        return np.asarray(data, dtype=float)


def test_simulate_and_load_standard_bank(tmp_path: Path) -> None:
    path = tmp_path / "bank.npz"
    bank = simulate_bank(ToySpec(), 20, 123, path)
    loaded = load_bank(path)
    assert bank.summaries.shape == (20, 1)
    assert loaded.theta.shape == (20, 1)
    assert loaded.model_index is None


def test_simulate_and_load_model_index_bank(tmp_path: Path) -> None:
    path = tmp_path / "bank_m.npz"
    bank = simulate_bank(ToyModelIndex(), 20, 123, path)
    loaded = load_bank(path)
    assert bank.model_index is not None
    assert loaded.model_index is not None
    assert loaded.theta.shape == (20, 1)
