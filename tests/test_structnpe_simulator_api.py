import numpy as np

from structnpe import ModelIndexSimulatorSpec, SimulatorSpec
from structnpe.simulator import ensure_summary_array, is_model_index_spec


class ToySpec(SimulatorSpec):
    def sample_prior(self, n, rng):
        return rng.normal(size=(n, 1))

    def simulate(self, theta, rng, **kwargs):
        return rng.normal(float(theta[0]), 1.0, size=5)

    def summarize(self, data):
        return np.array([np.mean(data)])


class ToyModelIndex(ModelIndexSimulatorSpec):
    def sample_model(self, n, rng):
        return rng.integers(0, 2, size=n)

    def sample_prior_given_model(self, m, rng):
        return np.array([float(m)])

    def simulate_given_model(self, m, theta_m, rng, **kwargs):
        return np.array([m, theta_m[0]])

    def summarize(self, data):
        return np.asarray(data, dtype=float)


def test_simulator_interfaces_and_summary_array() -> None:
    assert not is_model_index_spec(ToySpec())
    assert is_model_index_spec(ToyModelIndex())
    summary = ensure_summary_array([[1.0, 2.0]])
    assert summary.shape == (2,)
