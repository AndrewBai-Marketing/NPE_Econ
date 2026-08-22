from __future__ import annotations

import numpy as np

from structnpe import SimulatorSpec


class MyModel(SimulatorSpec):
    """Replace this toy model with your structural simulator."""

    def sample_prior(self, n: int, rng: np.random.Generator):
        return rng.normal(size=(n, 1))

    def simulate(self, theta, rng: np.random.Generator, **kwargs):
        return rng.normal(loc=float(theta[0]), scale=1.0, size=20)

    def summarize(self, data):
        arr = np.asarray(data, dtype=float)
        return np.array([arr.mean(), arr.std()], dtype=float)

    def counterfactual(self, theta, policy, rng=None):
        shift = float(policy.get("shift", 0.0)) if isinstance(policy, dict) else 0.0
        return float(theta[0]) + shift
