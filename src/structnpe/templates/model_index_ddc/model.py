from __future__ import annotations

import numpy as np

from structnpe import ModelIndexSimulatorSpec


class TinyModelIndexDDC(ModelIndexSimulatorSpec):
    """Tiny model-index example with two mechanisms and finite parameters."""

    grids = {
        0: np.array([[-0.5], [0.5]], dtype=float),
        1: np.array([[-0.5], [0.5]], dtype=float),
    }

    def sample_model(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=n)

    def sample_prior_given_model(self, m: int, rng: np.random.Generator):
        grid = self.grids[int(m)]
        return grid[int(rng.integers(0, len(grid)))]

    def simulate_given_model(self, m: int, theta_m, rng: np.random.Generator, **kwargs):
        theta = float(theta_m[0])
        state = rng.normal(size=10)
        persistence = 0.8 if int(m) == 1 else 0.0
        latent = np.zeros(10)
        for t in range(1, 10):
            latent[t] = persistence * latent[t - 1] + rng.normal(scale=0.5)
        probs = 1.0 / (1.0 + np.exp(-(theta + state + latent)))
        choices = rng.binomial(1, probs)
        return np.column_stack([state, choices])

    def summarize(self, data):
        arr = np.asarray(data, dtype=float)
        choices = arr[:, 1]
        return np.array([choices.mean(), np.abs(np.diff(choices)).mean()], dtype=float)

    def counterfactual(self, m: int, theta_m, policy, rng=None):
        tax = float(policy.get("tax", 0.0)) if isinstance(policy, dict) else 0.0
        multiplier = 1.0 if int(m) == 0 else 1.5
        return multiplier * (1.0 / (1.0 + np.exp(-(float(theta_m[0]) - tax))))
