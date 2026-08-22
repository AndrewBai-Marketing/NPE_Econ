from __future__ import annotations

import numpy as np

from structnpe import SimulatorSpec


class TinyDDC(SimulatorSpec):
    """Tiny two-action dynamic-choice-like simulator for smoke tests."""

    grid = np.array([[-1.0], [0.0], [1.0]], dtype=float)

    def sample_prior(self, n: int, rng: np.random.Generator):
        idx = rng.integers(0, len(self.grid), size=n)
        return self.grid[idx]

    def simulate(self, theta, rng: np.random.Generator, **kwargs):
        beta = float(theta[0])
        state = rng.normal(size=12)
        probs = 1.0 / (1.0 + np.exp(-(beta + state)))
        choices = rng.binomial(1, probs)
        return np.column_stack([state, choices])

    def summarize(self, data):
        arr = np.asarray(data, dtype=float)
        return np.array([arr[:, 1].mean(), arr[:, 0].mean(), arr[:, 0].std()], dtype=float)

    def counterfactual(self, theta, policy, rng=None):
        tax = float(policy.get("tax", 0.0)) if isinstance(policy, dict) else 0.0
        beta = float(theta[0]) - tax
        return 1.0 / (1.0 + np.exp(-beta))
