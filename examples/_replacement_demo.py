"""Fully synthetic replacement model shared by the public beta examples.

The saved demonstration estimator was trained on this exact model contract.
The model is deliberately small: five deterioration states, two actions, and
state-action counts from twelve synthetic agents over ten periods.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from structnpe import ArrayAdapter, ParameterSpec, StructuralModel


S_MAX = 4
N_AGENTS = 12
N_PERIODS = 10
DISCOUNT = 0.9
MAINTENANCE_GRID = np.linspace(0.1, 0.7, 7)
REPLACEMENT_GRID = np.linspace(1.25, 5.75, 9)
THETA_GRID = np.array(
    [
        (maintenance, replacement)
        for maintenance in MAINTENANCE_GRID
        for replacement in REPLACEMENT_GRID
    ],
    dtype=float,
)

# Fixed synthetic observation used in the validated structural-grid smoke run.
OBSERVED_COUNTS = np.array([32, 1, 27, 3, 18, 7, 12, 4, 6, 10], dtype=float)


def grid_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw uniformly from the declared finite parameter grid."""

    indices = rng.integers(0, len(THETA_GRID), size=int(n))
    return THETA_GRID[indices].copy()


def _logsumexp_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    maximum = np.maximum(a, b)
    return maximum + np.log(np.exp(a - maximum) + np.exp(b - maximum))


def choice_probabilities(theta: np.ndarray) -> np.ndarray:
    """Solve keep/replace probabilities for one parameter vector."""

    maintenance, replacement = np.asarray(theta, dtype=float).reshape(2)
    states = np.arange(S_MAX + 1, dtype=float)
    next_keep = np.minimum(np.arange(S_MAX + 1) + 1, S_MAX)
    value = np.zeros(S_MAX + 1, dtype=float)
    for _ in range(2_000):
        keep = -maintenance * states + DISCOUNT * value[next_keep]
        replace = -replacement + DISCOUNT * value[0]
        updated = _logsumexp_pair(keep, replace)
        if float(np.max(np.abs(updated - value))) <= 1.0e-11:
            value = updated
            break
        value = updated
    else:  # pragma: no cover - defensive convergence guard
        raise RuntimeError("replacement-model value iteration did not converge")
    keep = -maintenance * states + DISCOUNT * value[next_keep]
    replace = np.full(S_MAX + 1, -replacement + DISCOUNT * value[0])
    denominator = _logsumexp_pair(keep, replace)
    probabilities = np.column_stack(
        [np.exp(keep - denominator), np.exp(replace - denominator)]
    )
    return probabilities / probabilities.sum(axis=1, keepdims=True)


@lru_cache(maxsize=256)
def _cached_probabilities(maintenance: float, replacement: float) -> np.ndarray:
    return choice_probabilities(np.array([maintenance, replacement], dtype=float))


def replacement_simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate sufficient state-action counts for one synthetic panel."""

    maintenance, replacement = np.asarray(theta, dtype=float).reshape(2)
    probabilities = _cached_probabilities(float(maintenance), float(replacement))
    states = np.zeros(N_AGENTS, dtype=int)
    counts = np.zeros((S_MAX + 1, 2), dtype=float)
    for _ in range(N_PERIODS):
        replace = rng.random(N_AGENTS) < probabilities[states, 1]
        for state in range(S_MAX + 1):
            at_state = states == state
            counts[state, 1] += float(np.sum(at_state & replace))
            counts[state, 0] += float(np.sum(at_state & ~replace))
        states = np.where(replace, 0, np.minimum(states + 1, S_MAX))
    return counts.reshape(-1)


def build_model() -> StructuralModel:
    """Return the exact model contract stored with the demo estimator."""

    return StructuralModel(
        prior=grid_prior,
        simulator=replacement_simulator,
        parameters=[
            ParameterSpec(
                "maintenance_cost",
                lower=0.05,
                upper=0.75,
                transform="auto",
                description="Per-state maintenance cost slope",
                unit="utility units",
            ),
            ParameterSpec(
                "replacement_cost",
                lower=1.0,
                upper=6.0,
                transform="auto",
                description="Replacement cost",
                unit="utility units",
            ),
        ],
        observation_adapter=ArrayAdapter(expected_shape=(2 * (S_MAX + 1),)),
        prior_id="structnpe.validation.replacement_grid.prior.v1",
        simulator_id="structnpe.validation.replacement_grid.simulator.v1",
        prior_config={
            "maintenance_grid": MAINTENANCE_GRID.tolist(),
            "replacement_grid": REPLACEMENT_GRID.tolist(),
            "weights": "uniform",
        },
        simulator_config={
            "states": S_MAX + 1,
            "agents": N_AGENTS,
            "periods": N_PERIODS,
            "discount": DISCOUNT,
            "transition": "replace_to_zero_otherwise_deteriorate_one",
            "observation": "state_action_counts",
        },
    )

