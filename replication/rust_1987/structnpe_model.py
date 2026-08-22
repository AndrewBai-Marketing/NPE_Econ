"""Current-API StructuralModel for amortized Rust panel inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from structnpe import ArrayAdapter, ParameterSpec, StructuralModel

try:
    from .constants import (
        BUS_COUNT,
        CANONICAL_TRANSITION_PROBABILITIES,
        COST_SCALE,
        DISCOUNT_FACTOR,
        NUM_STATES,
        PERIOD_COUNT,
        PRIOR_BOUNDS,
    )
    from .model import simulate_choice_summary_batch
except ImportError:  # pragma: no cover - direct script execution
    from constants import (
        BUS_COUNT,
        CANONICAL_TRANSITION_PROBABILITIES,
        COST_SCALE,
        DISCOUNT_FACTOR,
        NUM_STATES,
        PERIOD_COUNT,
        PRIOR_BOUNDS,
    )
    from model import simulate_choice_summary_batch


@dataclass(frozen=True)
class UniformCostPrior:
    bounds: tuple[tuple[float, float], tuple[float, float]]

    def __call__(self, n: int, rng: np.random.Generator) -> np.ndarray:
        lower = np.asarray([item[0] for item in self.bounds], dtype=float)
        upper = np.asarray([item[1] for item in self.bounds], dtype=float)
        # Stay strictly inside transform boundaries.
        unit = rng.uniform(1e-6, 1.0 - 1e-6, size=(int(n), 2))
        return lower + (upper - lower) * unit


@dataclass(frozen=True)
class RustBatchSimulator:
    transition_probabilities: tuple[float, ...]
    num_states: int
    beta: float
    scale: float
    buses: int
    periods: int

    def __call__(self, theta: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
        output = simulate_choice_summary_batch(
            theta,
            rng,
            self.transition_probabilities,
            num_states=self.num_states,
            beta=self.beta,
            scale=self.scale,
            buses=self.buses,
            periods=self.periods,
        )
        return [row for row in output]


def build_model(
    *,
    transition_probabilities: Sequence[float] = CANONICAL_TRANSITION_PROBABILITIES,
    num_states: int = NUM_STATES,
    beta: float = DISCOUNT_FACTOR,
    scale: float = COST_SCALE,
    buses: int = BUS_COUNT,
    periods: int = PERIOD_COUNT,
    prior_bounds: tuple[tuple[float, float], tuple[float, float]] = PRIOR_BOUNDS,
) -> StructuralModel:
    probabilities = tuple(float(value) for value in transition_probabilities)
    prior = UniformCostPrior(prior_bounds)
    simulator = RustBatchSimulator(probabilities, num_states, beta, scale, buses, periods)
    return StructuralModel(
        prior=prior,
        simulator=simulator,
        parameters=(
            ParameterSpec(
                "replacement_cost",
                lower=prior_bounds[0][0],
                upper=prior_bounds[0][1],
                description="Utility-scaled engine replacement cost",
            ),
            ParameterSpec(
                "maintenance_slope",
                lower=prior_bounds[1][0],
                upper=prior_bounds[1][1],
                description="Linear mileage-state maintenance-cost coefficient",
            ),
        ),
        observation_adapter=ArrayAdapter(expected_shape=(2 * int(num_states),)),
        prior_id="structnpe.replication.rust.uniform_cost_prior.v1",
        simulator_id="structnpe.replication.rust.batched_panel.v1",
        prior_config={"bounds": [list(item) for item in prior_bounds]},
        simulator_config={
            "transition_probabilities": list(probabilities),
            "num_states": int(num_states),
            "discount_factor": float(beta),
            "cost_scale": float(scale),
            "buses": int(buses),
            "periods": int(periods),
            "initial_state": 0,
            "period_zero_excluded_from_summary": True,
        },
        batched_simulator=True,
    )

