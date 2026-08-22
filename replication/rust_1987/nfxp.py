"""Conventional nested fixed-point comparator for the Rust replication."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .constants import COST_SCALE, DISCOUNT_FACTOR, NUM_STATES, PRIOR_BOUNDS
    from .model import choice_log_likelihood, solve_bellman, transition_matrix
    from .preprocess import read_processed_csv
except ImportError:  # pragma: no cover - direct script execution
    from constants import COST_SCALE, DISCOUNT_FACTOR, NUM_STATES, PRIOR_BOUNDS
    from model import choice_log_likelihood, solve_bellman, transition_matrix
    from preprocess import read_processed_csv


@dataclass(frozen=True)
class NFXPResult:
    replacement_cost: float
    maintenance_slope: float
    negative_log_likelihood: float
    success: bool
    message: str
    optimizer_iterations: int
    objective_evaluations: int
    bellman_residual: float
    seconds: float


class NFXPObjective:
    """NFXP likelihood with an analytic implicit-function score."""

    def __init__(
        self,
        keep_counts: np.ndarray,
        replacement_counts: np.ndarray,
        probabilities: Sequence[float],
        *,
        num_states: int,
        beta: float,
        scale: float,
    ) -> None:
        self.keep_counts = np.asarray(keep_counts, dtype=float)
        self.replacement_counts = np.asarray(replacement_counts, dtype=float)
        if self.keep_counts.shape != (num_states,) or self.replacement_counts.shape != (num_states,):
            raise ValueError("Choice counts do not match num_states.")
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.num_states = int(num_states)
        self.beta = float(beta)
        self.scale = float(scale)
        self.matrix = transition_matrix(num_states, probabilities)
        self._last_parameters: np.ndarray | None = None
        self._last_value: np.ndarray | None = None
        self._last_output: tuple[float, np.ndarray, float] | None = None

    def value_and_gradient(self, parameters: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.asarray(parameters, dtype=float)
        if theta.shape != (2,) or np.any(theta <= 0) or not np.all(np.isfinite(theta)):
            return 1e100, np.zeros(2)
        if self._last_parameters is not None and np.array_equal(theta, self._last_parameters):
            assert self._last_output is not None
            return self._last_output[0], self._last_output[1].copy()
        solution = solve_bellman(
            theta[0],
            theta[1],
            self.probabilities,
            num_states=self.num_states,
            beta=self.beta,
            scale=self.scale,
            initial_value=self._last_value,
        )
        value = solution.relative_value
        replacement_probability = solution.replacement_probability
        keep_probability = 1.0 - replacement_probability
        state = np.arange(self.num_states, dtype=float)
        jacobian = np.eye(self.num_states - 1) - self.beta * (
            self.matrix[1:, 1:] - self.matrix[0:1, 1:]
        ) * keep_probability[None, 1:]
        direct_integrated = np.column_stack(
            (
                -replacement_probability,
                -keep_probability * self.scale * state,
            )
        )
        mapped_direct = self.matrix @ direct_integrated
        right_hand_side = mapped_direct[1:] - mapped_direct[0]
        value_derivative = np.zeros((self.num_states, 2), dtype=float)
        value_derivative[1:] = np.linalg.solve(jacobian, right_hand_side)
        logit_derivative = np.column_stack(
            (
                -np.ones(self.num_states),
                self.scale * state,
            )
        ) - self.beta * value_derivative
        score_weight = self.replacement_counts - (
            self.keep_counts + self.replacement_counts
        ) * replacement_probability
        gradient = -(score_weight[:, None] * logit_derivative).sum(axis=0)
        objective = -choice_log_likelihood(
            replacement_probability, self.keep_counts, self.replacement_counts
        )
        self._last_parameters = theta.copy()
        self._last_value = value.copy()
        self._last_output = (float(objective), gradient.copy(), solution.residual)
        return float(objective), gradient


def estimate_nfxp(
    keep_counts: np.ndarray,
    replacement_counts: np.ndarray,
    probabilities: Sequence[float],
    *,
    num_states: int,
    beta: float,
    scale: float,
    start: Sequence[float] = (10.0, 2.3),
    bounds: Sequence[tuple[float, float]] = PRIOR_BOUNDS,
) -> NFXPResult:
    """Estimate replacement and maintenance costs by NFXP maximum likelihood."""

    from scipy.optimize import minimize

    objective = NFXPObjective(
        keep_counts,
        replacement_counts,
        probabilities,
        num_states=num_states,
        beta=beta,
        scale=scale,
    )
    started = time.perf_counter()
    result = minimize(
        objective.value_and_gradient,
        np.asarray(start, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=list(bounds),
        options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 300, "maxls": 30},
    )
    elapsed = time.perf_counter() - started
    final_objective, _ = objective.value_and_gradient(result.x)
    assert objective._last_output is not None
    return NFXPResult(
        replacement_cost=float(result.x[0]),
        maintenance_slope=float(result.x[1]),
        negative_log_likelihood=float(final_objective),
        success=bool(result.success),
        message=str(result.message),
        optimizer_iterations=int(result.nit),
        objective_evaluations=int(result.nfev),
        bellman_residual=float(objective._last_output[2]),
        seconds=float(elapsed),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    panel = read_processed_csv(args.data)
    keep, replacement = panel.choice_counts(NUM_STATES)
    result = estimate_nfxp(
        keep,
        replacement,
        panel.transition_probabilities,
        num_states=NUM_STATES,
        beta=DISCOUNT_FACTOR,
        scale=COST_SCALE,
    )
    payload = {
        **asdict(result),
        "num_states": NUM_STATES,
        "discount_factor": DISCOUNT_FACTOR,
        "cost_scale": COST_SCALE,
        "transition_counts": panel.transition_counts.tolist(),
        "transition_probabilities": panel.transition_probabilities.tolist(),
        "mpec": {"supported": False, "reason": "This independent replication implements NFXP only."},
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

