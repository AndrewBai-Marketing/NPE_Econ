"""Dense-grid Bayesian reference under the exact Rust choice likelihood."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .constants import COST_SCALE, DISCOUNT_FACTOR, NUM_STATES, PRIOR_BOUNDS
    from .model import expected_replacements, solve_bellman
    from .preprocess import read_processed_csv
except ImportError:  # pragma: no cover - direct script execution
    from constants import COST_SCALE, DISCOUNT_FACTOR, NUM_STATES, PRIOR_BOUNDS
    from model import expected_replacements, solve_bellman
    from preprocess import read_processed_csv


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(np.interp(float(quantile), cumulative, sorted_values))


@dataclass(frozen=True)
class GridPosterior:
    parameters: np.ndarray
    weights: np.ndarray
    log_likelihood: np.ndarray
    policies: np.ndarray
    rc_axis: np.ndarray
    slope_axis: np.ndarray
    seconds: float

    def draw(self, draws: int, rng: np.random.Generator) -> np.ndarray:
        index = rng.choice(len(self.weights), size=int(draws), p=self.weights)
        return self.parameters[index]

    def parameter_summary(self) -> list[dict[str, float | str]]:
        output: list[dict[str, float | str]] = []
        for column, name in enumerate(("replacement_cost", "maintenance_slope")):
            values = self.parameters[:, column]
            mean = float(self.weights @ values)
            variance = float(self.weights @ ((values - mean) ** 2))
            output.append(
                {
                    "parameter": name,
                    "mean": mean,
                    "sd": float(np.sqrt(variance)),
                    "median": _weighted_quantile(values, self.weights, 0.5),
                    "q025": _weighted_quantile(values, self.weights, 0.025),
                    "q975": _weighted_quantile(values, self.weights, 0.975),
                }
            )
        return output

    def policy_summary(self) -> dict[str, list[float]]:
        means = self.weights @ self.policies
        lower = np.empty(self.policies.shape[1])
        upper = np.empty_like(lower)
        for state in range(self.policies.shape[1]):
            lower[state] = _weighted_quantile(self.policies[:, state], self.weights, 0.025)
            upper[state] = _weighted_quantile(self.policies[:, state], self.weights, 0.975)
        return {"mean": means.tolist(), "q025": lower.tolist(), "q975": upper.tolist()}


def grid_posterior(
    keep_counts: np.ndarray,
    replacement_counts: np.ndarray,
    probabilities: Sequence[float],
    *,
    num_states: int,
    beta: float,
    scale: float,
    rc_bounds: tuple[float, float] = PRIOR_BOUNDS[0],
    slope_bounds: tuple[float, float] = PRIOR_BOUNDS[1],
    rc_points: int = 141,
    slope_points: int = 117,
    bellman_tolerance: float = 1e-9,
) -> GridPosterior:
    """Evaluate a uniform-prior posterior by dense trapezoidal quadrature."""

    if rc_points < 3 or slope_points < 3:
        raise ValueError("Each dense-grid axis requires at least three points.")
    keep = np.asarray(keep_counts, dtype=float)
    replace = np.asarray(replacement_counts, dtype=float)
    if keep.shape != (num_states,) or replace.shape != (num_states,):
        raise ValueError("Choice counts do not match num_states.")
    rc_axis = np.linspace(*rc_bounds, int(rc_points))
    slope_axis = np.linspace(*slope_bounds, int(slope_points))
    rc_mesh, slope_mesh = np.meshgrid(rc_axis, slope_axis, indexing="ij")
    parameters = np.column_stack((rc_mesh.ravel(), slope_mesh.ravel()))
    started = time.perf_counter()
    # The Cartesian traversal warm-starts every exact normalized Newton solve
    # from its neighboring grid point.  This is considerably faster here than
    # a beta-near-one contraction over the whole grid.
    policies = np.empty((len(parameters), num_states), dtype=float)
    previous_value: np.ndarray | None = None
    for index, parameter in enumerate(parameters):
        solution = solve_bellman(
            parameter[0],
            parameter[1],
            probabilities,
            num_states=num_states,
            beta=beta,
            scale=scale,
            tolerance=bellman_tolerance,
            initial_value=previous_value,
        )
        policies[index] = solution.replacement_probability
        previous_value = solution.relative_value
    clipped = np.clip(policies, 1e-300, 1 - 1e-15)
    log_likelihood = (
        replace[None, :] * np.log(clipped) + keep[None, :] * np.log1p(-clipped)
    ).sum(axis=1)
    rc_quadrature = np.ones(rc_points)
    rc_quadrature[[0, -1]] = 0.5
    slope_quadrature = np.ones(slope_points)
    slope_quadrature[[0, -1]] = 0.5
    log_weight = log_likelihood + np.log(
        np.multiply.outer(rc_quadrature, slope_quadrature).ravel()
    )
    log_weight -= np.max(log_weight)
    weights = np.exp(log_weight)
    weights /= weights.sum()
    return GridPosterior(
        parameters=parameters,
        weights=weights,
        log_likelihood=log_likelihood,
        policies=policies,
        rc_axis=rc_axis,
        slope_axis=slope_axis,
        seconds=float(time.perf_counter() - started),
    )


def save_grid(reference: GridPosterior, path: str | Path) -> tuple[Path, Path]:
    """Save flat numeric arrays plus a human-readable JSON summary (no pickle)."""

    target = Path(path)
    if target.suffix != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        parameters=reference.parameters,
        weights=reference.weights,
        log_likelihood=reference.log_likelihood,
        policies=reference.policies,
        rc_axis=reference.rc_axis,
        slope_axis=reference.slope_axis,
    )
    summary_path = target.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "dense Cartesian trapezoidal quadrature",
                "prior": {
                    "replacement_cost": [float(reference.rc_axis[0]), float(reference.rc_axis[-1])],
                    "maintenance_slope": [
                        float(reference.slope_axis[0]),
                        float(reference.slope_axis[-1]),
                    ],
                    "density": "independent continuous uniform",
                },
                "grid_shape": [len(reference.rc_axis), len(reference.slope_axis)],
                "grid_points": len(reference.weights),
                "seconds": reference.seconds,
                "parameter_summary": reference.parameter_summary(),
                "policy_summary": reference.policy_summary(),
                "nonclaim": (
                    "This is a deterministic grid approximation to the posterior under the stated "
                    "prior and likelihood, not Rust's maximum-likelihood result."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return target, summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rc-points", type=int, default=141)
    parser.add_argument("--slope-points", type=int, default=117)
    args = parser.parse_args()
    panel = read_processed_csv(args.data)
    keep, replacement = panel.choice_counts(NUM_STATES)
    reference = grid_posterior(
        keep,
        replacement,
        panel.transition_probabilities,
        num_states=NUM_STATES,
        beta=DISCOUNT_FACTOR,
        scale=COST_SCALE,
        rc_points=args.rc_points,
        slope_points=args.slope_points,
    )
    npz, summary = save_grid(reference, args.output)
    print(json.dumps({"grid": str(npz), "summary": str(summary), "seconds": reference.seconds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
