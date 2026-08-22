"""Independent NumPy/SciPy implementation of the canonical Rust model.

The code implements a stationary two-action dynamic logit model.  Choosing
maintenance incurs ``scale * slope * state``; choosing replacement incurs
``replacement_cost``.  Mileage increments follow a finite discrete
distribution and are pooled at the final state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def transition_matrix(num_states: int, probabilities: Sequence[float]) -> np.ndarray:
    """Return the capped mileage transition matrix."""

    states = int(num_states)
    probs = np.asarray(probabilities, dtype=np.float64)
    if states < 2:
        raise ValueError("num_states must be at least two.")
    if probs.ndim != 1 or probs.size < 1 or np.any(probs < 0) or not np.all(np.isfinite(probs)):
        raise ValueError("Transition probabilities must be a finite nonnegative vector.")
    if not np.isclose(probs.sum(), 1.0, atol=1e-12, rtol=0.0):
        raise ValueError("Transition probabilities must sum to one.")
    matrix = np.zeros((states, states), dtype=np.float64)
    for current in range(states):
        for increment, probability in enumerate(probs):
            matrix[current, min(current + increment, states - 1)] += probability
    return matrix


def _utilities(
    relative_value: np.ndarray,
    replacement_cost: float,
    maintenance_slope: float,
    *,
    beta: float,
    scale: float,
) -> tuple[np.ndarray, float]:
    state = np.arange(relative_value.size, dtype=np.float64)
    maintain = beta * relative_value - scale * maintenance_slope * state
    replace = -replacement_cost + beta * relative_value[0]
    return maintain, float(replace)


@dataclass(frozen=True)
class BellmanSolution:
    relative_value: np.ndarray
    replacement_probability: np.ndarray
    residual: float
    iterations: int


def solve_bellman(
    replacement_cost: float,
    maintenance_slope: float,
    probabilities: Sequence[float],
    *,
    num_states: int,
    beta: float,
    scale: float,
    tolerance: float = 1e-12,
    max_iterations: int = 50,
    initial_value: np.ndarray | None = None,
) -> BellmanSolution:
    """Solve a normalized Bellman equation by damped Newton iteration.

    The integrated value is normalized to zero at state zero.  This removes the
    poorly identified additive level when beta is close to one while preserving
    exactly the choice probabilities.
    """

    from scipy.special import expit

    if not (0.0 <= beta < 1.0) or scale <= 0:
        raise ValueError("beta must lie in [0, 1) and scale must be positive.")
    if replacement_cost <= 0 or maintenance_slope <= 0:
        raise ValueError("Cost parameters must be positive.")
    matrix = transition_matrix(num_states, probabilities)
    if initial_value is None:
        value = np.zeros(int(num_states), dtype=np.float64)
    else:
        value = np.asarray(initial_value, dtype=np.float64).copy()
        if value.shape != (int(num_states),) or not np.all(np.isfinite(value)):
            raise ValueError("initial_value has the wrong shape or non-finite values.")
        value -= value[0]
    residual_norm = np.inf
    for iteration in range(1, int(max_iterations) + 1):
        maintain, replace = _utilities(
            value, replacement_cost, maintenance_slope, beta=beta, scale=scale
        )
        integrated = np.logaddexp(maintain, replace)
        mapped = matrix @ integrated
        residual = value[1:] - (mapped[1:] - mapped[0])
        residual_norm = float(np.max(np.abs(residual)))
        if residual_norm <= tolerance:
            break
        keep_probability = expit(maintain - replace)
        jacobian = np.eye(num_states - 1) - beta * (
            matrix[1:, 1:] - matrix[0:1, 1:]
        ) * keep_probability[None, 1:]
        step = np.linalg.solve(jacobian, residual)
        # A short backtracking line search makes the solver reliable over the
        # declared broad prior without changing the fixed point.
        accepted = False
        damping = 1.0
        for _ in range(12):
            candidate = value.copy()
            candidate[1:] -= damping * step
            c_maintain, c_replace = _utilities(
                candidate, replacement_cost, maintenance_slope, beta=beta, scale=scale
            )
            c_integrated = np.logaddexp(c_maintain, c_replace)
            c_mapped = matrix @ c_integrated
            candidate_residual = candidate[1:] - (c_mapped[1:] - c_mapped[0])
            if float(np.max(np.abs(candidate_residual))) < residual_norm:
                value = candidate
                accepted = True
                break
            damping *= 0.5
        if not accepted:
            # The relative contraction is a safe fallback step.
            value = mapped - mapped[0]
    else:
        raise RuntimeError(
            f"Bellman solver did not converge after {max_iterations} iterations "
            f"(residual={residual_norm:.3e})."
        )
    maintain, replace = _utilities(
        value, replacement_cost, maintenance_slope, beta=beta, scale=scale
    )
    probability = expit(replace - maintain)
    return BellmanSolution(value, probability, residual_norm, iteration)


def batched_replacement_probabilities(
    parameters: np.ndarray,
    probabilities: Sequence[float],
    *,
    num_states: int,
    beta: float,
    scale: float,
    tolerance: float = 1e-9,
    max_iterations: int = 50,
    chunk_size: int = 128,
) -> np.ndarray:
    """Vectorized normalized Newton solver for simulator/grid batches."""

    from scipy.special import expit

    theta = np.asarray(parameters, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[1] != 2 or len(theta) < 1:
        raise ValueError("parameters must have shape (n, 2).")
    if not np.all(np.isfinite(theta)) or np.any(theta <= 0):
        raise ValueError("Cost parameters must be finite and positive.")
    matrix = transition_matrix(num_states, probabilities)
    transition_difference = matrix[1:, 1:] - matrix[0:1, 1:]
    state = np.arange(num_states, dtype=np.float64)
    output = np.empty((len(theta), num_states), dtype=np.float64)
    for start in range(0, len(theta), int(chunk_size)):
        stop = min(start + int(chunk_size), len(theta))
        block = theta[start:stop]
        value = np.zeros((len(block), num_states), dtype=np.float64)
        error = np.inf
        for _ in range(int(max_iterations)):
            maintain = beta * value - scale * block[:, 1:2] * state[None, :]
            replace = -block[:, 0:1]
            integrated = np.logaddexp(maintain, replace)
            mapped = integrated @ matrix.T
            residual = value[:, 1:] - (mapped[:, 1:] - mapped[:, 0:1])
            error = float(np.max(np.abs(residual)))
            if error <= tolerance:
                break
            keep_probability = expit(maintain - replace)
            jacobian = np.eye(num_states - 1)[None, :, :] - beta * (
                transition_difference[None, :, :] * keep_probability[:, None, 1:]
            )
            step = np.linalg.solve(jacobian, residual[..., None])[..., 0]
            # One shared backtracking coefficient keeps the batched operation
            # deterministic and avoids a Python loop over simulations.
            damping = 1.0
            accepted = False
            for _ in range(12):
                candidate = value.copy()
                candidate[:, 1:] -= damping * step
                c_maintain = beta * candidate - scale * block[:, 1:2] * state[None, :]
                c_integrated = np.logaddexp(c_maintain, replace)
                c_mapped = c_integrated @ matrix.T
                c_residual = candidate[:, 1:] - (c_mapped[:, 1:] - c_mapped[:, 0:1])
                if float(np.max(np.abs(c_residual))) < error:
                    value = candidate
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                value = mapped - mapped[:, 0:1]
        else:
            raise RuntimeError(
                f"Batched Bellman solver did not converge after {max_iterations} iterations "
                f"(residual={error:.3e})."
            )
        maintain = beta * value - scale * block[:, 1:2] * state[None, :]
        output[start:stop] = expit(-block[:, 0:1] - maintain)
    return output


def choice_log_likelihood(
    replacement_probability: np.ndarray,
    keep_counts: np.ndarray,
    replacement_counts: np.ndarray,
) -> float:
    probability = np.clip(np.asarray(replacement_probability, dtype=float), 1e-300, 1 - 1e-15)
    keep = np.asarray(keep_counts, dtype=float)
    replace = np.asarray(replacement_counts, dtype=float)
    if probability.shape != keep.shape or probability.shape != replace.shape:
        raise ValueError("Probability and count arrays must have the same shape.")
    return float(np.sum(replace * np.log(probability) + keep * np.log1p(-probability)))


def simulate_choice_summary_batch(
    parameters: np.ndarray,
    rng: np.random.Generator,
    probabilities: Sequence[float],
    *,
    num_states: int,
    beta: float,
    scale: float,
    buses: int,
    periods: int,
) -> np.ndarray:
    """Simulate panels and return state visits plus replacement counts.

    Period zero initializes the Markov panel and is excluded from the summary,
    matching the empirical likelihood convention.
    """

    theta = np.asarray(parameters, dtype=np.float64)
    if buses < 1 or periods < 2:
        raise ValueError("Simulation requires positive buses and at least two periods.")
    policy = batched_replacement_probabilities(
        theta,
        probabilities,
        num_states=num_states,
        beta=beta,
        scale=scale,
    )
    simulations = len(theta)
    state = np.zeros((simulations, int(buses)), dtype=np.int64)
    visits = np.zeros((simulations, num_states), dtype=np.float64)
    replacements = np.zeros_like(visits)
    probs = np.asarray(probabilities, dtype=float)
    simulation_index = np.repeat(np.arange(simulations), int(buses))
    for period in range(int(periods)):
        probability = policy[np.arange(simulations)[:, None], state]
        decision = rng.random(state.shape) < probability
        if period > 0:
            flat_state = state.ravel()
            np.add.at(visits, (simulation_index, flat_state), 1.0)
            np.add.at(replacements, (simulation_index, flat_state), decision.ravel().astype(float))
        increments = rng.choice(len(probs), size=state.shape, p=probs)
        state = np.minimum(np.where(decision, increments, state + increments), num_states - 1)
    return np.concatenate((visits, replacements), axis=1)


def expected_replacements(
    replacement_probability: np.ndarray,
    probabilities: Sequence[float],
    *,
    periods: int,
    buses: int,
    initial_distribution: np.ndarray | None = None,
) -> float:
    """Compute expected finite-horizon replacements under a policy curve."""

    policy = np.asarray(replacement_probability, dtype=float)
    if policy.ndim != 1 or np.any((policy < 0) | (policy > 1)):
        raise ValueError("replacement_probability must be a one-dimensional probability vector.")
    transition = transition_matrix(len(policy), probabilities)
    reset = transition[0]
    if initial_distribution is None:
        distribution = np.zeros(len(policy), dtype=float)
        distribution[0] = 1.0
    else:
        distribution = np.asarray(initial_distribution, dtype=float).copy()
        if distribution.shape != policy.shape or np.any(distribution < 0):
            raise ValueError("initial_distribution is invalid.")
        distribution /= distribution.sum()
    expected = 0.0
    for _ in range(int(periods)):
        expected += float(distribution @ policy) * int(buses)
        keep_mass = distribution * (1.0 - policy)
        replacement_mass = float(distribution @ policy)
        distribution = keep_mass @ transition + replacement_mass * reset
    return expected


def policy_curves(
    parameters: np.ndarray,
    probabilities: Sequence[float],
    *,
    num_states: int,
    beta: float,
    scale: float,
) -> np.ndarray:
    return batched_replacement_probabilities(
        parameters,
        probabilities,
        num_states=num_states,
        beta=beta,
        scale=scale,
    )
