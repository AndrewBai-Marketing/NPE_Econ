"""Exact-grid validation on a self-contained dynamic replacement model.

This benchmark has no empirical inputs and imports no research model.  A
decision maker observes a finite deterioration state and chooses keep or
replace. Type-I extreme-value shocks give logit conditional choice
probabilities after solving the infinite-horizon Bellman equation. The prior is
uniform on a declared 7 x 9 parameter grid, making the posterior over that grid
exactly enumerable from sufficient state-action counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from structnpe import ArrayAdapter, ParameterSpec, StructuralModel, fit


S_MAX = 4
N_AGENTS = 12
N_PERIODS = 10
DISCOUNT = 0.9
VALUE_TOLERANCE = 1.0e-11
VALUE_MAX_ITERATIONS = 2_000
POLICY_SUBSIDY = 0.5
MAINTENANCE_GRID = np.linspace(0.1, 0.7, 7)
REPLACEMENT_GRID = np.linspace(1.25, 5.75, 9)
THETA_GRID = np.array(
    [(maintenance, replacement) for maintenance in MAINTENANCE_GRID for replacement in REPLACEMENT_GRID],
    dtype=float,
)
PARAMETER_RANGES = np.array(
    [MAINTENANCE_GRID[-1] - MAINTENANCE_GRID[0], REPLACEMENT_GRID[-1] - REPLACEMENT_GRID[0]],
    dtype=float,
)
ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_PATH = Path(__file__).with_name("thresholds.json")


def grid_prior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw uniformly from the declared finite parameter prior."""

    indices = rng.integers(0, len(THETA_GRID), size=int(n))
    return THETA_GRID[indices].copy()


def _logsumexp_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    maximum = np.maximum(a, b)
    return maximum + np.log(np.exp(a - maximum) + np.exp(b - maximum))


def choice_probabilities_batch(theta: np.ndarray, subsidy: float = 0.0) -> np.ndarray:
    """Solve conditional choice probabilities for one or many parameters."""

    parameters = np.asarray(theta, dtype=float)
    if parameters.ndim == 1:
        parameters = parameters.reshape(1, -1)
    if parameters.ndim != 2 or parameters.shape[1] != 2:
        raise ValueError("theta must have shape (n, 2)")

    maintenance = parameters[:, 0]
    replacement = parameters[:, 1] - float(subsidy)
    states = np.arange(S_MAX + 1, dtype=float)
    next_keep = np.minimum(np.arange(S_MAX + 1) + 1, S_MAX)
    value = np.zeros((parameters.shape[0], S_MAX + 1), dtype=float)
    for _ in range(VALUE_MAX_ITERATIONS):
        keep = -maintenance[:, None] * states[None, :] + DISCOUNT * value[:, next_keep]
        replace = -replacement[:, None] + DISCOUNT * value[:, [0]]
        updated = _logsumexp_pair(keep, replace)
        if float(np.max(np.abs(updated - value))) <= VALUE_TOLERANCE:
            value = updated
            break
        value = updated
    else:
        raise RuntimeError("replacement-model value iteration did not converge")

    keep = -maintenance[:, None] * states[None, :] + DISCOUNT * value[:, next_keep]
    replace = -replacement[:, None] + DISCOUNT * value[:, [0]]
    denominator = _logsumexp_pair(keep, replace)
    probabilities = np.stack([np.exp(keep - denominator), np.exp(replace - denominator)], axis=2)
    return probabilities / probabilities.sum(axis=2, keepdims=True)


@lru_cache(maxsize=256)
def _cached_choice_probabilities(maintenance: float, replacement: float) -> np.ndarray:
    return choice_probabilities_batch(np.array([maintenance, replacement]))[0]


def replacement_simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return sufficient state-action counts for one simulated panel."""

    maintenance, replacement = np.asarray(theta, dtype=float).reshape(-1)
    probabilities = _cached_choice_probabilities(float(maintenance), float(replacement))
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


def exact_grid_posterior(counts: np.ndarray) -> np.ndarray:
    """Enumerate exact posterior weights under the uniform grid prior.

    Initial states are fixed and transitions are deterministic conditional on
    the observed actions. Every action history with the same state-action
    counts therefore has the same product of CCPs; the count-event
    multiplicity is constant in theta. The counts are consequently sufficient
    for this deliberately small model.
    """

    count_matrix = np.asarray(counts, dtype=float).reshape(S_MAX + 1, 2)
    probabilities = choice_probabilities_batch(THETA_GRID)
    if not np.all(np.isfinite(probabilities)) or not np.all(probabilities > 0.0):
        raise RuntimeError("finite-grid CCPs must be finite and strictly positive")
    log_likelihood = np.sum(count_matrix[None, :, :] * np.log(probabilities), axis=(1, 2))
    weights = np.exp(log_likelihood - np.max(log_likelihood))
    return weights / weights.sum()


def expected_state_action_frequencies(theta: np.ndarray, subsidy: float = 0.0) -> np.ndarray:
    """Return expected panel state-action frequencies for each theta row."""

    parameters = np.asarray(theta, dtype=float)
    if parameters.ndim == 1:
        parameters = parameters.reshape(1, -1)
    probabilities = choice_probabilities_batch(parameters, subsidy=subsidy)
    distribution = np.zeros((parameters.shape[0], S_MAX + 1), dtype=float)
    distribution[:, 0] = 1.0
    counts = np.zeros((parameters.shape[0], S_MAX + 1, 2), dtype=float)
    for _ in range(N_PERIODS):
        action_mass = distribution[:, :, None] * probabilities
        counts += action_mass
        next_distribution = np.zeros_like(distribution)
        next_distribution[:, 0] += action_mass[:, :, 1].sum(axis=1)
        for state in range(S_MAX + 1):
            next_distribution[:, min(state + 1, S_MAX)] += action_mass[:, state, 0]
        distribution = next_distribution
    return counts.reshape(parameters.shape[0], -1) / N_PERIODS


def policy_replacement_share(theta: np.ndarray) -> np.ndarray:
    """Expected replacement share under the declared replacement subsidy."""

    frequencies = expected_state_action_frequencies(theta, subsidy=POLICY_SUBSIDY)
    return frequencies.reshape(-1, S_MAX + 1, 2)[:, :, 1].sum(axis=1)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(sorted_weights) / sorted_weights.sum()
    return sorted_values[np.minimum(np.searchsorted(cumulative, probabilities, side="left"), len(values) - 1)]


def _weighted_covariance(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(weights, dtype=float) @ np.asarray(values, dtype=float)
    centered = np.asarray(values, dtype=float) - mean[None, :]
    covariance = (centered * weights[:, None]).T @ centered
    return mean, covariance


def _correlation(covariance: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.maximum(np.diag(covariance), 1.0e-300))
    return covariance / np.outer(sd, sd)


def _evaluate(draws: np.ndarray, weights: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    samples = np.asarray(draws, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"inference draws must have shape (draws, 2), got {samples.shape}")

    exact_mean, exact_covariance = _weighted_covariance(THETA_GRID, weights)
    exact_sd = np.sqrt(np.diag(exact_covariance))
    approx_mean = samples.mean(axis=0)
    approx_covariance = np.cov(samples, rowvar=False, ddof=1)
    approx_sd = np.sqrt(np.diag(approx_covariance))
    exact_correlation = _correlation(exact_covariance)
    approx_correlation = _correlation(approx_covariance)

    probabilities = np.array([0.025, 0.975])
    exact_interval = np.column_stack(
        [_weighted_quantile(THETA_GRID[:, dim], weights, probabilities) for dim in range(2)]
    ).T
    uniform_draw_weights = np.ones(samples.shape[0], dtype=float)
    approx_interval = np.column_stack(
        [
            _weighted_quantile(samples[:, dim], uniform_draw_weights, probabilities)
            for dim in range(2)
        ]
    ).T

    quantile_grid = (np.arange(samples.shape[0], dtype=float) + 0.5) / samples.shape[0]
    marginal_quantile_grid_l1 = []
    for dim in range(2):
        exact_quantiles = _weighted_quantile(THETA_GRID[:, dim], weights, quantile_grid)
        marginal_quantile_grid_l1.append(
            float(np.mean(np.abs(np.sort(samples[:, dim]) - exact_quantiles)))
        )

    maintenance_cell = np.argmin(
        np.abs(samples[:, [0]] - MAINTENANCE_GRID[None, :]), axis=1
    )
    replacement_cell = np.argmin(
        np.abs(samples[:, [1]] - REPLACEMENT_GRID[None, :]), axis=1
    )
    joint_cell = maintenance_cell * len(REPLACEMENT_GRID) + replacement_cell
    approximate_grid_mass = np.bincount(joint_cell, minlength=len(THETA_GRID)).astype(float)
    approximate_grid_mass /= approximate_grid_mass.sum()
    joint_grid_total_variation = 0.5 * float(
        np.sum(np.abs(approximate_grid_mass - weights))
    )

    exact_policy = policy_replacement_share(THETA_GRID)
    approx_policy = policy_replacement_share(samples)
    exact_policy_mean = float(weights @ exact_policy)
    exact_policy_sd = float(np.sqrt(weights @ ((exact_policy - exact_policy_mean) ** 2)))
    exact_policy_interval = _weighted_quantile(exact_policy, weights, probabilities)
    approx_policy_interval = _weighted_quantile(
        approx_policy, uniform_draw_weights, probabilities
    )

    exact_predictive = weights @ expected_state_action_frequencies(THETA_GRID)
    approx_predictive = expected_state_action_frequencies(samples).mean(axis=0)
    support_ok = (
        (samples[:, 0] > 0.05)
        & (samples[:, 0] < 0.75)
        & (samples[:, 1] > 1.0)
        & (samples[:, 1] < 6.0)
    )

    covariance_scale = np.outer(PARAMETER_RANGES, PARAMETER_RANGES)
    metrics = {
        "parameter_mean_normalized_mae": float(np.mean(np.abs(approx_mean - exact_mean) / PARAMETER_RANGES)),
        "parameter_sd_normalized_mae": float(np.mean(np.abs(approx_sd - exact_sd) / PARAMETER_RANGES)),
        "central_95_endpoint_normalized_mae": float(
            np.mean(np.abs(approx_interval - exact_interval) / PARAMETER_RANGES[:, None])
        ),
        "covariance_normalized_max_abs_error": float(
            np.max(np.abs(approx_covariance - exact_covariance) / covariance_scale)
        ),
        "correlation_max_abs_error": float(np.max(np.abs(approx_correlation - exact_correlation))),
        "marginal_quantile_grid_l1_normalized_mean": float(
            np.mean(np.asarray(marginal_quantile_grid_l1) / PARAMETER_RANGES)
        ),
        "joint_grid_voronoi_total_variation": joint_grid_total_variation,
        "policy_mean_abs_error": abs(float(approx_policy.mean()) - exact_policy_mean),
        "policy_sd_abs_error": abs(float(approx_policy.std(ddof=1)) - exact_policy_sd),
        "policy_95_endpoint_mae": float(np.mean(np.abs(approx_policy_interval - exact_policy_interval))),
        "predictive_cell_probability_max_abs_error": float(np.max(np.abs(approx_predictive - exact_predictive))),
        "parameter_box_support_violation_rate": float(1.0 - np.mean(support_ok)),
    }
    parameter_names = ("maintenance_cost", "replacement_cost")
    comparison = {
        "parameters": [
            {
                "parameter": parameter_names[dim],
                "exact_mean": float(exact_mean[dim]),
                "approximate_mean": float(approx_mean[dim]),
                "exact_sd": float(exact_sd[dim]),
                "approximate_sd": float(approx_sd[dim]),
                "exact_q025": float(exact_interval[dim, 0]),
                "approximate_q025": float(approx_interval[dim, 0]),
                "exact_q975": float(exact_interval[dim, 1]),
                "approximate_q975": float(approx_interval[dim, 1]),
                "quantile_grid_l1": float(marginal_quantile_grid_l1[dim]),
            }
            for dim in range(2)
        ],
        "covariance": {
            "exact": exact_covariance.tolist(),
            "approximate": approx_covariance.tolist(),
        },
        "correlation": {
            "exact": exact_correlation.tolist(),
            "approximate": approx_correlation.tolist(),
        },
        "joint_grid_voronoi_projection": [
            {
                "maintenance_cost": float(THETA_GRID[index, 0]),
                "replacement_cost": float(THETA_GRID[index, 1]),
                "exact_probability": float(weights[index]),
                "approximate_probability": float(approximate_grid_mass[index]),
            }
            for index in range(len(THETA_GRID))
        ],
        "policy_replacement_share": {
            "subsidy": POLICY_SUBSIDY,
            "exact_mean": exact_policy_mean,
            "approximate_mean": float(approx_policy.mean()),
            "exact_sd": exact_policy_sd,
            "approximate_sd": float(approx_policy.std(ddof=1)),
            "exact_q025": float(exact_policy_interval[0]),
            "approximate_q025": float(approx_policy_interval[0]),
            "exact_q975": float(exact_policy_interval[1]),
            "approximate_q975": float(approx_policy_interval[1]),
        },
        "posterior_predictive_expected_state_action_frequencies": [
            {
                "state": int(index // 2),
                "action": "keep" if index % 2 == 0 else "replace",
                "exact": float(exact_predictive[index]),
                "approximate": float(approx_predictive[index]),
            }
            for index in range(len(exact_predictive))
        ],
    }
    return metrics, comparison


def _package_version() -> str:
    try:
        return version("structnpe")
    except PackageNotFoundError:
        return "source-checkout"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _load_profile(name: str) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    raw = THRESHOLD_PATH.read_bytes()
    payload = json.loads(raw)
    if not payload.get("declared_before_execution"):
        raise RuntimeError("threshold file is not marked as predeclared")
    try:
        profile = payload["profiles"][name]
    except KeyError as exc:
        raise ValueError(f"unknown validation profile {name!r}") from exc
    return (
        dict(profile["configuration"]),
        dict(profile["acceptance_maxima"]),
        {
            "threshold_file": str(THRESHOLD_PATH.relative_to(ROOT)),
            "threshold_sha256": hashlib.sha256(raw).hexdigest(),
            "threshold_schema_version": payload["schema_version"],
            "benchmark": payload["benchmark"],
            "target": payload["target"],
            "decision_rule": payload["decision_rule"],
        },
    )


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Structural exact-grid validation",
        "",
        f"Status: **{payload['status']}**",
        "",
        "The comparator is exact for the declared finite-grid prior and sufficient",
        "state-action-count observation. Acceptance applies only to the one",
        "predeclared midpoint-truth panel generated with the profile's fixed seed;",
        "it is not evidence of amortized accuracy across observations, identification,",
        "or specification validity for other structural models.",
        "",
        "| metric | value | maximum | pass |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, check in payload["checks"].items():
        lines.append(
            f"| {name} | {check['value']:.8g} | {check['maximum']:.8g} | {check['passed']} |"
        )
    lines.extend(
        [
            "",
            f"Threshold SHA-256: `{payload['thresholds']['threshold_sha256']}`",
            "",
            "Metrics are generated only when this script is explicitly run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, maxima, threshold_metadata = _load_profile(args.profile)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()

    training_start = time.perf_counter()
    estimator = fit(
        model,
        simulations=config["simulations"],
        seed=config["training_seed"],
        validation_fraction=config["validation_fraction"],
        hidden_dim=config["hidden_dim"],
        depth=config["depth"],
        components=config["components"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        patience=config["patience"],
        device="cpu",
        output_dir=args.output_dir / "estimator",
        progress=not args.quiet,
    )
    training_seconds = time.perf_counter() - training_start

    true_theta = THETA_GRID[len(THETA_GRID) // 2]
    observed = replacement_simulator(true_theta, np.random.default_rng(config["observed_seed"]))
    exact_weights = exact_grid_posterior(observed)
    inference_start = time.perf_counter()
    result = estimator.infer(
        observed,
        draws=config["posterior_draws"],
        seed=config["inference_seed"],
    )
    inference_seconds = time.perf_counter() - inference_start
    metrics, comparison = _evaluate(np.asarray(result.draws), exact_weights)

    checks = {
        name: {
            "value": metrics[name],
            "maximum": float(maximum),
            "passed": bool(np.isfinite(metrics[name]) and metrics[name] <= float(maximum)),
        }
        for name, maximum in maxima.items()
    }
    passed = all(check["passed"] for check in checks.values())
    payload = {
        "schema_version": 1,
        "benchmark": threshold_metadata["benchmark"],
        "profile": args.profile,
        "status": "PASS" if passed else "FAIL",
        "configuration": config,
        "structural_configuration": {
            "state_count": S_MAX + 1,
            "action_count": 2,
            "agents": N_AGENTS,
            "periods": N_PERIODS,
            "discount": DISCOUNT,
            "parameter_grid_cells": len(THETA_GRID),
            "true_theta": true_theta.tolist(),
            "policy_replacement_subsidy": POLICY_SUBSIDY,
        },
        "thresholds": threshold_metadata,
        "metrics": metrics,
        "comparison": comparison,
        "checks": checks,
        "timing_seconds": {"training": training_seconds, "inference": inference_seconds},
        "environment": {
            "structnpe_version": _package_version(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "processor": platform.processor() or "unknown",
        },
        "estimator": {
            "model_fingerprint": estimator.model_fingerprint,
            "training_metadata": _jsonable(estimator.training_metadata),
        },
        "nonclaim": (
            "This comparison checks one predeclared midpoint-truth synthetic panel. "
            "It does not establish amortized accuracy across observations or transfer "
            "to another structural model."
        ),
    }
    (args.output_dir / "validation_metrics.json").write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir / "report.md", payload)
    print(json.dumps({"status": payload["status"], "output_dir": str(args.output_dir)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
