"""Compare a trained MDN with the empirical dense-grid posterior and policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from structnpe import load_estimator

try:
    from .constants import (
        BUS_COUNT,
        CANONICAL_TRANSITION_PROBABILITIES,
        COST_SCALE,
        DISCOUNT_FACTOR,
        NUM_STATES,
    )
    from .model import expected_replacements, policy_curves
    from .preprocess import read_processed_csv
    from .structnpe_model import build_model
except ImportError:  # pragma: no cover - direct script execution
    from constants import (
        BUS_COUNT,
        CANONICAL_TRANSITION_PROBABILITIES,
        COST_SCALE,
        DISCOUNT_FACTOR,
        NUM_STATES,
    )
    from model import expected_replacements, policy_curves
    from preprocess import read_processed_csv
    from structnpe_model import build_model


def _weighted_moments(parameters: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = weights @ parameters
    centered = parameters - mean
    covariance = (centered * weights[:, None]).T @ centered
    return mean, covariance


def _weighted_cdf_distance(
    draws: np.ndarray, grid_values: np.ndarray, weights: np.ndarray
) -> float:
    order = np.argsort(grid_values)
    values = grid_values[order]
    cdf = np.cumsum(weights[order])
    empirical = np.searchsorted(np.sort(draws), values, side="right") / len(draws)
    return float(np.max(np.abs(empirical - cdf)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(quantile, cumulative, values[order]))


def _joint_coarsened_tv(
    draws: np.ndarray,
    parameters: np.ndarray,
    weights: np.ndarray,
    *,
    bounds: list[list[float]],
    bins: list[int],
) -> float:
    edges = [np.linspace(bound[0], bound[1], count + 1) for bound, count in zip(bounds, bins, strict=True)]
    grid_histogram = np.histogram2d(
        parameters[:, 0], parameters[:, 1], bins=edges, weights=weights
    )[0]
    npe_histogram = np.histogram2d(draws[:, 0], draws[:, 1], bins=edges)[0]
    npe_histogram /= len(draws)
    return float(0.5 * np.sum(np.abs(grid_histogram - npe_histogram)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimator", type=Path)
    parser.add_argument("grid", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(__file__).with_name("validation_config.json"),
    )
    args = parser.parse_args()
    model = build_model()
    estimator = load_estimator(args.estimator, model=model)
    panel = read_processed_csv(args.data)
    observed = panel.observation_summary(NUM_STATES)
    started = time.perf_counter()
    result = estimator.infer(observed, draws=args.draws, seed=args.seed)
    inference_seconds = time.perf_counter() - started
    npe_draws = np.asarray(result.draws, dtype=float)
    with np.load(args.grid, allow_pickle=False) as payload:
        parameters = np.asarray(payload["parameters"], dtype=float)
        weights = np.asarray(payload["weights"], dtype=float)
        grid_policies = np.asarray(payload["policies"], dtype=float)
    if parameters.shape != (len(weights), 2) or grid_policies.shape != (len(weights), NUM_STATES):
        raise ValueError("Dense-grid artifact has incompatible shapes.")
    grid_mean, grid_covariance = _weighted_moments(parameters, weights)
    npe_mean = npe_draws.mean(axis=0)
    npe_covariance = np.cov(npe_draws, rowvar=False)
    threshold_config = json.loads(args.thresholds.read_text(encoding="utf-8"))
    comparison_config = threshold_config["comparison"]
    policy_draw_count = min(len(npe_draws), int(comparison_config["policy_draws"]))
    policy_rng = np.random.default_rng(args.seed + 2)
    policy_index = policy_rng.choice(len(npe_draws), size=policy_draw_count, replace=False)
    npe_policy_draws = npe_draws[policy_index]
    npe_policies = policy_curves(
        npe_policy_draws,
        CANONICAL_TRANSITION_PROBABILITIES,
        num_states=NUM_STATES,
        beta=DISCOUNT_FACTOR,
        scale=COST_SCALE,
    )
    grid_policy_mean = weights @ grid_policies
    npe_policy_mean = npe_policies.mean(axis=0)
    grid_policy_lower = np.asarray(
        [_weighted_quantile(grid_policies[:, state], weights, 0.025) for state in range(NUM_STATES)]
    )
    grid_policy_upper = np.asarray(
        [_weighted_quantile(grid_policies[:, state], weights, 0.975) for state in range(NUM_STATES)]
    )
    npe_policy_lower, npe_policy_upper = np.quantile(npe_policies, [0.025, 0.975], axis=0)
    rng = np.random.default_rng(args.seed + 1)
    grid_index = rng.choice(len(weights), size=policy_draw_count, p=weights)
    grid_expected = np.asarray(
        [
            expected_replacements(
                grid_policies[index],
                CANONICAL_TRANSITION_PROBABILITIES,
                periods=12,
                buses=BUS_COUNT,
            )
            for index in grid_index
        ]
    )
    npe_expected = np.asarray(
        [
            expected_replacements(
                policy,
                CANONICAL_TRANSITION_PROBABILITIES,
                periods=12,
                buses=BUS_COUNT,
            )
            for policy in npe_policies
        ]
    )
    gates = comparison_config["gates"]
    grid_sd = np.sqrt(np.diag(grid_covariance))
    marginal_cdf = [
        _weighted_cdf_distance(npe_draws[:, j], parameters[:, j], weights) for j in range(2)
    ]
    joint_tv = _joint_coarsened_tv(
        npe_draws,
        parameters,
        weights,
        bounds=[
            threshold_config["prior"]["replacement_cost"],
            threshold_config["prior"]["maintenance_slope"],
        ],
        bins=comparison_config["joint_histogram_bins"],
    )
    mean_error_grid_sd = np.abs(npe_mean - grid_mean) / grid_sd
    policy_mean_error = float(np.max(np.abs(npe_policy_mean - grid_policy_mean)))
    policy_interval_error = float(
        max(
            np.max(np.abs(npe_policy_lower - grid_policy_lower)),
            np.max(np.abs(npe_policy_upper - grid_policy_upper)),
        )
    )
    expected_mean_error = float(abs(npe_expected.mean() - grid_expected.mean()))
    expected_interval_error = float(
        max(
            abs(np.quantile(npe_expected, 0.025) - np.quantile(grid_expected, 0.025)),
            abs(np.quantile(npe_expected, 0.975) - np.quantile(grid_expected, 0.975)),
        )
    )
    support_warning = any(bool(row.get("warning")) for row in result.support_diagnostics)
    full_profile = threshold_config["training_profiles"]["full"]
    training_simulations = int(estimator.training_metadata["number_of_simulations"])
    promotion_eligible = (
        training_simulations >= int(full_profile["simulations"])
        and args.draws >= int(comparison_config["posterior_draws"])
    )
    gate_results = {
        "marginal_cdf": max(marginal_cdf) <= gates["maximum_marginal_cdf_supremum"],
        "joint_coarsened_tv": joint_tv <= gates["joint_coarsened_total_variation"],
        "parameter_means": float(np.max(mean_error_grid_sd))
        <= gates["maximum_parameter_mean_error_in_grid_sd"],
        "policy_mean": policy_mean_error <= gates["policy_mean_maximum_absolute_error"],
        "policy_interval": policy_interval_error
        <= gates["policy_interval_endpoint_maximum_absolute_error"],
        "expected_replacements_mean": expected_mean_error
        <= gates["expected_replacements_mean_absolute_error"],
        "expected_replacements_interval": expected_interval_error
        <= gates["expected_replacements_interval_endpoint_maximum_absolute_error"],
        "empirical_support": (not support_warning)
        if not gates["empirical_support_warning_allowed"]
        else True,
    }
    metrics = {
        "parameter_mean_grid": grid_mean.tolist(),
        "parameter_mean_npe": npe_mean.tolist(),
        "parameter_mean_absolute_error": np.abs(npe_mean - grid_mean).tolist(),
        "covariance_grid": grid_covariance.tolist(),
        "covariance_npe": npe_covariance.tolist(),
        "parameter_mean_error_in_grid_sd": mean_error_grid_sd.tolist(),
        "marginal_cdf_supremum": marginal_cdf,
        "joint_coarsened_total_variation": joint_tv,
        "joint_histogram_bins": comparison_config["joint_histogram_bins"],
        "policy_mean_max_absolute_error": policy_mean_error,
        "policy_interval_endpoint_max_absolute_error": policy_interval_error,
        "expected_replacements_12_months": {
            "grid_mean": float(grid_expected.mean()),
            "grid_q025": float(np.quantile(grid_expected, 0.025)),
            "grid_q975": float(np.quantile(grid_expected, 0.975)),
            "npe_mean": float(npe_expected.mean()),
            "npe_q025": float(np.quantile(npe_expected, 0.025)),
            "npe_q975": float(np.quantile(npe_expected, 0.975)),
        },
        "inference_seconds": inference_seconds,
        "draws": args.draws,
        "policy_draws": policy_draw_count,
        "thresholds_predeclared": True,
        "threshold_config": "replication/rust_1987/validation_config.json",
        "training_simulations": training_simulations,
        "promotion_eligible": promotion_eligible,
        "gate_results": gate_results,
        "approximation_gates_pass": bool(all(gate_results.values())),
        "pass": bool(promotion_eligible and all(gate_results.values())),
        "nonclaim": (
            "Passing these approximation gates does not satisfy the separate calibration gate "
            "and does not establish accuracy for other structural models."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
