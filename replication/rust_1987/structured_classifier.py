"""Rust-specific simulation-estimated finite-grid classifier benchmark.

This benchmark targets the same conditional choice posterior as the dense-grid
Rust reference.  It is deliberately *not* the generic ``structnpe.fit`` MDN.
At each parameter-grid class it estimates state-specific replacement
probabilities from simulation, then applies the resulting class likelihood to
the empirical state-level choice counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .constants import (
        BUS_COUNT,
        CANONICAL_TRANSITION_COUNTS,
        CANONICAL_TRANSITION_PROBABILITIES,
        COST_SCALE,
        DISCOUNT_FACTOR,
        NUM_STATES,
        RAW_SHA256,
        SOURCE_COMMIT,
    )
    from .grid_posterior import GridPosterior, grid_posterior
    from .model import transition_matrix
    from .preprocess import read_processed_csv
except ImportError:  # pragma: no cover - direct script execution
    from constants import (
        BUS_COUNT,
        CANONICAL_TRANSITION_COUNTS,
        CANONICAL_TRANSITION_PROBABILITIES,
        COST_SCALE,
        DISCOUNT_FACTOR,
        NUM_STATES,
        RAW_SHA256,
        SOURCE_COMMIT,
    )
    from grid_posterior import GridPosterior, grid_posterior
    from model import transition_matrix
    from preprocess import read_processed_csv


DEFAULT_SEEDS = (1701, 1702, 1703, 1704, 1705)
DEFAULT_COARSE_GRID = (29, 30)
DEFAULT_REPLICATES_PER_CLASS = 500
PARAMETER_NAMES = ("replacement_cost", "maintenance_slope")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("validation_config.json")
DEFAULT_CONFIG_REPOSITORY_PATH = "replication/rust_1987/validation_config.json"
DEFAULT_CONFIG_SHA256 = "bc227e62b45f4c76f5a7e3131d761a279e694c159a65b7e7b98497a111941bac"
DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "processed" / "group4.csv"
DEFAULT_DATA_REPOSITORY_PATH = "replication/rust_1987/data/processed/group4.csv"
CANONICAL_PROCESSED_SHA256 = "7f9e169e5cb2f18204c342935cadab13df188af4275c45fe07dfe0f94840376b"
COMPACT_FLOAT_DECIMAL_PLACES = 14


def _reported_path(path: str | Path, default_path: Path, repository_path: str) -> str:
    supplied = Path(path).expanduser()
    if supplied.absolute() == default_path.absolute():
        return repository_path
    return str(supplied)


def describe_validation_config(path: str | Path, payload: bytes) -> dict[str, Any]:
    """Record the actual config and authenticate the one committed default."""

    source = Path(path)
    digest = hashlib.sha256(payload).hexdigest()
    default_path_match = source.expanduser().absolute() == DEFAULT_CONFIG_PATH.absolute()
    frozen = bool(default_path_match and digest == DEFAULT_CONFIG_SHA256)
    return {
        "path": _reported_path(
            source, DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_REPOSITORY_PATH
        ),
        "sha256": digest,
        "frozen_default": frozen,
        "gates_read_from_frozen_file": frozen,
    }


def describe_processed_data(
    path: str | Path,
    *,
    sha256: str,
    transition_counts: Sequence[int],
    transition_probabilities: Sequence[float],
) -> dict[str, Any]:
    """Record input provenance and authenticate the canonical processed panel."""

    counts = [int(value) for value in transition_counts]
    probabilities = [float(value) for value in transition_probabilities]
    transition_match = bool(
        counts == list(CANONICAL_TRANSITION_COUNTS)
        and np.allclose(
            probabilities,
            CANONICAL_TRANSITION_PROBABILITIES,
            atol=1e-15,
            rtol=0.0,
        )
    )
    canonical = bool(sha256 == CANONICAL_PROCESSED_SHA256 and transition_match)
    return {
        "path": _reported_path(path, DEFAULT_DATA_PATH, DEFAULT_DATA_REPOSITORY_PATH),
        "sha256": sha256,
        "canonical_processed_group4": canonical,
        "transition_counts": counts,
        "transition_probabilities": probabilities,
        "canonical_transition_match": transition_match,
        "canonical_reference": {
            "source_commit": SOURCE_COMMIT,
            "raw_sha256": RAW_SHA256,
            "processed_sha256": CANONICAL_PROCESSED_SHA256,
            "transition_counts": list(CANONICAL_TRANSITION_COUNTS),
            "transition_probabilities": list(CANONICAL_TRANSITION_PROBABILITIES),
        },
    }


def _normalized_weights(log_weight: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weight, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("log_weight must be a nonempty finite vector.")
    shifted = values - np.max(values)
    weights = np.exp(shifted)
    weights /= weights.sum()
    return weights


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(np.interp(float(quantile), cumulative, values[order]))


def _posterior_summary(parameters: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    mean = weights @ parameters
    centered = parameters - mean
    covariance = (centered * weights[:, None]).T @ centered
    sd = np.sqrt(np.diag(covariance))
    correlation = float(covariance[0, 1] / (sd[0] * sd[1]))
    intervals = []
    for column, name in enumerate(PARAMETER_NAMES):
        values = parameters[:, column]
        intervals.append(
            {
                "parameter": name,
                "q025": _weighted_quantile(values, weights, 0.025),
                "median": _weighted_quantile(values, weights, 0.5),
                "q975": _weighted_quantile(values, weights, 0.975),
            }
        )
    return {
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "correlation": correlation,
        "intervals": intervals,
    }


def aggregate_binomial_probability_estimate(
    policies: np.ndarray,
    exposures: np.ndarray,
    *,
    replicates_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Estimate every class/state replacement probability with Jeffreys smoothing.

    Conditional on a fixed state-exposure count ``N_s``, one simulated panel's
    replacement count is ``Binomial(N_s, p_gs)``.  The sum from ``M``
    independent panels is therefore exactly ``Binomial(M * N_s, p_gs)``.  One
    aggregate draw has the same law as materializing and summing all ``M``
    panels; it is not a deterministic plug-in of the exact policy.
    """

    probability = np.asarray(policies, dtype=float)
    count = np.asarray(exposures)
    if probability.ndim != 2 or np.any((probability < 0) | (probability > 1)):
        raise ValueError("policies must be a two-dimensional probability array.")
    if count.shape != (probability.shape[1],) or np.any(count < 0):
        raise ValueError("exposures must be a nonnegative vector matching the states.")
    if not np.all(np.equal(count, np.floor(count))):
        raise ValueError("exposures must contain integer counts.")
    if int(replicates_per_class) < 1:
        raise ValueError("replicates_per_class must be positive.")
    trials = int(replicates_per_class) * count.astype(np.int64)
    successes = rng.binomial(trials[None, :], probability)
    # Posterior mean under the independent Jeffreys Beta(1/2, 1/2) smoothing
    # law.  In particular, unvisited states remain finite at one half.
    return (successes + 0.5) / (trials[None, :] + 1.0)


def classifier_posterior_weights(
    estimated_probability: np.ndarray,
    keep_counts: np.ndarray,
    replacement_counts: np.ndarray,
    class_prior_weights: np.ndarray,
) -> np.ndarray:
    """Return normalized class probabilities for the observed choice counts."""

    probability = np.asarray(estimated_probability, dtype=float)
    keep = np.asarray(keep_counts, dtype=float)
    replacement = np.asarray(replacement_counts, dtype=float)
    prior = np.asarray(class_prior_weights, dtype=float)
    if probability.ndim != 2 or keep.shape != probability.shape[1:]:
        raise ValueError("Choice counts do not match the classifier state dimension.")
    if replacement.shape != keep.shape or prior.shape != probability.shape[:1]:
        raise ValueError("Replacement counts or class priors have incompatible shapes.")
    if np.any((probability <= 0) | (probability >= 1)):
        raise ValueError("Smoothed probabilities must lie strictly inside (0, 1).")
    if np.any(keep < 0) or np.any(replacement < 0) or np.any(prior <= 0):
        raise ValueError("Counts must be nonnegative and class priors positive.")
    log_weight = np.log(prior) + (
        replacement[None, :] * np.log(probability)
        + keep[None, :] * np.log1p(-probability)
    ).sum(axis=1)
    return _normalized_weights(log_weight)


def _trapezoid_class_weights(shape: tuple[int, int]) -> np.ndarray:
    rc = np.ones(shape[0], dtype=float)
    slope = np.ones(shape[1], dtype=float)
    rc[[0, -1]] = 0.5
    slope[[0, -1]] = 0.5
    return np.multiply.outer(rc, slope).ravel()


def _weighted_cdf_distance(
    candidate_parameters: np.ndarray,
    candidate_weights: np.ndarray,
    reference_parameters: np.ndarray,
    reference_weights: np.ndarray,
    column: int,
) -> float:
    # The 29-by-30 axes are mathematical subsets of the dense axes.  Round
    # away harmless ``linspace`` representation differences before grouping
    # equal support points; otherwise a value such as 1.6 can be treated as
    # lying infinitesimally beside 1.5999999999999999.
    candidate_values = np.round(candidate_parameters[:, column], 12)
    reference_values = np.round(reference_parameters[:, column], 12)
    support = np.unique(np.concatenate((candidate_values, reference_values)))

    def cdf(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        order = np.argsort(values)
        sorted_values = values[order]
        cumulative = np.cumsum(weights[order])
        location = np.searchsorted(sorted_values, support, side="right") - 1
        output = np.zeros(len(support), dtype=float)
        valid = location >= 0
        output[valid] = cumulative[location[valid]]
        return output

    return float(
        np.max(
            np.abs(
                cdf(candidate_values, candidate_weights)
                - cdf(reference_values, reference_weights)
            )
        )
    )


def _joint_coarsened_tv(
    candidate: GridPosterior,
    reference: GridPosterior,
    *,
    bounds: Sequence[Sequence[float]],
    bins: Sequence[int],
) -> float:
    edges = [
        np.linspace(float(bound[0]), float(bound[1]), int(count) + 1)
        for bound, count in zip(bounds, bins, strict=True)
    ]
    candidate_histogram = np.histogram2d(
        candidate.parameters[:, 0],
        candidate.parameters[:, 1],
        bins=edges,
        weights=candidate.weights,
    )[0]
    reference_histogram = np.histogram2d(
        reference.parameters[:, 0],
        reference.parameters[:, 1],
        bins=edges,
        weights=reference.weights,
    )[0]
    return float(0.5 * np.sum(np.abs(candidate_histogram - reference_histogram)))


def _expected_replacements_batch(
    policies: np.ndarray,
    probabilities: Sequence[float],
    *,
    periods: int,
    buses: int,
) -> np.ndarray:
    """Vectorized equivalent of ``model.expected_replacements`` for a grid."""

    policy = np.asarray(policies, dtype=float)
    if policy.ndim != 2 or np.any((policy < 0) | (policy > 1)):
        raise ValueError("policies must be a two-dimensional probability array.")
    probability = np.asarray(probabilities, dtype=float)
    transition = transition_matrix(policy.shape[1], probability)
    distribution = np.zeros_like(policy)
    distribution[:, 0] = 1.0
    expected = np.zeros(policy.shape[0], dtype=float)
    for _ in range(int(periods)):
        replacement_mass = np.sum(distribution * policy, axis=1)
        expected += replacement_mass * int(buses)
        keep_mass = distribution * (1.0 - policy)
        next_distribution = np.zeros_like(distribution)
        # Apply the capped increment kernel in O(grid * states * increments),
        # avoiding a dense grid-by-90-by-90 product at every horizon step.
        states = policy.shape[1]
        for increment, increment_probability in enumerate(probability):
            if increment == 0:
                next_distribution += increment_probability * keep_mass
            elif increment < states:
                next_distribution[:, increment:] += (
                    increment_probability * keep_mass[:, : states - increment]
                )
                next_distribution[:, -1] += increment_probability * np.sum(
                    keep_mass[:, states - increment :], axis=1
                )
            else:
                next_distribution[:, -1] += increment_probability * np.sum(
                    keep_mass, axis=1
                )
        next_distribution += replacement_mass[:, None] * transition[0][None, :]
        distribution = next_distribution
    return expected


def _scalar_summary(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(weights @ values),
        "q025": _weighted_quantile(values, weights, 0.025),
        "q975": _weighted_quantile(values, weights, 0.975),
    }


def _reference_cache(
    reference: GridPosterior,
    probabilities: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = config["comparison"]
    expected = _expected_replacements_batch(
        reference.policies,
        probabilities,
        periods=int(comparison["expected_replacement_horizon_months"]),
        buses=int(comparison["expected_replacement_buses"]),
    )
    return {
        "posterior": _posterior_summary(reference.parameters, reference.weights),
        "policy": reference.policy_summary(),
        "expected": _scalar_summary(expected, reference.weights),
    }


def _comparison_metrics(
    candidate: GridPosterior,
    reference: GridPosterior,
    *,
    probabilities: Sequence[float],
    config: Mapping[str, Any],
    reference_cache: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    comparison = config["comparison"]
    gates = comparison["gates"]
    candidate_summary = _posterior_summary(candidate.parameters, candidate.weights)
    cached = (
        _reference_cache(reference, probabilities, config)
        if reference_cache is None
        else reference_cache
    )
    reference_summary = cached["posterior"]
    candidate_mean = np.asarray(candidate_summary["mean"])
    reference_mean = np.asarray(reference_summary["mean"])
    reference_sd = np.asarray(reference_summary["sd"])
    mean_error = np.abs(candidate_mean - reference_mean) / reference_sd
    marginal_cdf = [
        _weighted_cdf_distance(
            candidate.parameters,
            candidate.weights,
            reference.parameters,
            reference.weights,
            column,
        )
        for column in range(2)
    ]
    joint_tv = _joint_coarsened_tv(
        candidate,
        reference,
        bounds=[config["prior"][name] for name in PARAMETER_NAMES],
        bins=comparison["joint_histogram_bins"],
    )
    candidate_policy = candidate.policy_summary()
    reference_policy = cached["policy"]
    policy_mean_error = float(
        np.max(
            np.abs(
                np.asarray(candidate_policy["mean"]) - np.asarray(reference_policy["mean"])
            )
        )
    )
    policy_interval_error = float(
        max(
            np.max(
                np.abs(
                    np.asarray(candidate_policy["q025"])
                    - np.asarray(reference_policy["q025"])
                )
            ),
            np.max(
                np.abs(
                    np.asarray(candidate_policy["q975"])
                    - np.asarray(reference_policy["q975"])
                )
            ),
        )
    )
    expected_candidate = _expected_replacements_batch(
        candidate.policies,
        probabilities,
        periods=int(comparison["expected_replacement_horizon_months"]),
        buses=int(comparison["expected_replacement_buses"]),
    )
    candidate_expected_summary = _scalar_summary(expected_candidate, candidate.weights)
    reference_expected_summary = cached["expected"]
    expected_mean_error = abs(
        candidate_expected_summary["mean"] - reference_expected_summary["mean"]
    )
    expected_interval_error = max(
        abs(candidate_expected_summary["q025"] - reference_expected_summary["q025"]),
        abs(candidate_expected_summary["q975"] - reference_expected_summary["q975"]),
    )
    bounds = [config["prior"][name] for name in PARAMETER_NAMES]
    support_ok = bool(
        np.all(
            (candidate.parameters >= np.asarray([bound[0] for bound in bounds]))
            & (candidate.parameters <= np.asarray([bound[1] for bound in bounds]))
        )
    )
    gate_results = {
        "marginal_cdf": max(marginal_cdf)
        <= gates["maximum_marginal_cdf_supremum"],
        "joint_coarsened_tv": joint_tv
        <= gates["joint_coarsened_total_variation"],
        "parameter_means": float(np.max(mean_error))
        <= gates["maximum_parameter_mean_error_in_grid_sd"],
        "policy_mean": policy_mean_error
        <= gates["policy_mean_maximum_absolute_error"],
        "policy_interval": policy_interval_error
        <= gates["policy_interval_endpoint_maximum_absolute_error"],
        "expected_replacements_mean": expected_mean_error
        <= gates["expected_replacements_mean_absolute_error"],
        "expected_replacements_interval": expected_interval_error
        <= gates["expected_replacements_interval_endpoint_maximum_absolute_error"],
        "empirical_support": support_ok
        if not gates["empirical_support_warning_allowed"]
        else True,
    }
    return {
        "posterior": candidate_summary,
        "parameter_mean_error_in_dense_sd": mean_error.tolist(),
        "marginal_cdf_supremum": marginal_cdf,
        "joint_coarsened_total_variation": joint_tv,
        "policy_mean_max_absolute_error": policy_mean_error,
        "policy_interval_endpoint_max_absolute_error": policy_interval_error,
        "expected_replacements_12_months": {
            "dense": reference_expected_summary,
            "candidate": candidate_expected_summary,
            "mean_absolute_error": expected_mean_error,
            "interval_endpoint_max_absolute_error": expected_interval_error,
        },
        "gate_results": gate_results,
        "all_configured_numerical_comparison_limits_pass": bool(
            all(gate_results.values())
        ),
    }


def _replace_weights(reference: GridPosterior, weights: np.ndarray) -> GridPosterior:
    return GridPosterior(
        parameters=reference.parameters,
        weights=np.asarray(weights, dtype=float),
        log_likelihood=reference.log_likelihood,
        policies=reference.policies,
        rc_axis=reference.rc_axis,
        slope_axis=reference.slope_axis,
        seconds=reference.seconds,
    )


def run_benchmark(
    keep_counts: np.ndarray,
    replacement_counts: np.ndarray,
    probabilities: Sequence[float],
    *,
    config: Mapping[str, Any],
    config_sha256: str | None = None,
    config_metadata: Mapping[str, Any] | None = None,
    data_metadata: Mapping[str, Any] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    coarse_grid: tuple[int, int] = DEFAULT_COARSE_GRID,
    replicates_per_class: int = DEFAULT_REPLICATES_PER_CLASS,
) -> dict[str, Any]:
    """Run deterministic references and the seeded structured classifier."""

    config_record = dict(config_metadata or {})
    if not config_record:
        config_record = {
            "path": "<in-memory>",
            "sha256": config_sha256 or "unrecorded",
            "frozen_default": False,
            "gates_read_from_frozen_file": False,
        }
    data_record = dict(data_metadata or {})
    if not data_record:
        data_record = {
            "path": "<in-memory aggregates>",
            "sha256": "unrecorded",
            "canonical_processed_group4": False,
            "canonical_transition_match": False,
        }
    frozen_config = bool(config_record.get("frozen_default", False))
    canonical_data = bool(data_record.get("canonical_processed_group4", False))
    prior = config["prior"]
    dense_spec = config["dense_grid"]
    common = {
        "num_states": NUM_STATES,
        "beta": DISCOUNT_FACTOR,
        "scale": COST_SCALE,
        "rc_bounds": tuple(prior["replacement_cost"]),
        "slope_bounds": tuple(prior["maintenance_slope"]),
        "bellman_tolerance": float(dense_spec["bellman_residual_tolerance"]),
    }
    dense = grid_posterior(
        keep_counts,
        replacement_counts,
        probabilities,
        rc_points=int(dense_spec["replacement_cost_points"]),
        slope_points=int(dense_spec["maintenance_slope_points"]),
        **common,
    )
    coarse = grid_posterior(
        keep_counts,
        replacement_counts,
        probabilities,
        rc_points=int(coarse_grid[0]),
        slope_points=int(coarse_grid[1]),
        **common,
    )
    reference_cache = _reference_cache(dense, probabilities, config)
    coarse_metrics = _comparison_metrics(
        coarse,
        dense,
        probabilities=probabilities,
        config=config,
        reference_cache=reference_cache,
    )
    exposures = np.asarray(keep_counts) + np.asarray(replacement_counts)
    class_prior = _trapezoid_class_weights(coarse_grid)
    runs = []
    for seed in seeds:
        estimated_probability = aggregate_binomial_probability_estimate(
            coarse.policies,
            exposures,
            replicates_per_class=int(replicates_per_class),
            rng=np.random.default_rng(int(seed)),
        )
        weights = classifier_posterior_weights(
            estimated_probability,
            keep_counts,
            replacement_counts,
            class_prior,
        )
        classifier = _replace_weights(coarse, weights)
        metrics = _comparison_metrics(
            classifier,
            dense,
            probabilities=probabilities,
            config=config,
            reference_cache=reference_cache,
        )
        runs.append({"seed": int(seed), **metrics})
    return {
        "schema_version": 1,
        "benchmark": (
            "Rust (1987) canonical group-4 structured finite-grid classifier"
            if canonical_data
            else "Rust (1987) noncanonical-input structured finite-grid classifier"
        ),
        "estimator": {
            "kind": "simulation-estimated finite-grid generative classifier",
            "generic_structnpe_fit_mdn": False,
            "target": (
                "posterior for (replacement_cost, maintenance_slope) conditional on "
                "observed state exposures"
            ),
            "parameter_order": list(PARAMETER_NAMES),
            "coarse_grid_shape": [int(coarse_grid[0]), int(coarse_grid[1])],
            "classes": int(coarse_grid[0] * coarse_grid[1]),
            "replicates_per_class": int(replicates_per_class),
            "equivalent_panel_simulations": int(
                coarse_grid[0] * coarse_grid[1] * replicates_per_class
            ),
            "simulation_unit": (
                "fixed-exposure conditional choice panel, not a jointly simulated state path"
            ),
            "seeds": [int(seed) for seed in seeds],
            "smoothing": "independent Jeffreys Beta(1/2, 1/2) posterior means",
            "aggregate_binomial_equivalence": (
                "For class g and state s, summing M independent "
                "Binomial(N_s,p_gs) panel counts is exactly "
                "Binomial(M*N_s,p_gs); the implementation draws the latter."
            ),
        },
        "data_summary": {
            "conditional_choice_observations": int(np.sum(exposures)),
            "observed_replacements": int(np.sum(replacement_counts)),
            "states": NUM_STATES,
            "buses": BUS_COUNT,
        },
        "data_provenance": data_record,
        "validation_config": config_record,
        "dense_reference": {
            "grid_shape": [len(dense.rc_axis), len(dense.slope_axis)],
            "posterior": _posterior_summary(dense.parameters, dense.weights),
            "expected_replacements_12_months": reference_cache["expected"],
        },
        "exact_coarse_reference": coarse_metrics,
        "classifier_runs": runs,
        "all_seeds_pass_configured_numerical_comparison_limits": bool(
            coarse_metrics["all_configured_numerical_comparison_limits_pass"]
            and all(
                run["all_configured_numerical_comparison_limits_pass"] for run in runs
            )
        ),
        "nonclaim": (
            "This is a model-specific conditional finite-grid classifier, not the generic "
            "structnpe.fit MDN, not an amortized estimator for arbitrary panels, and not "
            "evidence for other structural models. Passing the "
            + ("frozen" if frozen_config else "supplied, non-frozen")
            + " numerical comparison limits is not the MDN-specific promotion or "
            "calibration result."
        ),
    }


def _compact_posterior(
    summary: Mapping[str, Any], *, include_quantiles: bool
) -> dict[str, Any]:
    """Strip a posterior summary to the stable committed-evidence schema."""

    output = {
        "mean": list(summary["mean"]),
        "sd": list(summary["sd"]),
        "correlation": float(summary["correlation"]),
    }
    if include_quantiles:
        intervals = list(summary["intervals"])
        if [row["parameter"] for row in intervals] != list(PARAMETER_NAMES):
            raise ValueError("Posterior intervals do not follow the declared parameter order.")
        for key in ("q025", "median", "q975"):
            output[key] = [float(row[key]) for row in intervals]
    return output


def _compact_comparison(
    metrics: Mapping[str, Any], *, frozen_config: bool
) -> dict[str, Any]:
    """Retain all posterior/policy gate statistics without bulky curve arrays."""

    expected = metrics["expected_replacements_12_months"]
    pass_label = (
        "all_frozen_numerical_comparison_limits_pass"
        if frozen_config
        else "all_configured_numerical_comparison_limits_pass"
    )
    output = {
        "posterior": _compact_posterior(metrics["posterior"], include_quantiles=False),
        "parameter_mean_error_in_dense_sd": list(
            metrics["parameter_mean_error_in_dense_sd"]
        ),
        "marginal_cdf_supremum": list(metrics["marginal_cdf_supremum"]),
        "joint_coarsened_total_variation": float(
            metrics["joint_coarsened_total_variation"]
        ),
        "policy_mean_max_absolute_error": float(
            metrics["policy_mean_max_absolute_error"]
        ),
        "policy_interval_endpoint_max_absolute_error": float(
            metrics["policy_interval_endpoint_max_absolute_error"]
        ),
        "expected_replacements_mean_absolute_error": float(expected["mean_absolute_error"]),
        "expected_replacements_interval_endpoint_max_absolute_error": float(
            expected["interval_endpoint_max_absolute_error"]
        ),
        pass_label: bool(metrics["all_configured_numerical_comparison_limits_pass"]),
    }
    return output


def _round_compact_floats(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return round(float(value), COMPACT_FLOAT_DECIMAL_PLACES)
    if isinstance(value, dict):
        return {key: _round_compact_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_compact_floats(item) for item in value]
    return value


def compact_benchmark_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic compact JSON object committed under ``expected/``.

    ``run_benchmark`` retains detailed interval and gate structures for Python
    callers.  The CLI intentionally emits this audit-focused projection:
    dense posterior quantiles, every posterior moment and scalar comparison
    statistic, all seeds, config/data provenance, and the estimator nonclaims.
    Floats are rounded to ``COMPACT_FLOAT_DECIMAL_PLACES`` after calculation so
    harmless platform-level differences are not presented as exact scientific
    differences. JSON whitespace and key order are not part of the contract.
    """

    dense = result["dense_reference"]
    frozen_config = bool(result["validation_config"].get("frozen_default", False))
    pass_label = (
        "all_seeds_pass_frozen_numerical_comparison_limits"
        if frozen_config
        else "all_seeds_pass_configured_numerical_comparison_limits"
    )
    compact_runs = []
    for run in result["classifier_runs"]:
        compact_runs.append(
            {
                "seed": int(run["seed"]),
                **_compact_comparison(run, frozen_config=frozen_config),
            }
        )
    output = {
        "schema_version": int(result["schema_version"]),
        "compact_float_decimal_places": COMPACT_FLOAT_DECIMAL_PLACES,
        "benchmark": result["benchmark"],
        "estimator": dict(result["estimator"]),
        "data_summary": dict(result["data_summary"]),
        "data_provenance": dict(result["data_provenance"]),
        "validation_config": dict(result["validation_config"]),
        "dense_reference": {
            "grid_shape": list(dense["grid_shape"]),
            "posterior": _compact_posterior(
                dense["posterior"], include_quantiles=True
            ),
            "expected_replacements_12_months": dict(
                dense["expected_replacements_12_months"]
            ),
        },
        "exact_coarse_reference": _compact_comparison(
            result["exact_coarse_reference"], frozen_config=frozen_config
        ),
        "classifier_runs": compact_runs,
        pass_label: bool(
            result["all_seeds_pass_configured_numerical_comparison_limits"]
        ),
        "nonclaim": result["nonclaim"],
    }
    return _round_compact_floats(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument("--replicates-per-class", type=int, default=DEFAULT_REPLICATES_PER_CLASS)
    parser.add_argument("--rc-points", type=int, default=DEFAULT_COARSE_GRID[0])
    parser.add_argument("--slope-points", type=int, default=DEFAULT_COARSE_GRID[1])
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated classifier simulation seeds.",
    )
    args = parser.parse_args()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if not seeds:
        parser.error("--seeds must contain at least one integer.")
    config_bytes = args.thresholds.read_bytes()
    config = json.loads(config_bytes)
    config_record = describe_validation_config(args.thresholds, config_bytes)
    panel = read_processed_csv(args.data)
    data_record = describe_processed_data(
        args.data,
        sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        transition_counts=panel.transition_counts,
        transition_probabilities=panel.transition_probabilities,
    )
    keep, replacement = panel.choice_counts(NUM_STATES)
    full_result = run_benchmark(
        keep,
        replacement,
        panel.transition_probabilities,
        config=config,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        config_metadata=config_record,
        data_metadata=data_record,
        seeds=seeds,
        coarse_grid=(args.rc_points, args.slope_points),
        replicates_per_class=args.replicates_per_class,
    )
    result = compact_benchmark_result(full_result)
    configured_limits_pass = bool(
        full_result["all_seeds_pass_configured_numerical_comparison_limits"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "all_seeds_pass_configured_numerical_comparison_limits": (
                    configured_limits_pass
                ),
                "frozen_default_config": config_record["frozen_default"],
                "canonical_processed_group4": data_record[
                    "canonical_processed_group4"
                ],
                "classifier_runs": len(result["classifier_runs"]),
            }
        )
    )
    return 0 if configured_limits_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
