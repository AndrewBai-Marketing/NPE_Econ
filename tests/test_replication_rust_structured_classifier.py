from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from replication.rust_1987.constants import (
    CANONICAL_TRANSITION_COUNTS,
    CANONICAL_TRANSITION_PROBABILITIES,
    NUM_STATES,
)
from replication.rust_1987.model import expected_replacements
from replication.rust_1987.structured_classifier import (
    _expected_replacements_batch,
    CANONICAL_PROCESSED_SHA256,
    DEFAULT_CONFIG_PATH,
    DEFAULT_CONFIG_REPOSITORY_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_DATA_REPOSITORY_PATH,
    aggregate_binomial_probability_estimate,
    classifier_posterior_weights,
    compact_benchmark_result,
    describe_processed_data,
    describe_validation_config,
    run_benchmark,
)


def test_aggregate_binomial_draw_and_jeffreys_smoothing() -> None:
    policies = np.asarray([[0.1, 0.8], [0.4, 0.2]])
    exposures = np.asarray([0, 3])
    seed = 1701
    replicates = 7
    expected_successes = np.random.default_rng(seed).binomial(
        (replicates * exposures)[None, :], policies
    )
    estimate = aggregate_binomial_probability_estimate(
        policies,
        exposures,
        replicates_per_class=replicates,
        rng=np.random.default_rng(seed),
    )
    np.testing.assert_allclose(
        estimate,
        (expected_successes + 0.5) / ((replicates * exposures)[None, :] + 1.0),
    )
    np.testing.assert_array_equal(estimate[:, 0], np.asarray([0.5, 0.5]))


def test_classifier_weights_match_direct_smoothed_class_likelihood() -> None:
    probability = np.asarray([[0.2, 0.7], [0.6, 0.4], [0.8, 0.1]])
    keep = np.asarray([4.0, 2.0])
    replacement = np.asarray([1.0, 3.0])
    prior = np.asarray([0.5, 1.0, 0.5])
    actual = classifier_posterior_weights(probability, keep, replacement, prior)
    log_weight = np.log(prior) + (
        replacement[None, :] * np.log(probability)
        + keep[None, :] * np.log1p(-probability)
    ).sum(axis=1)
    expected = np.exp(log_weight - np.max(log_weight))
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)


def test_batched_policy_functional_matches_scalar_implementation() -> None:
    policies = np.asarray(
        [
            np.linspace(0.001, 0.2, 90),
            np.linspace(0.01, 0.4, 90),
        ]
    )
    actual = _expected_replacements_batch(
        policies,
        CANONICAL_TRANSITION_PROBABILITIES,
        periods=12,
        buses=37,
    )
    expected = np.asarray(
        [
            expected_replacements(
                policy,
                CANONICAL_TRANSITION_PROBABILITIES,
                periods=12,
                buses=37,
            )
            for policy in policies
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_compact_result_schema_is_a_deterministic_projection() -> None:
    posterior = {
        "mean": [10.0, 2.0],
        "sd": [1.0, 0.5],
        "correlation": 0.9,
        "intervals": [
            {
                "parameter": "replacement_cost",
                "q025": 8.0,
                "median": 10.0,
                "q975": 12.0,
            },
            {
                "parameter": "maintenance_slope",
                "q025": 1.0,
                "median": 2.0,
                "q975": 3.0,
            },
        ],
    }
    comparison = {
        "posterior": posterior,
        "parameter_mean_error_in_dense_sd": [0.01, 0.02],
        "marginal_cdf_supremum": [0.03, 0.04],
        "joint_coarsened_total_variation": 0.05,
        "policy_mean_max_absolute_error": 0.006,
        "policy_interval_endpoint_max_absolute_error": 0.007,
        "expected_replacements_12_months": {
            "mean_absolute_error": 0.008,
            "interval_endpoint_max_absolute_error": 0.009,
        },
        "all_configured_numerical_comparison_limits_pass": True,
    }
    full = {
        "schema_version": 1,
        "benchmark": "test",
        "estimator": {"generic_structnpe_fit_mdn": False},
        "data_summary": {"states": 90},
        "data_provenance": {"canonical_processed_group4": False},
        "validation_config": {"sha256": "abc", "frozen_default": False},
        "dense_reference": {
            "grid_shape": [3, 3],
            "posterior": posterior,
            "expected_replacements_12_months": {
                "mean": 0.1,
                "q025": 0.0,
                "q975": 0.2,
            },
        },
        "exact_coarse_reference": comparison,
        "classifier_runs": [{"seed": 1701, **comparison}],
        "all_seeds_pass_configured_numerical_comparison_limits": True,
        "nonclaim": "test only",
    }
    compact = compact_benchmark_result(full)
    assert compact["dense_reference"]["posterior"]["q025"] == [8.0, 1.0]
    assert compact["dense_reference"]["posterior"]["median"] == [10.0, 2.0]
    assert compact["dense_reference"]["posterior"]["q975"] == [12.0, 3.0]
    assert "intervals" not in compact["classifier_runs"][0]["posterior"]
    assert compact["classifier_runs"][0][
        "expected_replacements_mean_absolute_error"
    ] == pytest.approx(0.008)
    assert "all_seeds_pass_configured_numerical_comparison_limits" in compact
    assert "all_seeds_pass_frozen_numerical_comparison_limits" not in compact


def test_config_provenance_requires_default_path_and_bytes(tmp_path: Path) -> None:
    default_bytes = DEFAULT_CONFIG_PATH.read_bytes()
    canonical = describe_validation_config(DEFAULT_CONFIG_PATH, default_bytes)
    assert canonical["path"] == DEFAULT_CONFIG_REPOSITORY_PATH
    assert canonical["frozen_default"] is True
    assert canonical["gates_read_from_frozen_file"] is True

    copied_path = tmp_path / "copied-thresholds.json"
    copied_path.write_bytes(default_bytes)
    copied = describe_validation_config(copied_path, default_bytes)
    assert copied["path"] == str(copied_path)
    assert copied["sha256"] == canonical["sha256"]
    assert copied["frozen_default"] is False
    assert copied["gates_read_from_frozen_file"] is False

    changed = describe_validation_config(DEFAULT_CONFIG_PATH, default_bytes + b"\n")
    assert changed["path"] == DEFAULT_CONFIG_REPOSITORY_PATH
    assert changed["frozen_default"] is False


def test_data_provenance_authenticates_hash_and_transition_law(tmp_path: Path) -> None:
    canonical = describe_processed_data(
        DEFAULT_DATA_PATH,
        sha256=CANONICAL_PROCESSED_SHA256,
        transition_counts=CANONICAL_TRANSITION_COUNTS,
        transition_probabilities=CANONICAL_TRANSITION_PROBABILITIES,
    )
    assert canonical["path"] == DEFAULT_DATA_REPOSITORY_PATH
    assert canonical["canonical_processed_group4"] is True
    assert canonical["canonical_transition_match"] is True

    custom_path = tmp_path / "custom.csv"
    custom = describe_processed_data(
        custom_path,
        sha256="0" * 64,
        transition_counts=CANONICAL_TRANSITION_COUNTS,
        transition_probabilities=CANONICAL_TRANSITION_PROBABILITIES,
    )
    assert custom["path"] == str(custom_path)
    assert custom["canonical_transition_match"] is True
    assert custom["canonical_processed_group4"] is False

    different_law = describe_processed_data(
        custom_path,
        sha256=CANONICAL_PROCESSED_SHA256,
        transition_counts=(1683, 2554, 55),
        transition_probabilities=(1683 / 4292, 2554 / 4292, 55 / 4292),
    )
    assert different_law["canonical_transition_match"] is False
    assert different_law["canonical_processed_group4"] is False


def test_custom_sources_produce_noncanonical_nonfrozen_output(tmp_path: Path) -> None:
    config_bytes = DEFAULT_CONFIG_PATH.read_bytes()
    config = json.loads(config_bytes)
    config["dense_grid"]["replacement_cost_points"] = 3
    config["dense_grid"]["maintenance_slope_points"] = 3
    custom_config_path = tmp_path / "thresholds.json"
    custom_config = describe_validation_config(
        custom_config_path, json.dumps(config, sort_keys=True).encode("utf-8")
    )
    custom_data = describe_processed_data(
        tmp_path / "panel.csv",
        sha256="0" * 64,
        transition_counts=CANONICAL_TRANSITION_COUNTS,
        transition_probabilities=CANONICAL_TRANSITION_PROBABILITIES,
    )
    keep = np.zeros(NUM_STATES)
    replacement = np.zeros(NUM_STATES)
    keep[0] = 1
    full = run_benchmark(
        keep,
        replacement,
        CANONICAL_TRANSITION_PROBABILITIES,
        config=config,
        config_metadata=custom_config,
        data_metadata=custom_data,
        seeds=(1,),
        coarse_grid=(3, 3),
        replicates_per_class=1,
    )
    compact = compact_benchmark_result(full)
    assert "noncanonical-input" in compact["benchmark"]
    assert compact["validation_config"]["frozen_default"] is False
    assert compact["data_provenance"]["canonical_processed_group4"] is False
    assert "all_seeds_pass_configured_numerical_comparison_limits" in compact
    assert "all_seeds_pass_frozen_numerical_comparison_limits" not in compact
    assert all(
        "all_configured_numerical_comparison_limits_pass" in run
        and "all_frozen_numerical_comparison_limits_pass" not in run
        for run in compact["classifier_runs"]
    )


def test_committed_structured_classifier_campaign_is_reproducible_and_scoped() -> None:
    evidence_path = Path(
        "replication/rust_1987/expected/structured_classifier_metrics.json"
    )
    evidence_text = evidence_path.read_text(encoding="utf-8")
    evidence = json.loads(evidence_text)
    assert ("/" + "home" + "/") not in evidence_text
    estimator = evidence["estimator"]
    assert estimator["generic_structnpe_fit_mdn"] is False
    assert estimator["coarse_grid_shape"] == [29, 30]
    assert estimator["replicates_per_class"] == 500
    assert estimator["equivalent_panel_simulations"] == 435_000
    assert estimator["seeds"] == [1701, 1702, 1703, 1704, 1705]
    assert evidence["compact_float_decimal_places"] == 14
    assert evidence["data_provenance"]["canonical_processed_group4"] is True
    assert evidence["validation_config"]["frozen_default"] is True
    assert len(evidence["classifier_runs"]) == 5
    assert evidence["all_seeds_pass_frozen_numerical_comparison_limits"] is True
    assert all(
        run["all_frozen_numerical_comparison_limits_pass"]
        for run in evidence["classifier_runs"]
    )
    config_path = Path(evidence["validation_config"]["path"])
    assert hashlib.sha256(config_path.read_bytes()).hexdigest() == evidence[
        "validation_config"
    ]["sha256"]
    gates = json.loads(config_path.read_text(encoding="utf-8"))["comparison"]["gates"]
    for run in [evidence["exact_coarse_reference"], *evidence["classifier_runs"]]:
        assert max(run["marginal_cdf_supremum"]) <= gates[
            "maximum_marginal_cdf_supremum"
        ]
        assert run["joint_coarsened_total_variation"] <= gates[
            "joint_coarsened_total_variation"
        ]
        assert max(run["parameter_mean_error_in_dense_sd"]) <= gates[
            "maximum_parameter_mean_error_in_grid_sd"
        ]
        assert run["policy_mean_max_absolute_error"] <= gates[
            "policy_mean_maximum_absolute_error"
        ]
        assert run["policy_interval_endpoint_max_absolute_error"] <= gates[
            "policy_interval_endpoint_maximum_absolute_error"
        ]
        assert run["expected_replacements_mean_absolute_error"] <= gates[
            "expected_replacements_mean_absolute_error"
        ]
        assert run[
            "expected_replacements_interval_endpoint_max_absolute_error"
        ] <= gates["expected_replacements_interval_endpoint_maximum_absolute_error"]
