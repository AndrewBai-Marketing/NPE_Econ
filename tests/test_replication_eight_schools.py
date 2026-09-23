from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from replication.eight_schools.exact import (
    hyperparameter_moments,
    sample_exact_posterior,
    tau_quadrature,
)
from replication.eight_schools.model import (
    EFFECTS,
    STANDARD_ERRORS,
    build_model,
    marginalized_simulator,
)
from replication.eight_schools.run_validation import (
    _joint_quantile_grid_tv,
    _ks_distance,
    _run_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


class _NormalRecorder:
    def __init__(self) -> None:
        self.location: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    def normal(self, location, scale):  # noqa: ANN001 - NumPy-compatible test double
        self.location = np.asarray(location, dtype=float)
        self.scale = np.asarray(scale, dtype=float)
        return np.broadcast_arrays(self.location, self.scale)[0].copy()


def test_tau_quadrature_is_normalized_and_finite() -> None:
    tau, mass, mu_mean, mu_variance = tau_quadrature(points=1001)
    assert tau.shape == mass.shape == mu_mean.shape == mu_variance.shape
    assert np.all(tau > 0.0)
    assert np.all(mass >= 0.0)
    assert np.all(mu_variance > 0.0)
    assert np.isclose(mass.sum(), 1.0)


def test_exact_joint_draws_and_model_contract() -> None:
    first = sample_exact_posterior(200, seed=71, quadrature_points=1001)
    second = sample_exact_posterior(200, seed=71, quadrature_points=1001)
    assert first.shape == (200, 10)
    assert np.array_equal(first, second)
    assert np.all(first[:, 1] > 0.0)
    model = build_model()
    prior_draws = model.sample_prior(200, np.random.default_rng(72))
    assert prior_draws.shape == (200, 2)
    assert np.all(prior_draws[:, 1] > 0.0)
    simulated = model.simulate_one(first[0, :2], np.random.default_rng(73))
    assert simulated.shape == EFFECTS.shape
    assert np.all(np.isfinite(simulated))


def test_marginalized_simulator_uses_exact_gaussian_convolution() -> None:
    recorder = _NormalRecorder()
    simulated = marginalized_simulator(np.array([2.0, 3.0]), recorder)
    assert np.array_equal(simulated, np.full(8, 2.0))
    assert np.array_equal(recorder.location, np.array(2.0))
    assert np.allclose(recorder.scale, np.sqrt(3.0**2 + STANDARD_ERRORS**2))


def test_deterministic_hyperparameter_moments_match_reference() -> None:
    moments = hyperparameter_moments(points=40_001)
    assert np.allclose(moments["mean"], [6.47201703297, 4.75310343438], atol=1e-10)
    assert np.allclose(moments["sd"], [4.19116294483, 3.68381836174], atol=1e-10)
    assert np.isclose(moments["correlation"], -0.0464943171541, atol=1e-10)


def test_distribution_metrics_detect_exact_agreement() -> None:
    exact = sample_exact_posterior(1_000, seed=81, quadrature_points=1_001)[:, :2]
    moments = hyperparameter_moments(points=1_001)
    metrics, rows = _run_metrics(
        exact,
        exact,
        np.asarray(moments["mean"]),
        np.asarray(moments["sd"]),
        float(moments["correlation"]),
    )
    assert _ks_distance(exact[:, 0], exact[:, 0]) == 0.0
    assert _joint_quantile_grid_tv(exact, exact) == 0.0
    assert metrics["marginal_cdf_max_abs_error"] == 0.0
    assert metrics["central_95_endpoint_max_standardized_abs_error"] == 0.0
    assert len(rows) == 2


def test_joint_grid_metric_detects_strong_dependence_reversal() -> None:
    coordinate = np.linspace(-1.0, 1.0, 2_000)
    exact = np.column_stack([coordinate, coordinate])
    reversed_dependence = np.column_stack([coordinate, coordinate[::-1]])
    assert _joint_quantile_grid_tv(reversed_dependence, exact, bins=10) > 0.9


def test_committed_eight_schools_evidence_passes_all_fixed_seeds() -> None:
    benchmark_dir = ROOT / "replication/eight_schools"
    evidence = json.loads(
        (benchmark_dir / "expected_metrics.json").read_text(encoding="utf-8")
    )
    config_bytes = (benchmark_dir / "config.json").read_bytes()
    config = json.loads(config_bytes)
    assert evidence["canonical_configuration"] is True
    assert evidence["configuration"] == config
    assert evidence["configuration_file_sha256"] == hashlib.sha256(
        config_bytes
    ).hexdigest()
    effective = json.dumps(
        config, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert evidence["effective_configuration_sha256"] == hashlib.sha256(
        effective
    ).hexdigest()
    assert evidence["status"] == "PASS"
    assert evidence["all_seeds_pass"] is True
    assert [run["seed"] for run in evidence["runs"]] == [
        73001,
        73002,
        73003,
        73004,
        73005,
    ]
    assert all(run["status"] == "PASS" for run in evidence["runs"])
    assert len({run["estimator_fingerprint"] for run in evidence["runs"]}) == 5
    assert all(len(run["estimator_fingerprint"]) == 64 for run in evidence["runs"])
    assert evidence["environment"]["torch_num_threads"] == config[
        "torch_num_threads"
    ]
    assert all(
        check["passed"]
        for run in evidence["runs"]
        for check in run["checks"].values()
    )
    for run in evidence["runs"]:
        assert run["training_summary"]["number_of_simulations"] == config[
            "simulations"
        ]
        for name, maximum in config["thresholds"].items():
            assert run["metrics"][name] <= maximum
            assert run["checks"][name] == {
                "maximum": maximum,
                "passed": True,
                "value": run["metrics"][name],
            }


def test_spline_evidence_preserves_failures_and_matches_configuration() -> None:
    benchmark_dir = ROOT / "replication/eight_schools"
    config_bytes = (benchmark_dir / "spline_config.json").read_bytes()
    config = json.loads(config_bytes)
    evidence = json.loads((benchmark_dir / "spline_expected_metrics.json").read_text())
    legacy_config = json.loads((benchmark_dir / "config.json").read_text())
    assert config["backend"] == "spline"
    assert config["thresholds"] == legacy_config["thresholds"]
    assert evidence["configuration"] == config
    assert evidence["configuration_file_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert [run["seed"] for run in evidence["runs"]] == config["seeds"]
    assert evidence["status"] == "FAIL"
    assert evidence["all_seeds_pass"] is False
    for run in evidence["runs"]:
        for name, maximum in config["thresholds"].items():
            assert run["checks"][name]["passed"] == (run["metrics"][name] <= maximum)
        assert (run["status"] == "PASS") == all(item["passed"] for item in run["checks"].values())
        assert run["checks"]["hyperparameter_mean_max_standardized_abs_error"]["passed"]
    assert sum(run["status"] == "PASS" for run in evidence["runs"]) == 1
