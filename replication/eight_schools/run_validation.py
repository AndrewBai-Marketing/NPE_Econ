"""Validate the structnpe Eight Schools hyperparameter posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
for variable in THREAD_ENVIRONMENT:
    os.environ.setdefault(variable, "1")

import numpy as np

from structnpe import fit

try:
    from .exact import hyperparameter_moments, sample_exact_posterior
    from .model import EFFECTS, STANDARD_ERRORS, build_model
except ImportError:  # pragma: no cover - direct execution
    from exact import hyperparameter_moments, sample_exact_posterior
    from model import EFFECTS, STANDARD_ERRORS, build_model


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
PARAMETER_NAMES = ("mu", "tau")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round_floats(value: Any, *, digits: int = 14) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round_floats(item, digits=digits) for item in value]
    if isinstance(value, dict):
        return {
            key: _round_floats(item, digits=digits) for key, item in value.items()
        }
    return value


def _ks_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Return the exact two-sample Kolmogorov--Smirnov distance."""

    left = np.sort(np.asarray(first, dtype=float))
    right = np.sort(np.asarray(second, dtype=float))
    pooled = np.sort(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, pooled, side="right") / left.size
    right_cdf = np.searchsorted(right, pooled, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _joint_quantile_grid_tv(
    approximate: np.ndarray,
    exact_draws: np.ndarray,
    *,
    bins: int = 20,
) -> float:
    """Return joint TV on bins fixed by exact marginal quantiles."""

    if bins < 2:
        raise ValueError("bins must be at least two")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = []
    for index in range(2):
        parameter_edges = np.quantile(exact_draws[:, index], quantiles)
        if np.any(np.diff(parameter_edges) <= 0.0):
            raise ValueError("exact draws do not define distinct quantile bins")
        parameter_edges[0] = -np.inf
        parameter_edges[-1] = np.inf
        edges.append(parameter_edges)
    approximate_mass = np.histogram2d(
        approximate[:, 0], approximate[:, 1], bins=edges
    )[0]
    exact_mass = np.histogram2d(
        exact_draws[:, 0], exact_draws[:, 1], bins=edges
    )[0]
    approximate_mass /= approximate_mass.sum()
    exact_mass /= exact_mass.sum()
    return float(0.5 * np.abs(approximate_mass - exact_mass).sum())


def _run_metrics(
    approximate: np.ndarray,
    exact_draws: np.ndarray,
    exact_mean: np.ndarray,
    exact_sd: np.ndarray,
    exact_correlation: float,
) -> tuple[dict[str, Any], list[dict[str, float | str]]]:
    approximate = np.asarray(approximate, dtype=float)
    exact_draws = np.asarray(exact_draws, dtype=float)
    if approximate.ndim != 2 or approximate.shape[1] != 2:
        raise ValueError("approximate draws must have shape (draws, 2)")
    if exact_draws.ndim != 2 or exact_draws.shape[1] != 2:
        raise ValueError("exact draws must have shape (draws, 2)")

    approximate_mean = approximate.mean(axis=0)
    approximate_sd = approximate.std(axis=0, ddof=1)
    approximate_interval = np.quantile(approximate, [0.025, 0.975], axis=0).T
    exact_interval = np.quantile(exact_draws, [0.025, 0.975], axis=0).T
    standardized_mean_error = np.abs(approximate_mean - exact_mean) / exact_sd
    relative_sd_error = np.abs(approximate_sd - exact_sd) / exact_sd
    standardized_endpoint_error = (
        np.abs(approximate_interval - exact_interval) / exact_sd[:, None]
    )
    marginal_cdf_error = np.array(
        [
            _ks_distance(approximate[:, index], exact_draws[:, index])
            for index in range(2)
        ]
    )
    approximate_correlation = float(np.corrcoef(approximate, rowvar=False)[0, 1])
    joint_grid_tv = _joint_quantile_grid_tv(approximate, exact_draws)

    rows = [
        {
            "parameter": name,
            "quadrature_mean": float(exact_mean[index]),
            "approximate_mean": float(approximate_mean[index]),
            "quadrature_sd": float(exact_sd[index]),
            "approximate_sd": float(approximate_sd[index]),
            "exact_draw_q025": float(exact_interval[index, 0]),
            "approximate_q025": float(approximate_interval[index, 0]),
            "exact_draw_q975": float(exact_interval[index, 1]),
            "approximate_q975": float(approximate_interval[index, 1]),
            "mean_error_in_exact_sd": float(standardized_mean_error[index]),
            "sd_relative_error": float(relative_sd_error[index]),
            "marginal_cdf_supremum": float(marginal_cdf_error[index]),
        }
        for index, name in enumerate(PARAMETER_NAMES)
    ]
    metrics: dict[str, Any] = {
        "hyperparameter_mean_max_standardized_abs_error": float(
            np.max(standardized_mean_error)
        ),
        "hyperparameter_sd_max_relative_abs_error": float(np.max(relative_sd_error)),
        "central_95_endpoint_max_standardized_abs_error": float(
            np.max(standardized_endpoint_error)
        ),
        "marginal_cdf_max_abs_error": float(np.max(marginal_cdf_error)),
        "correlation_abs_error": float(
            abs(approximate_correlation - exact_correlation)
        ),
        "joint_20x20_quantile_grid_total_variation": joint_grid_tv,
        "tau_support_violation_rate": float(np.mean(approximate[:, 1] <= 0.0)),
        "marginal_cdf_supremum": marginal_cdf_error.tolist(),
        "approximate_correlation": approximate_correlation,
    }
    return metrics, rows


def _checks(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    return {
        name: {
            "value": float(metrics[name]),
            "maximum": float(maximum),
            "passed": bool(np.isfinite(metrics[name]) and metrics[name] <= maximum),
        }
        for name, maximum in thresholds.items()
    }


def compact_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop timings and output paths while retaining reproducibility metadata."""

    compact = {
        "schema_version": payload["schema_version"],
        "benchmark": payload["benchmark"],
        "status": payload["status"],
        "canonical_configuration": payload["canonical_configuration"],
        "configuration_file_sha256": payload["configuration_file_sha256"],
        "effective_configuration_sha256": payload[
            "effective_configuration_sha256"
        ],
        "configuration": payload["configuration"],
        "data": payload["data"],
        "environment": payload["environment"],
        "exact_reference": payload["exact_reference"],
        "runs": [
            {
                "seed": run["seed"],
                "status": run["status"],
                "metrics": run["metrics"],
                "checks": run["checks"],
                "parameters": run["parameters"],
                "model_fingerprint": run["model_fingerprint"],
                "estimator_fingerprint": run["estimator_fingerprint"],
                "training_summary": run["training_summary"],
            }
            for run in payload["runs"]
        ],
        "all_seeds_pass": payload["all_seeds_pass"],
        "nonclaim": payload["nonclaim"],
    }
    return _round_floats(compact)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--posterior-draws", type=int, default=None)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = dict(default_config)
    for argument, key in (
        (args.simulations, "simulations"),
        (args.epochs, "epochs"),
        (args.posterior_draws, "posterior_draws"),
    ):
        if argument is not None:
            config[key] = int(argument)
    if args.seeds:
        config["seeds"] = [int(seed) for seed in args.seeds]

    import torch

    torch.set_num_threads(int(config["torch_num_threads"]))

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    exact_start = time.perf_counter()
    moments = hyperparameter_moments(points=int(config["quadrature_points"]))
    exact_mean = np.asarray(moments["mean"], dtype=float)
    exact_sd = np.asarray(moments["sd"], dtype=float)
    exact_draws = sample_exact_posterior(
        int(config["exact_draws"]),
        seed=int(config["exact_seed"]),
        quadrature_points=int(config["quadrature_points"]),
    )[:, :2]
    exact_seconds = time.perf_counter() - exact_start

    thresholds = {key: float(value) for key, value in config["thresholds"].items()}
    runs: list[dict[str, Any]] = []
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        run_output = output / f"seed_{seed}"
        training_start = time.perf_counter()
        estimator = fit(
            build_model(),
            simulations=int(config["simulations"]),
            seed=seed,
            hidden_dim=int(config["hidden_dim"]),
            depth=int(config["depth"]),
            components=int(config["components"]),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            patience=int(config["patience"]),
            device="cpu",
            output_dir=run_output / "estimator",
            progress=not args.quiet,
        )
        training_seconds = time.perf_counter() - training_start
        inference_start = time.perf_counter()
        approximate = np.asarray(
            estimator.infer(
                EFFECTS,
                draws=int(config["posterior_draws"]),
                seed=seed + int(config["inference_seed_offset"]),
            ).draws
        )
        inference_seconds = time.perf_counter() - inference_start
        metrics, parameters = _run_metrics(
            approximate,
            exact_draws,
            exact_mean,
            exact_sd,
            float(moments["correlation"]),
        )
        checks = _checks(metrics, thresholds)
        status = "PASS" if all(item["passed"] for item in checks.values()) else "FAIL"
        runs.append(
            {
                "seed": seed,
                "status": status,
                "metrics": metrics,
                "checks": checks,
                "parameters": parameters,
                "model_fingerprint": estimator.model_fingerprint,
                "estimator_fingerprint": estimator.estimator_fingerprint,
                "training_summary": {
                    "best_epoch": int(estimator.training_metadata["best_epoch"]),
                    "best_validation_nll": float(
                        estimator.training_metadata["best_validation_nll"]
                    ),
                    "number_of_simulations": int(
                        estimator.training_metadata["number_of_simulations"]
                    ),
                    "simulation_seed": int(
                        estimator.training_metadata["simulation_seed"]
                    ),
                    "training_seed": int(estimator.training_metadata["training_seed"]),
                },
                "timing_seconds": {
                    "training": training_seconds,
                    "inference": inference_seconds,
                },
            }
        )

    all_seeds_pass = all(run["status"] == "PASS" for run in runs)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": "eight_schools_marginal_hyperposterior_v2",
        "status": "PASS" if all_seeds_pass else "FAIL",
        "all_seeds_pass": all_seeds_pass,
        "canonical_configuration": config == default_config,
        "configuration_file_sha256": _sha256(CONFIG_PATH),
        "effective_configuration_sha256": _json_sha256(config),
        "configuration": config,
        "data": {
            "reported_effects": EFFECTS.tolist(),
            "reported_standard_errors": STANDARD_ERRORS.tolist(),
        },
        "exact_reference": {
            "method": "dense log-tau quadrature with conditional Gaussian mu",
            "mean": exact_mean.tolist(),
            "sd": exact_sd.tolist(),
            "correlation": float(moments["correlation"]),
            "draws_for_distributional_metrics": int(config["exact_draws"]),
            "draw_seed": int(config["exact_seed"]),
        },
        "runs": runs,
        "timing_seconds": {"exact_reference": exact_seconds},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENVIRONMENT
            },
        },
        "nonclaim": (
            "This validates the marginalized (mu, tau) posterior for one public "
            "dataset and stated prior. It does not validate arbitrary simulators, "
            "latent-variable parameterizations, or posterior geometries."
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "compact_metrics.json").write_text(
        json.dumps(compact_result(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "seeds": [
                    {
                        "seed": run["seed"],
                        "status": run["status"],
                        "metrics": run["metrics"],
                    }
                    for run in runs
                ],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if all_seeds_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
