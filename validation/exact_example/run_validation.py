"""Validate the beta estimator against an analytic conjugate-normal posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from exact_toy import (  # noqa: E402
    N_OBSERVATIONS,
    OBSERVATION_SD,
    PRIOR_SD,
    build_model,
    exact_posterior,
    normal_simulator,
)
from structnpe import fit  # noqa: E402


THRESHOLD_PATH = Path(__file__).with_name("thresholds.json")


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
    metadata = {
        "threshold_file": str(THRESHOLD_PATH.relative_to(ROOT)),
        "threshold_sha256": hashlib.sha256(raw).hexdigest(),
        "threshold_schema_version": payload["schema_version"],
        "benchmark": payload["benchmark"],
        "target": payload["target"],
        "decision_rule": payload["decision_rule"],
    }
    return dict(profile["configuration"]), dict(profile["acceptance_maxima"]), metadata


def _normal_quantile_offsets(n_draws: int) -> np.ndarray:
    normal = NormalDist()
    probabilities = (np.arange(int(n_draws), dtype=float) + 0.5) / int(n_draws)
    return np.array([normal.inv_cdf(float(q)) for q in probabilities], dtype=float)


def _evaluate(
    draws: np.ndarray,
    observed: np.ndarray,
    truth: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    samples = np.asarray(draws, dtype=float)
    if samples.ndim != 3 or samples.shape[0] != observed.shape[0] or samples.shape[2] != 1:
        raise ValueError(
            "batch inference must return draws with shape "
            f"(cases, draws, 1); got {samples.shape}"
        )

    exact = np.array([exact_posterior(row) for row in observed], dtype=float)
    exact_mean = exact[:, 0]
    exact_sd = exact[:, 1]
    approx = samples[:, :, 0]
    approx_mean = approx.mean(axis=1)
    approx_sd = approx.std(axis=1, ddof=1)
    approx_variance = approx.var(axis=1, ddof=1)
    exact_variance = exact_sd**2
    approx_lo, approx_hi = np.quantile(approx, [0.025, 0.975], axis=1)
    z975 = NormalDist().inv_cdf(0.975)
    exact_lo = exact_mean - z975 * exact_sd
    exact_hi = exact_mean + z975 * exact_sd

    offsets = _normal_quantile_offsets(approx.shape[1])
    exact_quantiles = exact_mean[:, None] + exact_sd[:, None] * offsets[None, :]
    quantile_grid_l1 = np.mean(np.abs(np.sort(approx, axis=1) - exact_quantiles), axis=1)

    predictive_exact_sd = np.sqrt(exact_variance + OBSERVATION_SD**2 / N_OBSERVATIONS)
    predictive_approx_sd = np.sqrt(approx_variance + OBSERVATION_SD**2 / N_OBSERVATIONS)
    empirical_coverage = float(np.mean((truth >= approx_lo) & (truth <= approx_hi)))
    exact_coverage = float(np.mean((truth >= exact_lo) & (truth <= exact_hi)))

    metrics = {
        "posterior_mean_mae": float(np.mean(np.abs(approx_mean - exact_mean))),
        "posterior_sd_mae": float(np.mean(np.abs(approx_sd - exact_sd))),
        "posterior_variance_mae": float(np.mean(np.abs(approx_variance - exact_variance))),
        "central_95_endpoint_mae": float(
            np.mean(np.column_stack([np.abs(approx_lo - exact_lo), np.abs(approx_hi - exact_hi)]))
        ),
        "marginal_quantile_grid_l1_mean": float(np.mean(quantile_grid_l1)),
        "coverage_95": empirical_coverage,
        "coverage_95_abs_error": abs(empirical_coverage - 0.95),
        "predictive_sd_mae": float(np.mean(np.abs(predictive_approx_sd - predictive_exact_sd))),
    }
    comparison = {
        "aggregate": {
            "exact_posterior_mean_average": float(np.mean(exact_mean)),
            "approximate_posterior_mean_average": float(np.mean(approx_mean)),
            "exact_posterior_sd_average": float(np.mean(exact_sd)),
            "approximate_posterior_sd_average": float(np.mean(approx_sd)),
            "exact_posterior_variance_average": float(np.mean(exact_variance)),
            "approximate_posterior_variance_average": float(np.mean(approx_variance)),
            "exact_95_coverage": exact_coverage,
            "approximate_95_coverage": empirical_coverage,
            "exact_predictive_sd_average": float(np.mean(predictive_exact_sd)),
            "approximate_predictive_sd_average": float(np.mean(predictive_approx_sd)),
        },
        "cases": [
            {
                "case": int(i),
                "truth": float(truth[i]),
                "observed_mean": float(np.mean(observed[i])),
                "exact_mean": float(exact_mean[i]),
                "approximate_mean": float(approx_mean[i]),
                "exact_sd": float(exact_sd[i]),
                "approximate_sd": float(approx_sd[i]),
                "exact_q025": float(exact_lo[i]),
                "approximate_q025": float(approx_lo[i]),
                "exact_q975": float(exact_hi[i]),
                "approximate_q975": float(approx_hi[i]),
                "quantile_grid_l1": float(quantile_grid_l1[i]),
            }
            for i in range(len(truth))
        ],
    }
    return metrics, comparison


def _sbc_diagnostics(draws: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Summarize ranks and central-interval coverage for the scalar target."""

    samples = np.asarray(draws, dtype=float)[:, :, 0]
    values = np.asarray(truth, dtype=float)
    posterior_mean = np.mean(samples, axis=1)
    ranks = np.mean(samples <= values[:, None], axis=1)
    levels = (0.5, 0.8, 0.9, 0.95)

    def row(label: str, mask: np.ndarray) -> dict[str, Any]:
        error = posterior_mean[mask] - values[mask]
        result: dict[str, Any] = {
            "parameter": "theta",
            "region": label,
            "n": int(np.sum(mask)),
            "bias": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mean_rank_fraction": float(np.mean(ranks[mask])),
        }
        for level in levels:
            alpha = (1.0 - level) / 2.0
            lo, hi = np.quantile(samples[mask], [alpha, 1.0 - alpha], axis=1)
            result[f"coverage_{int(level * 100)}"] = float(
                np.mean((values[mask] >= lo) & (values[mask] <= hi))
            )
        return result

    cuts = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    regions = [
        row("all", np.ones(len(values), dtype=bool)),
        row("lower", values <= cuts[0]),
        row("middle", (values > cuts[0]) & (values <= cuts[1])),
        row("upper", values > cuts[1]),
    ]
    histogram, edges = np.histogram(ranks, bins=np.linspace(0.0, 1.0, 11))
    return {
        "configuration": {
            "cases": int(len(values)),
            "posterior_draws_per_case": int(samples.shape[1]),
            "rank_histogram_bins": 10,
            "profile_role": "tiny smoke" if len(values) < 100 else "full manual",
        },
        "by_parameter_and_region": regions,
        "rank_histogram": {
            "edges": edges.tolist(),
            "counts": histogram.tolist(),
        },
        "note": (
            "Coverage and ranks are descriptive at this finite case count and are conditional on the "
            "declared prior, simulator, representation, and fitted approximation."
        ),
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Exact posterior validation",
        "",
        f"Status: **{payload['status']}**",
        "",
        "The target is the analytic posterior under the declared Normal prior,",
        "Normal simulator, and sample-mean representation. Passing this benchmark",
        "does not establish structural identification or correct specification.",
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
            "",
            "## Simulation-based calibration smoke summary",
            "",
            "| region | n | bias | RMSE | cov. 50 | cov. 80 | cov. 90 | cov. 95 | mean rank |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["sbc"]["by_parameter_and_region"]:
        lines.append(
            f"| {row['region']} | {row['n']} | {row['bias']:.6g} | {row['rmse']:.6g} | "
            f"{row['coverage_50']:.3f} | {row['coverage_80']:.3f} | {row['coverage_90']:.3f} | "
            f"{row['coverage_95']:.3f} | {row['mean_rank_fraction']:.3f} |"
        )
    lines.extend(["", payload["sbc"]["note"]])
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
    train_start = time.perf_counter()
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
    training_seconds = time.perf_counter() - train_start

    rng = np.random.default_rng(config["evaluation_seed"])
    truth = rng.normal(0.0, PRIOR_SD, size=config["evaluation_cases"])
    observed = np.vstack([normal_simulator(np.array([theta]), rng) for theta in truth])
    inference_start = time.perf_counter()
    result = estimator.infer(
        observed,
        draws=config["posterior_draws"],
        seed=config["inference_seed"],
        batch=True,
    )
    inference_seconds = time.perf_counter() - inference_start
    metrics, comparison = _evaluate(np.asarray(result.draws), observed, truth)

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
        "thresholds": threshold_metadata,
        "metrics": metrics,
        "comparison": comparison,
        "sbc": _sbc_diagnostics(np.asarray(result.draws), truth),
        "checks": checks,
        "timing_seconds": {
            "training": training_seconds,
            "batch_inference": inference_seconds,
        },
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
            "Agreement here validates posterior approximation only for this declared "
            "simulator, prior, representation, configuration, and threshold profile."
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
