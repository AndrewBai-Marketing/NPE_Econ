"""Compare structnpe with the quadrature Eight Schools posterior."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from structnpe import fit

try:
    from .exact import sample_exact_posterior
    from .model import EFFECTS, SCHOOL_NAMES, build_model
except ImportError:  # pragma: no cover - direct execution
    from exact import sample_exact_posterior
    from model import EFFECTS, SCHOOL_NAMES, build_model


HERE = Path(__file__).resolve().parent


def _correlation(draws: np.ndarray) -> np.ndarray:
    return np.corrcoef(np.asarray(draws, dtype=float), rowvar=False)


def _metrics(approximate: np.ndarray, exact: np.ndarray) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    exact_mean = exact.mean(axis=0)
    approximate_mean = approximate.mean(axis=0)
    exact_sd = exact.std(axis=0, ddof=1)
    approximate_sd = approximate.std(axis=0, ddof=1)
    scale = np.maximum(exact_sd, 1.0)
    exact_interval = np.quantile(exact, [0.025, 0.975], axis=0).T
    approximate_interval = np.quantile(approximate, [0.025, 0.975], axis=0).T
    names = ["mu", "tau", *[f"theta_{name}" for name in SCHOOL_NAMES]]
    rows = [
        {
            "parameter": name,
            "exact_mean": float(exact_mean[index]),
            "approximate_mean": float(approximate_mean[index]),
            "exact_sd": float(exact_sd[index]),
            "approximate_sd": float(approximate_sd[index]),
            "exact_q025": float(exact_interval[index, 0]),
            "approximate_q025": float(approximate_interval[index, 0]),
            "exact_q975": float(exact_interval[index, 1]),
            "approximate_q975": float(approximate_interval[index, 1]),
        }
        for index, name in enumerate(names)
    ]
    return {
        "parameter_mean_standardized_mae": float(np.mean(np.abs(approximate_mean - exact_mean) / scale)),
        "parameter_sd_standardized_mae": float(np.mean(np.abs(approximate_sd - exact_sd) / scale)),
        "central_95_endpoint_standardized_mae": float(
            np.mean(np.abs(approximate_interval - exact_interval) / scale[:, None])
        ),
        "correlation_max_abs_error": float(
            np.max(np.abs(_correlation(approximate) - _correlation(exact)))
        ),
        "tau_mean_abs_error": float(abs(approximate_mean[1] - exact_mean[1])),
        "predictive_mean_max_abs_error": float(
            np.max(np.abs(approximate_mean[2:] - exact_mean[2:]))
        ),
        "tau_support_violation_rate": float(np.mean(approximate[:, 1] <= 0.0)),
    }, rows


def _write_forest_plot(path: Path, rows: list[dict[str, float | str]]) -> str:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return "not generated; install matplotlib"
    school_rows = rows[2:]
    y = np.arange(len(school_rows))
    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    exact_mean = np.array([float(row["exact_mean"]) for row in school_rows])
    exact_low = np.array([float(row["exact_q025"]) for row in school_rows])
    exact_high = np.array([float(row["exact_q975"]) for row in school_rows])
    approximate_mean = np.array([float(row["approximate_mean"]) for row in school_rows])
    approximate_low = np.array([float(row["approximate_q025"]) for row in school_rows])
    approximate_high = np.array([float(row["approximate_q975"]) for row in school_rows])
    axis.errorbar(
        exact_mean,
        y - 0.12,
        xerr=[exact_mean - exact_low, exact_high - exact_mean],
        fmt="o",
        label="Quadrature reference",
    )
    axis.errorbar(
        approximate_mean,
        y + 0.12,
        xerr=[approximate_mean - approximate_low, approximate_high - approximate_mean],
        fmt="s",
        label="structnpe",
    )
    axis.scatter(EFFECTS, y, marker="x", color="black", label="Reported estimate")
    axis.axvline(0.0, color="0.6", linewidth=1)
    axis.set_yticks(y, SCHOOL_NAMES)
    axis.set_xlabel("Coaching effect (SAT points)")
    axis.set_ylabel("School")
    axis.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "smoke")
    parser.add_argument("--simulations", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--posterior-draws", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = HERE / f"config_{args.profile}.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.profile == "full" and config.get("run_status") == "UNRUN_FUTURE_WORK":
        raise SystemExit("The full Eight Schools profile is frozen as UNRUN_FUTURE_WORK.")
    for argument, key in (
        (args.simulations, "simulations"),
        (args.epochs, "epochs"),
        (args.posterior_draws, "posterior_draws"),
    ):
        if argument is not None:
            config[key] = int(argument)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    exact_start = time.perf_counter()
    exact_draws = sample_exact_posterior(
        int(config["exact_draws"]),
        seed=int(config["seed"]) + 1,
        quadrature_points=int(config["quadrature_points"]),
    )
    exact_seconds = time.perf_counter() - exact_start

    model = build_model()
    training_start = time.perf_counter()
    estimator = fit(
        model,
        simulations=int(config["simulations"]),
        seed=int(config["seed"]),
        hidden_dim=int(config["hidden_dim"]),
        depth=int(config["depth"]),
        components=int(config["components"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        patience=int(config["patience"]),
        device="cpu",
        output_dir=output / "estimator",
        progress=not args.quiet,
    )
    training_seconds = time.perf_counter() - training_start
    inference_start = time.perf_counter()
    result = estimator.infer(
        EFFECTS,
        draws=int(config["posterior_draws"]),
        seed=int(config["seed"]) + 2,
    )
    inference_seconds = time.perf_counter() - inference_start
    metrics, parameter_rows = _metrics(np.asarray(result.draws), exact_draws)
    thresholds = dict(config["thresholds"])
    checks = {
        name: {
            "value": metrics[name],
            "maximum": float(maximum),
            "passed": bool(np.isfinite(metrics[name]) and metrics[name] <= float(maximum)),
        }
        for name, maximum in thresholds.items()
    }
    status = "PASS" if all(item["passed"] for item in checks.values()) else "FAIL"
    figure = _write_forest_plot(output / "eight_schools_forest.png", parameter_rows)
    payload = {
        "schema_version": 1,
        "benchmark": "eight_schools_quadrature_v1",
        "profile": args.profile,
        "status": status,
        "configuration": config,
        "data": {
            "reported_effects": EFFECTS.tolist(),
            "reported_standard_errors": [15, 10, 16, 11, 9, 11, 10, 18],
        },
        "metrics": metrics,
        "checks": checks,
        "parameters": parameter_rows,
        "timing_seconds": {
            "exact_reference": exact_seconds,
            "training": training_seconds,
            "inference": inference_seconds,
        },
        "model_fingerprint": estimator.model_fingerprint,
        "figure": figure,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "nonclaim": (
            "This is one canonical hierarchical-model benchmark under the stated "
            "priors. It does not establish accuracy for other simulators or posterior geometries."
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "metrics": metrics, "output": str(output)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

