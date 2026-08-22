"""Reproducible component benchmark for the public ``structnpe`` beta.

This benchmark uses the synthetic conjugate-normal example. It reports
component timings and an honest break-even calculation against the analytic
posterior; it does not make an unconditional speed-superiority claim.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.exact_toy import build_model, exact_posterior, make_observed  # noqa: E402
from structnpe import fit, load_estimator  # noqa: E402


def _measure(function: Callable[[], Any], repeats: int) -> tuple[list[float], Any]:
    times: list[float] = []
    output: Any = None
    for _ in range(repeats):
        started = time.perf_counter()
        output = function()
        times.append(time.perf_counter() - started)
    return times, output


def _summary(values: list[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    return {
        "repeats": len(values),
        "median_seconds": median,
        "mad_seconds": statistics.median(abs(value - median) for value in values),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulations", type=int, default=2_500)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32, help="Number of observed datasets in batch benchmark.")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=61001)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("benchmark_results.json"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 3 or args.draws < 10 or args.batch_size < 1:
        raise ValueError("Use at least 3 repeats, 10 draws, and batch size 1.")
    model = build_model()
    observed = make_observed(seed=args.seed + 1)
    batch_observed = [make_observed(theta=0.35 + 0.01 * index, seed=args.seed + 100 + index) for index in range(args.batch_size)]

    total_fit_start = time.perf_counter()
    estimator = fit(
        model,
        simulations=args.simulations,
        seed=args.seed,
        validation_fraction=0.2,
        hidden_dim=32,
        depth=2,
        components=5,
        epochs=args.epochs,
        batch_size=128,
        patience=7,
        device="cpu",
        progress=not args.quiet,
    )
    total_fit_wall = time.perf_counter() - total_fit_start

    with tempfile.TemporaryDirectory(prefix="structnpe_benchmark_") as temporary:
        artifact = Path(temporary) / "estimator"
        estimator.save(artifact)

        load_times, loaded = _measure(lambda: load_estimator(artifact), args.repeats)
        # The first inference on a newly loaded estimator includes ordinary
        # process-resident initialization but not a new Python interpreter or
        # a guaranteed cold OS page cache. It is labeled accordingly.
        new_load_start = time.perf_counter()
        newly_loaded = load_estimator(artifact)
        load_once = time.perf_counter() - new_load_start
        first_start = time.perf_counter()
        first_result = newly_loaded.infer(observed, draws=args.draws, seed=args.seed + 2)
        first_inference = time.perf_counter() - first_start

        warm_times, warm_result = _measure(
            lambda: loaded.infer(observed, draws=args.draws, seed=args.seed + 2),
            args.repeats,
        )
        batch_times, batch_result = _measure(
            lambda: loaded.infer(batch_observed, draws=args.draws, seed=args.seed + 3, batch=True),
            args.repeats,
        )
        summary_times, _ = _measure(lambda: warm_result.summary(), args.repeats)

    comparator_times, comparator_output = _measure(lambda: exact_posterior(observed), args.repeats * 10)
    conventional_per_dataset = statistics.median(comparator_times)
    amortized_per_dataset = statistics.median(warm_times)
    training_cost = total_fit_wall
    if conventional_per_dataset > amortized_per_dataset:
        break_even: float | None = training_cost / (conventional_per_dataset - amortized_per_dataset)
        break_even_note = "Finite break-even under the measured per-dataset times."
    else:
        break_even = None
        break_even_note = (
            "No finite break-even against this analytic comparator: the exact formula is faster per dataset "
            "than MDN draw generation, before amortizing training."
        )

    single_metadata = dict(warm_result.metadata)
    batch_metadata = dict(batch_result.metadata)
    payload = {
        "schema_version": 1,
        "benchmark": "conjugate_normal_component_timing_v1",
        "configuration": {
            "raw_dataset_shape": list(np.asarray(observed).shape),
            "representation_dimension": 1,
            "parameter_dimension": 1,
            "posterior_draws": args.draws,
            "batch_size": args.batch_size,
            "simulation_budget": args.simulations,
            "epochs_requested": args.epochs,
            "repeats": args.repeats,
            "seed": args.seed,
            "estimator": "five-component diagonal-Gaussian MDN",
        },
        "environment": {
            "platform": platform.platform(),
            "operating_system": platform.system(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "processor": platform.processor() or "unknown",
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "training_device": estimator.training_metadata.get("device"),
            "inference_device": single_metadata.get("device"),
        },
        "timing": {
            "simulation_seconds": estimator.training_metadata["simulation_seconds"],
            "training_excluding_simulation_seconds": estimator.training_metadata["training_seconds"],
            "fit_wall_including_simulation_and_training_seconds": total_fit_wall,
            "estimator_load_warm_process": _summary(load_times),
            "newly_loaded_first_inference": {
                "load_seconds": load_once,
                "inference_seconds": first_inference,
                "combined_seconds": load_once + first_inference,
                "cold_definition": "new estimator object in an existing Python process; OS cache not controlled",
            },
            "one_dataset_warm_inference": _summary(warm_times),
            "batch_warm_inference": _summary(batch_times),
            "one_dataset_preprocessing_seconds_last": single_metadata["preprocessing_seconds"],
            "one_dataset_posterior_draw_seconds_last": single_metadata["posterior_draw_seconds"],
            "batch_preprocessing_seconds_last": batch_metadata["preprocessing_seconds"],
            "batch_posterior_draw_seconds_last": batch_metadata["posterior_draw_seconds"],
            "summary_table": _summary(summary_times),
        },
        "comparator": {
            "name": "analytic conjugate-normal posterior",
            "configuration": {
                "prior": "Normal(0, 1)",
                "likelihood": "20 iid Normal(theta, 1)",
                "output": "closed-form posterior mean and standard deviation",
                "posterior_sampling": False,
                "effective_sample_size": None,
            },
            "timing": _summary(comparator_times),
            "output": {"posterior_mean": comparator_output[0], "posterior_sd": comparator_output[1]},
        },
        "break_even": {
            "formula": "training_seconds / (comparator_seconds_per_dataset - amortized_seconds_per_dataset)",
            "repeated_datasets": break_even,
            "note": break_even_note,
        },
        "claims": {
            "includes_model_loading_in_warm_inference": False,
            "training_reported_separately": True,
            "unconditional_speedup_claim": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "break_even": payload["break_even"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
