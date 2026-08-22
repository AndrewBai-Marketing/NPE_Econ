"""Evaluate frozen simulator-calibration gates for a trained Rust estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from structnpe import load_estimator

try:
    from .structnpe_model import build_model
except ImportError:  # pragma: no cover - direct script execution
    from structnpe_model import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimator", type=Path)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path(__file__).with_name("validation_config.json"),
    )
    args = parser.parse_args()
    model = build_model()
    estimator = load_estimator(args.estimator, model=model)
    result = estimator.validate(
        model,
        simulations=args.simulations,
        draws=args.draws,
        seed=args.seed,
        coverages=(0.9,),
    )
    config = json.loads(args.thresholds.read_text(encoding="utf-8"))["calibration"]
    gates = config["gates"]
    widths = {"replacement_cost": 14.0, "maintenance_slope": 5.8}
    all_rows = {row["parameter"]: row for row in result.metrics if row["region"] == "all"}
    gate_results = {
        "minimum_simulations": args.simulations >= config["minimum_simulations_for_acceptance"],
        "bias": all(
            abs(row["bias"]) / widths[name]
            <= gates["maximum_absolute_bias_as_prior_width_fraction"]
            for name, row in all_rows.items()
        ),
        "coverage_90": all(
            gates["coverage_90_lower"] <= row["coverage_90"] <= gates["coverage_90_upper"]
            for row in all_rows.values()
        ),
        "rank_histogram": all(
            row["rank_histogram_l1"] <= gates["maximum_rank_histogram_l1"]
            for row in all_rows.values()
        ),
    }
    payload = {
        "schema_version": 1,
        "simulations": args.simulations,
        "draws_per_simulation": args.draws,
        "metrics": list(result.metrics),
        "gate_results": gate_results,
        "pass": bool(all(gate_results.values())),
        "threshold_config": "replication/rust_1987/validation_config.json",
        "nonclaim": "Calibration is conditional on the maintained prior, simulator, and representation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
