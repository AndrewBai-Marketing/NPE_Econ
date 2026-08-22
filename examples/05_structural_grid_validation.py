"""Run the predeclared structural exact-grid smoke validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "validation" / "structural_example" / "run_validation.py"
DEFAULT_OUTPUT = ROOT / "replication" / "results" / "structural_grid_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--profile",
            "smoke",
            "--output-dir",
            str(args.output_dir),
            "--quiet",
        ],
        check=True,
        cwd=ROOT,
    )
    metrics_path = args.output_dir / "validation_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    checks = payload["checks"]
    if payload["status"] != "PASS" or len(checks) != 12:
        raise RuntimeError("structural validation did not pass all twelve declared checks")
    if not all(bool(check["passed"]) for check in checks.values()):
        raise RuntimeError("at least one declared structural check failed")

    for name, check in checks.items():
        print(
            f"{name}: value={check['value']:.8g}, "
            f"maximum={check['maximum']:.8g}, passed={check['passed']}"
        )
    comparison = payload["comparison"]["policy_replacement_share"]
    print(
        "joint_grid_voronoi_total_variation="
        f"{payload['metrics']['joint_grid_voronoi_total_variation']:.8g}"
    )
    print(f"policy_mean_exact={comparison['exact_mean']:.8g}")
    print(f"policy_mean_approximate={comparison['approximate_mean']:.8g}")
    print(
        "predictive_cell_probability_max_abs_error="
        f"{payload['metrics']['predictive_cell_probability_max_abs_error']:.8g}"
    )
    print("STRUCTURAL_GRID_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

