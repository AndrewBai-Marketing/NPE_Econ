"""Exercise the saved-estimator support warning without importing Torch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from _replacement_demo import OBSERVED_COUNTS  # noqa: E402
from structnpe import load_estimator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=100.0)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--seed", type=int, default=62001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scale <= 1.0 or args.draws < 1:
        raise ValueError("scale must exceed one and draws must be positive")
    torch_loaded_before = "torch" in sys.modules
    estimator = load_estimator(args.artifact)
    shifted = (np.asarray(OBSERVED_COUNTS, dtype=float) + 1.0) * args.scale
    result = estimator.infer(shifted, draws=args.draws, seed=args.seed, device="cpu")
    diagnostic = result.diagnostics().iloc[0].to_dict()
    torch_loaded_after = "torch" in sys.modules
    warning = str(diagnostic.get("warning") or "")
    passed = bool(diagnostic.get("support_warning")) and "Distribution-support warning" in warning
    passed = passed and not torch_loaded_before and not torch_loaded_after
    payload = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "synthetic_observation": True,
        "observation_scale": args.scale,
        "posterior_draws": args.draws,
        "seed": args.seed,
        "torch_imported": torch_loaded_after,
        "diagnostic": diagnostic,
        "interpretation": "This fixed-distance flag is a diagnostic, not a hypothesis test or guarantee of invalid inference.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
