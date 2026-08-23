"""Load the bundled synthetic estimator and draw an approximate posterior."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structnpe import load_estimator


DEFAULT_ESTIMATOR = Path(__file__).resolve().parent / "assets" / "demo_estimator"
OBSERVED_COUNTS = np.array([32, 1, 27, 3, 18, 7, 12, 4, 6, 10], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    estimator = load_estimator(DEFAULT_ESTIMATOR)
    posterior = estimator.infer(OBSERVED_COUNTS, draws=args.draws, seed=args.seed)
    columns = ["parameter", "posterior_mean", "posterior_sd", "q025", "q975"]
    print(posterior.summary()[columns].to_string(index=False))
    print(f"joint_draws={posterior.draws.shape}")
    print("QUICKSTART_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
