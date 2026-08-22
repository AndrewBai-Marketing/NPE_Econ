"""Load the synthetic demo estimator and obtain a joint posterior on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from structnpe import load_estimator

from _replacement_demo import OBSERVED_COUNTS


DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "assets" / "demo_estimator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimator = load_estimator(args.artifact)
    observed_data = OBSERVED_COUNTS
    draws_requested = args.draws
    seed = args.seed

    # README:quickstart:start
    result = estimator.infer(
        observed_data,
        draws=draws_requested,
        seed=seed,
        device="cpu",
    )

    columns = [
        "parameter",
        "posterior_mean",
        "posterior_sd",
        "median",
        "q025",
        "q975",
    ]
    print(result.summary()[columns].to_string(index=False))
    draws = result.to_dataframe()
    # README:quickstart:end

    repeated = estimator.infer(
        observed_data,
        draws=draws_requested,
        seed=seed,
        device="cpu",
    )
    if not np.array_equal(result.draws, repeated.draws):
        raise RuntimeError("fixed-seed posterior draws were not reproducible")
    if draws.shape != (draws_requested, 3):
        raise RuntimeError(f"unexpected draw table shape: {draws.shape}")

    print(f"joint_draw_table_shape={draws.shape}")
    print("QUICKSTART_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
