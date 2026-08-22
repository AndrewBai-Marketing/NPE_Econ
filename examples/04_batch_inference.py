"""Reuse one saved estimator for a batch of synthetic datasets."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from structnpe import load_estimator

from _replacement_demo import build_model, replacement_simulator


DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "assets" / "demo_estimator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--datasets", type=int, default=8)
    parser.add_argument("--draws", type=int, default=2_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.datasets < 1:
        raise ValueError("datasets must be positive")
    model = build_model()
    estimator = load_estimator(args.artifact)
    rng = np.random.default_rng(404)
    theta = model.sample_prior(args.datasets, rng)
    observations = [replacement_simulator(row, rng) for row in theta]

    estimator.infer(observations[:1], draws=32, seed=1, batch=True)
    start = time.perf_counter()
    result = estimator.infer(
        observations,
        draws=args.draws,
        seed=405,
        batch=True,
    )
    elapsed = time.perf_counter() - start
    repeated = estimator.infer(
        observations,
        draws=args.draws,
        seed=405,
        batch=True,
    )
    if not np.array_equal(result.draws, repeated.draws):
        raise RuntimeError("fixed-seed batch inference was not reproducible")
    expected_shape = (args.datasets, args.draws, 2)
    if result.draws.shape != expected_shape:
        raise RuntimeError(f"unexpected batch draw shape {result.draws.shape}")

    summary = result.summary()
    draw_table = result.to_dataframe()
    print(summary.head(2 * min(args.datasets, 3)).to_string(index=False))
    print(f"one_estimator_datasets={args.datasets}")
    print(f"draw_shape={result.draws.shape}")
    print(f"draw_table_rows={len(draw_table)}")
    print(f"inference_seconds_excluding_load_and_training={elapsed:.6f}")
    print("BATCH_INFERENCE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

