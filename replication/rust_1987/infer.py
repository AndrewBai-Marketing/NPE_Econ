"""Run single empirical or batched simulated inference with a trained estimator."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from structnpe import load_estimator

try:
    from .constants import NUM_STATES
    from .preprocess import read_processed_csv
    from .structnpe_model import build_model
except ImportError:  # pragma: no cover - direct script execution
    from constants import NUM_STATES
    from preprocess import read_processed_csv
    from structnpe_model import build_model


def _summary_rows(result: object) -> list[dict[str, object]]:
    table = result.summary(point="mean")
    return json.loads(table.to_json(orient="records"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimator", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--batch-simulations", type=int, default=0)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1988)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.data is None) == (args.batch_simulations > 0):
        raise SystemExit("Specify exactly one of --data or a positive --batch-simulations.")
    model = build_model()
    estimator = load_estimator(args.estimator, model=model)
    if args.data is not None:
        panel = read_processed_csv(args.data)
        observed: object = panel.observation_summary(NUM_STATES)
        batch = False
    else:
        rng = np.random.default_rng(args.seed)
        truth = model.sample_prior(args.batch_simulations, rng)
        observed = model.simulate_batch(truth, rng)
        batch = True
    started = time.perf_counter()
    result = estimator.infer(
        observed,
        draws=args.draws,
        seed=args.seed + 1,
        batch=batch,
        device="cpu",
    )
    payload = {
        "estimator_fingerprint": estimator.estimator_fingerprint,
        "batch": batch,
        "datasets": int(result.metadata["datasets"]),
        "draws_per_dataset": args.draws,
        "wall_seconds": time.perf_counter() - started,
        "summary": _summary_rows(result),
        "posterior_correlation": np.asarray(result.correlation()).tolist(),
        "support_diagnostics": list(result.support_diagnostics),
        "nonclaim": (
            "These are approximate posterior draws conditional on the simulator, prior, "
            "representation, and trained MDN; they are not Rust's MLE."
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

