"""Train the five-component structnpe MDN on batched Rust simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from structnpe import fit

try:
    from .structnpe_model import build_model
except ImportError:  # pragma: no cover - direct script execution
    from structnpe_model import build_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=20_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1987)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    model = build_model()
    estimator = fit(
        model,
        simulations=args.simulations,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        components=args.components,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        progress=not args.quiet,
    )
    estimator.save(args.output, overwrite=False)
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "estimator_fingerprint": estimator.estimator_fingerprint,
                "simulations": args.simulations,
                "simulation_seconds": estimator.training_metadata["simulation_seconds"],
                "training_seconds": estimator.training_metadata["training_seconds"],
                "best_epoch": estimator.training_metadata["best_epoch"],
                "nonclaim": (
                    "Training completion alone does not establish agreement with the dense-grid posterior."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

