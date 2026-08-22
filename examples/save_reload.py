"""Train once, save, and reproduce inference in a fresh Python process."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from structnpe import fit, load_estimator

if __package__:
    from .exact_toy import build_model, make_observed
else:
    from exact_toy import build_model, make_observed


@contextmanager
def _workspace(path: Path | None) -> Iterator[Path]:
    """Yield a requested persistent workspace or a temporary one."""

    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        yield path
        return
    with tempfile.TemporaryDirectory(prefix="structnpe_save_reload_") as tmp:
        yield Path(tmp)


def _child(args: argparse.Namespace) -> int:
    model = build_model()
    estimator = load_estimator(args.artifact, model=model)
    observed = np.load(args.observed, allow_pickle=False)
    result = estimator.infer(observed, draws=args.draws, seed=args.inference_seed)
    np.save(args.child_draws, np.asarray(result.draws, dtype=float), allow_pickle=False)
    args.child_metadata.write_text(
        json.dumps({"model_fingerprint": estimator.model_fingerprint}, indent=2),
        encoding="utf-8",
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--simulations", type=int, default=3_000)
    parser.add_argument("--draws", type=int, default=1_000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=97531)
    parser.add_argument("--quiet", action="store_true")

    # Private child-process arguments used by this same script.
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--artifact", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--observed", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-draws", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--child-metadata", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inference-seed", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        required = [args.artifact, args.observed, args.child_draws, args.child_metadata, args.inference_seed]
        if any(value is None for value in required):
            raise ValueError("incomplete child-process arguments")
        return _child(args)

    with _workspace(args.work_dir) as work:
        artifact = work / "estimator"
        observed_path = work / "observed.npy"
        child_draws_path = work / "child_draws.npy"
        child_metadata_path = work / "child_metadata.json"

        model = build_model()
        estimator = fit(
            model,
            simulations=args.simulations,
            seed=args.seed,
            validation_fraction=0.15,
            hidden_dim=32,
            depth=2,
            components=5,
            epochs=args.epochs,
            batch_size=128,
            patience=8,
            device="cpu",
            progress=not args.quiet,
        )
        observed = make_observed(seed=args.seed + 10_000)
        inference_seed = args.seed + 1
        before = estimator.infer(observed, draws=args.draws, seed=inference_seed)
        estimator.save(artifact)
        np.save(observed_path, observed, allow_pickle=False)

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--artifact",
                str(artifact),
                "--observed",
                str(observed_path),
                "--child-draws",
                str(child_draws_path),
                "--child-metadata",
                str(child_metadata_path),
                "--draws",
                str(args.draws),
                "--inference-seed",
                str(inference_seed),
            ],
            check=True,
        )

        after_draws = np.load(child_draws_path, allow_pickle=False)
        child_metadata = json.loads(child_metadata_path.read_text(encoding="utf-8"))
        draws_equal = np.array_equal(np.asarray(before.draws), after_draws)
        fingerprints_equal = child_metadata["model_fingerprint"] == estimator.model_fingerprint
        if not draws_equal or not fingerprints_equal:
            raise RuntimeError(
                "save/reload reproducibility failed: "
                f"draws_equal={draws_equal}, fingerprints_equal={fingerprints_equal}"
            )

        print("Fresh-process save/reload check passed.")
        print(f"Artifact: {artifact}")
        print(f"Model fingerprint: {estimator.model_fingerprint}")
        print(f"Reproduced draws: {args.draws}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
