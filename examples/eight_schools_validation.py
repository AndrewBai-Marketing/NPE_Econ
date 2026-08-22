"""Run the canonical Eight Schools smoke comparison."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "replication" / "eight_schools" / "results" / "smoke",
    )
    args = parser.parse_args()
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "replication" / "eight_schools" / "run_validation.py"),
            "--profile",
            "smoke",
            "--output-dir",
            str(args.output_dir),
            "--quiet",
        ],
        cwd=ROOT,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
