#!/usr/bin/env python3
"""Run the two public examples against an installed wheel."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = (ROOT / "src" / "structnpe").resolve()


@dataclass(frozen=True)
class Example:
    script: str
    arguments: tuple[str, ...]
    marker: str
    timeout: int


BASE = (Example("quickstart.py", ("--draws", "512"), "QUICKSTART_OK", 30),)
NEURAL = (
    Example(
        "custom_model.py",
        (
            "--simulations",
            "256",
            "--epochs",
            "2",
            "--draws",
            "128",
            "--output",
            "{workspace}/custom-estimator",
        ),
        "CUSTOM_MODEL_OK",
        180,
    ),
)


def _check_inventory() -> None:
    executable = {
        path.name
        for path in (ROOT / "examples").glob("*.py")
        if 'if __name__ == "__main__"' in path.read_text(encoding="utf-8")
    }
    expected = {example.script for example in BASE + NEURAL}
    if executable != expected:
        raise RuntimeError(
            f"example inventory mismatch: unexpected={sorted(executable - expected)}, "
            f"missing={sorted(expected - executable)}"
        )


def _installed_package() -> Path:
    spec = importlib.util.find_spec("structnpe")
    if spec is None or spec.origin is None:
        raise RuntimeError("structnpe is not installed")
    origin = Path(spec.origin).resolve()
    if origin == SOURCE_PACKAGE or SOURCE_PACKAGE in origin.parents:
        raise RuntimeError(f"structnpe resolved to the source tree: {origin}")
    return origin


def _run(example: Example, workspace: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    arguments = [value.format(workspace=workspace) for value in example.arguments]
    completed = subprocess.run(
        [sys.executable, str(ROOT / "examples" / example.script), *arguments],
        cwd=workspace,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=example.timeout,
        check=False,
    )
    print(f"--- {example.script} ---")
    print(completed.stdout.rstrip())
    if completed.returncode != 0:
        raise RuntimeError(f"{example.script} exited with {completed.returncode}")
    if example.marker not in completed.stdout:
        raise RuntimeError(f"{example.script} did not emit {example.marker!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("base", "neural", "all"), default="base")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--require-no-torch", action="store_true")
    args = parser.parse_args()

    _check_inventory()
    if args.inventory_only:
        print("PUBLIC_EXAMPLE_INVENTORY_OK")
        return 0

    origin = _installed_package()
    if args.require_no_torch and importlib.util.find_spec("torch") is not None:
        raise RuntimeError("Torch is installed in the base-only environment")

    examples = BASE if args.profile == "base" else NEURAL
    if args.profile == "all":
        examples = BASE + NEURAL
    with tempfile.TemporaryDirectory(prefix="structnpe-examples-") as directory:
        workspace = Path(directory)
        for example in examples:
            _run(example, workspace)

    print(f"installed_structnpe={origin}")
    print("INSTALLED_WHEEL_EXAMPLES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
