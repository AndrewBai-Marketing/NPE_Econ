#!/usr/bin/env python3
"""Run bounded public examples while importing only an installed wheel."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = (ROOT / "src" / "structnpe").resolve()


@dataclass(frozen=True)
class Example:
    script: str
    arguments: tuple[str, ...]
    success_marker: str
    timeout_seconds: int


BASE_EXAMPLES = (
    Example("01_quickstart_saved_estimator.py", ("--draws", "512"), "QUICKSTART_OK", 30),
    Example(
        "02_bayesian_workflow.py",
        ("--draws", "512", "--predictive-replications", "8"),
        "BAYESIAN_WORKFLOW_OK",
        60,
    ),
    Example(
        "04_batch_inference.py",
        ("--datasets", "3", "--draws", "128"),
        "BATCH_INFERENCE_OK",
        30,
    ),
)

NEURAL_EXAMPLES = (
    Example(
        "03_custom_simulator.py",
        ("--simulations", "256", "--epochs", "2", "--draws", "128", "--quiet"),
        "CUSTOM_SIMULATOR_OK",
        180,
    ),
    Example(
        "exact_toy.py",
        (
            "--simulations", "256", "--epochs", "2", "--draws", "128",
            "--output-dir", "{workspace}/exact_toy", "--quiet",
        ),
        "Exact posterior:",
        180,
    ),
    Example(
        "custom_simulator.py",
        ("--simulations", "256", "--epochs", "2", "--draws", "128", "--quiet"),
        "Posterior draws shape:",
        180,
    ),
    Example(
        "save_reload.py",
        (
            "--simulations", "256", "--epochs", "2", "--draws", "128",
            "--work-dir", "{workspace}/save_reload", "--quiet",
        ),
        "Fresh-process save/reload check passed.",
        240,
    ),
)

VALIDATION_EXAMPLES = (
    Example(
        "05_structural_grid_validation.py",
        ("--output-dir", "{workspace}/structural_grid"),
        "STRUCTURAL_GRID_VALIDATION_OK",
        240,
    ),
    Example(
        "eight_schools_validation.py",
        ("--output-dir", "{workspace}/eight_schools"),
        '"status": "PASS"',
        240,
    ),
)

ALL_EXAMPLES = BASE_EXAMPLES + NEURAL_EXAMPLES + VALIDATION_EXAMPLES


def _validate_inventory() -> None:
    executable = {
        path.name
        for path in (ROOT / "examples").glob("*.py")
        if not path.name.startswith("_")
        and 'if __name__ == "__main__"' in path.read_text(encoding="utf-8")
    }
    mapped = {example.script for example in ALL_EXAMPLES}
    if executable != mapped:
        raise RuntimeError(
            "installed-wheel example inventory mismatch: "
            f"unmapped={sorted(executable - mapped)}, missing={sorted(mapped - executable)}"
        )


def _installed_package_origin() -> Path:
    spec = importlib.util.find_spec("structnpe")
    if spec is None or spec.origin is None:
        raise RuntimeError("structnpe is not installed for this interpreter")
    origin = Path(spec.origin).resolve()
    if origin == SOURCE_PACKAGE or SOURCE_PACKAGE in origin.parents:
        raise RuntimeError(f"structnpe resolved to the source tree, not an installed wheel: {origin}")
    return origin


def _run(example: Example, workspace: Path) -> None:
    script = ROOT / "examples" / example.script
    if not script.is_file():
        raise FileNotFoundError(f"missing public example: {script}")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    arguments = tuple(value.format(workspace=str(workspace)) for value in example.arguments)
    command = [sys.executable, str(script), *arguments]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=workspace,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=example.timeout_seconds,
        check=False,
    )
    elapsed = time.perf_counter() - started
    print(f"--- {example.script} ---")
    print(completed.stdout.rstrip())
    print(f"example_wall_seconds={elapsed:.6f}")
    if completed.returncode != 0:
        raise RuntimeError(f"{example.script} failed with exit code {completed.returncode}")
    if example.success_marker not in completed.stdout:
        raise RuntimeError(
            f"{example.script} did not emit its success marker {example.success_marker!r}"
        )


def _run_legacy_tiny_ddc(workspace: Path) -> None:
    """Exercise the shipped legacy project-file example from the installed wheel."""

    source = ROOT / "examples" / "tiny_ddc"
    project = workspace / "tiny_ddc"
    shutil.copytree(source, project)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    commands = (
        ("simulate", "--config", "config.yaml"),
        ("train", "--config", "config.yaml"),
        ("infer", "--config", "config.yaml", "--observed", "observed_example.csv"),
    )
    started = time.perf_counter()
    for arguments in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "structnpe.cli", *arguments],
            cwd=project,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"tiny_ddc {' '.join(arguments)} failed:\n{completed.stdout}"
            )
    summary = project / "runs" / "tiny_ddc" / "posterior_summary.csv"
    if not summary.is_file():
        raise RuntimeError("tiny_ddc did not produce posterior_summary.csv")
    elapsed = time.perf_counter() - started
    print("--- tiny_ddc legacy CLI project ---")
    print(f"example_wall_seconds={elapsed:.6f}")
    print("TINY_DDC_INSTALLED_WHEEL_OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("base", "neural", "validation", "all"), default="base"
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="verify every executable top-level public example is mapped, then exit",
    )
    parser.add_argument(
        "--require-no-torch",
        action="store_true",
        help="fail unless Torch is absent (used by the base-wheel CI job)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_inventory()
    if args.inventory_only:
        print("PUBLIC_EXAMPLE_INVENTORY_OK")
        return 0
    origin = _installed_package_origin()
    if args.require_no_torch and importlib.util.find_spec("torch") is not None:
        raise RuntimeError("base-wheel profile unexpectedly has Torch installed")
    examples: tuple[Example, ...] = ()
    if args.profile in {"base", "all"}:
        examples += BASE_EXAMPLES
    if args.profile in {"neural", "all"}:
        examples += NEURAL_EXAMPLES
    if args.profile in {"validation", "all"}:
        examples += VALIDATION_EXAMPLES
    with tempfile.TemporaryDirectory(prefix="structnpe-wheel-examples-") as directory:
        workspace = Path(directory)
        for example in examples:
            _run(example, workspace)
        if args.profile in {"base", "all"}:
            _run_legacy_tiny_ddc(workspace)
    print(f"installed_structnpe={origin}")
    print(f"profile={args.profile}")
    print("INSTALLED_WHEEL_EXAMPLES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
