"""Standard-library helpers for the public beta replication runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
REPLICATION = ROOT / "replication"
CONFIGS = REPLICATION / "configs"
EXPECTED = REPLICATION / "expected" / "release_0.1.0b1_metrics.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a complete JSON object without leaving a partial report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_version() -> str:
    version_file = ROOT / "src" / "structnpe" / "_version.py"
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
        version_file.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not read __version__ from {version_file}")
    return match.group(1)


def _git_output(arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def git_metadata() -> dict[str, Any]:
    commit = _git_output(["rev-parse", "HEAD"])
    if commit is None:
        return {
            "available": False,
            "commit": None,
            "dirty": None,
            "note": "No readable Git metadata was available to the runner.",
        }
    status = _git_output(["status", "--porcelain"])
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
    }


def machine_metadata() -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in ("structnpe", "numpy", "pandas", "torch", "pytest", "build")
    }
    return {
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_name": Path(sys.executable).name,
        "package_versions": packages,
        "structnpe_source_version": source_version(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYTHONHASHSEED",
            )
        },
    }


def replication_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment.setdefault(key, "1")
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("CUDA_VISIBLE_DEVICES", "")
    return environment


def run_command(
    *,
    name: str,
    command: Sequence[str],
    log_dir: Path,
    timeout_seconds: int,
    display_command: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one bounded stage and retain its combined stdout/stderr."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started_at = utc_now()
    started = time.perf_counter()
    print(f"[{name}] starting", flush=True)
    try:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env=dict(environment or replication_environment()),
            check=False,
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return_code: int | None = int(completed.returncode)
        status = "PASS" if completed.returncode == 0 else "FAIL"
        error = None
    except subprocess.TimeoutExpired as exc:
        captured_stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        captured_stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = captured_stdout + captured_stderr
        return_code = None
        status = "FAIL"
        error = f"Stage exceeded its frozen {timeout_seconds}-second timeout."
    duration = time.perf_counter() - started
    rendered = list(display_command or command)
    log_path.write_text(
        "$ " + " ".join(rendered) + "\n\n" + output,
        encoding="utf-8",
    )
    print(f"[{name}] {status} in {duration:.3f}s", flush=True)
    return {
        "name": name,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": duration,
        "timeout_seconds": int(timeout_seconds),
        "return_code": return_code,
        "command": rendered,
        "log": repository_relative(log_path),
        "error": error,
    }


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_replication_configs() -> dict[str, Any]:
    """Verify frozen replication configs against the predeclared source files."""

    exact = load_json(CONFIGS / "conjugate_normal_smoke.json")
    exact_source_path = ROOT / exact["source_threshold_file"]
    exact_source = load_json(exact_source_path)
    _assert(sha256_file(exact_source_path) == exact["source_threshold_sha256"], "exact threshold hash changed")
    _assert(exact_source.get("declared_before_execution") is True, "exact thresholds are not predeclared")
    exact_profile = exact_source["profiles"][exact["source_profile"]]
    _assert(exact["configuration"] == exact_profile["configuration"], "exact configuration drift")
    _assert(exact["acceptance_maxima"] == exact_profile["acceptance_maxima"], "exact threshold drift")

    structural = load_json(CONFIGS / "structural_grid_smoke.json")
    structural_source_path = ROOT / structural["source_threshold_file"]
    structural_source = load_json(structural_source_path)
    _assert(
        sha256_file(structural_source_path) == structural["source_threshold_sha256"],
        "structural threshold hash changed",
    )
    _assert(structural_source.get("declared_before_execution") is True, "structural thresholds are not predeclared")
    structural_profile = structural_source["profiles"][structural["source_profile"]]
    _assert(structural["configuration"] == structural_profile["configuration"], "structural configuration drift")
    _assert(
        structural["acceptance_maxima"] == structural_profile["acceptance_maxima"],
        "structural threshold drift",
    )

    sbc = load_json(CONFIGS / "sbc_32.json")
    _assert(sbc["cases"] == exact["configuration"]["evaluation_cases"], "SBC case count drift")
    _assert(
        sbc["posterior_draws_per_case"] == exact["configuration"]["posterior_draws"],
        "SBC draw count drift",
    )
    _assert(sbc["acceptance"]["gating"] is False, "tiny SBC must remain descriptive")

    benchmark = load_json(CONFIGS / "cpu_benchmark.json")
    recorded_benchmark = load_json(ROOT / "benchmarks" / "benchmark_results.json")
    recorded_configuration = recorded_benchmark["configuration"]
    mapped = {
        "simulations": recorded_configuration["simulation_budget"],
        "epochs": recorded_configuration["epochs_requested"],
        "posterior_draws": recorded_configuration["posterior_draws"],
        "batch_size": recorded_configuration["batch_size"],
        "repeats": recorded_configuration["repeats"],
        "seed": recorded_configuration["seed"],
        "estimator": recorded_configuration["estimator"],
    }
    _assert(benchmark["configuration"] == mapped, "benchmark configuration drift")
    _assert(benchmark["acceptance"]["gating"] is False, "timing must remain non-gating")

    pipeline = load_json(CONFIGS / "pipeline_smoke.json")
    for stage, stage_config in pipeline["stages"].items():
        _assert(int(stage_config["timeout_seconds"]) > 0, f"invalid timeout for {stage}")
    full = load_json(CONFIGS / "full_profile.json")
    _assert(
        full["status"] == "UNRUN_FUTURE_WORK",
        "full profile must remain explicitly UNRUN_FUTURE_WORK",
    )

    expected = load_json(EXPECTED)
    _assert(expected["release"] == pipeline["release"], "release version mismatch")
    for name, metric in expected["metrics"].items():
        _assert(metric["comparison"] in {"maximum", "descriptive"}, f"unknown comparison for {name}")
        if metric["comparison"] == "maximum":
            _assert(metric["gating"] is True and metric["tolerance"] is not None, f"invalid gate for {name}")
        else:
            _assert(metric["gating"] is False and metric["tolerance"] is None, f"invalid descriptive metric {name}")

    return {
        "status": "PASS",
        "checked_configs": sorted(path.name for path in CONFIGS.glob("*.json")),
        "exact_threshold_sha256": exact["source_threshold_sha256"],
        "structural_threshold_sha256": structural["source_threshold_sha256"],
    }


def deep_get(payload: Any, path: Iterable[str | int]) -> Any:
    current = payload
    for component in path:
        current = current[component]
    return current


def compare_expected(
    expected: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
    active_stages: set[str],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for name, specification in expected["metrics"].items():
        stage = str(specification["stage"])
        if stage not in active_stages:
            continue
        source = str(specification["source"])
        base = {
            "metric": name,
            "stage": stage,
            "historical": specification["historical"],
            "tolerance": specification["tolerance"],
            "comparison": specification["comparison"],
            "gating": bool(specification["gating"]),
        }
        if source not in evidence:
            comparisons.append({**base, "current": None, "difference": None, "passed": False if base["gating"] else None, "status": "UNAVAILABLE"})
            continue
        try:
            current = float(deep_get(evidence[source], specification["path"]))
        except (KeyError, IndexError, TypeError, ValueError):
            comparisons.append({**base, "current": None, "difference": None, "passed": False if base["gating"] else None, "status": "UNAVAILABLE"})
            continue
        historical = float(specification["historical"])
        if specification["comparison"] == "maximum":
            passed: bool | None = current <= float(specification["tolerance"])
        else:
            passed = None
        comparisons.append(
            {
                **base,
                "current": current,
                "difference": current - historical,
                "passed": passed,
                "status": "PASS" if passed is True else ("FAIL" if passed is False else "DESCRIPTIVE"),
            }
        )
    return comparisons
