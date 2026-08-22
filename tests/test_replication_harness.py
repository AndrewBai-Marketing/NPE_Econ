from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "replication" / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from helpers import (  # noqa: E402
    EXPECTED,
    compare_expected,
    load_json,
    machine_metadata,
    validate_replication_configs,
)
from run_all import STAGE_ORDER  # noqa: E402


def _run_runner(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "run_all.py"), *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_frozen_configs_match_predeclared_sources_and_stage_contract() -> None:
    audit = validate_replication_configs()
    assert audit["status"] == "PASS"
    assert len(audit["exact_threshold_sha256"]) == 64
    assert len(audit["structural_threshold_sha256"]) == 64
    assert STAGE_ORDER == (
        "package_tests",
        "build_distributions",
        "wheel_install",
        "sdist_install",
        "conjugate_normal",
        "sbc_32",
        "structural_grid",
        "support_warning",
        "save_reload",
        "cpu_neural",
        "benchmark",
    )


def test_dry_run_is_selectable_and_does_not_execute_stages(tmp_path: Path) -> None:
    report = tmp_path / "dry-run.json"
    completed = _run_runner(
        "--profile",
        "smoke",
        "--only",
        "conjugate_normal,sbc_32,support_warning",
        "--skip",
        "support_warning",
        "--dry-run",
        "--results-dir",
        str(tmp_path / "results"),
        "--report",
        str(report),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "PLANNED"
    statuses = {row["name"]: row["status"] for row in payload["stages"]}
    assert statuses["conjugate_normal"] == "PLANNED"
    assert statuses["sbc_32"] == "PLANNED"
    assert statuses["support_warning"] == "SKIPPED"
    assert statuses["benchmark"] == "NOT_SELECTED"
    assert not (tmp_path / "results" / "conjugate_normal").exists()


def test_full_profile_is_explicit_future_work_and_launches_nothing(tmp_path: Path) -> None:
    report = tmp_path / "full.json"
    completed = _run_runner(
        "--profile",
        "full",
        "--only",
        "conjugate_normal,structural_grid",
        "--results-dir",
        str(tmp_path / "results"),
        "--report",
        str(report),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "UNRUN_FUTURE_WORK"
    assert payload["full_profile"]["status"] == "UNRUN_FUTURE_WORK"
    selected = {
        row["name"]: row
        for row in payload["stages"]
        if row["name"] in {"conjugate_normal", "structural_grid"}
    }
    assert {row["status"] for row in selected.values()} == {"UNRUN_FUTURE_WORK"}
    assert not (tmp_path / "results").exists()


def test_recorded_metrics_have_declared_gating_and_descriptive_semantics() -> None:
    expected = load_json(EXPECTED)
    evidence = {
        "conjugate_normal": load_json(
            ROOT / "validation" / "exact_example" / "output" / "validation_metrics.json"
        ),
        "structural_grid": load_json(
            ROOT / "validation" / "structural_example" / "output" / "validation_metrics.json"
        ),
        "benchmark": load_json(ROOT / "benchmarks" / "benchmark_results.json"),
    }
    comparisons = compare_expected(expected, evidence, set(STAGE_ORDER))
    assert comparisons
    gating = [row for row in comparisons if row["gating"]]
    descriptive = [row for row in comparisons if not row["gating"]]
    assert gating and all(row["passed"] is True for row in gating)
    assert descriptive and all(row["passed"] is None for row in descriptive)
    assert all(row["status"] == "DESCRIPTIVE" for row in descriptive)


def test_support_warning_uses_numpy_inference_without_torch(tmp_path: Path) -> None:
    output = tmp_path / "support.json"
    environment = dict(os.environ)
    python_path = [str(ROOT / "src"), str(ROOT / "examples")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "support_warning.py"),
            "--artifact",
            str(ROOT / "examples" / "assets" / "demo_estimator"),
            "--output",
            str(output),
            "--scale",
            "100",
            "--draws",
            "4",
            "--seed",
            "62001",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["torch_imported"] is False
    assert payload["diagnostic"]["support_warning"] is True
    assert "Distribution-support warning" in payload["diagnostic"]["warning"]


def test_shareable_machine_metadata_omits_absolute_python_path() -> None:
    metadata = machine_metadata()
    assert "python_executable" not in metadata
    assert metadata["python_executable_name"] == Path(sys.executable).name
    assert "/" not in metadata["python_executable_name"]
