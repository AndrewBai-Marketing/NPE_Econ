from __future__ import annotations

import os
import re
import stat
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIST = Path(os.environ.get("STRUCTNPE_DIST_DIR", ROOT / "dist"))
VERSION = "0.1.0b1"
_WINDOWS_DRIVE_PATTERN = re.compile(r"(?:^|/)[A-Za-z]:")

FORBIDDEN_COMPONENTS = {
    ".lake",
    ".pytest_cache",
    "NpeCompilerLean",
    "airline_compiler",
    "artifacts",
    "data_manifest",
    "ddc_npe",
    "marketing_compiler",
    "paper",
    "paper_artifacts",
    "paper_latex",
    "research",
}


def _single_artifact(pattern: str) -> Path:
    matches = sorted(DIST.glob(pattern))
    if not matches:
        pytest.skip("Build artifacts are not present; run `python -m build` first.")
    assert len(matches) == 1, f"expected one {pattern} artifact, found: {matches}"
    return matches[0]


def _assert_release_boundary(names: list[str]) -> None:
    violations: list[str] = []
    unrelated_tests: list[str] = []
    normalized: set[str] = set()
    for raw_name in names:
        path = PurePosixPath(raw_name)
        if (
            not raw_name
            or "\x00" in raw_name
            or "\\" in raw_name
            or _WINDOWS_DRIVE_PATTERN.search(raw_name)
            or path.is_absolute()
            or ".." in path.parts
            or not path.parts
        ):
            violations.append(raw_name)
            continue
        canonical = path.as_posix().rstrip("/")
        if canonical in normalized:
            violations.append(raw_name)
        normalized.add(canonical)
        parts = set(path.parts)
        basename = path.name
        if parts & FORBIDDEN_COMPONENTS:
            violations.append(raw_name)
        if "replication" in parts and ({"data", "results"} & parts):
            violations.append(raw_name)
        if basename.startswith("CLOUD_") or basename.startswith("NpeCompilerLean"):
            violations.append(raw_name)
        if "__pycache__" in parts or basename.endswith((".pyc", ".pyo")):
            violations.append(raw_name)
        if "output" in parts and "estimator" in parts:
            violations.append(raw_name)
        if (
            "tests" in parts
            and basename.startswith("test_")
            and not basename.startswith(("test_structnpe", "test_replication", "test_public_ci"))
        ):
            unrelated_tests.append(raw_name)
    assert not violations, f"protected or generated paths entered the distribution: {violations}"
    assert not unrelated_tests, f"unrelated research tests entered the source distribution: {unrelated_tests}"


def test_wheel_contains_only_structnpe_package() -> None:
    wheel = _single_artifact(f"structnpe-{VERSION}-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        _assert_release_boundary(names)
        assert all(
            PurePosixPath(name).parts[0] == "structnpe"
            or PurePosixPath(name).parts[0].startswith(f"structnpe-{VERSION}.dist-info")
            for name in names
        )
        for info in archive.infolist():
            mode = (info.external_attr >> 16) & 0xFFFF
            assert stat.S_IFMT(mode) in {0, stat.S_IFREG, stat.S_IFDIR}

        python_roots = {
            PurePosixPath(name).parts[0]
            for name in names
            if name.endswith(".py") and ".dist-info/" not in name
        }
        assert python_roots == {"structnpe"}
        assert "structnpe/__init__.py" in names
        assert "structnpe/templates/custom_simulator/config.yaml" in names
        assert "structnpe/templates/finite_ddc/model.py" in names
        assert "structnpe/templates/model_index_ddc/observed_example.csv" in names

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        assert len(metadata_names) == 1
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        assert metadata["Version"] == VERSION
        python_specifiers = {part.strip() for part in metadata["Requires-Python"].split(",")}
        assert python_specifiers == {">=3.11", "<3.14"}
        requirements = metadata.get_all("Requires-Dist", [])
        assert "numpy>=1.26" in requirements
        assert "pandas>=2.2" in requirements
        assert metadata["License-Expression"] == "MIT"
        assert {"build", "dev", "neural", "replication", "torch"} <= set(
            metadata.get_all("Provides-Extra", [])
        )
        entry_points = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        assert len(entry_points) == 1
        assert "structnpe = structnpe.cli:main" in archive.read(entry_points[0]).decode("utf-8")


def test_sdist_respects_public_release_boundary() -> None:
    sdist = _single_artifact(f"structnpe-{VERSION}.tar.gz")
    prefix = f"structnpe-{VERSION}/"
    with tarfile.open(sdist, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        _assert_release_boundary(names)
        assert all(
            PurePosixPath(name).parts[0] == f"structnpe-{VERSION}" for name in names
        )
        for member in members:
            assert member.uid == 0 and member.gid == 0
            assert member.uname == "" and member.gname == ""
            assert member.mode == (0o755 if member.isdir() else 0o644)
        allowed_children = {
            "BENCHMARKS.md",
            "CHANGELOG.md",
            "CITATION.cff",
            "CONTRIBUTING.md",
            "LICENSE",
            "MANIFEST.in",
            "PKG-INFO",
            "README.md",
            "SECURITY.md",
            "THEORY.md",
            "examples",
            "pyproject.toml",
            "setup.cfg",
            "src",
        }
        assert {
            PurePosixPath(name).parts[1]
            for name in names
            if len(PurePosixPath(name).parts) > 1
        } <= allowed_children
        assert all(member.isfile() or member.isdir() for member in members)
        assert f"{prefix}LICENSE" in names
        assert f"{prefix}README.md" in names
        assert f"{prefix}BENCHMARKS.md" in names
        assert f"{prefix}THEORY.md" in names
        assert f"{prefix}pyproject.toml" in names
        assert f"{prefix}src/structnpe/__init__.py" in names
        assert f"{prefix}examples/quickstart.py" in names
        assert f"{prefix}examples/custom_model.py" in names
        assert f"{prefix}examples/assets/demo_estimator/manifest.json" in names
        assert not any(f"{prefix}replication/" in name for name in names)
        assert not any(f"{prefix}tests/" in name for name in names)
