from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from scripts.normalize_sdist import (
    EXPECTED_SDIST_FILENAME,
    EXPECTED_SDIST_ROOT,
    _validated_member_path,
    normalize_sdist,
)


ROOT = Path(__file__).resolve().parents[1]


def test_public_benchmark_table_matches_frozen_evidence() -> None:
    eight = json.loads(
        (ROOT / "replication/eight_schools/expected_smoke_metrics.json").read_text(
            encoding="utf-8"
        )
    )["posterior_hyperparameters"]
    rust_mdn = json.loads(
        (ROOT / "replication/rust_1987/expected/smoke_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    rust = json.loads(
        (
            ROOT
            / "replication/rust_1987/expected/structured_classifier_metrics.json"
        ).read_text(encoding="utf-8")
    )
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    benchmarks = (ROOT / "BENCHMARKS.md").read_text(encoding="utf-8")

    nfxp = rust_mdn["nfxp"]
    dense_mean = rust["dense_reference"]["posterior"]["mean"]
    first_run = rust["classifier_runs"][0]
    classifier_mean = first_run["posterior"]["mean"]
    expected_rows = (
        f"| Population mean `mu` | {eight['mu_exact_mean']:.4f} | "
        f"{eight['mu_approximate_mean']:.4f} |",
        f"| Heterogeneity `tau` | {eight['tau_exact_mean']:.4f} | "
        f"{eight['tau_approximate_mean']:.4f} |",
        f"| This repository's NFXP maximum likelihood | "
        f"{nfxp['replacement_cost']:.4f} | {nfxp['maintenance_slope']:.4f} |",
        f"| Dense-grid posterior mean | {dense_mean[0]:.4f} | "
        f"{dense_mean[1]:.4f} |",
        f"| Simulation-trained grid posterior mean (seed {first_run['seed']}) | "
        f"{classifier_mean[0]:.4f} | {classifier_mean[1]:.4f} |",
    )
    for row in expected_rows:
        assert row in readme
        assert row in benchmarks

    assert rust["all_seeds_pass_frozen_numerical_comparison_limits"] is True
    assert all(
        run["all_frozen_numerical_comparison_limits_pass"]
        for run in rust["classifier_runs"]
    )
    worst_cdf = max(
        max(run["marginal_cdf_supremum"]) for run in rust["classifier_runs"]
    )
    worst_tv = max(
        run["joint_coarsened_total_variation"]
        for run in rust["classifier_runs"]
    )
    assert f"{worst_cdf:.5f}" in readme
    assert f"{worst_tv:.5f}" in readme
    for run in rust["classifier_runs"]:
        row = (
            f"| {run['seed']} | {run['posterior']['mean'][0]:.4f} | "
            f"{run['posterior']['mean'][1]:.4f} | "
            f"{max(run['marginal_cdf_supremum']):.5f} | "
            f"{run['joint_coarsened_total_variation']:.5f} |"
        )
        assert row in benchmarks
    assert "not the generic `structnpe.fit` MDN" in readme
    assert "12.8636" in benchmarks
    assert "3.2528" in benchmarks


def test_every_executable_public_example_has_an_installed_wheel_profile() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_installed_wheel_examples.py"),
            "--inventory-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert "PUBLIC_EXAMPLE_INVENTORY_OK" in completed.stdout


def test_sdist_normalizer_removes_identity_and_executable_file_modes(tmp_path: Path) -> None:
    archive = tmp_path / EXPECTED_SDIST_FILENAME
    payload = b"public source\n"
    with tarfile.open(archive, mode="w:gz") as output:
        directory = tarfile.TarInfo("structnpe-0.1.0b1")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o777
        directory.uid = 123
        directory.gid = 456
        directory.uname = "private-user"
        directory.gname = "private-group"
        output.addfile(directory)
        member = tarfile.TarInfo("structnpe-0.1.0b1/README.md")
        member.size = len(payload)
        member.mode = 0o777
        member.uid = 123
        member.gid = 456
        member.uname = "private-user"
        member.gname = "private-group"
        output.addfile(member, io.BytesIO(payload))

    command = [
        sys.executable,
        str(ROOT / "scripts" / "normalize_sdist.py"),
        str(archive),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    first_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run(command, cwd=ROOT, check=True)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == first_hash

    with tarfile.open(archive, mode="r:gz") as normalized:
        members = normalized.getmembers()
        assert normalized.extractfile(members[1]).read() == payload
    assert [(item.uid, item.gid, item.uname, item.gname) for item in members] == [
        (0, 0, "", ""),
        (0, 0, "", ""),
    ]
    assert [item.mode for item in members] == [0o755, 0o644]


@pytest.mark.parametrize(
    "name",
    [
        f"{EXPECTED_SDIST_ROOT}/bad\x00name",
        f"{EXPECTED_SDIST_ROOT}\\outside.txt",
        "C:/outside.txt",
        f"{EXPECTED_SDIST_ROOT}/C:drive-relative.txt",
        "//server/share/outside.txt",
    ],
)
def test_sdist_normalizer_rejects_nonportable_or_unsafe_member_names(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe sdist member path"):
        _validated_member_path(name, expected_root=EXPECTED_SDIST_ROOT)


def test_sdist_normalizer_enforces_exact_filename_and_archive_root(tmp_path: Path) -> None:
    wrong_filename = tmp_path / "renamed.tar.gz"
    with tarfile.open(wrong_filename, mode="w:gz") as output:
        root = tarfile.TarInfo(EXPECTED_SDIST_ROOT)
        root.type = tarfile.DIRTYPE
        output.addfile(root)
    with pytest.raises(ValueError, match="filename must be exactly"):
        normalize_sdist(wrong_filename)

    wrong_root = tmp_path / EXPECTED_SDIST_FILENAME
    with tarfile.open(wrong_root, mode="w:gz") as output:
        root = tarfile.TarInfo("other-project-0.1.0b1")
        root.type = tarfile.DIRTYPE
        output.addfile(root)
    with pytest.raises(ValueError, match="outside the expected root"):
        normalize_sdist(wrong_root)


def test_sdist_normalizer_rejects_canonical_duplicate_members(tmp_path: Path) -> None:
    archive = tmp_path / EXPECTED_SDIST_FILENAME
    payload = b"same canonical path\n"
    with tarfile.open(archive, mode="w:gz") as output:
        root = tarfile.TarInfo(EXPECTED_SDIST_ROOT)
        root.type = tarfile.DIRTYPE
        output.addfile(root)
        for name in (
            f"{EXPECTED_SDIST_ROOT}/duplicate.txt",
            f"{EXPECTED_SDIST_ROOT}/./duplicate.txt",
        ):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            output.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="duplicate sdist member"):
        normalize_sdist(archive)


def test_sdist_normalizer_rejects_input_symlink_without_touching_target(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    link_dir = tmp_path / "link"
    target_dir.mkdir()
    link_dir.mkdir()
    target = target_dir / EXPECTED_SDIST_FILENAME
    with tarfile.open(target, mode="w:gz") as output:
        root = tarfile.TarInfo(EXPECTED_SDIST_ROOT)
        root.type = tarfile.DIRTYPE
        output.addfile(root)
    target_before = target.read_bytes()
    link = link_dir / EXPECTED_SDIST_FILENAME
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symlink or special file"):
        normalize_sdist(link)

    assert link.is_symlink()
    assert target.read_bytes() == target_before


def test_model_validation_form_requests_the_public_reproduction_contract() -> None:
    template = (
        ROOT / ".github" / "ISSUE_TEMPLATE" / "model_validation.yml"
    ).read_text(encoding="utf-8")
    for phrase in (
        "Simulator and parameter vector",
        "Prior",
        "Observed-data representation",
        "Simulation and training budget",
        "Comparator and posterior diagnostics",
        "Package version",
        "Model fingerprint",
        "Minimal reproducible example",
    ):
        assert phrase in template
    assert "Do not upload private data" in template
