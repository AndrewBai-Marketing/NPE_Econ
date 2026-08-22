from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile
from pathlib import Path

import numpy as np
import pytest

import structnpe._artifacts as artifacts
from structnpe._artifacts import (
    ArtifactError,
    read_directory_header,
    read_directory_manifest,
    read_npz_arrays,
    write_directory_artifact,
)


KIND = "artifact_security_test"


def _writer(root: Path, *, names: tuple[str, ...] = ("payload.npz",)) -> list[str]:
    for name in set(names):
        np.savez_compressed(root / name, value=np.arange(4, dtype=np.float64))
    return list(names)


def _write_safe(path: Path, *, overwrite: bool = False) -> Path:
    return write_directory_artifact(
        path,
        manifest={"artifact_kind": KIND, "note": "safe"},
        write_payloads=_writer,
        overwrite=overwrite,
    )


def _npy_bytes(value: np.ndarray | None = None) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array(
        handle,
        np.arange(4, dtype=np.float64) if value is None else value,
        allow_pickle=False,
    )
    return handle.getvalue()


def _install_raw_payload(root: Path, payload: bytes, *, name: str = "payload.npz") -> Path:
    root.mkdir()
    (root / name).write_bytes(payload)
    manifest = {
        "artifact_format_version": 1,
        "artifact_kind": KIND,
        "files": {name: hashlib.sha256(payload).hexdigest()},
    }
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (root / "manifest.json").write_bytes(manifest_data)
    (root / "COMPLETE").write_text(
        hashlib.sha256(manifest_data).hexdigest() + "\n", encoding="ascii"
    )
    return root


def test_header_full_verification_same_descriptor_reader_and_existing_file_overwrite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "artifact"
    target.write_text("old file", encoding="utf-8")
    artifact = _write_safe(target, overwrite=True)
    assert artifact.is_dir()
    root, header = read_directory_header(artifact, expected_kind=KIND)
    assert root == artifact
    assert header["files"].keys() == {"payload.npz"}
    read_directory_manifest(artifact, expected_kind=KIND)
    arrays = read_npz_arrays(
        artifact / "payload.npz",
        expected_names={"value"},
        expected_sha256=header["files"]["payload.npz"],
    )
    np.testing.assert_array_equal(arrays["value"], np.arange(4, dtype=np.float64))
    with pytest.raises(ArtifactError, match="array names"):
        read_npz_arrays(artifact / "payload.npz", expected_names={"wrong"})
    with pytest.raises(ArtifactError, match="not a valid lowercase digest"):
        read_npz_arrays(artifact / "payload.npz", expected_sha256="not-a-digest")
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        read_npz_arrays(artifact / "payload.npz", expected_sha256="0" * 64)
    assert not list(tmp_path.glob(".artifact.previous-*"))
    with pytest.raises(FileExistsError):
        _write_safe(target)


def test_header_defers_payload_body_but_full_verifier_rejects_invalid_npz(tmp_path: Path) -> None:
    artifact = _install_raw_payload(tmp_path / "artifact", b"not a zip archive")
    read_directory_header(artifact, expected_kind=KIND)
    with pytest.raises(ArtifactError, match="valid safe NPZ"):
        read_directory_manifest(artifact, expected_kind=KIND)


def test_manifest_and_payload_limits_are_enforced_before_unbounded_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _write_safe(tmp_path / "artifact")
    manifest_size = (artifact / "manifest.json").stat().st_size
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "MAX_MANIFEST_BYTES", manifest_size - 1)
        with pytest.raises(ArtifactError, match="too large"):
            read_directory_header(artifact, expected_kind=KIND)
    payload_size = (artifact / "payload.npz").stat().st_size
    with monkeypatch.context() as patch:
        patch.setattr(artifacts, "MAX_PAYLOAD_BYTES", payload_size - 1)
        with pytest.raises(ArtifactError, match="too large"):
            read_directory_manifest(artifact, expected_kind=KIND)


@pytest.mark.parametrize(
    ("limit", "value", "message"),
    [
        ("MAX_ZIP_ARCHIVE_BYTES", 1, "archive-size"),
        ("MAX_NPZ_MEMBERS", 0, "too many ZIP members"),
        ("MAX_NPZ_MEMBER_COMPRESSED_BYTES", 1, "compressed-size"),
        ("MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES", 1, "decompressed-size"),
        ("MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES", 1, "total decompressed-size"),
        ("MAX_ARRAY_ELEMENTS", 3, "array-element"),
        ("MAX_ARRAY_BYTES", 1, "array-byte"),
    ],
)
def test_npz_archive_member_decompression_and_array_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    value: int,
    message: str,
) -> None:
    artifact = _write_safe(tmp_path / "artifact")
    monkeypatch.setattr(artifacts, limit, value)
    with pytest.raises(ArtifactError, match=message):
        read_directory_manifest(artifact, expected_kind=KIND)


@pytest.mark.parametrize("member_name", ["../escape.npy", "/absolute.npy", "nested/value.npy"])
def test_npz_member_path_traversal_is_rejected(tmp_path: Path, member_name: str) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(member_name, _npy_bytes())
    artifact = _install_raw_payload(tmp_path / "artifact", payload.getvalue())
    with pytest.raises(ArtifactError, match="unsafe ZIP member"):
        read_directory_manifest(artifact, expected_kind=KIND)


def test_duplicate_and_symlink_zip_members_are_rejected(tmp_path: Path) -> None:
    duplicate = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr("value.npy", _npy_bytes())
            archive.writestr("value.npy", _npy_bytes())
    duplicate_artifact = _install_raw_payload(tmp_path / "duplicate", duplicate.getvalue())
    with pytest.raises(ArtifactError, match="duplicate ZIP member"):
        read_directory_manifest(duplicate_artifact, expected_kind=KIND)

    symlink = io.BytesIO()
    info = zipfile.ZipInfo("value.npy")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(info, b"elsewhere")
    symlink_artifact = _install_raw_payload(tmp_path / "symlink", symlink.getvalue())
    with pytest.raises(ArtifactError, match="symlink or special ZIP member"):
        read_directory_manifest(symlink_artifact, expected_kind=KIND)


def test_duplicate_json_keys_payload_traversal_and_unlisted_entries_are_rejected(
    tmp_path: Path,
) -> None:
    artifact = _write_safe(tmp_path / "artifact")
    manifest = (artifact / "manifest.json").read_text(encoding="utf-8")
    duplicate = manifest.replace(
        '"artifact_kind": "artifact_security_test",',
        '"artifact_kind": "artifact_security_test",\n  "artifact_kind": "artifact_security_test",',
    ).encode()
    (artifact / "manifest.json").write_bytes(duplicate)
    (artifact / "COMPLETE").write_text(
        hashlib.sha256(duplicate).hexdigest() + "\n", encoding="ascii"
    )
    with pytest.raises(ArtifactError, match="duplicate JSON key"):
        read_directory_header(artifact, expected_kind=KIND)

    traversal = tmp_path / "traversal"
    traversal.mkdir()
    traversal_manifest = {
        "artifact_format_version": 1,
        "artifact_kind": KIND,
        "files": {"../outside.npz": "0" * 64},
    }
    traversal_data = (json.dumps(traversal_manifest) + "\n").encode()
    (traversal / "manifest.json").write_bytes(traversal_data)
    (traversal / "COMPLETE").write_text(
        hashlib.sha256(traversal_data).hexdigest() + "\n", encoding="ascii"
    )
    with pytest.raises(ArtifactError, match="Unsafe artifact payload name"):
        read_directory_header(traversal, expected_kind=KIND)

    clean = _write_safe(tmp_path / "unlisted")
    (clean / "surprise.txt").write_text("not declared", encoding="utf-8")
    with pytest.raises(ArtifactError, match="unlisted entry"):
        read_directory_header(clean, expected_kind=KIND)


def test_directory_symlinks_special_files_and_writer_schema_are_rejected(tmp_path: Path) -> None:
    artifact = _write_safe(tmp_path / "artifact")
    alias = tmp_path / "alias"
    alias.symlink_to(artifact, target_is_directory=True)
    with pytest.raises(ArtifactError, match="real directory"):
        read_directory_header(alias, expected_kind=KIND)

    payload = artifact / "payload.npz"
    outside = tmp_path / "outside.npz"
    payload.replace(outside)
    payload.symlink_to(outside)
    with pytest.raises(ArtifactError, match="symlink"):
        read_directory_header(artifact, expected_kind=KIND)

    def duplicate_writer(root: Path) -> list[str]:
        np.savez(root / "same.npz", value=np.arange(1))
        return ["same.npz", "same.npz"]

    with pytest.raises(ArtifactError, match="duplicate payload"):
        write_directory_artifact(
            tmp_path / "duplicate-writer",
            manifest={"artifact_kind": KIND},
            write_payloads=duplicate_writer,
        )

    with pytest.raises(ArtifactError, match="Unsafe artifact payload name"):
        write_directory_artifact(
            tmp_path / "traversal-writer",
            manifest={"artifact_kind": KIND},
            write_payloads=lambda _root: ["../outside.npz"],
        )


def test_object_array_and_extra_writer_output_are_rejected(tmp_path: Path) -> None:
    def object_writer(root: Path) -> list[str]:
        np.savez(root / "object.npz", value=np.array([object()], dtype=object))
        return ["object.npz"]

    with pytest.raises(ArtifactError, match="object array"):
        write_directory_artifact(
            tmp_path / "object",
            manifest={"artifact_kind": KIND},
            write_payloads=object_writer,
        )

    def extra_writer(root: Path) -> list[str]:
        np.savez(root / "safe.npz", value=np.arange(1))
        (root / "extra.txt").write_text("extra", encoding="utf-8")
        return ["safe.npz"]

    with pytest.raises(ArtifactError, match="unlisted entry"):
        write_directory_artifact(
            tmp_path / "extra",
            manifest={"artifact_kind": KIND},
            write_payloads=extra_writer,
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO creation is unavailable")
def test_special_files_inside_directory_artifact_are_rejected(tmp_path: Path) -> None:
    artifact = _write_safe(tmp_path / "artifact")
    payload = artifact / "payload.npz"
    payload.unlink()
    os.mkfifo(payload)
    with pytest.raises(ArtifactError, match="special file"):
        read_directory_header(artifact, expected_kind=KIND)
