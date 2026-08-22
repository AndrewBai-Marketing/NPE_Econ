"""Safe, versioned directory-artifact primitives."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterable, Iterator

import numpy as np

from ._fingerprints import canonical_json


ARTIFACT_FORMAT_VERSION = 1

# These limits bound all data parsed automatically by the public artifact
# loaders. They are intentionally generous for the beta's small MDNs and
# posterior-draw files, while making resource use finite for untrusted input.
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_COMPLETE_BYTES = 128
MAX_PAYLOAD_FILES = 32
MAX_PAYLOAD_BYTES = 512 * 1024 * 1024
MAX_TOTAL_PAYLOAD_BYTES = 1024 * 1024 * 1024
MAX_PAYLOAD_NAME_BYTES = 255
MAX_ZIP_ARCHIVE_BYTES = MAX_PAYLOAD_BYTES
MAX_NPZ_MEMBERS = 1024
MAX_NPZ_MEMBER_COMPRESSED_BYTES = 512 * 1024 * 1024
MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 64 * 1024
MAX_ARRAY_ELEMENTS = 100_000_000
MAX_ARRAY_BYTES = 512 * 1024 * 1024
MAX_COMPRESSION_RATIO = 10_000

_HASH_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_READ_BLOCK_BYTES = 1024 * 1024
_RESERVED_NAMES = frozenset({"manifest.json", "COMPLETE"})


class ArtifactError(ValueError):
    """Raised when a public artifact is missing, corrupt, or incompatible."""


def _entry_exists(path: Path) -> bool:
    """Return whether a directory entry exists, including dangling symlinks."""

    return os.path.lexists(path)


def _remove_entry(path: Path) -> None:
    """Remove one exact entry without following a symlink."""

    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _validate_payload_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise ArtifactError(f"Unsafe artifact payload name: {name!r}.")
    if (
        name in _RESERVED_NAMES
        or "\x00" in name
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
        or PurePosixPath(name).name != name
        or len(name.encode("utf-8")) > MAX_PAYLOAD_NAME_BYTES
    ):
        raise ArtifactError(f"Unsafe artifact payload name: {name!r}.")
    return name


@contextmanager
def _open_regular_file(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> Iterator[tuple[BinaryIO, int]]:
    """Open a bounded regular file and reject symlink-swap races."""

    try:
        before = os.lstat(path)
    except (FileNotFoundError, OSError) as exc:
        raise ArtifactError(f"{label} is missing or inaccessible.") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ArtifactError(f"{label} must be a regular file, not a symlink or special file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactError(f"{label} could not be opened safely.") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ArtifactError(f"{label} must be a regular file.")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ArtifactError(f"{label} changed while it was being opened.")
        if opened.st_size > max_bytes:
            raise ArtifactError(
                f"{label} is too large ({opened.st_size} bytes; limit {max_bytes})."
            )
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            yield handle, opened.st_size
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_bounded_regular_file(path: Path, *, label: str, max_bytes: int) -> bytes:
    with _open_regular_file(path, label=label, max_bytes=max_bytes) as (handle, size):
        payload = handle.read(size + 1)
        if len(payload) != size:
            raise ArtifactError(f"{label} changed while it was being read.")
        return payload


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactError(f"Artifact manifest contains duplicate JSON key {key!r}.")
        value[key] = item
    return value


def _nonfinite_json_number(token: str) -> Any:
    raise ArtifactError(f"Artifact manifest contains non-finite number {token!r}.")


def _parse_manifest(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_nonfinite_json_number,
        )
    except ArtifactError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise ArtifactError("Artifact manifest is not valid bounded UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ArtifactError("Artifact manifest root must be a JSON object.")
    try:
        canonical_json(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ArtifactError("Artifact manifest contains unsupported metadata.") from exc
    return value


def _zip_member_type_is_safe(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if file_type not in {0, stat.S_IFREG}:
        return False
    # Some creators provide only DOS attributes.
    if info.create_system == 0 and (info.external_attr & 0x10):
        return False
    return not info.is_dir()


def _validate_npy_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    try:
        with archive.open(info, "r") as member:
            version = np.lib.format.read_magic(member)
            if version == (1, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(
                    member, max_header_size=MAX_NPY_HEADER_BYTES
                )
            elif version == (2, 0):
                shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(
                    member, max_header_size=MAX_NPY_HEADER_BYTES
                )
            else:
                raise ArtifactError(
                    f"NPZ member {info.filename!r} uses unsupported NPY version {version!r}."
                )
            dtype = np.dtype(dtype)
            if dtype.hasobject:
                raise ArtifactError(f"NPZ member {info.filename!r} contains an object array.")
            if dtype.itemsize <= 0:
                raise ArtifactError(f"NPZ member {info.filename!r} has a zero-width dtype.")
            elements = 1
            for dimension in shape:
                if not isinstance(dimension, int) or dimension < 0:
                    raise ArtifactError(f"NPZ member {info.filename!r} has an invalid shape.")
                elements *= dimension
                if elements > MAX_ARRAY_ELEMENTS:
                    raise ArtifactError(
                        f"NPZ member {info.filename!r} exceeds the array-element limit."
                    )
            array_bytes = elements * dtype.itemsize
            if array_bytes > MAX_ARRAY_BYTES:
                raise ArtifactError(f"NPZ member {info.filename!r} exceeds the array-byte limit.")
            header_bytes = member.tell()
            if header_bytes > MAX_NPY_HEADER_BYTES + 12:
                raise ArtifactError(f"NPZ member {info.filename!r} has an oversized NPY header.")
            if header_bytes + array_bytes != info.file_size:
                raise ArtifactError(
                    f"NPZ member {info.filename!r} size disagrees with its NPY header."
                )
            remaining = array_bytes
            while remaining:
                block = member.read(min(_READ_BLOCK_BYTES, remaining))
                if not block:
                    raise ArtifactError(f"NPZ member {info.filename!r} is truncated.")
                remaining -= len(block)
            if member.read(1):
                raise ArtifactError(f"NPZ member {info.filename!r} contains trailing data.")
    except ArtifactError:
        raise
    except (EOFError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise ArtifactError(f"Could not validate safe NPZ member {info.filename!r}.") from exc


def _validate_npz(handle: BinaryIO, *, label: str, archive_bytes: int) -> None:
    if archive_bytes > MAX_ZIP_ARCHIVE_BYTES:
        raise ArtifactError(f"{label} exceeds the ZIP archive-size limit.")
    try:
        with zipfile.ZipFile(handle, "r") as archive:
            members = archive.infolist()
            if not members:
                raise ArtifactError(f"{label} contains no arrays.")
            if len(members) > MAX_NPZ_MEMBERS:
                raise ArtifactError(f"{label} contains too many ZIP members.")
            names: set[str] = set()
            total_uncompressed = 0
            for info in members:
                name = info.filename
                if name in names:
                    raise ArtifactError(f"{label} contains duplicate ZIP member {name!r}.")
                names.add(name)
                if (
                    not isinstance(name, str)
                    or not name.endswith(".npy")
                    or "/" in name
                    or "\\" in name
                    or "\x00" in name
                    or PurePosixPath(name).name != name
                    or len(name.encode("utf-8")) > MAX_PAYLOAD_NAME_BYTES
                ):
                    raise ArtifactError(f"{label} contains unsafe ZIP member name {name!r}.")
                if not _zip_member_type_is_safe(info):
                    raise ArtifactError(
                        f"{label} contains a symlink or special ZIP member {name!r}."
                    )
                if info.flag_bits & 0x1:
                    raise ArtifactError(f"{label} contains encrypted ZIP member {name!r}.")
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ArtifactError(f"{label} uses an unsupported ZIP compression method.")
                if info.compress_size > MAX_NPZ_MEMBER_COMPRESSED_BYTES:
                    raise ArtifactError(f"ZIP member {name!r} exceeds the compressed-size limit.")
                if info.file_size > MAX_NPZ_MEMBER_UNCOMPRESSED_BYTES:
                    raise ArtifactError(f"ZIP member {name!r} exceeds the decompressed-size limit.")
                if (
                    info.file_size > _READ_BLOCK_BYTES
                    and info.file_size > max(1, info.compress_size) * MAX_COMPRESSION_RATIO
                ):
                    raise ArtifactError(f"ZIP member {name!r} exceeds the compression-ratio limit.")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
                    raise ArtifactError(f"{label} exceeds the total decompressed-size limit.")
            for info in members:
                _validate_npy_member(archive, info)
    except ArtifactError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise ArtifactError(f"{label} is not a valid safe NPZ archive.") from exc


def _validate_and_hash_payload(path: Path, *, name: str) -> tuple[str, int]:
    label = f"Artifact payload {name!r}"
    with _open_regular_file(path, label=label, max_bytes=MAX_PAYLOAD_BYTES) as (handle, size):
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(_READ_BLOCK_BYTES), b""):
            digest.update(block)
        if name.endswith(".npz"):
            handle.seek(0)
            _validate_npz(handle, label=label, archive_bytes=size)
        return digest.hexdigest(), size


def read_npz_arrays(
    path: str | Path,
    *,
    expected_names: Iterable[str] | None = None,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    """Authenticate, validate, and eagerly load a bounded NPZ from one descriptor."""

    source = Path(path)
    expected: set[str] | None = None
    if expected_names is not None:
        expected_list = list(expected_names)
        if (
            len(expected_list) > MAX_NPZ_MEMBERS
            or len(expected_list) != len(set(expected_list))
            or any(not isinstance(name, str) or not name for name in expected_list)
        ):
            raise ArtifactError("Expected NPZ array names are invalid or duplicated.")
        expected = set(expected_list)
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or not _HASH_PATTERN.fullmatch(expected_sha256)
    ):
        raise ArtifactError("Expected NPZ SHA-256 is not a valid lowercase digest.")
    label = f"NPZ payload {source.name!r}"
    with _open_regular_file(source, label=label, max_bytes=MAX_PAYLOAD_BYTES) as (handle, size):
        if expected_sha256 is not None:
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(_READ_BLOCK_BYTES), b""):
                digest.update(block)
            if digest.hexdigest() != expected_sha256:
                raise ArtifactError(f"Checksum mismatch for {label}.")
            handle.seek(0)
        _validate_npz(handle, label=label, archive_bytes=size)
        handle.seek(0)
        try:
            with np.load(handle, allow_pickle=False) as payload:
                actual = set(payload.files)
                if expected is not None and actual != expected:
                    missing = sorted(expected - actual)
                    unexpected = sorted(actual - expected)
                    raise ArtifactError(
                        "NPZ array names do not match the required schema; "
                        f"missing={missing}, unexpected={unexpected}."
                    )
                return {name: np.array(payload[name], copy=True) for name in payload.files}
        except ArtifactError:
            raise
        except (EOFError, KeyError, OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
            raise ArtifactError(f"{label} could not be loaded safely.") from exc


def _validate_directory_entries(root: Path, allowed_names: set[str]) -> None:
    seen: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name in seen:
                    raise ArtifactError(
                        f"Artifact directory contains duplicate entry {entry.name!r}."
                    )
                seen.add(entry.name)
                if len(seen) > MAX_PAYLOAD_FILES + len(_RESERVED_NAMES):
                    raise ArtifactError("Artifact directory contains too many entries.")
                if entry.name not in allowed_names:
                    raise ArtifactError(
                        f"Artifact directory contains unlisted entry {entry.name!r}."
                    )
                entry_stat = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise ArtifactError(
                        f"Artifact entry {entry.name!r} is a symlink, directory, or special file."
                    )
    except ArtifactError:
        raise
    except OSError as exc:
        raise ArtifactError("Artifact directory could not be inspected safely.") from exc
    if seen != allowed_names:
        missing = sorted(allowed_names - seen)
        raise ArtifactError(f"Artifact directory is missing required entries: {missing}.")


def write_directory_artifact(
    path: str | Path,
    *,
    manifest: dict[str, Any],
    write_payloads: Callable[[Path], list[str]],
    overwrite: bool = False,
) -> Path:
    """Atomically write bounded payloads, checksums, manifest, and completion marker."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _entry_exists(target) and not overwrite:
        raise FileExistsError(f"Artifact path already exists: {target}")
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        payload_names = write_payloads(temp)
        if not isinstance(payload_names, list):
            raise ArtifactError("Artifact writer must return a list of payload names.")
        names = [_validate_payload_name(name) for name in payload_names]
        if not names:
            raise ArtifactError("Artifact writer returned no payloads.")
        if len(names) > MAX_PAYLOAD_FILES:
            raise ArtifactError("Artifact writer returned too many payloads.")
        if len(names) != len(set(names)):
            raise ArtifactError("Artifact writer returned duplicate payload names.")
        _validate_directory_entries(temp, set(names))
        checksums: dict[str, str] = {}
        total_payload_bytes = 0
        for name in sorted(names):
            checksum, size = _validate_and_hash_payload(temp / name, name=name)
            total_payload_bytes += size
            if total_payload_bytes > MAX_TOTAL_PAYLOAD_BYTES:
                raise ArtifactError("Artifact payloads exceed the total size limit.")
            checksums[name] = checksum
        full_manifest = dict(manifest)
        full_manifest["artifact_format_version"] = ARTIFACT_FORMAT_VERSION
        full_manifest["files"] = checksums
        try:
            canonical_json(full_manifest)
            manifest_data = (
                json.dumps(full_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise ArtifactError("Artifact manifest contains unsupported metadata.") from exc
        if len(manifest_data) > MAX_MANIFEST_BYTES:
            raise ArtifactError("Artifact manifest exceeds the size limit.")
        manifest_path = temp / "manifest.json"
        manifest_path.write_bytes(manifest_data)
        complete_data = hashlib.sha256(manifest_data).hexdigest().encode("ascii") + b"\n"
        (temp / "COMPLETE").write_bytes(complete_data)
        _validate_directory_entries(temp, set(names) | _RESERVED_NAMES)

        if _entry_exists(target):
            if not overwrite:
                raise FileExistsError(f"Artifact path already exists: {target}")
            backup = target.with_name(
                f".{target.name}.previous-{os.getpid()}-{secrets.token_hex(8)}"
            )
            if _entry_exists(backup):
                raise FileExistsError(f"Cannot create safe overwrite backup: {backup}")
            os.replace(target, backup)
            try:
                os.replace(temp, target)
            except Exception:
                os.replace(backup, target)
                raise
            _remove_entry(backup)
        else:
            os.replace(temp, target)
        return target
    except Exception:
        if _entry_exists(temp):
            _remove_entry(temp)
        raise


def read_directory_header(path: str | Path, *, expected_kind: str) -> tuple[Path, dict[str, Any]]:
    """Authenticate bounded metadata and entry types without reading payload bodies."""

    root = Path(path)
    try:
        root_mode = os.lstat(root).st_mode
    except (FileNotFoundError, OSError) as exc:
        raise ArtifactError(f"Artifact directory not found: {root}") from exc
    if not stat.S_ISDIR(root_mode):
        raise ArtifactError(f"Artifact path must be a real directory, not a symlink: {root}")

    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE"
    manifest_data = _read_bounded_regular_file(
        manifest_path,
        label="Artifact is incomplete: manifest.json",
        max_bytes=MAX_MANIFEST_BYTES,
    )
    complete_data = _read_bounded_regular_file(
        complete_path,
        label="Artifact is incomplete: COMPLETE marker",
        max_bytes=MAX_COMPLETE_BYTES,
    )
    try:
        expected_manifest_hash = complete_data.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ArtifactError("Artifact COMPLETE marker is not ASCII.") from exc
    if not _HASH_PATTERN.fullmatch(expected_manifest_hash):
        raise ArtifactError("Artifact COMPLETE marker is not a valid SHA-256 digest.")
    if expected_manifest_hash != hashlib.sha256(manifest_data).hexdigest():
        raise ArtifactError("Artifact manifest checksum does not match COMPLETE marker.")
    manifest = _parse_manifest(manifest_data)
    if manifest.get("artifact_format_version") != ARTIFACT_FORMAT_VERSION:
        raise ArtifactError(
            f"Unsupported artifact format {manifest.get('artifact_format_version')!r}; "
            f"expected {ARTIFACT_FORMAT_VERSION}."
        )
    if manifest.get("artifact_kind") != expected_kind:
        raise ArtifactError(
            f"Artifact kind is {manifest.get('artifact_kind')!r}; expected {expected_kind!r}."
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactError("Artifact manifest has no payload checksum map.")
    if len(files) > MAX_PAYLOAD_FILES:
        raise ArtifactError("Artifact manifest declares too many payload files.")
    names = {_validate_payload_name(name) for name in files}
    for name in names:
        expected_hash = files[name]
        if not isinstance(expected_hash, str) or not _HASH_PATTERN.fullmatch(expected_hash):
            raise ArtifactError(f"Invalid checksum for artifact payload: {name}")
    _validate_directory_entries(root, names | _RESERVED_NAMES)
    return root, manifest


def read_directory_manifest(path: str | Path, *, expected_kind: str) -> tuple[Path, dict[str, Any]]:
    """Authenticate metadata and fully verify every bounded artifact payload."""

    root, manifest = read_directory_header(path, expected_kind=expected_kind)
    files = manifest["files"]
    names = set(files)
    total_payload_bytes = 0
    for name in sorted(names):
        expected_hash = files[name]
        actual_hash, size = _validate_and_hash_payload(root / name, name=name)
        total_payload_bytes += size
        if total_payload_bytes > MAX_TOTAL_PAYLOAD_BYTES:
            raise ArtifactError("Artifact payloads exceed the total size limit.")
        if actual_hash != expected_hash:
            raise ArtifactError(f"Checksum mismatch for artifact payload: {name}")
    return root, manifest


__all__ = [
    "ARTIFACT_FORMAT_VERSION",
    "ArtifactError",
    "read_directory_header",
    "read_directory_manifest",
    "read_npz_arrays",
    "write_directory_artifact",
]
