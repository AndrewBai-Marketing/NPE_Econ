#!/usr/bin/env python3
"""Normalize a built sdist's public tar metadata without changing its files."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import os
import re
import stat
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


MAX_MEMBERS = 10_000
MAX_TOTAL_BYTES = 100 * 1024 * 1024
EXPECTED_SDIST_ROOT = "structnpe-0.1.0b1"
EXPECTED_SDIST_FILENAME = f"{EXPECTED_SDIST_ROOT}.tar.gz"
_WINDOWS_DRIVE_PATTERN = re.compile(r"(?:^|/)[A-Za-z]:")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_member_path(name: str, *, expected_root: str) -> PurePosixPath:
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or _WINDOWS_DRIVE_PATTERN.search(name)
    ):
        raise ValueError(f"unsafe sdist member path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe sdist member path: {name!r}")
    if path.parts[0] != expected_root:
        raise ValueError(
            f"sdist member is outside the expected root {expected_root!r}: {name!r}"
        )
    return path


def _validated_members(
    archive: tarfile.TarFile, *, expected_root: str
) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    if not members or len(members) > MAX_MEMBERS:
        raise ValueError("sdist has an invalid member count")
    canonical_names: set[str] = set()
    total = 0
    root_member: tarfile.TarInfo | None = None
    for member in members:
        path = _validated_member_path(member.name, expected_root=expected_root)
        canonical_name = path.as_posix()
        if canonical_name in canonical_names:
            raise ValueError(f"duplicate sdist member: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsupported sdist member type: {member.name!r}")
        canonical_names.add(canonical_name)
        total += int(member.size)
        if len(path.parts) == 1:
            root_member = member
    if root_member is None or not root_member.isdir():
        raise ValueError(f"sdist must contain the expected root directory {expected_root!r}")
    if total > MAX_TOTAL_BYTES:
        raise ValueError("sdist must have a bounded total payload size")
    return sorted(members, key=lambda item: item.name)


def normalize_sdist(path: str | Path, *, epoch: int = 0) -> str:
    """Rewrite *path* atomically with neutral ownership and conventional modes."""

    source = Path(path)
    if source.name != EXPECTED_SDIST_FILENAME:
        raise ValueError(f"sdist filename must be exactly {EXPECTED_SDIST_FILENAME!r}")
    try:
        source_stat = os.lstat(source)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError("sdist must be a regular .tar.gz file") from exc
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError("sdist must be a regular .tar.gz file, not a symlink or special file")
    expected_root = EXPECTED_SDIST_ROOT
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=source.parent, prefix=f".{source.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with tarfile.open(source, mode="r:gz") as input_archive:
            members = _validated_members(input_archive, expected_root=expected_root)
            with temporary.open("wb") as raw_output:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw_output, mtime=int(epoch)
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                    ) as output_archive:
                        for original in members:
                            member = copy.copy(original)
                            member.uid = 0
                            member.gid = 0
                            member.uname = ""
                            member.gname = ""
                            member.mtime = int(epoch)
                            member.mode = 0o755 if member.isdir() else 0o644
                            member.pax_headers = {}
                            payload = input_archive.extractfile(original) if original.isfile() else None
                            output_archive.addfile(member, payload)
        with tarfile.open(temporary, mode="r:gz") as check_archive:
            normalized = _validated_members(check_archive, expected_root=expected_root)
            for member in normalized:
                expected_mode = 0o755 if member.isdir() else 0o644
                if (
                    member.uid != 0
                    or member.gid != 0
                    or member.uname
                    or member.gname
                    or member.mode != expected_mode
                ):
                    raise ValueError("normalized sdist metadata verification failed")
        os.replace(temporary, source)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256(source)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument(
        "--epoch",
        type=int,
        default=int(os.environ.get("SOURCE_DATE_EPOCH", "0")),
        help="normalized tar and gzip timestamp (default: SOURCE_DATE_EPOCH or zero)",
    )
    args = parser.parse_args()
    digest = normalize_sdist(args.sdist, epoch=args.epoch)
    print(f"normalized_sdist={args.sdist}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
