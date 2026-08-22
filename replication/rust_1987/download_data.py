"""Download the pinned group-4 ASCII file and verify its exact bytes.

No raw data are redistributed with this repository.  This downloader accepts
only the immutable URL and digest declared in :mod:`constants`, limits the
response size, verifies ASCII decoding and the expected token count, and writes
the result atomically.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import tempfile
import urllib.request
from pathlib import Path

try:
    from .constants import RAW_BYTES, RAW_FILENAME, RAW_SHA256, RAW_URL, RAW_VALUES
except ImportError:  # pragma: no cover - direct script execution
    from constants import RAW_BYTES, RAW_FILENAME, RAW_SHA256, RAW_URL, RAW_VALUES

MAX_DOWNLOAD_BYTES = 1_000_000


def _existing_regular_file(path: Path) -> os.stat_result | None:
    """Inspect one existing directory entry without following symlinks."""

    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("Existing raw-data target is inaccessible.") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(
            "Existing raw-data target must be a regular file, not a symlink or special file."
        )
    return metadata


def _read_existing_payload(path: Path, metadata: os.stat_result) -> bytes:
    """Read a size-bounded regular file while detecting entry-swap races."""

    if metadata.st_size > MAX_DOWNLOAD_BYTES:
        raise ValueError("Refusing an unexpectedly large existing raw-data file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("Existing raw-data target could not be opened safely.") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("Existing raw-data target must be a regular file.")
        if (metadata.st_dev, metadata.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("Existing raw-data target changed while it was being opened.")
        if opened.st_size > MAX_DOWNLOAD_BYTES:
            raise ValueError("Refusing an unexpectedly large existing raw-data file.")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            payload = stream.read(opened.st_size + 1)
        if len(payload) != opened.st_size:
            raise ValueError("Existing raw-data target changed while it was being read.")
        return payload
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validated_payload(payload: bytes) -> str:
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError("Refusing an unexpectedly large raw-data response.")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != RAW_SHA256:
        raise ValueError(f"Raw-data SHA-256 mismatch: expected {RAW_SHA256}, received {digest}.")
    if len(payload) != RAW_BYTES:
        raise ValueError(f"Raw-data byte count mismatch: expected {RAW_BYTES}, received {len(payload)}.")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("The pinned raw data are not valid ASCII.") from exc
    if len(text.split()) != RAW_VALUES:
        raise ValueError(f"Expected {RAW_VALUES} ASCII numeric fields.")
    return digest


def download(target: str | Path, *, overwrite: bool = False, timeout: float = 30.0) -> Path:
    """Download and authenticate the pinned immutable source file."""

    destination = Path(target)
    existing = _existing_regular_file(destination)
    if existing is not None and not overwrite:
        _validated_payload(_read_existing_payload(destination, existing))
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        RAW_URL,
        headers={"User-Agent": "structnpe-rust-replication/0.1"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=float(timeout)) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Refusing an unexpectedly large raw-data response.")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    _validated_payload(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "raw" / RAW_FILENAME,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = download(args.output, overwrite=args.overwrite)
    print(f"verified {output} sha256={RAW_SHA256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
