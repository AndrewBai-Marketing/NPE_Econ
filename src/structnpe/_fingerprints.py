"""Canonical structured fingerprints used by public beta artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value with stable ordering and formatting."""

    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Fingerprinted metadata cannot contain NaN or infinity.")
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        non_string = [key for key in value if not isinstance(key, str)]
        if non_string:
            raise TypeError(
                "Fingerprinted metadata mappings must use string keys; implicit key coercion "
                "could create compatibility-fingerprint collisions."
            )
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(
        f"Metadata of type {type(value).__name__} is not safely JSON serializable; "
        "supply a structured identifier/config instead."
    )


__all__ = ["canonical_json", "fingerprint", "sha256_file"]
