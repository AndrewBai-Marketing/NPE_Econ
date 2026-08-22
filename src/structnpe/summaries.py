"""Small summary helpers for examples and user projects."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def numeric_summary(data: object) -> np.ndarray:
    """Return a finite one-dimensional numeric summary for array-like data."""

    arr = np.asarray(data, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    flat = arr.reshape(-1)
    return np.array(
        [
            float(flat.mean()) if flat.size else 0.0,
            float(flat.std()) if flat.size else 0.0,
            float(flat.min()) if flat.size else 0.0,
            float(flat.max()) if flat.size else 0.0,
        ],
        dtype=float,
    )


def csv_numeric_summary(rows: Iterable[dict[str, object]], columns: list[str] | None = None) -> np.ndarray:
    """Summarize numeric columns in a list of CSV rows."""

    materialized = list(rows)
    if not materialized:
        return np.zeros(4, dtype=float)
    cols = columns or list(materialized[0].keys())
    values: list[float] = []
    for row in materialized:
        for col in cols:
            try:
                values.append(float(row.get(col, 0.0)))
            except (TypeError, ValueError):
                continue
    return numeric_summary(np.asarray(values, dtype=float))
