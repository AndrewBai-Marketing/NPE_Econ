"""Simulation-bank creation and storage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .simulator import ensure_summary_array, is_model_index_spec


@dataclass
class SimulationBank:
    summaries: np.ndarray
    theta: np.ndarray
    model_index: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "summaries": np.asarray(self.summaries, dtype=float),
            "theta": np.asarray(self.theta),
            "metadata_json": json.dumps(self.metadata or {}),
        }
        if self.model_index is not None:
            arrays["model_index"] = np.asarray(self.model_index, dtype=int)
        np.savez_compressed(path, **arrays)
        return path


def simulate_bank(spec: Any, n: int, seed: int, out_path: str | Path | None = None) -> SimulationBank:
    """Simulate a training/validation bank from a user simulator."""

    rng = np.random.default_rng(seed)
    summaries: list[np.ndarray] = []
    theta_rows: list[Any] = []
    model_rows: list[int] = []

    if is_model_index_spec(spec):
        models = np.asarray(spec.sample_model(n, rng), dtype=int)
        if len(models) != n:
            raise ValueError("sample_model(n, rng) must return n model indices.")
        for m in models:
            theta_m = spec.sample_prior_given_model(int(m), rng)
            data = spec.simulate_given_model(int(m), theta_m, rng)
            summaries.append(ensure_summary_array(spec.summarize(data)))
            theta_rows.append(theta_m)
            model_rows.append(int(m))
    else:
        theta_draws = spec.sample_prior(n, rng)
        if len(theta_draws) != n:
            raise ValueError("sample_prior(n, rng) must return n primitive draws.")
        for theta in theta_draws:
            data = spec.simulate(theta, rng)
            summaries.append(ensure_summary_array(spec.summarize(data)))
            theta_rows.append(theta)

    bank = SimulationBank(
        summaries=np.vstack(summaries).astype(float),
        theta=_as_theta_array(theta_rows),
        model_index=np.asarray(model_rows, dtype=int) if model_rows else None,
        metadata={"n": n, "seed": seed, "model_index": bool(model_rows)},
    )
    if out_path is not None:
        bank.save(out_path)
    return bank


def load_bank(path: str | Path) -> SimulationBank:
    """Load a bank written by :func:`simulate_bank`."""

    with np.load(Path(path), allow_pickle=True) as data:
        metadata_raw = data["metadata_json"]
        metadata_text = str(metadata_raw.item() if getattr(metadata_raw, "shape", ()) == () else metadata_raw)
        return SimulationBank(
            summaries=np.asarray(data["summaries"], dtype=float),
            theta=np.asarray(data["theta"], dtype=float),
            model_index=np.asarray(data["model_index"], dtype=int) if "model_index" in data.files else None,
            metadata=json.loads(metadata_text) if metadata_text else {},
        )


def _as_theta_array(theta_rows: list[Any]) -> np.ndarray:
    try:
        arr = np.asarray(theta_rows, dtype=float)
    except (TypeError, ValueError):
        arr = np.asarray(theta_rows, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(len(theta_rows), -1)
    return arr
