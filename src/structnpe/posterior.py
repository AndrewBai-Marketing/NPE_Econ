"""Posterior estimator factory and persistence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import PosteriorConfig
from .estimators import FiniteGridClassifier, GaussianPosterior, MDNPosterior, ModelIndexPosterior
from .simulation_bank import SimulationBank


Estimator = FiniteGridClassifier | GaussianPosterior | MDNPosterior | ModelIndexPosterior


def make_estimator(config: PosteriorConfig | str) -> Estimator:
    posterior_type = config.type if isinstance(config, PosteriorConfig) else str(config)
    if posterior_type == "finite_grid":
        return FiniteGridClassifier()
    if posterior_type == "gaussian":
        return GaussianPosterior()
    if posterior_type == "mdn":
        return MDNPosterior()
    if posterior_type == "model_index":
        return ModelIndexPosterior()
    raise ValueError(f"Unknown posterior estimator type: {posterior_type}")


def fit_estimator(train_bank: SimulationBank, valid_bank: SimulationBank | None, config: PosteriorConfig | str) -> Estimator:
    est = make_estimator(config)
    est.fit(train_bank, valid_bank, config)
    return est


def load_posterior(path: str | Path) -> Estimator:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        estimator_type = str(data["estimator_type"].item() if getattr(data["estimator_type"], "shape", ()) == () else data["estimator_type"])
    if estimator_type == "finite_grid":
        return FiniteGridClassifier.load(path)
    if estimator_type == "gaussian":
        return GaussianPosterior.load(path)
    if estimator_type == "mdn":
        return MDNPosterior.load(path)
    if estimator_type == "model_index":
        return ModelIndexPosterior.load(path)
    raise ValueError(f"Unknown saved estimator type: {estimator_type}")


def posterior_summary(draws: np.ndarray) -> list[dict[str, object]]:
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows: list[dict[str, object]] = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        rows.append(
            {
                "parameter": f"theta_{j}",
                "mean": float(col.mean()),
                "sd": float(col.std()),
                "q05": float(np.quantile(col, 0.05)),
                "q50": float(np.quantile(col, 0.50)),
                "q95": float(np.quantile(col, 0.95)),
            }
        )
    return rows
