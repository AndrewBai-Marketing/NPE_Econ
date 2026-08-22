"""Finite-grid posterior classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..simulation_bank import SimulationBank


class FiniteGridClassifier:
    """Class-conditional Gaussian classifier over finite theta cells."""

    estimator_type = "finite_grid"

    def __init__(self) -> None:
        self.theta_cells: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.class_means: np.ndarray | None = None
        self.class_priors: np.ndarray | None = None
        self.summary_scale: np.ndarray | None = None

    def fit(self, train_bank: SimulationBank, valid_bank: SimulationBank | None = None, config: Any | None = None) -> "FiniteGridClassifier":
        summaries = np.asarray(train_bank.summaries, dtype=float)
        theta = np.asarray(train_bank.theta, dtype=float)
        cells, inverse = np.unique(theta, axis=0, return_inverse=True)
        self.theta_cells = cells
        self.labels = np.arange(len(cells), dtype=int)
        means: list[np.ndarray] = []
        priors: list[float] = []
        global_scale = np.where(summaries.std(axis=0) > 1.0e-8, summaries.std(axis=0), 1.0)
        for label in range(len(cells)):
            mask = inverse == label
            means.append(summaries[mask].mean(axis=0))
            priors.append(float(mask.mean()))
        self.class_means = np.vstack(means)
        self.class_priors = np.asarray(priors, dtype=float)
        self.class_priors = self.class_priors / self.class_priors.sum()
        self.summary_scale = global_scale
        return self

    def predict(self, observed_summary: np.ndarray) -> dict[str, np.ndarray]:
        self._check_fit()
        x = np.asarray(observed_summary, dtype=float).reshape(-1)
        means = self.class_means  # type: ignore[assignment]
        scale = self.summary_scale  # type: ignore[assignment]
        priors = self.class_priors  # type: ignore[assignment]
        distances = np.sum(((means - x[None, :]) / scale[None, :]) ** 2, axis=1)
        logits = -0.5 * distances + np.log(np.maximum(priors, 1.0e-12))
        weights = _softmax(logits)
        mean = weights @ self.theta_cells  # type: ignore[operator]
        return {"weights": weights, "theta": self.theta_cells, "mean": mean}

    def sample(self, observed_summary: np.ndarray, n_draws: int, seed: int | None = None) -> np.ndarray:
        pred = self.predict(observed_summary)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pred["weights"]), size=n_draws, p=pred["weights"])
        return np.asarray(pred["theta"])[idx]

    def save(self, path: str | Path) -> Path:
        self._check_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            estimator_type=self.estimator_type,
            theta_cells=self.theta_cells,
            class_means=self.class_means,
            class_priors=self.class_priors,
            summary_scale=self.summary_scale,
            metadata_json=json.dumps({"estimator_type": self.estimator_type}),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FiniteGridClassifier":
        est = cls()
        with np.load(Path(path), allow_pickle=True) as data:
            est.theta_cells = np.asarray(data["theta_cells"], dtype=float)
            est.class_means = np.asarray(data["class_means"], dtype=float)
            est.class_priors = np.asarray(data["class_priors"], dtype=float)
            est.summary_scale = np.asarray(data["summary_scale"], dtype=float)
            est.labels = np.arange(len(est.theta_cells), dtype=int)
        return est

    def _check_fit(self) -> None:
        if self.theta_cells is None or self.class_means is None or self.class_priors is None or self.summary_scale is None:
            raise RuntimeError("FiniteGridClassifier has not been fit.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / exp.sum()
