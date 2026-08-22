"""Model-index posterior classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..simulation_bank import SimulationBank
from .finite_grid import _softmax


class ModelIndexPosterior:
    """Finite classifier over ``(m, theta_m)`` cells with model marginals."""

    estimator_type = "model_index"

    def __init__(self) -> None:
        self.cells: np.ndarray | None = None
        self.models: np.ndarray | None = None
        self.theta_cells: np.ndarray | None = None
        self.class_means: np.ndarray | None = None
        self.class_priors: np.ndarray | None = None
        self.summary_scale: np.ndarray | None = None

    def fit(self, train_bank: SimulationBank, valid_bank: SimulationBank | None = None, config: Any | None = None) -> "ModelIndexPosterior":
        if train_bank.model_index is None:
            raise ValueError("ModelIndexPosterior requires bank.model_index.")
        summaries = np.asarray(train_bank.summaries, dtype=float)
        models = np.asarray(train_bank.model_index, dtype=int).reshape(-1, 1)
        theta = np.asarray(train_bank.theta, dtype=float)
        cells, inverse = np.unique(np.column_stack([models, theta]), axis=0, return_inverse=True)
        self.cells = cells
        self.models = cells[:, 0].astype(int)
        self.theta_cells = cells[:, 1:]
        self.summary_scale = np.where(summaries.std(axis=0) > 1.0e-8, summaries.std(axis=0), 1.0)
        means: list[np.ndarray] = []
        priors: list[float] = []
        for label in range(len(cells)):
            mask = inverse == label
            means.append(summaries[mask].mean(axis=0))
            priors.append(float(mask.mean()))
        self.class_means = np.vstack(means)
        self.class_priors = np.asarray(priors, dtype=float)
        self.class_priors = self.class_priors / self.class_priors.sum()
        return self

    def predict(self, observed_summary: np.ndarray) -> dict[str, np.ndarray]:
        self._check_fit()
        x = np.asarray(observed_summary, dtype=float).reshape(-1)
        distances = np.sum(((self.class_means - x[None, :]) / self.summary_scale[None, :]) ** 2, axis=1)  # type: ignore[operator]
        logits = -0.5 * distances + np.log(np.maximum(self.class_priors, 1.0e-12))  # type: ignore[arg-type]
        weights = _softmax(logits)
        unique_models = np.unique(self.models)  # type: ignore[arg-type]
        model_probs = np.array([weights[self.models == m].sum() for m in unique_models])  # type: ignore[operator]
        theta_mean = weights @ self.theta_cells  # type: ignore[operator]
        return {
            "weights": weights,
            "models": self.models,
            "theta": self.theta_cells,
            "mean": theta_mean,
            "model_values": unique_models,
            "model_probs": model_probs,
        }

    def sample(self, observed_summary: np.ndarray, n_draws: int, seed: int | None = None) -> np.ndarray:
        pred = self.predict(observed_summary)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pred["weights"]), size=n_draws, p=pred["weights"])
        return np.column_stack([np.asarray(pred["models"])[idx], np.asarray(pred["theta"])[idx]])

    def save(self, path: str | Path) -> Path:
        self._check_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            estimator_type=self.estimator_type,
            cells=self.cells,
            class_means=self.class_means,
            class_priors=self.class_priors,
            summary_scale=self.summary_scale,
            metadata_json=json.dumps({"estimator_type": self.estimator_type}),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ModelIndexPosterior":
        est = cls()
        with np.load(Path(path), allow_pickle=True) as data:
            est.cells = np.asarray(data["cells"], dtype=float)
            est.models = est.cells[:, 0].astype(int)
            est.theta_cells = est.cells[:, 1:]
            est.class_means = np.asarray(data["class_means"], dtype=float)
            est.class_priors = np.asarray(data["class_priors"], dtype=float)
            est.summary_scale = np.asarray(data["summary_scale"], dtype=float)
        return est

    def _check_fit(self) -> None:
        if (
            self.cells is None
            or self.models is None
            or self.theta_cells is None
            or self.class_means is None
            or self.class_priors is None
            or self.summary_scale is None
        ):
            raise RuntimeError("ModelIndexPosterior has not been fit.")
