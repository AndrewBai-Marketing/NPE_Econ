"""Gaussian posterior baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..simulation_bank import SimulationBank


class GaussianPosterior:
    """Ridge-linear Gaussian approximation ``theta | summary``."""

    estimator_type = "gaussian"

    def __init__(self, ridge: float = 1.0e-6) -> None:
        self.ridge = ridge
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.resid_sd: np.ndarray | None = None

    def fit(self, train_bank: SimulationBank, valid_bank: SimulationBank | None = None, config: Any | None = None) -> "GaussianPosterior":
        x = np.asarray(train_bank.summaries, dtype=float)
        y = np.asarray(train_bank.theta, dtype=float)
        self.x_mean = x.mean(axis=0)
        self.x_scale = np.where(x.std(axis=0) > 1.0e-8, x.std(axis=0), 1.0)
        xs = (x - self.x_mean[None, :]) / self.x_scale[None, :]
        design = np.column_stack([np.ones(xs.shape[0]), xs])
        gram = design.T @ design + self.ridge * np.eye(design.shape[1])
        self.coef = np.linalg.solve(gram, design.T @ y)
        resid = y - design @ self.coef
        self.resid_sd = np.where(resid.std(axis=0) > 1.0e-8, resid.std(axis=0), 1.0e-3)
        return self

    def predict(self, observed_summary: np.ndarray) -> dict[str, np.ndarray]:
        self._check_fit()
        x = np.asarray(observed_summary, dtype=float).reshape(1, -1)
        xs = (x - self.x_mean[None, :]) / self.x_scale[None, :]  # type: ignore[index]
        design = np.column_stack([np.ones(xs.shape[0]), xs])
        mean = design @ self.coef  # type: ignore[operator]
        return {"mean": mean.reshape(-1), "sd": self.resid_sd}

    def sample(self, observed_summary: np.ndarray, n_draws: int, seed: int | None = None) -> np.ndarray:
        pred = self.predict(observed_summary)
        rng = np.random.default_rng(seed)
        return rng.normal(pred["mean"][None, :], pred["sd"][None, :], size=(n_draws, len(pred["mean"])))

    def save(self, path: str | Path) -> Path:
        self._check_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            estimator_type=self.estimator_type,
            x_mean=self.x_mean,
            x_scale=self.x_scale,
            coef=self.coef,
            resid_sd=self.resid_sd,
            metadata_json=json.dumps({"estimator_type": self.estimator_type}),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "GaussianPosterior":
        est = cls()
        with np.load(Path(path), allow_pickle=True) as data:
            est.x_mean = np.asarray(data["x_mean"], dtype=float)
            est.x_scale = np.asarray(data["x_scale"], dtype=float)
            est.coef = np.asarray(data["coef"], dtype=float)
            est.resid_sd = np.asarray(data["resid_sd"], dtype=float)
        return est

    def _check_fit(self) -> None:
        if self.x_mean is None or self.x_scale is None or self.coef is None or self.resid_sd is None:
            raise RuntimeError("GaussianPosterior has not been fit.")
