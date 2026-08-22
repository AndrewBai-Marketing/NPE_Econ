"""Simulator interfaces for ``structnpe``.

Users provide the economics. The package only assumes it can sample from a
prior, simulate data, turn simulated or observed data into a numerical summary,
and optionally evaluate counterfactuals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class SimulatorSpec(ABC):
    """Base class for a simulator-defined structural model.

    If ``summarize`` returns a representation ``T(x)``, the learned posterior
    target is the simulator posterior conditional on ``T(x)``, not necessarily
    the posterior conditional on the full raw data ``x``.
    """

    @abstractmethod
    def sample_prior(self, n: int, rng: np.random.Generator) -> Any:
        """Draw ``n`` primitive values from the prior."""

    @abstractmethod
    def simulate(self, theta: Any, rng: np.random.Generator, **kwargs: Any) -> Any:
        """Simulate one dataset from primitives ``theta``."""

    @abstractmethod
    def summarize(self, data: Any) -> np.ndarray:
        """Return a one-dimensional numerical summary array ``T(data)``."""

    def counterfactual(self, theta: Any, policy: Any, rng: np.random.Generator | None = None) -> Any:
        """Evaluate a counterfactual policy effect.

        Subclasses may override this. The default makes counterfactual commands
        fail explicitly rather than silently returning a meaningless value.
        """

        raise NotImplementedError("This simulator does not define counterfactual(theta, policy).")


class ModelIndexSimulatorSpec(ABC):
    """Base class for model-index or mechanism-index structural simulators.

    The posterior target is over ``(m, theta_m)`` conditional on the supplied
    data representation. This is the interface for latent-mechanism ambiguity
    examples.
    """

    @abstractmethod
    def sample_model(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` model/mechanism indices from the prior over ``m``."""

    @abstractmethod
    def sample_prior_given_model(self, m: int, rng: np.random.Generator) -> Any:
        """Draw primitives conditional on model index ``m``."""

    @abstractmethod
    def simulate_given_model(self, m: int, theta_m: Any, rng: np.random.Generator, **kwargs: Any) -> Any:
        """Simulate one dataset conditional on ``(m, theta_m)``."""

    @abstractmethod
    def summarize(self, data: Any) -> np.ndarray:
        """Return a one-dimensional numerical summary array ``T(data)``."""

    def counterfactual(self, m: int, theta_m: Any, policy: Any, rng: np.random.Generator | None = None) -> Any:
        """Evaluate a counterfactual policy effect under model ``m``."""

        raise NotImplementedError("This simulator does not define counterfactual(m, theta_m, policy).")


def is_model_index_spec(spec: Any) -> bool:
    """Return whether an object implements the model-index simulator methods."""

    return all(
        hasattr(spec, name)
        for name in ("sample_model", "sample_prior_given_model", "simulate_given_model", "summarize")
    )


def ensure_summary_array(summary: Any) -> np.ndarray:
    """Convert a simulator summary to a finite one-dimensional float array."""

    arr = np.asarray(summary, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Simulator summary contains non-finite values.")
    return arr.astype(float)
