"""Posterior estimator implementations for ``structnpe``."""

from .finite_grid import FiniteGridClassifier
from .gaussian import GaussianPosterior
from .mdn import MDNPosterior
from .model_index import ModelIndexPosterior

__all__ = ["FiniteGridClassifier", "GaussianPosterior", "MDNPosterior", "ModelIndexPosterior"]
