"""Amortized posterior inference for simulator-defined structural models."""

from ._artifacts import ArtifactError
from ._version import __version__
from .api import infer, run_training, validate
from .estimator import TrainedEstimator, ValidationResult, fit, load_estimator
from .results import InferenceResult, PredictiveCheckResult, load_result
from .schema import ArrayAdapter, ObservationAdapter, ParameterSpec, StructuralModel, SummaryAdapter
from .simulator import ModelIndexSimulatorSpec, SimulatorSpec

__all__ = [
    "ArrayAdapter",
    "ArtifactError",
    "InferenceResult",
    "ModelIndexSimulatorSpec",
    "ObservationAdapter",
    "ParameterSpec",
    "PredictiveCheckResult",
    "SimulatorSpec",
    "StructuralModel",
    "SummaryAdapter",
    "TrainedEstimator",
    "ValidationResult",
    "__version__",
    "fit",
    "infer",
    "load_estimator",
    "load_result",
    "run_training",
    "validate",
]
