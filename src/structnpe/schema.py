"""Public model, parameter, and observation contracts for the beta API."""

from __future__ import annotations

import importlib
import inspect
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np


_TRANSFORMS = {"auto", "identity", "log", "lower_log", "upper_log", "logit"}


@dataclass(frozen=True)
class ParameterSpec:
    """Definition of one ordered structural parameter.

    Bounds are metadata and compatibility constraints. Transforms operate on
    the interior of bounded supports and are inverted before draws are returned
    to users.
    """

    name: str
    lower: float | None = None
    upper: float | None = None
    transform: str = "auto"
    description: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Parameter names must be non-empty strings.")
        if self.lower is not None and not math.isfinite(float(self.lower)):
            raise ValueError(f"lower bound for {self.name!r} must be finite or None.")
        if self.upper is not None and not math.isfinite(float(self.upper)):
            raise ValueError(f"upper bound for {self.name!r} must be finite or None.")
        if self.lower is not None and self.upper is not None and self.lower >= self.upper:
            raise ValueError(f"lower must be less than upper for {self.name!r}.")
        if self.transform not in _TRANSFORMS:
            raise ValueError(f"Unknown transform {self.transform!r} for {self.name!r}.")
        kind = self.resolved_transform
        if kind == "identity" and (self.lower is not None or self.upper is not None):
            raise ValueError(
                "transform='identity' cannot enforce finite declared bounds; use transform='auto'."
            )
        if kind == "log" and (self.lower not in (0, 0.0) or self.upper is not None):
            raise ValueError("transform='log' requires lower=0 and upper=None.")
        if kind == "lower_log" and (self.lower is None or self.upper is not None):
            raise ValueError("transform='lower_log' requires only a finite lower bound.")
        if kind == "upper_log" and (self.upper is None or self.lower is not None):
            raise ValueError("transform='upper_log' requires only a finite upper bound.")
        if kind == "logit" and (self.lower is None or self.upper is None):
            raise ValueError("transform='logit' requires finite lower and upper bounds.")

    @property
    def resolved_transform(self) -> str:
        if self.transform != "auto":
            return self.transform
        if self.lower is not None and self.upper is not None:
            return "logit"
        if self.lower is not None:
            return "lower_log"
        if self.upper is not None:
            return "upper_log"
        return "identity"

    def transform_values(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        self.validate_values(x, transformed=False, strict=self.resolved_transform != "identity")
        kind = self.resolved_transform
        if kind == "identity":
            return x.copy()
        if kind == "log":
            if np.any(x <= 0):
                raise ValueError(f"Values for {self.name!r} must be strictly positive for transform='log'.")
            return np.log(x)
        if kind == "lower_log":
            return np.log(x - float(self.lower))
        if kind == "upper_log":
            return np.log(float(self.upper) - x)
        lower, upper = float(self.lower), float(self.upper)
        unit = (x - lower) / (upper - lower)
        return np.log(unit) - np.log1p(-unit)

    def inverse_values(self, values: np.ndarray) -> np.ndarray:
        z = np.asarray(values, dtype=float)
        kind = self.resolved_transform
        if kind == "identity":
            out = z.copy()
        elif kind == "log":
            out = np.exp(z)
        elif kind == "lower_log":
            out = float(self.lower) + np.exp(z)
        elif kind == "upper_log":
            out = float(self.upper) - np.exp(z)
        else:
            # Stable logistic avoids overflow warnings for extreme neural draws.
            positive = z >= 0
            unit = np.empty_like(z, dtype=float)
            unit[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
            ez = np.exp(z[~positive])
            unit[~positive] = ez / (1.0 + ez)
            out = float(self.lower) + (float(self.upper) - float(self.lower)) * unit
        return out

    def validate_values(self, values: np.ndarray, *, transformed: bool = False, strict: bool = False) -> None:
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            location = "transformed " if transformed else ""
            raise ValueError(f"{location}values for {self.name!r} contain non-finite entries.")
        if transformed:
            return
        if self.lower is not None:
            invalid = arr <= self.lower if strict else arr < self.lower
            if np.any(invalid):
                relation = "strictly above" if strict else "at or above"
                raise ValueError(f"Values for {self.name!r} must be {relation} {self.lower}.")
        if self.upper is not None:
            invalid = arr >= self.upper if strict else arr > self.upper
            if np.any(invalid):
                relation = "strictly below" if strict else "at or below"
                raise ValueError(f"Values for {self.name!r} must be {relation} {self.upper}.")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolved_transform"] = self.resolved_transform
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParameterSpec":
        return cls(
            name=str(data["name"]),
            lower=_optional_float(data.get("lower")),
            upper=_optional_float(data.get("upper")),
            transform=str(data.get("transform", data.get("resolved_transform", "auto"))),
            description=_optional_str(data.get("description")),
            unit=_optional_str(data.get("unit")),
        )


class ObservationAdapter:
    """Base class for deterministic raw-data representations.

    Custom portable adapters must be importable classes, implement
    :meth:`transform`, and expose structured constructor state through
    :meth:`get_config` / :meth:`from_config`.
    """

    output_dim: int | None = None

    def fit(self, observations: Sequence[Any]) -> "ObservationAdapter":
        if not observations:
            raise ValueError("Cannot fit an observation adapter without observations.")
        rows = [self._checked_transform(item) for item in observations]
        dims = {row.size for row in rows}
        if len(dims) != 1:
            raise ValueError(f"Observation adapter produced inconsistent dimensions: {sorted(dims)}.")
        self.output_dim = rows[0].size
        return self

    def transform(self, observation: Any) -> np.ndarray:
        raise NotImplementedError

    def transform_one(self, observation: Any) -> np.ndarray:
        row = self._checked_transform(observation)
        if self.output_dim is None:
            self.output_dim = row.size
        if row.size != self.output_dim:
            raise ValueError(
                f"Observation representation has dimension {row.size}; expected {self.output_dim}."
            )
        return row

    def transform_many(self, observations: Sequence[Any]) -> np.ndarray:
        rows = [self.transform_one(item) for item in observations]
        if not rows:
            raise ValueError("Observed-data batch must not be empty.")
        return np.vstack(rows).astype(np.float64, copy=False)

    def _checked_transform(self, observation: Any) -> np.ndarray:
        arr = np.asarray(self.transform(observation), dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(
                f"Observation adapter must return one-dimensional output; got shape {arr.shape}."
            )
        if arr.size == 0:
            raise ValueError("Observation adapter returned an empty representation.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Observation representation contains non-finite values.")
        return arr

    def get_config(self) -> dict[str, Any]:
        return {}

    def get_state(self) -> dict[str, Any]:
        return {"output_dim": self.output_dim}

    def set_state(self, state: Mapping[str, Any]) -> None:
        output_dim = state.get("output_dim")
        self.output_dim = int(output_dim) if output_dim is not None else None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ObservationAdapter":
        return cls(**dict(config))


class ArrayAdapter(ObservationAdapter):
    """Flatten a fixed-shape numeric observation into a vector."""

    def __init__(self, expected_shape: Sequence[int] | None = None) -> None:
        self._declared_expected_shape = (
            tuple(int(v) for v in expected_shape) if expected_shape is not None else None
        )
        self.expected_shape = self._declared_expected_shape
        self.output_dim = int(np.prod(self.expected_shape)) if self.expected_shape is not None else None

    def fit(self, observations: Sequence[Any]) -> "ArrayAdapter":
        if not observations:
            raise ValueError("Cannot fit ArrayAdapter without observations.")
        first = np.asarray(observations[0])
        if self.expected_shape is None:
            self.expected_shape = tuple(first.shape)
        for item in observations:
            self.transform(item)
        self.output_dim = int(np.prod(self.expected_shape, dtype=int))
        return self

    def transform(self, observation: Any) -> np.ndarray:
        arr = np.asarray(observation, dtype=np.float64)
        if self.expected_shape is not None and tuple(arr.shape) != self.expected_shape:
            raise ValueError(f"Observed array has shape {arr.shape}; expected {self.expected_shape}.")
        return arr.reshape(-1)

    def get_config(self) -> dict[str, Any]:
        return {
            "expected_shape": (
                list(self._declared_expected_shape) if self._declared_expected_shape is not None else None
            )
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "output_dim": self.output_dim,
            "fitted_expected_shape": list(self.expected_shape) if self.expected_shape is not None else None,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        super().set_state(state)
        fitted = state.get("fitted_expected_shape")
        self.expected_shape = tuple(int(v) for v in fitted) if fitted is not None else self._declared_expected_shape


class SummaryAdapter(ObservationAdapter):
    """Return selected deterministic scalar summaries of a numeric array."""

    _ALLOWED = {"mean", "std", "min", "max", "size"}

    def __init__(
        self,
        statistics: Sequence[str] = ("mean", "std"),
        expected_shape: Sequence[int] | None = None,
    ) -> None:
        self.statistics = tuple(str(item) for item in statistics)
        unknown = set(self.statistics) - self._ALLOWED
        if not self.statistics or unknown:
            raise ValueError(f"Unknown or empty summary statistics: {sorted(unknown)}.")
        self._declared_expected_shape = (
            tuple(int(v) for v in expected_shape) if expected_shape is not None else None
        )
        self.expected_shape = self._declared_expected_shape
        self.output_dim = len(self.statistics)

    def fit(self, observations: Sequence[Any]) -> "SummaryAdapter":
        if not observations:
            raise ValueError("Cannot fit SummaryAdapter without observations.")
        if self.expected_shape is None:
            self.expected_shape = tuple(np.asarray(observations[0]).shape)
        for item in observations:
            self.transform(item)
        return self

    def transform(self, observation: Any) -> np.ndarray:
        arr = np.asarray(observation, dtype=np.float64)
        if self.expected_shape is not None and tuple(arr.shape) != self.expected_shape:
            raise ValueError(f"Observed array has shape {arr.shape}; expected {self.expected_shape}.")
        flat = arr.reshape(-1)
        if flat.size == 0:
            raise ValueError("Cannot summarize an empty observation.")
        functions: dict[str, Callable[[], float]] = {
            "mean": lambda: float(np.mean(flat)),
            "std": lambda: float(np.std(flat)),
            "min": lambda: float(np.min(flat)),
            "max": lambda: float(np.max(flat)),
            "size": lambda: float(flat.size),
        }
        return np.asarray([functions[name]() for name in self.statistics], dtype=np.float64)

    def get_config(self) -> dict[str, Any]:
        return {
            "statistics": list(self.statistics),
            "expected_shape": (
                list(self._declared_expected_shape) if self._declared_expected_shape is not None else None
            ),
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "output_dim": self.output_dim,
            "fitted_expected_shape": list(self.expected_shape) if self.expected_shape is not None else None,
        }

    def set_state(self, state: Mapping[str, Any]) -> None:
        super().set_state(state)
        fitted = state.get("fitted_expected_shape")
        self.expected_shape = tuple(int(v) for v in fitted) if fitted is not None else self._declared_expected_shape


def adapter_to_dict(adapter: ObservationAdapter) -> dict[str, Any]:
    cls = adapter.__class__
    qualname = cls.__qualname__
    if "<locals>" in qualname or cls.__module__ == "__main__":
        raise ValueError(
            "Portable estimator artifacts require an importable observation-adapter class; "
            "move it out of __main__/a local scope or use a built-in adapter."
        )
    return {
        "module": cls.__module__,
        "qualname": qualname,
        "config": adapter.get_config(),
        "state": adapter.get_state(),
    }


def adapter_from_dict(
    data: Mapping[str, Any],
    *,
    allow_custom: bool = False,
    trusted_class: type[ObservationAdapter] | None = None,
) -> ObservationAdapter:
    """Reconstruct an adapter from structured metadata.

    Built-in adapters are allowlisted and require no dynamic import. Custom
    adapter imports are disabled by default because importing an
    artifact-selected module and invoking ``from_config`` executes installed
    Python code. A caller may instead supply an already imported
    ``trusted_class`` or explicitly opt in with ``allow_custom=True``.
    """

    module_name = str(data["module"])
    qualname = str(data["qualname"])
    identity = (module_name, qualname)
    builtins: dict[tuple[str, str], type[ObservationAdapter]] = {
        (__name__, "ArrayAdapter"): ArrayAdapter,
        (__name__, "SummaryAdapter"): SummaryAdapter,
    }
    obj: Any
    if trusted_class is not None:
        trusted_identity = (trusted_class.__module__, trusted_class.__qualname__)
        if trusted_identity != identity:
            raise ValueError(
                "The supplied trusted observation-adapter class does not match the serialized adapter."
            )
        obj = trusted_class
    elif identity in builtins:
        obj = builtins[identity]
    elif not allow_custom:
        raise ValueError(
            f"Custom observation adapter {module_name}.{qualname} is not imported by default. "
            "Pass a compatible StructuralModel to load_estimator, or explicitly set "
            "allow_custom_adapter=True only for trusted artifacts and installed code."
        )
    else:
        if (
            not module_name
            or any(not part.isidentifier() for part in module_name.split("."))
            or not qualname
            or "<locals>" in qualname
            or any(not part.isidentifier() for part in qualname.split("."))
        ):
            raise ValueError("Serialized custom adapter has an invalid import path.")
        try:
            obj = importlib.import_module(module_name)
            for part in qualname.split("."):
                obj = getattr(obj, part)
        except (ImportError, AttributeError) as exc:
            raise ValueError(f"Cannot reconstruct observation adapter {module_name}.{qualname}.") from exc
    if not inspect.isclass(obj) or not issubclass(obj, ObservationAdapter):
        raise ValueError(f"Serialized adapter {module_name}.{qualname} is not an ObservationAdapter class.")
    adapter = obj.from_config(dict(data.get("config") or {}))
    # ``output_dim`` was used by an early beta-development artifact. Accept it
    # only as structured state; no pickle or arbitrary object migration occurs.
    state = dict(data.get("state") or {})
    if "output_dim" in data and "output_dim" not in state:
        state["output_dim"] = data.get("output_dim")
    adapter.set_state(state)
    return adapter


class StructuralModel:
    """Bring-your-own-simulator model definition for amortized inference."""

    def __init__(
        self,
        *,
        prior: Any,
        simulator: Any,
        parameter_names: Sequence[str] | None = None,
        parameters: Sequence[ParameterSpec | Mapping[str, Any]] | None = None,
        observation_adapter: ObservationAdapter | None = None,
        prior_id: str | None = None,
        simulator_id: str | None = None,
        prior_config: Mapping[str, Any] | None = None,
        simulator_config: Mapping[str, Any] | None = None,
        batched_simulator: bool = False,
    ) -> None:
        if parameters is not None and parameter_names is not None:
            raise ValueError("Specify parameters or parameter_names, not both.")
        if parameters is None:
            if not parameter_names:
                raise ValueError("At least one parameter name or ParameterSpec is required.")
            self.parameters = tuple(ParameterSpec(str(name)) for name in parameter_names)
        else:
            self.parameters = tuple(
                item if isinstance(item, ParameterSpec) else ParameterSpec.from_dict(item) for item in parameters
            )
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Parameter names must be unique and ordered.")
        self.prior = prior
        self.simulator = simulator
        self.observation_adapter = observation_adapter or ArrayAdapter()
        if not isinstance(self.observation_adapter, ObservationAdapter):
            raise TypeError("observation_adapter must be an ObservationAdapter instance.")
        _validate_callable_spec(prior, "prior", prior_id, prior_config)
        _validate_callable_spec(simulator, "simulator", simulator_id, simulator_config)
        self.prior_id = prior_id or _callable_identifier(prior, "prior")
        self.simulator_id = simulator_id or _callable_identifier(simulator, "simulator")
        self.prior_config = dict(prior_config or {})
        self.simulator_config = dict(simulator_config or {})
        self.batched_simulator = bool(batched_simulator)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if n < 1:
            raise ValueError("Prior batch size must be positive.")
        sampler = getattr(self.prior, "sample", self.prior)
        if not callable(sampler):
            raise TypeError("prior must be callable or expose sample(n, rng).")
        raw = sampler(int(n), rng)
        if isinstance(raw, Mapping):
            missing = [name for name in self.parameter_names if name not in raw]
            if missing:
                raise ValueError(f"Prior mapping is missing parameters: {missing}.")
            arr = np.column_stack([np.asarray(raw[name], dtype=float) for name in self.parameter_names])
        else:
            arr = np.asarray(raw, dtype=float)
            if arr.ndim == 1 and len(self.parameters) == 1 and arr.size == n:
                arr = arr.reshape(n, 1)
        if arr.shape != (n, len(self.parameters)):
            raise ValueError(
                f"Prior must return shape {(n, len(self.parameters))}; received {arr.shape}."
            )
        for j, parameter in enumerate(self.parameters):
            parameter.validate_values(arr[:, j], strict=parameter.resolved_transform != "identity")
        return arr.astype(np.float64, copy=False)

    def simulate_one(self, theta: np.ndarray, rng: np.random.Generator) -> Any:
        simulator = getattr(self.simulator, "simulate", self.simulator)
        if not callable(simulator):
            raise TypeError("simulator must be callable or expose simulate(theta, rng).")
        row = np.asarray(theta, dtype=float)
        if row.shape != (len(self.parameters),):
            raise ValueError(f"One parameter draw must have shape {(len(self.parameters),)}; got {row.shape}.")
        return simulator(row, rng)

    def simulate_batch(self, theta: np.ndarray, rng: np.random.Generator) -> list[Any]:
        arr = np.asarray(theta, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != len(self.parameters):
            raise ValueError(f"Parameter batch must have shape (n, {len(self.parameters)}); got {arr.shape}.")
        simulator = getattr(self.simulator, "simulate", self.simulator)
        if self.batched_simulator:
            raw = simulator(arr, rng)
            if len(raw) != len(arr):
                raise ValueError("A batched simulator must return one observation per parameter row.")
            return list(raw)
        return [self.simulate_one(row, rng) for row in arr]

    def transform_parameters(self, theta: np.ndarray) -> np.ndarray:
        arr = np.asarray(theta, dtype=float)
        if arr.shape[-1] != len(self.parameters):
            raise ValueError("Parameter array has the wrong final dimension.")
        return np.stack(
            [parameter.transform_values(arr[..., j]) for j, parameter in enumerate(self.parameters)], axis=-1
        )

    def inverse_parameters(self, transformed: np.ndarray) -> np.ndarray:
        arr = np.asarray(transformed, dtype=float)
        if arr.shape[-1] != len(self.parameters):
            raise ValueError("Transformed parameter array has the wrong final dimension.")
        return np.stack(
            [parameter.inverse_values(arr[..., j]) for j, parameter in enumerate(self.parameters)], axis=-1
        )

    def specification(self, *, include_adapter_state: bool = True) -> dict[str, Any]:
        adapter = adapter_to_dict(self.observation_adapter)
        if not include_adapter_state:
            adapter = dict(adapter)
            adapter.pop("state", None)
        return {
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "prior": {"identifier": self.prior_id, "config": self.prior_config},
            "simulator": {"identifier": self.simulator_id, "config": self.simulator_config},
            "observation_adapter": adapter,
            "batched_simulator": self.batched_simulator,
        }


def _callable_identifier(value: Any, label: str) -> str:
    target = value if inspect.isfunction(value) or inspect.isclass(value) else None
    if target is None:
        raise ValueError(
            f"{label}_id and {label}_config are required for callable instances, methods, "
            "partials, and built-ins; structnpe does not introspect their state."
        )
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if (
        not module
        or not qualname
        or qualname == "<lambda>"
        or "<locals>" in qualname
        or module == "__main__"
    ):
        raise ValueError(
            f"{label}_id is required when the {label} does not have a stable importable identifier."
        )
    return f"{module}.{qualname}"


def _validate_callable_spec(
    value: Any,
    label: str,
    identifier: str | None,
    config: Mapping[str, Any] | None,
) -> None:
    if identifier is not None and (not isinstance(identifier, str) or not identifier.strip()):
        raise ValueError(f"{label}_id must be a non-empty string when supplied.")
    if config is not None and not isinstance(config, Mapping):
        raise TypeError(f"{label}_config must be a mapping when supplied.")
    plain_named_callable = inspect.isfunction(value) or inspect.isclass(value)
    qualname = getattr(value, "__qualname__", "") if plain_named_callable else ""
    plain_named_callable = bool(
        plain_named_callable
        and qualname
        and qualname != "<lambda>"
        and "<locals>" not in qualname
    )
    if not plain_named_callable and (identifier is None or config is None):
        raise ValueError(
            f"{label}_id and {label}_config must both be supplied for a callable without a "
            "stable stateless function/class identity. This prevents stateful fingerprint collisions."
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "ArrayAdapter",
    "ObservationAdapter",
    "ParameterSpec",
    "StructuralModel",
    "SummaryAdapter",
    "adapter_from_dict",
    "adapter_to_dict",
]
