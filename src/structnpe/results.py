"""Conventional posterior results and predictive diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ._artifacts import ArtifactError, read_directory_header, read_npz_arrays, write_directory_artifact
from .schema import ParameterSpec, adapter_to_dict

if TYPE_CHECKING:
    from .schema import StructuralModel


def _pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - packaging makes pandas required
        raise RuntimeError("pandas is required for tabular result output.") from exc
    return pd


@dataclass
class PredictiveCheckResult:
    """Posterior-predictive representation comparison."""

    table: Any
    metadata: dict[str, Any]
    warning: str = (
        "Passing selected posterior predictive checks does not establish correct model specification."
    )

    def to_dataframe(self) -> Any:
        return self.table.copy()


@dataclass
class InferenceResult:
    """Posterior draws, familiar summaries, and inference diagnostics."""

    draws: np.ndarray
    parameters: tuple[ParameterSpec, ...]
    metadata: dict[str, Any]
    representation: np.ndarray
    support_diagnostics: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    _model: "StructuralModel | None" = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.draws = np.asarray(self.draws, dtype=np.float64)
        self.representation = np.asarray(self.representation, dtype=np.float64)
        self.parameters = tuple(self.parameters)
        self.metadata = dict(self.metadata)
        if self.draws.ndim not in (2, 3):
            raise ValueError("Posterior draws must have shape (draw, parameter) or (dataset, draw, parameter).")
        if self.draws.size == 0 or any(size == 0 for size in self.draws.shape):
            raise ValueError("Posterior draws must contain at least one dataset, draw, and parameter.")
        if not np.all(np.isfinite(self.draws)):
            raise ValueError("Posterior draws contain non-finite values.")
        if self.draws.shape[-1] != len(self.parameters):
            raise ValueError("Posterior draws do not match the stored parameter ordering.")
        if self.representation.ndim not in (1, 2):
            raise ValueError(
                "Stored observation representation must have shape (feature,) or (dataset, feature)."
            )
        if self.representation.size == 0 or any(size == 0 for size in self.representation.shape):
            raise ValueError("Stored observation representation must not be empty.")
        if not np.all(np.isfinite(self.representation)):
            raise ValueError("Stored observation representation contains non-finite values.")
        expected_datasets = 1 if self.draws.ndim == 2 else self.draws.shape[0]
        reps = self.representation.reshape(1, -1) if self.representation.ndim == 1 else self.representation
        if reps.ndim != 2 or reps.shape[0] != expected_datasets:
            raise ValueError("Stored observation representation does not match the draw batch.")
        representation_dim = int(reps.shape[1])
        stored_dim = self.metadata.get("representation_dim")
        if stored_dim is not None and int(stored_dim) != representation_dim:
            raise ValueError(
                "Stored observation representation width does not match result metadata."
            )
        self.metadata.setdefault("representation_dim", representation_dim)
        if self._model is not None:
            adapter = self._model.observation_adapter
            if adapter.output_dim is None or int(adapter.output_dim) != representation_dim:
                raise ValueError(
                    "Fitted observation-adapter width does not match the stored representation."
                )
            from ._fingerprints import fingerprint

            fitted_adapter_fingerprint = fingerprint(adapter_to_dict(adapter))
            stored_adapter_fingerprint = self.metadata.get("observation_adapter_fingerprint")
            if (
                stored_adapter_fingerprint
                and stored_adapter_fingerprint != fitted_adapter_fingerprint
                and not self.metadata.get("compatibility_override", False)
            ):
                raise ValueError(
                    "Attached model's fitted observation adapter does not match result metadata."
                )
            self.metadata.setdefault(
                "observation_adapter_fingerprint", fitted_adapter_fingerprint
            )

    @property
    def is_batch(self) -> bool:
        return self.draws.ndim == 3

    def summary(self, point: str = "mean") -> Any:
        """Return conventional posterior summaries in user parameter units."""

        point_key = point.lower()
        if point_key == "map":
            raise NotImplementedError(
                "MAP is not exposed: the beta does not use a sample-mode approximation as a MAP estimate."
            )
        if point_key not in {"mean", "median"}:
            raise ValueError("point must be 'mean', 'median', or 'MAP'.")
        arrays = self.draws[None, ...] if self.draws.ndim == 2 else self.draws
        rows: list[dict[str, Any]] = []
        for dataset, data in enumerate(arrays):
            global_warning = self._warning_for_dataset(dataset)
            for j, parameter in enumerate(self.parameters):
                column = data[:, j]
                posterior_mean = float(np.mean(column))
                median = float(np.median(column))
                row: dict[str, Any] = {
                    "parameter": parameter.name,
                    "estimate": posterior_mean if point_key == "mean" else median,
                    "posterior_mean": posterior_mean,
                    "posterior_sd": float(np.std(column, ddof=1)) if len(column) > 1 else 0.0,
                    "median": median,
                    "q025": float(np.quantile(column, 0.025)),
                    "q975": float(np.quantile(column, 0.975)),
                    # MDN draws are independently generated conditional on the
                    # fitted approximation. This ESS is about draw dependence,
                    # not approximation quality.
                    "ess": int(len(column)),
                    "warning": global_warning,
                }
                if self.is_batch:
                    row = {"dataset": dataset, **row}
                rows.append(row)
        return _pandas().DataFrame(rows)

    def covariance(self) -> np.ndarray:
        """Return sample covariance, requiring at least two draws per dataset."""

        arrays = self.draws[None, ...] if self.draws.ndim == 2 else self.draws
        if arrays.shape[1] < 2:
            raise ValueError("Covariance requires at least two posterior draws per dataset.")
        values = []
        for data in arrays:
            cov = np.asarray(np.cov(data, rowvar=False, ddof=1), dtype=float)
            values.append(cov.reshape(len(self.parameters), len(self.parameters)))
        return values[0] if not self.is_batch else np.stack(values)

    def correlation(self) -> np.ndarray:
        """Return sample correlation, requiring at least two draws per dataset."""

        covariance = self.covariance()
        arrays = covariance[None, ...] if covariance.ndim == 2 else covariance
        results: list[np.ndarray] = []
        for cov in arrays:
            scale = np.sqrt(np.maximum(np.diag(cov), 0.0))
            denom = scale[:, None] * scale[None, :]
            corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom > 0)
            np.fill_diagonal(corr, 1.0)
            results.append(corr)
        return results[0] if covariance.ndim == 2 else np.stack(results)

    def to_dataframe(self) -> Any:
        arrays = self.draws[None, ...] if self.draws.ndim == 2 else self.draws
        rows: list[dict[str, Any]] = []
        names = [parameter.name for parameter in self.parameters]
        for dataset, data in enumerate(arrays):
            for draw_index, draw in enumerate(data):
                row: dict[str, Any] = {"draw": draw_index}
                if self.is_batch:
                    row["dataset"] = dataset
                row.update({name: float(value) for name, value in zip(names, draw, strict=True)})
                rows.append(row)
        return _pandas().DataFrame(rows)

    def diagnostics(self) -> Any:
        rows = [dict(row) for row in self.support_diagnostics]
        if not rows:
            rows = [{"dataset": 0, "support_warning": False, "warning": ""}]
        return _pandas().DataFrame(rows)

    def predictive_check(
        self,
        model: "StructuralModel | None" = None,
        *,
        replications: int = 200,
        seed: int = 123,
        dataset: int = 0,
    ) -> PredictiveCheckResult:
        """Simulate replications and compare coordinates using mid-distribution tails."""

        active_model = model or self._model
        if active_model is None:
            raise RuntimeError(
                "Posterior predictive checks require an executable StructuralModel; "
                "pass model=... or load the estimator with model=...."
            )
        compatibility_override = bool(self.metadata.get("compatibility_override", False))
        stored_contract = self.metadata.get("model_contract_fingerprint")
        if stored_contract and not compatibility_override:
            from ._fingerprints import fingerprint

            package_version = str(self.metadata.get("package_version", ""))
            pieces = package_version.split(".")
            compatibility = ".".join(pieces[:2]) if len(pieces) >= 2 else package_version
            current_contract = fingerprint(
                {
                    "package_compatibility": compatibility,
                    "model_contract": active_model.specification(include_adapter_state=False),
                }
            )
            if current_contract != stored_contract:
                raise ValueError(
                    "Predictive-check model does not match the estimator's stored model contract."
                )
        stored_adapter_fingerprint = self.metadata.get("observation_adapter_fingerprint")
        if not compatibility_override:
            if not stored_adapter_fingerprint:
                raise ValueError(
                    "Predictive-check compatibility cannot be verified because this result lacks "
                    "a fitted observation-adapter fingerprint. Re-run inference with the current "
                    "estimator before requesting a predictive check."
                )
            from ._fingerprints import fingerprint

            current_adapter_fingerprint = fingerprint(
                adapter_to_dict(active_model.observation_adapter)
            )
            if current_adapter_fingerprint != stored_adapter_fingerprint:
                raise ValueError(
                    "Predictive-check model's fitted observation adapter does not match the "
                    "adapter used for inference."
                )
        if replications < 1:
            raise ValueError("replications must be positive.")
        arrays = self.draws[None, ...] if self.draws.ndim == 2 else self.draws
        reps_obs = self.representation.reshape(1, -1) if self.representation.ndim == 1 else self.representation
        if dataset < 0 or dataset >= len(arrays):
            raise IndexError("dataset index is outside the inference batch.")
        rng = np.random.default_rng(seed)
        chosen = rng.integers(0, arrays.shape[1], size=replications)
        replicated: list[np.ndarray] = []
        expected_width = int(reps_obs.shape[1])
        for index in chosen:
            raw = active_model.simulate_one(arrays[dataset, index], rng)
            replicated.append(
                _transform_replication(
                    active_model.observation_adapter,
                    raw,
                    expected_width=expected_width,
                )
            )
        matrix = np.vstack(replicated)
        if matrix.shape != (replications, expected_width):
            raise ValueError(
                "Posterior-predictive representations do not have the exact stored width."
            )
        observed = reps_obs[dataset]
        rows: list[dict[str, Any]] = []
        for j, value in enumerate(observed):
            less_fraction = float(np.mean(matrix[:, j] < value))
            equal_fraction = float(np.mean(matrix[:, j] == value))
            # The mid-distribution convention assigns half of exact equality
            # mass to each tail, so discrete statistics are not spuriously
            # extreme merely because they have ties.
            lower_mid_fraction = less_fraction + 0.5 * equal_fraction
            rows.append(
                {
                    "representation": j,
                    "observed": float(value),
                    "replicated_mean": float(np.mean(matrix[:, j])),
                    "replicated_sd": float(np.std(matrix[:, j], ddof=1)) if replications > 1 else 0.0,
                    "replicated_q025": float(np.quantile(matrix[:, j], 0.025)),
                    "replicated_q975": float(np.quantile(matrix[:, j], 0.975)),
                    "replicated_less_fraction": less_fraction,
                    "replicated_equal_fraction": equal_fraction,
                    "two_sided_tail_fraction": float(
                        min(1.0, 2.0 * min(lower_mid_fraction, 1.0 - lower_mid_fraction))
                    ),
                }
            )
        return PredictiveCheckResult(
            table=_pandas().DataFrame(rows),
            metadata={
                "replications": replications,
                "seed": seed,
                "dataset": dataset,
                "tie_handling": "mid-distribution: half of exact equality mass in each tail",
            },
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Save draws and metadata as a checksummed, pickle-free artifact."""

        manifest = {
            "artifact_kind": "structnpe_inference_result",
            "package_version": self.metadata.get("package_version"),
            "model_fingerprint": self.metadata.get("model_fingerprint"),
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "metadata": self.metadata,
            "support_diagnostics": list(self.support_diagnostics),
        }

        def writer(root: Path) -> list[str]:
            np.savez_compressed(root / "result.npz", draws=self.draws, representation=self.representation)
            return ["result.npz"]

        return write_directory_artifact(path, manifest=manifest, write_payloads=writer, overwrite=overwrite)

    def _warning_for_dataset(self, dataset: int) -> str:
        if dataset < len(self.support_diagnostics):
            return str(self.support_diagnostics[dataset].get("warning") or "")
        return ""


def load_result(path: str | Path) -> InferenceResult:
    root, manifest = read_directory_header(path, expected_kind="structnpe_inference_result")
    files = dict(manifest.get("files") or {})
    if set(files) != {"result.npz"}:
        raise ArtifactError("Inference-result artifact must contain exactly result.npz.")
    arrays = read_npz_arrays(
        root / "result.npz",
        expected_names={"draws", "representation"},
        expected_sha256=files["result.npz"],
    )
    draws = np.asarray(arrays["draws"], dtype=float)
    representation = np.asarray(arrays["representation"], dtype=float)
    parameters = tuple(ParameterSpec.from_dict(item) for item in manifest["parameters"])
    return InferenceResult(
        draws=draws,
        parameters=parameters,
        metadata=dict(manifest.get("metadata") or {}),
        representation=representation,
        support_diagnostics=tuple(dict(row) for row in manifest.get("support_diagnostics") or []),
    )


def _transform_replication(
    adapter: Any,
    observation: Any,
    *,
    expected_width: int,
) -> np.ndarray:
    """Apply an adapter without implicitly fitting or mutating its base state."""

    transformed = np.asarray(adapter.transform(observation), dtype=np.float64)
    if transformed.ndim != 1:
        raise ValueError(
            "Posterior-predictive observation adapter must return one-dimensional output; "
            f"got shape {transformed.shape}."
        )
    if transformed.size != expected_width:
        raise ValueError(
            "Posterior-predictive observation representation has dimension "
            f"{transformed.size}; expected exactly {expected_width}."
        )
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Posterior-predictive observation representation contains non-finite values.")
    return transformed


__all__ = ["InferenceResult", "PredictiveCheckResult", "load_result"]
