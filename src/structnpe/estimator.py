"""High-level training, inference, compatibility, and persistence facade."""

from __future__ import annotations

import copy
import hashlib
import math
import platform
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._artifacts import (
    MAX_ARRAY_BYTES,
    MAX_ARRAY_ELEMENTS,
    MAX_NPZ_MEMBERS,
    MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES,
    ArtifactError,
    read_directory_header,
    read_npz_arrays,
    write_directory_artifact,
)
from ._fingerprints import fingerprint
from ._version import __version__
from .results import InferenceResult
from .schema import (
    ObservationAdapter,
    ParameterSpec,
    StructuralModel,
    adapter_from_dict,
    adapter_to_dict,
)


ESTIMATOR_TYPE = "mdn_diagonal_gaussian"
ESTIMATOR_FORMAT_VERSION = 1
LOG_VARIANCE_BOUNDS = (-7.0, 5.0)


@dataclass(frozen=True)
class ValidationResult:
    """Machine-readable simulation-based calibration result."""

    metrics: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]

    def to_dataframe(self) -> Any:
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError("pandas is required for validation tables.") from exc
        return pd.DataFrame([dict(row) for row in self.metrics])

    def save(self, path: str | Path) -> Path:
        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps({"metadata": self.metadata, "metrics": list(self.metrics)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target


class TrainedEstimator:
    """A fitted MDN with safe NumPy inference and artifact persistence."""

    def __init__(
        self,
        *,
        parameters: Sequence[ParameterSpec],
        observation_adapter: ObservationAdapter,
        architecture: Mapping[str, Any],
        weights: Mapping[str, np.ndarray],
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        theta_mean: np.ndarray,
        theta_scale: np.ndarray,
        support_reference: np.ndarray,
        support_threshold: float,
        training_metadata: Mapping[str, Any],
        validation_metadata: Mapping[str, Any] | None,
        model_fingerprint: str,
        model_contract_fingerprint: str,
        model_contract: Mapping[str, Any],
        model: StructuralModel | None = None,
        compatibility_warnings: Sequence[str] = (),
        load_warnings: Sequence[str] = (),
    ) -> None:
        self.parameters = tuple(parameters)
        self.observation_adapter = observation_adapter
        self.architecture = dict(architecture)
        self.weights = {str(key): np.asarray(value, dtype=np.float32) for key, value in weights.items()}
        self.x_mean = np.asarray(x_mean, dtype=np.float64)
        self.x_scale = np.asarray(x_scale, dtype=np.float64)
        self.theta_mean = np.asarray(theta_mean, dtype=np.float64)
        self.theta_scale = np.asarray(theta_scale, dtype=np.float64)
        self.support_reference = np.asarray(support_reference, dtype=np.float64)
        self.support_threshold = float(support_threshold)
        self.training_metadata = dict(training_metadata)
        self.validation_metadata = dict(validation_metadata or {})
        self.model_fingerprint = str(model_fingerprint)
        self.model_contract_fingerprint = str(model_contract_fingerprint)
        self.model_contract = dict(model_contract)
        self.compatibility_warnings = tuple(str(item) for item in compatibility_warnings)
        self.load_warnings = tuple(str(item) for item in load_warnings)
        self._model = model
        self._validate_internal_shapes()
        self.estimator_fingerprint = _estimator_fingerprint(self)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def infer(
        self,
        observed_data: Any,
        *,
        draws: int = 20_000,
        seed: int = 5678,
        batch: bool = False,
        device: str = "cpu",
    ) -> InferenceResult:
        """Draw from the fitted approximate posterior for one or more datasets."""

        if draws < 1:
            raise ValueError("draws must be positive.")
        total_start = time.perf_counter()
        preprocess_start = time.perf_counter()
        if batch:
            try:
                observations = list(observed_data)
            except TypeError as exc:
                raise ValueError("batch=True requires an iterable of observed datasets.") from exc
            representation = self.observation_adapter.transform_many(observations)
        else:
            representation = self.observation_adapter.transform_one(observed_data).reshape(1, -1)
        preprocess_seconds = time.perf_counter() - preprocess_start
        support = self._support_diagnostics(representation)
        resolved_device, device_warning = _resolve_inference_device(device)
        draw_start = time.perf_counter()
        raw = self._forward(representation, resolved_device)
        transformed_draws = self._sample_raw(raw, draws=draws, seed=seed)
        posterior_draws = _inverse_parameters(transformed_draws, self.parameters)
        draw_seconds = time.perf_counter() - draw_start
        if device_warning:
            support = tuple(
                {**row, "warning": _join_warnings(str(row.get("warning") or ""), device_warning)}
                for row in support
            )
        if self.compatibility_warnings:
            override_warning = " ".join(self.compatibility_warnings)
            support = tuple(
                {**row, "warning": _join_warnings(str(row.get("warning") or ""), override_warning)}
                for row in support
            )
        if self.load_warnings:
            load_warning = " ".join(self.load_warnings)
            support = tuple(
                {**row, "warning": _join_warnings(str(row.get("warning") or ""), load_warning)}
                for row in support
            )
        output = posterior_draws if batch else posterior_draws[0]
        rep_output = representation if batch else representation[0]
        metadata = {
            "package_version": __version__,
            "estimator_type": ESTIMATOR_TYPE,
            "estimator_fingerprint": self.estimator_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "model_contract_fingerprint": self.model_contract_fingerprint,
            "observation_adapter_fingerprint": fingerprint(
                adapter_to_dict(self.observation_adapter)
            ),
            "parameter_order": list(self.parameter_names),
            "representation_dim": int(representation.shape[1]),
            "draws": int(draws),
            "datasets": int(len(representation)),
            "seed": int(seed),
            "device": resolved_device,
            "preprocessing_seconds": preprocess_seconds,
            "posterior_draw_seconds": draw_seconds,
            "inference_seconds": time.perf_counter() - total_start,
            "compatibility_override": bool(self.compatibility_warnings),
            "artifact_load_warnings": list(self.load_warnings),
            "credible_interval_note": "Intervals are Bayesian credible intervals conditional on the fitted simulator, prior, and representation.",
        }
        return InferenceResult(
            draws=output,
            parameters=self.parameters,
            metadata=metadata,
            representation=rep_output,
            support_diagnostics=support,
            _model=self._model,
        )

    def validate(
        self,
        model: StructuralModel | None = None,
        *,
        simulations: int = 100,
        draws: int = 500,
        seed: int = 2468,
        coverages: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
        output_path: str | Path | None = None,
    ) -> ValidationResult:
        """Run bounded SBC-style checks under the maintained prior and simulator."""

        active_model = model or self._model
        if active_model is None:
            raise RuntimeError("Validation requires an executable StructuralModel.")
        _check_model_contract(active_model, self, allow_incompatible=False)
        if simulations < 3 or draws < 20:
            raise ValueError("Validation requires at least 3 simulations and 20 posterior draws.")
        levels = tuple(float(level) for level in coverages)
        if not levels or any(level <= 0 or level >= 1 for level in levels):
            raise ValueError("Coverage levels must lie strictly between zero and one.")
        seeds = np.random.SeedSequence(int(seed)).spawn(4)
        prior_rng = np.random.default_rng(seeds[0])
        sim_rng = np.random.default_rng(seeds[1])
        draw_rng = np.random.default_rng(seeds[2])
        rank_rng = np.random.default_rng(seeds[3])
        truth = active_model.sample_prior(simulations, prior_rng)
        raw = active_model.simulate_batch(truth, sim_rng)
        estimates = np.empty_like(truth)
        rank_fractions = np.empty_like(truth)
        covered = {level: np.empty_like(truth, dtype=bool) for level in levels}
        for i, observation in enumerate(raw):
            result = self.infer(
                observation,
                draws=draws,
                seed=int(draw_rng.integers(0, np.iinfo(np.uint32).max)),
            )
            sample = np.asarray(result.draws, dtype=float)
            estimates[i] = np.mean(sample, axis=0)
            less = np.sum(sample < truth[i][None, :], axis=0)
            equal = np.sum(sample == truth[i][None, :], axis=0)
            for j in range(sample.shape[1]):
                randomized_rank = int(
                    rank_rng.integers(int(less[j]), int(less[j] + equal[j]) + 1)
                )
                rank_fractions[i, j] = randomized_rank / float(draws)
            for level in levels:
                alpha = (1.0 - level) / 2.0
                lo, hi = np.quantile(sample, [alpha, 1.0 - alpha], axis=0)
                covered[level][i] = (truth[i] >= lo) & (truth[i] <= hi)
        rows: list[dict[str, Any]] = []
        rank_edges = np.linspace(0.0, 1.0, 11)
        rank_histograms: list[dict[str, Any]] = []
        for j, parameter in enumerate(self.parameters):
            error = estimates[:, j] - truth[:, j]
            rank_counts, _ = np.histogram(rank_fractions[:, j], bins=rank_edges)
            rank_histograms.append(
                {
                    "parameter": parameter.name,
                    "bin_edges": rank_edges.tolist(),
                    "counts": rank_counts.astype(int).tolist(),
                    "n": int(simulations),
                }
            )
            base = {
                "parameter": parameter.name,
                "region": "all",
                "n": simulations,
                "bias": float(np.mean(error)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mean_rank_fraction": float(np.mean(rank_fractions[:, j])),
                "rank_histogram_l1": float(
                    np.sum(np.abs(rank_counts / float(simulations) - 0.1))
                ),
            }
            for level in levels:
                base[f"coverage_{int(round(level * 100))}"] = float(np.mean(covered[level][:, j]))
            rows.append(base)
            cuts = np.quantile(truth[:, j], [1.0 / 3.0, 2.0 / 3.0])
            regions = (
                ("lower", truth[:, j] <= cuts[0]),
                ("middle", (truth[:, j] > cuts[0]) & (truth[:, j] <= cuts[1])),
                ("upper", truth[:, j] > cuts[1]),
            )
            for region, mask in regions:
                if int(mask.sum()) < 3:
                    continue
                region_error = error[mask]
                entry: dict[str, Any] = {
                    "parameter": parameter.name,
                    "region": region,
                    "n": int(mask.sum()),
                    "bias": float(np.mean(region_error)),
                    "rmse": float(np.sqrt(np.mean(region_error**2))),
                    "mean_rank_fraction": float(np.mean(rank_fractions[mask, j])),
                }
                for level in levels:
                    entry[f"coverage_{int(round(level * 100))}"] = float(np.mean(covered[level][mask, j]))
                rows.append(entry)
        result = ValidationResult(
            metrics=tuple(rows),
            metadata={
                "package_version": __version__,
                "model_fingerprint": self.model_fingerprint,
                "simulations": int(simulations),
                "draws_per_simulation": int(draws),
                "seed": int(seed),
                "coverage_levels": list(levels),
                "rank_histograms": rank_histograms,
                "rank_tie_rule": (
                    "The truth rank among equal posterior draws is randomized uniformly over "
                    "its admissible discrete positions; ranks are divided by draws_per_simulation."
                ),
                "note": "SBC is conditional on the maintained prior, simulator, representation, and fitted posterior approximation.",
            },
        )
        self.validation_metadata = {
            "latest_sbc": {
                "metadata": dict(result.metadata),
                "metrics": [dict(row) for row in result.metrics],
            }
        }
        if output_path is not None:
            result.save(output_path)
        return result

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        """Write a checksum-verified, pickle-free estimator directory."""

        manifest = {
            "artifact_kind": "structnpe_estimator",
            "estimator_format_version": ESTIMATOR_FORMAT_VERSION,
            "package_version": __version__,
            "package_compatibility": _compatibility_line(__version__),
            "estimator_type": ESTIMATOR_TYPE,
            "architecture": self.architecture,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "observation_adapter": adapter_to_dict(self.observation_adapter),
            "model_fingerprint": self.model_fingerprint,
            "model_contract_fingerprint": self.model_contract_fingerprint,
            "model_contract": self.model_contract,
            "estimator_fingerprint": self.estimator_fingerprint,
            "support_threshold": self.support_threshold,
            "training_metadata": self.training_metadata,
            "validation_metadata": self.validation_metadata,
            "compatibility_warnings": list(self.compatibility_warnings),
            "load_warnings": list(self.load_warnings),
        }

        def writer(root: Path) -> list[str]:
            np.savez_compressed(root / "weights.npz", **self.weights)
            np.savez_compressed(
                root / "state.npz",
                x_mean=self.x_mean,
                x_scale=self.x_scale,
                theta_mean=self.theta_mean,
                theta_scale=self.theta_scale,
                support_reference=self.support_reference,
            )
            return ["weights.npz", "state.npz"]

        return write_directory_artifact(path, manifest=manifest, write_payloads=writer, overwrite=overwrite)

    def _forward(self, representation: np.ndarray, device: str) -> np.ndarray:
        standardized_64 = (representation - self.x_mean[None, :]) / self.x_scale[None, :]
        if not np.all(np.isfinite(standardized_64)) or np.any(
            np.abs(standardized_64) > np.finfo(np.float32).max
        ):
            raise ValueError(
                "Standardized observation is non-finite or outside the numeric range supported by the MDN."
            )
        standardized = standardized_64.astype(np.float32)
        if device == "cpu":
            with np.errstate(over="ignore", invalid="ignore"):
                hidden = standardized
                for layer in range(int(self.architecture["depth"])):
                    hidden = (
                        hidden @ self.weights[f"linear_{layer}_weight"].T
                        + self.weights[f"linear_{layer}_bias"]
                    )
                    hidden = np.maximum(hidden, 0.0)
                output = hidden @ self.weights["head_weight"].T + self.weights["head_bias"]
            if not np.all(np.isfinite(output)):
                raise RuntimeError(
                    "MDN forward propagation produced non-finite values for this observation."
                )
            return output
        torch, model = _torch_model(self.architecture, device=device)
        _load_numpy_weights_into_torch(model, self.weights)
        torch_device = torch.device(device)
        model.to(torch_device).eval()
        with torch.no_grad():
            output = model(torch.as_tensor(standardized, device=torch_device)).cpu().numpy()
        if not np.all(np.isfinite(output)):
            raise RuntimeError("MDN forward propagation produced non-finite values for this observation.")
        return output

    def _sample_raw(self, raw: np.ndarray, *, draws: int, seed: int) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        batch = raw.shape[0]
        components = int(self.architecture["components"])
        dimension = len(self.parameters)
        expected_width = components * (1 + 2 * dimension)
        if raw.ndim != 2 or raw.shape[1] != expected_width or not np.all(np.isfinite(raw)):
            raise RuntimeError("MDN output has an invalid shape or contains non-finite values.")
        logits = raw[:, :components]
        offset = components
        means = raw[:, offset : offset + components * dimension].reshape(batch, components, dimension)
        offset += components * dimension
        lower_logvar, upper_logvar = (
            float(value) for value in self.architecture["log_variance_bounds"]
        )
        logvar = np.clip(
            raw[:, offset:].reshape(batch, components, dimension),
            lower_logvar,
            upper_logvar,
        )
        weights = _softmax_rows(logits)
        if not np.all(np.isfinite(weights)):
            raise RuntimeError("MDN mixture probabilities are non-finite.")
        rng = np.random.default_rng(seed)
        sampled = np.empty((batch, draws, dimension), dtype=np.float64)
        for i in range(batch):
            labels = rng.choice(components, size=draws, p=weights[i])
            eps = rng.normal(size=(draws, dimension))
            standardized = means[i, labels] + eps * np.exp(0.5 * logvar[i, labels])
            sampled[i] = standardized * self.theta_scale[None, :] + self.theta_mean[None, :]
        return sampled

    def _support_diagnostics(self, representation: np.ndarray) -> tuple[dict[str, Any], ...]:
        standardized = (representation - self.x_mean[None, :]) / self.x_scale[None, :]
        if not np.all(np.isfinite(standardized)) or np.any(
            np.abs(standardized) > np.finfo(np.float32).max
        ):
            raise ValueError(
                "Standardized observation is non-finite or outside the numeric range supported by the MDN."
            )
        distances = _nearest_distances(standardized, self.support_reference)
        rows: list[dict[str, Any]] = []
        for dataset, distance in enumerate(distances):
            warning = ""
            flagged = bool(distance > self.support_threshold)
            if flagged:
                warning = (
                    "Distribution-support warning: the observed representation is farther from the "
                    "stored training reference than the fixed validation-calibrated threshold. "
                    "This is not a hypothesis test."
                )
            rows.append(
                {
                    "dataset": dataset,
                    "nearest_standardized_distance": float(distance),
                    "support_threshold": self.support_threshold,
                    "distance_to_threshold": float(distance / self.support_threshold),
                    "support_warning": flagged,
                    "warning": warning,
                }
            )
        return tuple(rows)

    def _validate_internal_shapes(self) -> None:
        input_dim, theta_dim, _, _, _ = _validated_architecture(self.architecture)
        if not self.parameters:
            raise ArtifactError("Estimator has no parameter definitions.")
        if theta_dim != len(self.parameters):
            raise ArtifactError("Estimator architecture dimensions are inconsistent.")
        if self.observation_adapter.output_dim != input_dim:
            raise ArtifactError("Observation-adapter output dimension does not match the estimator.")
        if self.x_mean.shape != (input_dim,) or self.x_scale.shape != (input_dim,):
            raise ArtifactError("Observation normalization arrays have incompatible dimensions.")
        if self.theta_mean.shape != (theta_dim,) or self.theta_scale.shape != (theta_dim,):
            raise ArtifactError("Parameter normalization arrays have incompatible dimensions.")
        if (
            self.support_reference.ndim != 2
            or self.support_reference.shape[0] < 1
            or self.support_reference.shape[1] != input_dim
        ):
            raise ArtifactError("Support-reference array has incompatible dimensions.")
        if not np.isfinite(self.support_threshold) or self.support_threshold <= 0:
            raise ArtifactError("Support threshold must be finite and positive.")
        expected = _weight_shapes(self.architecture)
        if set(self.weights) != set(expected):
            raise ArtifactError(
                f"Estimator weights differ from architecture; missing={sorted(set(expected)-set(self.weights))}, "
                f"extra={sorted(set(self.weights)-set(expected))}."
            )
        for name, shape in expected.items():
            if self.weights[name].shape != shape or not np.all(np.isfinite(self.weights[name])):
                raise ArtifactError(f"Weight {name!r} has invalid shape or non-finite values.")
        for name, array in (
            ("x_mean", self.x_mean),
            ("x_scale", self.x_scale),
            ("theta_mean", self.theta_mean),
            ("theta_scale", self.theta_scale),
            ("support_reference", self.support_reference),
        ):
            if not np.all(np.isfinite(array)):
                raise ArtifactError(f"State array {name!r} contains non-finite values.")
        if np.any(self.x_scale <= 0) or np.any(self.theta_scale <= 0):
            raise ArtifactError("Normalization scales must be positive.")


def fit(
    model: StructuralModel,
    *,
    simulations: int = 200_000,
    seed: int = 1234,
    validation_fraction: float = 0.1,
    hidden_dim: int = 64,
    depth: int = 2,
    components: int = 5,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    patience: int = 15,
    device: str = "cpu",
    output_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
    progress: bool | Callable[[dict[str, Any]], None] = True,
    support_reference_size: int = 2048,
) -> TrainedEstimator:
    """Simulate, train, and return the primary public-beta MDN estimator."""

    if not isinstance(model, StructuralModel):
        raise TypeError("model must be a StructuralModel.")
    if simulations < 10:
        raise ValueError("simulations must be at least 10.")
    if not 0.01 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must lie in [0.01, 0.5).")
    if hidden_dim < 1 or depth < 1 or components < 1:
        raise ValueError("hidden_dim, depth, and components must be positive.")
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size, and patience must be positive.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative.")
    if support_reference_size < 2:
        raise ValueError("support_reference_size must be at least two.")

    training_start = time.perf_counter()
    created = datetime.now(timezone.utc).isoformat()
    seed_children = np.random.SeedSequence(int(seed)).spawn(4)
    simulation_seed = int(seed_children[0].generate_state(1, dtype=np.uint32)[0])
    split_seed = int(seed_children[1].generate_state(1, dtype=np.uint32)[0])
    training_seed = int(seed_children[2].generate_state(1, dtype=np.uint32)[0])
    support_seed = int(seed_children[3].generate_state(1, dtype=np.uint32)[0])
    model_contract = model.specification(include_adapter_state=False)
    model_contract_fingerprint = fingerprint(
        {"package_compatibility": _compatibility_line(__version__), "model_contract": model_contract}
    )

    simulation_start = time.perf_counter()
    simulation_rng = np.random.default_rng(simulation_seed)
    theta = model.sample_prior(simulations, simulation_rng)
    raw_observations = model.simulate_batch(theta, simulation_rng)
    split_rng = np.random.default_rng(split_seed)
    order = split_rng.permutation(simulations)
    n_valid = max(1, int(round(simulations * validation_fraction)))
    valid_index, train_index = order[:n_valid], order[n_valid:]
    if len(train_index) < 2:
        raise ValueError("Training split is too small.")
    adapter_fit_observations = [raw_observations[int(index)] for index in train_index]
    model.observation_adapter.fit(adapter_fit_observations)
    representation = model.observation_adapter.transform_many(raw_observations)
    simulation_seconds = time.perf_counter() - simulation_start
    if representation.shape[0] != simulations:
        raise RuntimeError("Observation adapter did not return one representation per simulation.")
    transformed_theta = model.transform_parameters(theta)
    if not np.all(np.isfinite(transformed_theta)):
        raise ValueError("Transformed prior draws contain non-finite values.")

    x_train, x_valid = representation[train_index], representation[valid_index]
    theta_train, theta_valid = transformed_theta[train_index], transformed_theta[valid_index]
    x_mean, x_scale = _standardization(x_train)
    theta_mean, theta_scale = _standardization(theta_train)
    x_train_std = ((x_train - x_mean) / x_scale).astype(np.float32)
    x_valid_std = ((x_valid - x_mean) / x_scale).astype(np.float32)
    theta_train_std = ((theta_train - theta_mean) / theta_scale).astype(np.float32)
    theta_valid_std = ((theta_valid - theta_mean) / theta_scale).astype(np.float32)
    architecture = {
        "estimator_type": ESTIMATOR_TYPE,
        "activation": "relu",
        "input_dim": int(x_train.shape[1]),
        "theta_dim": int(theta.shape[1]),
        "hidden_dim": int(hidden_dim),
        "depth": int(depth),
        "components": int(components),
        "log_variance_bounds": list(LOG_VARIANCE_BOUNDS),
    }
    full_spec = {
        "package_compatibility": _compatibility_line(__version__),
        "model": model.specification(include_adapter_state=True),
        "architecture": architecture,
    }
    model_fingerprint = fingerprint(full_spec)

    resolved_device, device_warning = _resolve_training_device(device)
    torch, torch_model = _torch_model(architecture, device=resolved_device, seed=training_seed)
    if resume_from is not None:
        resumed = load_estimator(resume_from, model=model, allow_incompatible=False)
        if resumed.architecture != architecture:
            raise ValueError("Resume checkpoint architecture is incompatible with requested training configuration.")
        normalization_pairs = (
            ("x_mean", resumed.x_mean, x_mean),
            ("x_scale", resumed.x_scale, x_scale),
            ("theta_mean", resumed.theta_mean, theta_mean),
            ("theta_scale", resumed.theta_scale, theta_scale),
        )
        for name, previous, current in normalization_pairs:
            if not np.array_equal(previous, current):
                raise ValueError(
                    "Resume checkpoint normalization is incompatible with the newly simulated bank "
                    f"({name} differs). Reuse the same seed, simulation budget, split, and model, or start fresh."
                )
        _load_numpy_weights_into_torch(torch_model, resumed.weights)
    weights, history, best_epoch, best_valid = _train_torch_mdn(
        torch,
        torch_model,
        x_train_std,
        theta_train_std,
        x_valid_std,
        theta_valid_std,
        architecture=architecture,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        seed=training_seed,
        device=resolved_device,
        progress=progress,
    )

    reference_rng = np.random.default_rng(support_seed)
    reference_count = min(support_reference_size, len(x_train_std))
    reference_index = reference_rng.choice(len(x_train_std), size=reference_count, replace=False)
    support_reference = np.asarray(x_train_std[reference_index], dtype=np.float64)
    calibration_distances = _nearest_distances(np.asarray(x_valid_std, dtype=float), support_reference)
    support_threshold = max(float(np.quantile(calibration_distances, 0.99)), 1.0e-8)
    checkpoint_path = str(Path(output_dir) / "best_estimator") if output_dir is not None else None
    metadata: dict[str, Any] = {
        "package_version": __version__,
        "estimator_type": ESTIMATOR_TYPE,
        "architecture": architecture,
        "seed": int(seed),
        "training_seed": training_seed,
        "simulation_seed": simulation_seed,
        "seed_scheme": "numpy.SeedSequence(seed).spawn(4)",
        "number_of_simulations": int(simulations),
        "prior_fingerprint": fingerprint(model_contract["prior"]),
        "simulator_fingerprint": fingerprint(model_contract["simulator"]),
        "observation_adapter_fingerprint": fingerprint(adapter_to_dict(model.observation_adapter)),
        "parameter_order": list(model.parameter_names),
        "parameter_transforms": [parameter.resolved_transform for parameter in model.parameters],
        "training_validation_split": {
            "training": int(len(train_index)),
            "validation": int(len(valid_index)),
            "validation_fraction": float(validation_fraction),
            "split_seed": split_seed,
        },
        "observation_adapter_fit": {
            "sample": "complete training split only",
            "count": int(len(train_index)),
            "split_seed": split_seed,
        },
        "simulation_seconds": simulation_seconds,
        "training_seconds": time.perf_counter() - training_start - simulation_seconds,
        "total_fit_seconds": time.perf_counter() - training_start,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "python": platform.python_version(),
        },
        "requested_device": device,
        "device": resolved_device,
        "device_warning": device_warning,
        "best_checkpoint": checkpoint_path,
        "best_epoch": best_epoch,
        "best_validation_nll": best_valid,
        "validation_loss_history": history,
        "creation_timestamp": created,
        "resume_from": str(resume_from) if resume_from is not None else None,
        "resume_optimizer_state": "not retained; compatible weights are resumed with a fresh AdamW optimizer",
        "accelerator_determinism": (
            "not guaranteed" if resolved_device != "cpu" else "CPU seeded path; platform/library changes may still alter floating-point results"
        ),
        "support_diagnostic": {
            "method": "validation-calibrated nearest-neighbor distance in standardized representation",
            "reference_size": int(reference_count),
            "calibration_quantile": 0.99,
            "threshold": support_threshold,
            "formal_test": False,
        },
    }
    estimator = TrainedEstimator(
        parameters=model.parameters,
        observation_adapter=model.observation_adapter,
        architecture=architecture,
        weights=weights,
        x_mean=x_mean,
        x_scale=x_scale,
        theta_mean=theta_mean,
        theta_scale=theta_scale,
        support_reference=support_reference,
        support_threshold=support_threshold,
        training_metadata=metadata,
        validation_metadata={},
        model_fingerprint=model_fingerprint,
        model_contract_fingerprint=model_contract_fingerprint,
        model_contract=model_contract,
        model=model,
    )
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        estimator.save(output / "best_estimator", overwrite=True)
    return estimator


def load_estimator(
    path: str | Path,
    *,
    model: StructuralModel | None = None,
    allow_incompatible: bool = False,
    allow_custom_adapter: bool = False,
) -> TrainedEstimator:
    """Load and verify a public beta estimator artifact without pickle.

    Built-in adapters are reconstructed from an allowlist. For custom
    adapters, pass a compatible model so its already imported class can be
    used. ``allow_custom_adapter=True`` enables artifact-directed imports and
    is appropriate only for trusted artifacts and installed adapter code.
    """

    root, manifest = read_directory_header(path, expected_kind="structnpe_estimator")
    if manifest.get("estimator_format_version") != ESTIMATOR_FORMAT_VERSION:
        raise ArtifactError("Unsupported estimator format version.")
    artifact_version = str(manifest.get("package_version", ""))
    artifact_compatibility = str(manifest.get("package_compatibility", ""))
    if artifact_compatibility != _compatibility_line(artifact_version):
        raise ArtifactError("Estimator manifest has inconsistent package version metadata.")
    runtime_compatibility = _compatibility_line(__version__)
    incompatible_reasons: list[str] = []
    if artifact_compatibility != runtime_compatibility:
        reason = (
            f"artifact package compatibility {artifact_compatibility!r} differs from "
            f"runtime {runtime_compatibility!r}."
        )
        if not allow_incompatible:
            # Reject before reading numeric payloads or importing any custom
            # adapter selected by the artifact.
            raise ArtifactError("Incompatible estimator: " + reason)
        incompatible_reasons.append(reason)
    architecture = dict(manifest.get("architecture") or {})
    _validated_architecture(architecture)
    parameter_payload = list(manifest.get("parameters") or [])
    adapter_payload = dict(manifest.get("observation_adapter") or {})
    model_contract = dict(manifest.get("model_contract") or {})
    stored_contract_fingerprint = str(manifest.get("model_contract_fingerprint", ""))
    recomputed_contract_fingerprint = fingerprint(
        {"package_compatibility": artifact_compatibility, "model_contract": model_contract}
    )
    if stored_contract_fingerprint != recomputed_contract_fingerprint:
        raise ArtifactError("Estimator model-contract fingerprint is internally inconsistent.")
    if fingerprint(parameter_payload) != fingerprint(model_contract.get("parameters")):
        raise ArtifactError("Top-level parameter definitions disagree with the model contract.")
    adapter_static = {key: value for key, value in adapter_payload.items() if key != "state"}
    if fingerprint(adapter_static) != fingerprint(model_contract.get("observation_adapter")):
        raise ArtifactError("Top-level observation adapter disagrees with the model contract.")
    fitted_model_spec = {**model_contract, "observation_adapter": adapter_payload}
    recomputed_model_fingerprint = fingerprint(
        {
            "package_compatibility": artifact_compatibility,
            "model": fitted_model_spec,
            "architecture": architecture,
        }
    )
    if str(manifest.get("model_fingerprint", "")) != recomputed_model_fingerprint:
        raise ArtifactError("Estimator model fingerprint is internally inconsistent.")
    if model is not None:
        current_contract = model.specification(include_adapter_state=False)
        current_contract_fingerprint = fingerprint(
            {
                "package_compatibility": artifact_compatibility,
                "model_contract": current_contract,
            }
        )
        if current_contract_fingerprint != stored_contract_fingerprint:
            reason = (
                "provided StructuralModel does not match the stored "
                "prior/simulator/parameter/adapter contract."
            )
            if not allow_incompatible:
                raise ArtifactError("Incompatible estimator: " + reason)
            incompatible_reasons.append(reason)
    payload_checksums = dict(manifest.get("files") or {})
    if set(payload_checksums) != {"weights.npz", "state.npz"}:
        raise ArtifactError("Estimator artifact must contain exactly weights.npz and state.npz.")
    expected_weights = set(_weight_shapes(architecture))
    required_state = {"x_mean", "x_scale", "theta_mean", "theta_scale", "support_reference"}
    weights_payload = read_npz_arrays(
        root / "weights.npz",
        expected_names=expected_weights,
        expected_sha256=payload_checksums["weights.npz"],
    )
    state_payload = read_npz_arrays(
        root / "state.npz",
        expected_names=required_state,
        expected_sha256=payload_checksums["state.npz"],
    )
    weights = {key: np.asarray(value, dtype=np.float32) for key, value in weights_payload.items()}
    state = {key: np.asarray(state_payload[key], dtype=float) for key in required_state}
    parameters = tuple(ParameterSpec.from_dict(item) for item in parameter_payload)
    if not parameters:
        raise ArtifactError("Estimator manifest has no parameter definitions.")
    trusted_adapter_class: type[ObservationAdapter] | None = None
    adapter_identity = (
        str(adapter_payload.get("module", "")),
        str(adapter_payload.get("qualname", "")),
    )
    builtin_adapter_identities = {
        ("structnpe.schema", "ArrayAdapter"),
        ("structnpe.schema", "SummaryAdapter"),
    }
    if model is not None and adapter_identity not in builtin_adapter_identities:
        supplied_adapter = adapter_to_dict(model.observation_adapter)
        supplied_adapter.pop("state", None)
        if fingerprint(supplied_adapter) == fingerprint(adapter_static):
            trusted_adapter_class = model.observation_adapter.__class__
    custom_import_warning = ""
    try:
        if adapter_identity in builtin_adapter_identities:
            adapter = adapter_from_dict(adapter_payload)
        elif trusted_adapter_class is not None:
            adapter = adapter_from_dict(adapter_payload, trusted_class=trusted_adapter_class)
        elif allow_custom_adapter:
            custom_import_warning = (
                "CUSTOM ADAPTER IMPORT ENABLED: loading this artifact imports its declared "
                "adapter module and executes that installed class's from_config/set_state methods. "
                "Use only trusted artifacts and code."
            )
            warnings.warn(custom_import_warning, RuntimeWarning, stacklevel=2)
            adapter = adapter_from_dict(adapter_payload, allow_custom=True)
        else:
            raise ValueError(
                "A custom adapter requires a compatible StructuralModel or the explicit "
                "allow_custom_adapter=True trusted-code opt-in."
            )
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"Could not safely reconstruct the observation adapter: {exc}") from exc
    stored_warnings = tuple(str(item) for item in manifest.get("compatibility_warnings") or [])
    load_warnings = tuple(str(item) for item in manifest.get("load_warnings") or [])
    if custom_import_warning:
        load_warnings = (*load_warnings, custom_import_warning)
    estimator = TrainedEstimator(
        parameters=parameters,
        observation_adapter=adapter,
        architecture=architecture,
        weights=weights,
        x_mean=state["x_mean"],
        x_scale=state["x_scale"],
        theta_mean=state["theta_mean"],
        theta_scale=state["theta_scale"],
        support_reference=state["support_reference"],
        support_threshold=float(manifest["support_threshold"]),
        training_metadata=dict(manifest.get("training_metadata") or {}),
        validation_metadata=dict(manifest.get("validation_metadata") or {}),
        model_fingerprint=recomputed_model_fingerprint,
        model_contract_fingerprint=recomputed_contract_fingerprint,
        model_contract=model_contract,
        model=None,
        compatibility_warnings=stored_warnings,
        load_warnings=load_warnings,
    )
    if estimator.estimator_fingerprint != manifest.get("estimator_fingerprint"):
        raise ArtifactError("Estimator fingerprint does not match the verified payload and metadata.")
    if model is not None:
        # Attach the serialized fitted adapter to a shallow model copy. Loading
        # must not mutate caller-owned model or adapter state.
        attached_model = copy.copy(model)
        attached_model.observation_adapter = adapter
        estimator._model = attached_model
    if incompatible_reasons:
        prominent = "COMPATIBILITY CHECK BYPASSED: " + " ".join(incompatible_reasons)
        warnings.warn(prominent, RuntimeWarning, stacklevel=2)
        estimator.compatibility_warnings = (*estimator.compatibility_warnings, prominent)
    return estimator


def _check_model_contract(
    model: StructuralModel,
    estimator: TrainedEstimator,
    *,
    allow_incompatible: bool,
) -> list[str]:
    current = model.specification(include_adapter_state=False)
    current_fingerprint = fingerprint(
        {"package_compatibility": _compatibility_line(__version__), "model_contract": current}
    )
    if current_fingerprint == estimator.model_contract_fingerprint:
        return []
    reason = "provided StructuralModel does not match the stored prior/simulator/parameter/adapter contract."
    if not allow_incompatible:
        raise ArtifactError("Incompatible estimator: " + reason)
    return [reason]


def _train_torch_mdn(
    torch: Any,
    model: Any,
    x_train: np.ndarray,
    theta_train: np.ndarray,
    x_valid: np.ndarray,
    theta_valid: np.ndarray,
    *,
    architecture: Mapping[str, Any],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    device: str,
    progress: bool | Callable[[dict[str, Any]], None],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], int, float]:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    order = np.arange(len(x_train))
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_valid = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    def nll(raw: Any, target: Any) -> Any:
        components = int(architecture["components"])
        dimension = int(architecture["theta_dim"])
        lower_logvar, upper_logvar = (
            float(value) for value in architecture["log_variance_bounds"]
        )
        logits = raw[:, :components]
        offset = components
        means = raw[:, offset : offset + components * dimension].reshape(-1, components, dimension)
        offset += components * dimension
        logvar = torch.clamp(
            raw[:, offset:].reshape(-1, components, dimension),
            lower_logvar,
            upper_logvar,
        )
        centered = target[:, None, :] - means
        component_log_probability = -0.5 * torch.sum(
            math.log(2.0 * math.pi) + logvar + centered**2 / torch.exp(logvar), dim=2
        )
        return -torch.mean(torch.logsumexp(torch.log_softmax(logits, dim=1) + component_log_probability, dim=1))

    train_x = torch.from_numpy(x_train)
    train_y = torch.from_numpy(theta_train)
    valid_x = torch.from_numpy(x_valid)
    valid_y = torch.from_numpy(theta_valid)
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        rng.shuffle(order)
        model.train()
        running_loss = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_x = train_x[indices].to(torch_device)
            batch_y = train_y[indices].to(torch_device)
            loss = nll(model(batch_x), batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        model.eval()
        with torch.no_grad():
            validation_total = 0.0
            validation_seen = 0
            for start in range(0, len(valid_x), batch_size):
                batch_valid_x = valid_x[start : start + batch_size].to(torch_device)
                batch_valid_y = valid_y[start : start + batch_size].to(torch_device)
                batch_validation_loss = nll(model(batch_valid_x), batch_valid_y)
                batch_count = len(batch_valid_x)
                validation_total += float(batch_validation_loss.detach().cpu()) * batch_count
                validation_seen += batch_count
            validation_loss = validation_total / max(validation_seen, 1)
        row = {
            "epoch": epoch,
            "train_nll": running_loss / max(seen, 1),
            "validation_nll": validation_loss,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if validation_loss < best_valid - 1.0e-6:
            best_valid = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if callable(progress):
            progress(dict(row))
        elif progress and (epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0):
            print(
                f"structnpe epoch {epoch}/{epochs}: train_nll={row['train_nll']:.4f}, "
                f"validation_nll={validation_loss:.4f}"
            )
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("MDN training did not produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    return _numpy_weights_from_torch(model), history, best_epoch, best_valid


def _torch_model(
    architecture: Mapping[str, Any],
    *,
    device: str,
    seed: int | None = None,
) -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch is required for MDN training or accelerator inference. Install structnpe[neural]."
        ) from exc
    if seed is not None:
        torch.manual_seed(int(seed))
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))
    input_dim, theta_dim, hidden_dim, depth, components = _validated_architecture(architecture)

    class MDN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            width = input_dim
            for _ in range(depth):
                layers.extend([nn.Linear(width, hidden_dim), nn.ReLU()])
                width = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(width, components * (1 + 2 * theta_dim))

        def forward(self, value: Any) -> Any:
            return self.head(self.backbone(value))

    return torch, MDN()


def _numpy_weights_from_torch(model: Any) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    linear_index = 0
    for module in model.backbone:
        if hasattr(module, "weight"):
            result[f"linear_{linear_index}_weight"] = module.weight.detach().cpu().numpy().astype(np.float32)
            result[f"linear_{linear_index}_bias"] = module.bias.detach().cpu().numpy().astype(np.float32)
            linear_index += 1
    result["head_weight"] = model.head.weight.detach().cpu().numpy().astype(np.float32)
    result["head_bias"] = model.head.bias.detach().cpu().numpy().astype(np.float32)
    return result


def _load_numpy_weights_into_torch(model: Any, weights: Mapping[str, np.ndarray]) -> None:
    import torch

    linear_index = 0
    with torch.no_grad():
        for module in model.backbone:
            if hasattr(module, "weight"):
                module.weight.copy_(torch.from_numpy(np.asarray(weights[f"linear_{linear_index}_weight"])))
                module.bias.copy_(torch.from_numpy(np.asarray(weights[f"linear_{linear_index}_bias"])))
                linear_index += 1
        model.head.weight.copy_(torch.from_numpy(np.asarray(weights["head_weight"])))
        model.head.bias.copy_(torch.from_numpy(np.asarray(weights["head_bias"])))


def _weight_shapes(architecture: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    input_dim, theta_dim, hidden_dim, depth, components = _validated_architecture(architecture)
    shapes: dict[str, tuple[int, ...]] = {}
    width = input_dim
    for layer in range(depth):
        shapes[f"linear_{layer}_weight"] = (hidden_dim, width)
        shapes[f"linear_{layer}_bias"] = (hidden_dim,)
        width = hidden_dim
    output_dim = components * (1 + 2 * theta_dim)
    shapes["head_weight"] = (output_dim, width)
    shapes["head_bias"] = (output_dim,)
    return shapes


def _validated_architecture(
    architecture: Mapping[str, Any],
) -> tuple[int, int, int, int, int]:
    if architecture.get("estimator_type") != ESTIMATOR_TYPE:
        raise ArtifactError("Estimator architecture has an unsupported estimator type.")
    if architecture.get("activation") != "relu":
        raise ArtifactError("Estimator architecture has an unsupported activation.")
    dimensions: list[int] = []
    for key in ("input_dim", "theta_dim", "hidden_dim", "depth", "components"):
        raw = architecture.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)) or int(raw) < 1:
            raise ArtifactError(f"Estimator architecture field {key!r} must be a positive integer.")
        dimensions.append(int(raw))
    try:
        bounds = tuple(float(value) for value in architecture.get("log_variance_bounds", ()))
    except (TypeError, ValueError) as exc:
        raise ArtifactError("Estimator log-variance bounds are invalid.") from exc
    if bounds != LOG_VARIANCE_BOUNDS:
        raise ArtifactError(
            f"Estimator log-variance bounds must be {list(LOG_VARIANCE_BOUNDS)} for this format."
        )
    validated = tuple(dimensions)
    _validate_architecture_resource_bounds(*validated)
    return validated  # type: ignore[return-value]


def _validate_architecture_resource_bounds(
    input_dim: int,
    theta_dim: int,
    hidden_dim: int,
    depth: int,
    components: int,
) -> None:
    """Reject manifests whose declared MDN cannot fit the artifact limits.

    This check is deliberately arithmetic-only: in particular, it must run
    before :func:`_weight_shapes` enumerates one pair of names per layer.
    The limits are the same element, byte, member, and total-uncompressed-byte
    ceilings enforced when the corresponding NPZ payload is read.
    """

    float32_bytes = np.dtype(np.float32).itemsize
    float64_bytes = np.dtype(np.float64).itemsize
    weight_members = 2 * depth + 2
    if weight_members > MAX_NPZ_MEMBERS:
        raise ArtifactError(
            "Estimator architecture exceeds the NPZ array-member resource limit."
        )

    # The state archive contains float64 normalization vectors with these
    # declared widths, so reject impossible widths before inspecting either
    # numeric payload.
    for name, dimension in (("input_dim", input_dim), ("theta_dim", theta_dim)):
        if dimension > MAX_ARRAY_ELEMENTS or dimension * float64_bytes > MAX_ARRAY_BYTES:
            raise ArtifactError(
                f"Estimator architecture field {name!r} exceeds artifact array resource limits."
            )

    output_dim = components * (1 + 2 * theta_dim)
    first_layer_elements = hidden_dim * input_dim
    hidden_layer_elements = hidden_dim * hidden_dim
    head_weight_elements = output_dim * hidden_dim
    array_elements = (
        ("first-layer weight", first_layer_elements),
        ("hidden-layer weight", hidden_layer_elements if depth > 1 else 0),
        ("hidden-layer bias", hidden_dim),
        ("output-layer weight", head_weight_elements),
        ("output-layer bias", output_dim),
    )
    for name, elements in array_elements:
        if elements > MAX_ARRAY_ELEMENTS or elements * float32_bytes > MAX_ARRAY_BYTES:
            raise ArtifactError(
                f"Estimator architecture {name} exceeds artifact array resource limits."
            )

    total_weight_elements = (
        first_layer_elements
        + (depth - 1) * hidden_layer_elements
        + depth * hidden_dim
        + head_weight_elements
        + output_dim
    )
    if total_weight_elements * float32_bytes > MAX_NPZ_TOTAL_UNCOMPRESSED_BYTES:
        raise ArtifactError(
            "Estimator architecture exceeds the NPZ total-uncompressed-byte resource limit."
        )


def _estimator_fingerprint(estimator: TrainedEstimator) -> str:
    weight_hashes: dict[str, str] = {}
    for name, value in sorted(estimator.weights.items()):
        array = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        weight_hashes[name] = digest.hexdigest()
    return fingerprint(
        {
            "model_fingerprint": estimator.model_fingerprint,
            "architecture": estimator.architecture,
            "weights": weight_hashes,
            "state": {
                "x_mean": estimator.x_mean.tolist(),
                "x_scale": estimator.x_scale.tolist(),
                "theta_mean": estimator.theta_mean.tolist(),
                "theta_scale": estimator.theta_scale.tolist(),
                "support_reference_hash": hashlib.sha256(
                    np.ascontiguousarray(estimator.support_reference).tobytes()
                ).hexdigest(),
                "support_threshold": estimator.support_threshold,
            },
        }
    )


def _standardization(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(array, dtype=np.float64)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1.0e-8, scale, 1.0)
    return mean, scale


def _nearest_distances(query: np.ndarray, reference: np.ndarray, *, chunk_size: int = 256) -> np.ndarray:
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if q.ndim != 2 or r.ndim != 2 or q.shape[1] != r.shape[1] or len(r) == 0:
        raise ValueError("Nearest-neighbor arrays must be nonempty 2D arrays with matching width.")
    result = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), chunk_size):
        block = q[start : start + chunk_size]
        squared = np.sum((block[:, None, :] - r[None, :, :]) ** 2, axis=2)
        result[start : start + len(block)] = np.sqrt(np.min(squared, axis=1))
    return result


def _inverse_parameters(values: np.ndarray, parameters: Sequence[ParameterSpec]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.stack(
        [parameter.inverse_values(array[..., j]) for j, parameter in enumerate(parameters)], axis=-1
    )
    if not np.all(np.isfinite(result)):
        raise RuntimeError(
            "Posterior inverse transformation produced non-finite draws; the fitted approximation "
            "is numerically unstable for this observation."
        )
    for j, parameter in enumerate(parameters):
        parameter.validate_values(result[..., j], strict=False)
    return result


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _resolve_training_device(requested: str) -> tuple[str, str]:
    name = str(requested).lower()
    if name == "cpu":
        return "cpu", ""
    if name != "auto" and name != "mps" and not _is_cuda_device(name):
        raise ValueError("Training device must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>'.")
    try:
        import torch
    except ModuleNotFoundError:
        if name == "auto":
            return "cpu", "Torch accelerator discovery unavailable; using CPU."
        raise RuntimeError("Torch is required for accelerator training. Install structnpe[neural].")
    if name == "auto":
        return ("cuda", "") if torch.cuda.is_available() else ("cpu", "No accelerator detected; using CPU.")
    if _is_cuda_device(name) and not torch.cuda.is_available():
        return "cpu", f"Requested device {requested!r} is unavailable; using CPU."
    if name == "mps" and not bool(getattr(torch.backends, "mps", None)):
        return "cpu", f"Requested device {requested!r} is unavailable; using CPU."
    if name == "mps" and not torch.backends.mps.is_available():
        return "cpu", f"Requested device {requested!r} is unavailable; using CPU."
    return name, ""


def _resolve_inference_device(requested: str) -> tuple[str, str]:
    name = str(requested).lower()
    if name in {"cpu", "auto"}:
        return "cpu", ""
    if name != "mps" and not _is_cuda_device(name):
        raise ValueError("Inference device must be 'auto', 'cpu', 'mps', 'cuda', or 'cuda:<index>'.")
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu", f"Requested device {requested!r} requires Torch; using NumPy CPU inference."
    if _is_cuda_device(name) and not torch.cuda.is_available():
        return "cpu", f"Requested device {requested!r} is unavailable; using NumPy CPU inference."
    if name == "mps" and (
        not bool(getattr(torch.backends, "mps", None)) or not torch.backends.mps.is_available()
    ):
        return "cpu", f"Requested device {requested!r} is unavailable; using NumPy CPU inference."
    return name, ""


def _is_cuda_device(name: str) -> bool:
    if name == "cuda":
        return True
    if not name.startswith("cuda:"):
        return False
    index = name.removeprefix("cuda:")
    return index.isdigit() and int(index) >= 0


def _compatibility_line(version: str) -> str:
    parts = str(version).split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return str(version)
    return f"{parts[0]}.{parts[1]}"


def _join_warnings(first: str, second: str) -> str:
    return " ".join(item for item in (first.strip(), second.strip()) if item)


__all__ = ["TrainedEstimator", "ValidationResult", "fit", "load_estimator"]
