from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pytest
import structnpe.estimator as estimator_module

from structnpe import (
    ArrayAdapter,
    ArtifactError,
    ParameterSpec,
    StructuralModel,
    TrainedEstimator,
    load_estimator,
)
from structnpe._fingerprints import fingerprint
from structnpe._version import __version__


def _prior(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n, 1))


def _simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([theta[0] + rng.normal()])


def _model(simulator_id: str = "tests.artifact.simulator.v1") -> StructuralModel:
    return StructuralModel(
        prior=_prior,
        simulator=_simulator,
        parameters=[ParameterSpec("theta")],
        observation_adapter=ArrayAdapter(expected_shape=(1,)),
        prior_id="tests.artifact.prior.v1",
        simulator_id=simulator_id,
    )


def _estimator() -> TrainedEstimator:
    model = _model()
    contract = model.specification(include_adapter_state=False)
    contract_fingerprint = fingerprint(
        {"package_compatibility": "0.1", "model_contract": contract}
    )
    model.observation_adapter.fit([np.array([0.0])])
    architecture = {
        "estimator_type": "mdn_diagonal_gaussian",
        "activation": "relu",
        "input_dim": 1,
        "theta_dim": 1,
        "hidden_dim": 2,
        "depth": 1,
        "components": 2,
        "log_variance_bounds": [-7.0, 5.0],
    }
    # Equal-weight components at standardized locations -1 and +1.
    weights = {
        "linear_0_weight": np.array([[1.0], [-1.0]], dtype=np.float32),
        "linear_0_bias": np.zeros(2, dtype=np.float32),
        "head_weight": np.zeros((6, 2), dtype=np.float32),
        "head_bias": np.array([0.0, 0.0, -1.0, 1.0, -2.0, -2.0], dtype=np.float32),
    }
    model_fingerprint = fingerprint(
        {
            "package_compatibility": "0.1",
            "model": model.specification(include_adapter_state=True),
            "architecture": architecture,
        }
    )
    return TrainedEstimator(
        parameters=model.parameters,
        observation_adapter=model.observation_adapter,
        architecture=architecture,
        weights=weights,
        x_mean=np.array([0.0]),
        x_scale=np.array([1.0]),
        theta_mean=np.array([0.0]),
        theta_scale=np.array([1.0]),
        support_reference=np.array([[-1.0], [0.0], [1.0]]),
        support_threshold=2.0,
        training_metadata={"package_version": __version__, "device": "cpu"},
        validation_metadata={},
        model_fingerprint=model_fingerprint,
        model_contract_fingerprint=contract_fingerprint,
        model_contract=contract,
        model=model,
    )


def test_deterministic_draws_batch_and_distribution_shift_warning() -> None:
    estimator = _estimator()
    first = estimator.infer(np.array([0.0]), draws=50, seed=22)
    second = estimator.infer(np.array([0.0]), draws=50, seed=22)
    np.testing.assert_array_equal(first.draws, second.draws)
    batch = estimator.infer([np.array([0.0]), np.array([0.5])], draws=12, seed=9, batch=True)
    assert batch.draws.shape == (2, 12, 1)
    shifted = estimator.infer(np.array([100.0]), draws=10, seed=1)
    assert bool(shifted.diagnostics().loc[0, "support_warning"])
    assert "Distribution-support warning" in shifted.summary().loc[0, "warning"]


def test_save_reload_checksums_completeness_and_model_compatibility(tmp_path: Path) -> None:
    estimator = _estimator()
    artifact = estimator.save(tmp_path / "estimator")
    loaded = load_estimator(artifact, model=_model())
    assert loaded.estimator_fingerprint == estimator.estimator_fingerprint
    expected = estimator.infer(np.array([0.2]), draws=30, seed=8).draws
    actual = loaded.infer(np.array([0.2]), draws=30, seed=8).draws
    np.testing.assert_array_equal(expected, actual)
    with pytest.raises(ArtifactError, match="Incompatible estimator"):
        load_estimator(artifact, model=_model("different.simulator"))
    with pytest.warns(RuntimeWarning, match="COMPATIBILITY CHECK BYPASSED"):
        overridden = load_estimator(
            artifact,
            model=_model("different.simulator"),
            allow_incompatible=True,
        )
    assert overridden.compatibility_warnings

    (artifact / "COMPLETE").unlink()
    with pytest.raises(ArtifactError, match="incomplete"):
        load_estimator(artifact)


def test_weight_corruption_and_wrong_observation_shape_fail(tmp_path: Path) -> None:
    artifact = _estimator().save(tmp_path / "estimator")
    with (artifact / "weights.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        load_estimator(artifact)
    with pytest.raises(ValueError, match="shape"):
        _estimator().infer(np.array([1.0, 2.0]), draws=10)


def test_self_consistent_manifest_checksum_cannot_hide_cross_field_mismatch(tmp_path: Path) -> None:
    artifact = _estimator().save(tmp_path / "estimator")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["parameters"][0]["name"] = "silently_reordered"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact / "COMPLETE").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    with pytest.raises(ArtifactError, match="parameter definitions disagree"):
        load_estimator(artifact)


def test_unavailable_cuda_falls_back_with_visible_warning() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("This test exercises CPU fallback when CUDA is unavailable.")
    result = _estimator().infer(np.array([0.0]), draws=10, device="cuda")
    assert "using NumPy CPU inference" in result.summary().loc[0, "warning"]


def test_package_mismatch_is_rejected_before_corrupt_payload_read(tmp_path: Path) -> None:
    artifact = _estimator().save(tmp_path / "estimator")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["package_version"] = "9.0.0"
    manifest["package_compatibility"] = "9.0"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact / "COMPLETE").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    (artifact / "weights.npz").write_bytes(b"not an NPZ")
    with pytest.raises(ArtifactError, match="Incompatible estimator"):
        load_estimator(artifact)


@pytest.mark.parametrize(
    ("field", "malicious_value"),
    [
        ("depth", 10**100),
        ("input_dim", 10**100),
        ("theta_dim", 10**100),
        ("hidden_dim", 10**100),
        ("components", 10**100),
    ],
)
def test_load_rejects_huge_architecture_before_weight_name_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    malicious_value: int,
) -> None:
    artifact = _estimator().save(tmp_path / "estimator")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["architecture"][field] = malicious_value
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (artifact / "COMPLETE").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )

    def forbidden_weight_shape_enumeration(_architecture: object) -> object:
        pytest.fail("_weight_shapes ran before bounded architecture validation")

    monkeypatch.setattr(
        estimator_module, "_weight_shapes", forbidden_weight_shape_enumeration
    )
    with pytest.raises(ArtifactError, match="resource limit"):
        load_estimator(artifact)


def test_architecture_support_and_extreme_input_validation() -> None:
    estimator = _estimator()
    bad_architecture = {**estimator.architecture, "activation": "tanh"}
    with pytest.raises(ArtifactError, match="activation"):
        TrainedEstimator(
            parameters=estimator.parameters,
            observation_adapter=estimator.observation_adapter,
            architecture=bad_architecture,
            weights=estimator.weights,
            x_mean=estimator.x_mean,
            x_scale=estimator.x_scale,
            theta_mean=estimator.theta_mean,
            theta_scale=estimator.theta_scale,
            support_reference=estimator.support_reference,
            support_threshold=estimator.support_threshold,
            training_metadata=estimator.training_metadata,
            validation_metadata=estimator.validation_metadata,
            model_fingerprint=estimator.model_fingerprint,
            model_contract_fingerprint=estimator.model_contract_fingerprint,
            model_contract=estimator.model_contract,
        )
    with pytest.raises(ArtifactError, match="Support-reference"):
        TrainedEstimator(
            parameters=estimator.parameters,
            observation_adapter=estimator.observation_adapter,
            architecture=estimator.architecture,
            weights=estimator.weights,
            x_mean=estimator.x_mean,
            x_scale=estimator.x_scale,
            theta_mean=estimator.theta_mean,
            theta_scale=estimator.theta_scale,
            support_reference=np.empty((0, 1)),
            support_threshold=estimator.support_threshold,
            training_metadata=estimator.training_metadata,
            validation_metadata=estimator.validation_metadata,
            model_fingerprint=estimator.model_fingerprint,
            model_contract_fingerprint=estimator.model_contract_fingerprint,
            model_contract=estimator.model_contract,
        )
    with pytest.raises(ValueError, match="numeric range"):
        estimator.infer(np.array([np.finfo(np.float64).max]), draws=2)
    with pytest.raises(ValueError, match="Inference device"):
        estimator.infer(np.array([0.0]), draws=2, device="not-a-device")
