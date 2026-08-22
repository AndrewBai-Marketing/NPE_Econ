from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from structnpe import (
    ArtifactError,
    ObservationAdapter,
    ParameterSpec,
    StructuralModel,
    SummaryAdapter,
    fit,
    load_estimator,
)


def _prior(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n, 1))


def _simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(theta[0], 1.0, size=5)


def _model() -> StructuralModel:
    return StructuralModel(
        prior=_prior,
        simulator=_simulator,
        parameters=[ParameterSpec("location")],
        observation_adapter=SummaryAdapter(("mean",), expected_shape=(5,)),
        prior_id="tests.integration.prior.v1",
        simulator_id="tests.integration.simulator.v1",
    )


class _CenteredMeanAdapter(ObservationAdapter):
    def __init__(self) -> None:
        self.center = 0.0
        self.output_dim = 1

    def fit(self, observations) -> "_CenteredMeanAdapter":
        if not observations:
            raise ValueError("observations required")
        self.center = float(np.mean([np.mean(np.asarray(item, dtype=float)) for item in observations]))
        return self

    def transform(self, observation) -> np.ndarray:
        return np.asarray([np.mean(np.asarray(observation, dtype=float)) - self.center])

    def get_state(self) -> dict[str, float | int]:
        return {"output_dim": 1, "center": self.center}

    def set_state(self, state) -> None:
        super().set_state(state)
        self.center = float(state["center"])


def _custom_adapter_model() -> StructuralModel:
    return StructuralModel(
        prior=_prior,
        simulator=_simulator,
        parameters=[ParameterSpec("location")],
        observation_adapter=_CenteredMeanAdapter(),
        prior_id="tests.integration.prior.v1",
        simulator_id="tests.integration.simulator.v1",
    )


@pytest.fixture(scope="module")
def tiny_estimator():
    pytest.importorskip("torch")
    return fit(
        _model(),
        simulations=64,
        seed=777,
        validation_fraction=0.2,
        hidden_dim=8,
        depth=1,
        components=5,
        epochs=1,
        batch_size=32,
        patience=1,
        device="cpu",
        progress=False,
        support_reference_size=32,
    )


def test_tiny_end_to_end_cpu_batch_and_exact_smoke(tiny_estimator) -> None:
    observed = np.array([0.2, 0.1, 0.4, -0.1, 0.3])
    result = tiny_estimator.infer(observed, draws=100, seed=13)
    assert result.draws.shape == (100, 1)
    assert np.all(np.isfinite(result.draws))
    assert list(result.summary()["parameter"]) == ["location"]
    batch = tiny_estimator.infer([observed, observed + 0.1], draws=25, seed=14, batch=True)
    assert batch.draws.shape == (2, 25, 1)
    # Broad CI-scale conjugate check only. The substantive predeclared
    # validation run is separate and much stronger.
    exact_mean = (5.0 / 6.0) * observed.mean()
    assert abs(float(result.draws.mean()) - exact_mean) < 2.0


def test_save_reload_and_predictive_check(tiny_estimator, tmp_path: Path) -> None:
    artifact = tiny_estimator.save(tmp_path / "estimator")
    loaded = load_estimator(artifact, model=_model())
    observed = np.zeros(5)
    before = tiny_estimator.infer(observed, draws=30, seed=55)
    after = loaded.infer(observed, draws=30, seed=55)
    np.testing.assert_array_equal(before.draws, after.draws)
    predictive = after.predictive_check(replications=5, seed=56)
    assert len(predictive.to_dataframe()) == 1
    assert "does not establish correct model specification" in predictive.warning


def test_validation_smoke_reports_required_coverages(tiny_estimator) -> None:
    validation = tiny_estimator.validate(_model(), simulations=3, draws=20, seed=99)
    table = validation.to_dataframe()
    assert {"coverage_50", "coverage_80", "coverage_90", "coverage_95"} <= set(table.columns)
    assert {"bias", "rmse", "mean_rank_fraction", "rank_histogram_l1"} <= set(table.columns)
    histograms = validation.metadata["rank_histograms"]
    assert len(histograms) == 1
    assert sum(histograms[0]["counts"]) == 3
    assert len(histograms[0]["bin_edges"]) == 11
    assert "latest_sbc" in tiny_estimator.validation_metadata


def test_training_seed_reproduces_weights_and_draws(tiny_estimator) -> None:
    repeated = fit(
        _model(),
        simulations=64,
        seed=777,
        validation_fraction=0.2,
        hidden_dim=8,
        depth=1,
        components=5,
        epochs=1,
        batch_size=32,
        patience=1,
        device="cpu",
        progress=False,
        support_reference_size=32,
    )
    for name in tiny_estimator.weights:
        np.testing.assert_array_equal(tiny_estimator.weights[name], repeated.weights[name])
    observed = np.zeros(5)
    np.testing.assert_array_equal(
        tiny_estimator.infer(observed, draws=20, seed=123).draws,
        repeated.infer(observed, draws=20, seed=123).draws,
    )


def test_custom_adapter_load_uses_supplied_class_without_mutating_model(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    estimator = fit(
        _custom_adapter_model(),
        simulations=32,
        seed=812,
        validation_fraction=0.2,
        hidden_dim=4,
        depth=1,
        components=2,
        epochs=1,
        batch_size=16,
        patience=1,
        progress=False,
        support_reference_size=16,
    )
    split = estimator.training_metadata["training_validation_split"]
    adapter_fit = estimator.training_metadata["observation_adapter_fit"]
    assert adapter_fit["count"] == split["training"]
    assert adapter_fit["sample"] == "complete training split only"
    artifact = estimator.save(tmp_path / "custom_estimator")
    with pytest.raises(ArtifactError, match="custom adapter"):
        load_estimator(artifact)
    supplied_model = _custom_adapter_model()
    before_state = supplied_model.observation_adapter.get_state()
    loaded = load_estimator(artifact, model=supplied_model)
    assert supplied_model.observation_adapter.get_state() == before_state
    assert loaded._model is not supplied_model
    assert loaded.observation_adapter.get_state() == estimator.observation_adapter.get_state()
