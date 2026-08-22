from __future__ import annotations

import numpy as np
import pytest

from structnpe import (
    ArrayAdapter,
    InferenceResult,
    ObservationAdapter,
    ParameterSpec,
    StructuralModel,
)


def _prior(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.zeros((n, 1))


def _constant_simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([1.0])


def _wide_simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.array([1.0, 2.0])


class _LooseAdapter(ObservationAdapter):
    """Test adapter whose raw transform does not enforce its fitted width."""

    def transform(self, observation: object) -> np.ndarray:
        return np.asarray(observation, dtype=float).reshape(-1)


def _predictive_model(
    *,
    adapter: ObservationAdapter | None = None,
    simulator: object = _constant_simulator,
) -> StructuralModel:
    return StructuralModel(
        prior=_prior,
        simulator=simulator,
        parameters=[ParameterSpec("alpha")],
        observation_adapter=adapter or ArrayAdapter(),
        prior_id="tests.results.prior.v1",
        simulator_id="tests.results.simulator.v1",
    )


def _result(batch: bool = False) -> InferenceResult:
    draws = np.array([[0.0, 2.0], [1.0, 4.0], [2.0, 6.0], [3.0, 8.0]])
    if batch:
        draws = np.stack([draws, draws + 1.0])
    return InferenceResult(
        draws=draws,
        parameters=(ParameterSpec("alpha"), ParameterSpec("beta", unit="utils")),
        metadata={"package_version": "0.1.0b1", "model_fingerprint": "abc"},
        representation=np.array([[0.0], [1.0]]) if batch else np.array([0.0]),
        support_diagnostics=tuple(
            {
                "dataset": index,
                "support_warning": index == 1,
                "warning": "outside support" if index == 1 else "",
            }
            for index in range(2 if batch else 1)
        ),
    )


def test_summary_has_conventional_bayesian_columns_and_point_choices() -> None:
    result = _result()
    table = result.summary()
    assert list(table.columns) == [
        "parameter",
        "estimate",
        "posterior_mean",
        "posterior_sd",
        "median",
        "q025",
        "q975",
        "ess",
        "warning",
    ]
    assert table.loc[0, "estimate"] == pytest.approx(1.5)
    assert result.summary(point="median").loc[1, "estimate"] == pytest.approx(5.0)
    assert table.loc[0, "ess"] == 4
    with pytest.raises(NotImplementedError, match="MAP"):
        result.summary(point="MAP")


def test_covariance_correlation_draw_table_and_batch_warning() -> None:
    result = _result()
    assert result.covariance().shape == (2, 2)
    np.testing.assert_allclose(result.correlation(), np.ones((2, 2)))
    assert list(result.to_dataframe().columns) == ["draw", "alpha", "beta"]
    batch = _result(batch=True)
    assert batch.covariance().shape == (2, 2, 2)
    assert "dataset" in batch.summary().columns
    assert set(batch.to_dataframe()["dataset"]) == {0, 1}
    assert batch.summary().query("dataset == 1")["warning"].str.contains("outside").all()


def test_result_rejects_empty_or_nonfinite_numeric_payloads() -> None:
    parameter = (ParameterSpec("alpha"),)
    metadata = {"package_version": "0.1.0b1"}
    with pytest.raises(ValueError, match="at least one"):
        InferenceResult(np.empty((0, 1)), parameter, metadata, np.array([0.0]))
    with pytest.raises(ValueError, match="non-finite"):
        InferenceResult(np.array([[np.inf]]), parameter, metadata, np.array([0.0]))
    with pytest.raises(ValueError, match="must not be empty"):
        InferenceResult(np.array([[0.0]]), parameter, metadata, np.array([]))
    with pytest.raises(ValueError, match="non-finite"):
        InferenceResult(np.array([[0.0]]), parameter, metadata, np.array([np.nan]))


def test_one_draw_covariance_and_correlation_fail_explicitly() -> None:
    result = InferenceResult(
        draws=np.array([[0.0, 1.0]]),
        parameters=(ParameterSpec("alpha"), ParameterSpec("beta")),
        metadata={"package_version": "0.1.0b1"},
        representation=np.array([0.0]),
    )
    with pytest.raises(ValueError, match="at least two"):
        result.covariance()
    with pytest.raises(ValueError, match="at least two"):
        result.correlation()


def test_predictive_check_uses_fitted_adapter_state_and_mid_ties() -> None:
    model = _predictive_model()
    model.observation_adapter.fit([np.array([1.0])])
    result = InferenceResult(
        draws=np.zeros((4, 1)),
        parameters=model.parameters,
        metadata={"package_version": "0.1.0b1"},
        representation=np.array([1.0]),
        _model=model,
    )
    state_before = model.observation_adapter.get_state()
    check = result.predictive_check(replications=8, seed=3)
    assert model.observation_adapter.get_state() == state_before
    assert check.table.loc[0, "replicated_less_fraction"] == 0.0
    assert check.table.loc[0, "replicated_equal_fraction"] == 1.0
    assert check.table.loc[0, "two_sided_tail_fraction"] == 1.0
    assert "half of exact equality" in check.metadata["tie_handling"]

    mismatched = _predictive_model()
    mismatched.observation_adapter.fit([np.array([1.0, 2.0])])
    with pytest.raises(ValueError, match="fitted observation adapter"):
        result.predictive_check(model=mismatched, replications=2)


def test_predictive_check_validates_exact_transformed_width_without_fitting() -> None:
    adapter = _LooseAdapter()
    adapter.output_dim = 1
    model = _predictive_model(adapter=adapter, simulator=_wide_simulator)
    result = InferenceResult(
        draws=np.zeros((2, 1)),
        parameters=model.parameters,
        metadata={"package_version": "0.1.0b1"},
        representation=np.array([1.0]),
        _model=model,
    )
    state_before = adapter.get_state()
    with pytest.raises(ValueError, match="expected exactly 1"):
        result.predictive_check(replications=1)
    assert adapter.get_state() == state_before


def test_result_artifact_round_trip_and_checksum_failure(tmp_path) -> None:
    from structnpe import ArtifactError, load_result

    result = _result()
    artifact = result.save(tmp_path / "result")
    loaded = load_result(artifact)
    np.testing.assert_array_equal(loaded.draws, result.draws)
    with (artifact / "result.npz").open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ArtifactError, match="Checksum mismatch"):
        load_result(artifact)
