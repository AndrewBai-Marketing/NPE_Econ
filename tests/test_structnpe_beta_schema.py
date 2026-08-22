from __future__ import annotations

import numpy as np
import pytest

from structnpe import ArrayAdapter, ObservationAdapter, ParameterSpec, StructuralModel, SummaryAdapter
from structnpe.schema import adapter_from_dict, adapter_to_dict
from structnpe._fingerprints import fingerprint


def _prior(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {"scale": rng.uniform(0.2, 1.8, n), "location": rng.uniform(-1.0, 1.0, n)}


def _simulator(theta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(theta[0], theta[1], size=4)


class _StatefulPrior:
    def __init__(self, location: float) -> None:
        self.location = float(location)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.location, 1.0, size=(n, 1))


class _CustomAdapter(ObservationAdapter):
    def __init__(self, multiplier: float = 1.0) -> None:
        self.multiplier = float(multiplier)
        self.output_dim = 1

    def transform(self, observation: object) -> np.ndarray:
        return np.asarray([float(observation) * self.multiplier])

    def get_config(self) -> dict[str, float]:
        return {"multiplier": self.multiplier}


def test_parameter_transforms_round_trip_and_validate_support() -> None:
    specs = [
        ParameterSpec("identity"),
        ParameterSpec("positive", lower=0.0),
        ParameterSpec("upper", upper=2.0),
        ParameterSpec("bounded", lower=-2.0, upper=3.0),
    ]
    values = [np.array([-2.0, 0.5]), np.array([0.2, 3.0]), np.array([-1.0, 1.5]), np.array([-1.5, 2.5])]
    assert [spec.resolved_transform for spec in specs] == ["identity", "lower_log", "upper_log", "logit"]
    for spec, value in zip(specs, values, strict=True):
        np.testing.assert_allclose(spec.inverse_values(spec.transform_values(value)), value)
    with pytest.raises(ValueError, match="strictly above"):
        specs[1].transform_values(np.array([0.0]))
    with pytest.raises(ValueError, match="Unknown transform"):
        ParameterSpec("bad", transform="sqrt")
    with pytest.raises(ValueError, match="cannot enforce"):
        ParameterSpec("bounded_identity", lower=0.0, upper=1.0, transform="identity")
    with pytest.raises(ValueError, match="requires lower=0"):
        ParameterSpec("implicit_positive", transform="log")


def test_prior_mapping_uses_declared_order_and_explicit_rng() -> None:
    model = StructuralModel(
        prior=_prior,
        simulator=_simulator,
        parameters=[
            ParameterSpec("location", lower=-2.0, upper=2.0),
            ParameterSpec("scale", lower=0.1, upper=2.0),
        ],
        observation_adapter=ArrayAdapter(expected_shape=(4,)),
        prior_id="tests.prior.v1",
        simulator_id="tests.simulator.v1",
    )
    first = model.sample_prior(5, np.random.default_rng(7))
    second = model.sample_prior(5, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    assert first.shape == (5, 2)
    assert np.all(np.abs(first[:, 0]) <= 1.0)
    assert np.all((first[:, 1] >= 0.2) & (first[:, 1] <= 1.8))


def test_adapters_freeze_shape_and_reject_nonfinite_or_silent_reshape() -> None:
    array = ArrayAdapter()
    array.fit([np.ones((2, 2)), np.zeros((2, 2))])
    assert array.transform_one(np.eye(2)).shape == (4,)
    with pytest.raises(ValueError, match="shape"):
        array.transform_one(np.ones(4))
    summary = SummaryAdapter(("mean", "std"), expected_shape=(3,))
    summary.fit([np.array([1.0, 2.0, 3.0])])
    np.testing.assert_allclose(summary.transform_one([1.0, 2.0, 3.0]), [2.0, np.std([1.0, 2.0, 3.0])])
    with pytest.raises(ValueError, match="non-finite"):
        summary.transform_one([1.0, np.nan, 3.0])


def test_parameter_names_are_unique_and_prior_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="unique"):
        StructuralModel(
            prior=lambda n, rng: np.zeros((n, 2)),
            simulator=_simulator,
            parameter_names=["x", "x"],
            prior_id="duplicate.prior",
            simulator_id="duplicate.simulator",
        )
    model = StructuralModel(
        prior=lambda n, rng: np.zeros(n),
        simulator=lambda theta, rng: np.zeros(2),
        parameter_names=["x", "y"],
        prior_id="shape.prior",
        simulator_id="shape.simulator",
        prior_config={},
        simulator_config={},
    )
    with pytest.raises(ValueError, match="Prior must return shape"):
        model.sample_prior(3, np.random.default_rng(1))


def test_stateful_callable_requires_explicit_identifier_and_config() -> None:
    with pytest.raises(ValueError, match="must both be supplied"):
        StructuralModel(
            prior=_StatefulPrior(0.0),
            simulator=_simulator,
            parameter_names=["theta"],
        )
    first = StructuralModel(
        prior=_StatefulPrior(0.0),
        simulator=_simulator,
        parameter_names=["theta"],
        prior_id="tests.stateful_prior.v1",
        prior_config={"location": 0.0},
    )
    second = StructuralModel(
        prior=_StatefulPrior(2.0),
        simulator=_simulator,
        parameter_names=["theta"],
        prior_id="tests.stateful_prior.v1",
        prior_config={"location": 2.0},
    )
    assert first.specification()["prior"] != second.specification()["prior"]


def test_custom_adapter_reconstruction_requires_trust_boundary() -> None:
    adapter = _CustomAdapter(multiplier=2.0)
    payload = adapter_to_dict(adapter)
    with pytest.raises(ValueError, match="not imported by default"):
        adapter_from_dict(payload)
    reconstructed = adapter_from_dict(payload, trusted_class=_CustomAdapter)
    np.testing.assert_allclose(reconstructed.transform_one(3.0), [6.0])


def test_fingerprints_reject_mapping_key_coercion() -> None:
    with pytest.raises(TypeError, match="string keys"):
        fingerprint({1: "numeric key", "1": "string key"})
