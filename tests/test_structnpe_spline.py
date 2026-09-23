"""Spline backend checks against a tractable conditional posterior."""

from __future__ import annotations

import numpy as np
import pytest

from structnpe import ArtifactError, ParameterSpec, StructuralModel, SummaryAdapter, fit, load_estimator
from structnpe._spline import SPLINE_TYPE, validate_architecture


def model():
    return StructuralModel(
        prior=lambda n, rng: rng.normal(size=(n, 1)),
        simulator=lambda theta, rng: rng.normal(theta[0], 1, size=5),
        parameters=[ParameterSpec("location")],
        observation_adapter=SummaryAdapter(("mean",), expected_shape=(5,)),
        prior_id="tests.spline.normal.prior", simulator_id="tests.spline.normal.simulator",
        prior_config={"sd": 1}, simulator_config={"sd": 1, "observations": 5},
    )


@pytest.fixture(scope="module")
def trained():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nflows")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield fit(model(), backend="spline", simulations=2000, epochs=60, batch_size=128, seed=9031,
                  hidden_dim=24, depth=1, flow_layers=2, patience=12, progress=False)
    finally:
        torch.set_num_threads(old_threads)


def test_spline_recovers_conditional_normal(trained):
    assert trained.architecture["estimator_type"] == SPLINE_TYPE
    for observed_mean in [-0.4, 0.4]:
        draws = trained.infer(np.full(5, observed_mean), draws=4000, seed=31).draws[:, 0]
        assert abs(draws.mean() - 5 * observed_mean / 6) < 0.25
        assert abs(draws.std() - np.sqrt(1 / 6)) < 0.18


def test_flow_roundtrip_batch_and_rng(trained, tmp_path):
    import torch
    path = trained.save(tmp_path / "flow")
    loaded = load_estimator(path, model=model())
    observed = np.zeros(5)
    rng_state = torch.random.get_rng_state().clone()
    before = trained.infer(observed, draws=41, seed=77)
    assert torch.equal(rng_state, torch.random.get_rng_state())
    after = loaded.infer(observed, draws=41, seed=77)
    np.testing.assert_array_equal(before.draws, after.draws)
    assert after.metadata["estimator_type"] == SPLINE_TYPE
    batched = loaded.infer([observed, observed + 0.2], draws=23, seed=11, batch=True)
    assert batched.draws.shape == (2, 23, 1)
    assert np.isfinite(batched.draws).all()


def test_spline_architecture_rejects_oversized_allocation():
    with pytest.raises(ArtifactError, match="input_dim"):
        validate_architecture({"input_dim": 10**12})


def test_first_coordinate_has_a_context_path():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nflows")
    from structnpe._spline import make_flow
    architecture = {
        "input_dim": 3, "theta_dim": 2, "hidden_dim": 12, "depth": 1,
        "flow_layers": 1, "num_bins": 8, "tail_bound": 6,
        "activation": "silu", "nflows_version": "0.14",
        "conditioner": "made_with_context_skip_v1",
    }
    _, flow = make_flow(architecture, seed=22)
    conditioner = flow._transform._transforms[1].autoregressive_net
    # In stock nflows 0.14 this first-coordinate slice is context-independent.
    zero_context = conditioner(torch.zeros(1, 2), context=torch.zeros(1, 3))
    one_context = conditioner(torch.zeros(1, 2), context=torch.ones(1, 3))
    assert not torch.allclose(zero_context[:, :23], one_context[:, :23])
    # Context-only connections must not introduce dependence on later parameters.
    perturbed = conditioner(torch.ones(1, 2), context=torch.ones(1, 3))
    torch.testing.assert_close(one_context[:, :23], perturbed[:, :23])
