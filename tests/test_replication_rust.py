from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from replication.rust_1987.constants import (
    CANONICAL_NFXP,
    CANONICAL_TRANSITION_COUNTS,
    CANONICAL_TRANSITION_PROBABILITIES,
    CHOICE_OBSERVATION_COUNT,
    COST_SCALE,
    DISCOUNT_FACTOR,
    NUM_STATES,
    OBSERVATION_COUNT,
    RAW_FILENAME,
)
from replication.rust_1987.download_data import MAX_DOWNLOAD_BYTES, download
from replication.rust_1987.model import (
    batched_replacement_probabilities,
    expected_replacements,
    simulate_choice_summary_batch,
    solve_bellman,
    transition_matrix,
)
from replication.rust_1987.nfxp import NFXPObjective, estimate_nfxp
from replication.rust_1987.preprocess import load_raw_matrix, preprocess_file
from replication.rust_1987.structnpe_model import build_model


# These are state-level choice aggregates, not redistributed bus histories.
# They make the exact conventional comparator a network-independent regression
# test while the downloader/preprocessor is checked separately below.
CANONICAL_KEEP_COUNTS = np.asarray(
    [
        101, 105, 115, 123, 121, 121, 103, 97, 98, 99, 96, 98, 93, 102, 84,
        92, 92, 91, 83, 79, 68, 74, 73, 72, 64, 68, 62, 64, 55, 54, 57, 49,
        49, 54, 54, 50, 52, 51, 50, 51, 57, 59, 47, 52, 48, 46, 45, 49, 47,
        47, 44, 49, 40, 44, 35, 35, 36, 29, 34, 26, 26, 26, 23, 23, 22, 21,
        13, 15, 17, 16, 7, 5, 3, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0,
    ],
    dtype=float,
)
CANONICAL_REPLACEMENT_COUNTS = np.asarray(
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0,
        0, 1, 2, 0, 1, 0, 0, 1, 0, 1, 1, 2, 2, 1, 3, 0, 1, 2, 1, 1,
        1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ],
    dtype=float,
)


def test_canonical_aggregates_have_expected_choice_count() -> None:
    assert CANONICAL_KEEP_COUNTS.shape == (NUM_STATES,)
    assert CANONICAL_REPLACEMENT_COUNTS.shape == (NUM_STATES,)
    assert int((CANONICAL_KEEP_COUNTS + CANONICAL_REPLACEMENT_COUNTS).sum()) == CHOICE_OBSERVATION_COUNT
    assert int(CANONICAL_REPLACEMENT_COUNTS.sum()) == 33


def test_transition_matrix_is_stochastic_and_caps_last_state() -> None:
    matrix = transition_matrix(NUM_STATES, CANONICAL_TRANSITION_PROBABILITIES)
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-15)
    assert matrix[-1, -1] == pytest.approx(1.0)
    assert np.count_nonzero(matrix[0]) == 3


def test_bellman_batch_matches_independent_single_solver() -> None:
    parameters = np.asarray([[10.07494, 2.29309], [11.7257, 2.4569]])
    batch = batched_replacement_probabilities(
        parameters,
        CANONICAL_TRANSITION_PROBABILITIES,
        num_states=NUM_STATES,
        beta=DISCOUNT_FACTOR,
        scale=COST_SCALE,
    )
    for index, theta in enumerate(parameters):
        single = solve_bellman(
            *theta,
            CANONICAL_TRANSITION_PROBABILITIES,
            num_states=NUM_STATES,
            beta=DISCOUNT_FACTOR,
            scale=COST_SCALE,
        )
        assert single.residual < 1e-10
        np.testing.assert_allclose(batch[index], single.replacement_probability, atol=2e-10, rtol=0)


def test_nfxp_analytic_score_matches_centered_difference() -> None:
    objective = NFXPObjective(
        CANONICAL_KEEP_COUNTS,
        CANONICAL_REPLACEMENT_COUNTS,
        CANONICAL_TRANSITION_PROBABILITIES,
        num_states=NUM_STATES,
        beta=DISCOUNT_FACTOR,
        scale=COST_SCALE,
    )
    theta = np.asarray([10.2, 2.4])
    _, gradient = objective.value_and_gradient(theta)
    numerical = np.empty(2)
    for j in range(2):
        step = np.zeros(2)
        step[j] = 1e-5
        plus = objective.value_and_gradient(theta + step)[0]
        minus = objective.value_and_gradient(theta - step)[0]
        numerical[j] = (plus - minus) / (2e-5)
    np.testing.assert_allclose(gradient, numerical, atol=2e-6, rtol=2e-6)


def test_canonical_nfxp_comparator() -> None:
    result = estimate_nfxp(
        CANONICAL_KEEP_COUNTS,
        CANONICAL_REPLACEMENT_COUNTS,
        CANONICAL_TRANSITION_PROBABILITIES,
        num_states=NUM_STATES,
        beta=DISCOUNT_FACTOR,
        scale=COST_SCALE,
    )
    assert result.success
    np.testing.assert_allclose(
        [result.replacement_cost, result.maintenance_slope],
        CANONICAL_NFXP,
        atol=2e-5,
        rtol=0,
    )
    assert result.bellman_residual < 1e-10


def test_safe_loader_rejects_unpinned_file(tmp_path: Path) -> None:
    source = tmp_path / RAW_FILENAME
    source.write_text("0\n" * 10, encoding="ascii")
    with pytest.raises(ValueError, match="pinned official file"):
        load_raw_matrix(source)


@pytest.mark.parametrize("overwrite", [False, True])
def test_download_rejects_existing_symlink_and_preserves_target(
    tmp_path: Path, overwrite: bool
) -> None:
    target = tmp_path / "private-target.txt"
    target.write_bytes(b"must remain unchanged")
    before = target.read_bytes()
    link = tmp_path / RAW_FILENAME
    link.symlink_to(target)

    with pytest.raises(ValueError, match="regular file, not a symlink"):
        download(link, overwrite=overwrite)

    assert link.is_symlink()
    assert target.read_bytes() == before


def test_download_rejects_existing_nonregular_or_oversized_target(tmp_path: Path) -> None:
    directory = tmp_path / "directory-target"
    directory.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        download(directory)

    oversized = tmp_path / RAW_FILENAME
    with oversized.open("wb") as stream:
        stream.truncate(MAX_DOWNLOAD_BYTES + 1)
    with pytest.raises(ValueError, match="large existing raw-data file"):
        download(oversized)


def test_official_download_and_preprocessing_when_explicitly_enabled(tmp_path: Path) -> None:
    if os.environ.get("STRUCTNPE_RUN_NETWORK_REPLICATION") != "1":
        pytest.skip("set STRUCTNPE_RUN_NETWORK_REPLICATION=1 for the pinned network regression")
    raw = download(tmp_path / RAW_FILENAME)
    panel = preprocess_file(raw)
    assert len(panel.state) == OBSERVATION_COUNT
    assert tuple(panel.transition_counts) == CANONICAL_TRANSITION_COUNTS
    np.testing.assert_allclose(panel.transition_probabilities, CANONICAL_TRANSITION_PROBABILITIES)
    keep, replacement = panel.choice_counts(NUM_STATES)
    np.testing.assert_array_equal(keep, CANONICAL_KEEP_COUNTS)
    np.testing.assert_array_equal(replacement, CANONICAL_REPLACEMENT_COUNTS)


def test_simulator_summary_and_policy_functionals_are_finite() -> None:
    parameters = np.asarray([[11.7257, 2.4569], [10.0, 2.0]])
    summaries = simulate_choice_summary_batch(
        parameters,
        np.random.default_rng(123),
        [0.0937, 0.4475, 0.4459, 0.0127, 0.0002],
        num_states=175,
        beta=0.975,
        scale=COST_SCALE,
        buses=4,
        periods=8,
    )
    assert summaries.shape == (2, 350)
    np.testing.assert_allclose(summaries[:, :175].sum(axis=1), 4 * 7)
    policy = solve_bellman(
        11.7257,
        2.4569,
        [0.0937, 0.4475, 0.4459, 0.0127, 0.0002],
        num_states=175,
        beta=0.975,
        scale=COST_SCALE,
    ).replacement_probability
    demand = expected_replacements(
        policy,
        [0.0937, 0.4475, 0.4459, 0.0127, 0.0002],
        periods=12,
        buses=50,
    )
    assert np.isfinite(demand) and 0 <= demand <= 600


def test_current_structnpe_model_contract_uses_batched_simulator() -> None:
    model = build_model(buses=2, periods=4)
    assert model.batched_simulator is True
    theta = model.sample_prior(2, np.random.default_rng(11))
    observations = model.simulate_batch(theta, np.random.default_rng(12))
    assert len(observations) == 2
    assert all(np.asarray(item).shape == (2 * NUM_STATES,) for item in observations)


def test_full_iskhakov_campaign_is_configured_but_unrun() -> None:
    path = Path("replication/iskhakov_2016/full_campaign.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["status"] == "configured_not_run"
    assert config["design"]["discount_factors"] == [0.975, 0.985, 0.995, 0.999, 0.9995, 0.9999]
    assert config["design"]["repetitions_per_discount_factor"] == 250
    assert config["design"]["truth"]["replacement_cost"] == 11.7257
    assert config["design"]["mpec"]["supported"] is False


def test_committed_rust_evidence_is_sanitized_and_matches_frozen_config() -> None:
    evidence_path = Path("replication/rust_1987/expected/smoke_metrics.json")
    evidence_text = evidence_path.read_text(encoding="utf-8")
    evidence = json.loads(evidence_text)
    assert ("/" + "home" + "/") not in evidence_text
    assert evidence["full_empirical_npe_campaign"]["executed"] is True
    assert evidence["full_empirical_npe_campaign"]["calibration"]["pass"] is True
    assert evidence["full_empirical_npe_campaign"]["empirical_comparison"]["pass"] is False
    config_path = Path(evidence["validation_config"]["path"])
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert digest == evidence["validation_config"]["sha256"]


def test_committed_iskhakov_evidence_is_sanitized_and_full_campaign_unrun() -> None:
    evidence_path = Path("replication/iskhakov_2016/expected/smoke_metrics.json")
    evidence_text = evidence_path.read_text(encoding="utf-8")
    evidence = json.loads(evidence_text)
    assert ("/" + "home" + "/") not in evidence_text
    assert evidence["configuration"]["full_status"] == "configured_not_run"
    assert evidence["design"]["repetitions"] == 3
    assert evidence["mpec"]["supported"] is False
