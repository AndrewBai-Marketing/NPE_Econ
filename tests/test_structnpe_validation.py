from pathlib import Path

import numpy as np

from structnpe.estimators import GaussianPosterior
from structnpe.simulation_bank import SimulationBank
from structnpe.validation import prior_predictive_support, simulation_based_calibration


def test_prior_predictive_support_outputs_expected_rows() -> None:
    bank = SimulationBank(summaries=np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]), theta=np.array([[0.0], [1.0], [2.0]]))
    rows = prior_predictive_support(bank, np.array([1.0, 2.0]))
    diagnostics = {row["diagnostic"] for row in rows}
    assert "summary_0_z" in diagnostics
    assert "nearest_neighbor_distance" in diagnostics


def test_simulation_based_calibration_smoke() -> None:
    x = np.linspace(-1, 1, 30).reshape(-1, 1)
    bank = SimulationBank(summaries=x, theta=x)
    posterior = GaussianPosterior().fit(bank)
    rows = simulation_based_calibration(posterior, bank, n_draws=20, seed=5)
    assert {row["diagnostic"] for row in rows} == {"sbc_90pct_joint_coverage", "sbc_rank_mean_theta0"}
