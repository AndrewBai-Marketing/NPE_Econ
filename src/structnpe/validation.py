"""Validation diagnostics for ``structnpe`` projects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .inference import load_simulator, observed_summary
from .posterior import load_posterior
from .reports import write_csv, write_validation_report
from .simulation_bank import SimulationBank, load_bank


def prior_predictive_support(bank: SimulationBank, obs_summary: np.ndarray) -> list[dict[str, object]]:
    sims = np.asarray(bank.summaries, dtype=float)
    obs = np.asarray(obs_summary, dtype=float).reshape(-1)
    mean = sims.mean(axis=0)
    sd = np.where(sims.std(axis=0) > 1.0e-8, sims.std(axis=0), 1.0)
    z = (obs - mean) / sd
    rows: list[dict[str, object]] = []
    for j, value in enumerate(obs):
        percentile = float((sims[:, j] <= value).mean())
        rows.append(
            {
                "diagnostic": f"summary_{j}_z",
                "value": round(float(z[j]), 6),
                "notes": f"prior-predictive percentile={percentile:.3f}",
            }
        )
    distances = np.sqrt(np.sum(((sims - obs[None, :]) / sd[None, :]) ** 2, axis=1))
    rows.append(
        {
            "diagnostic": "nearest_neighbor_distance",
            "value": round(float(distances.min()), 6),
            "notes": "standardized nearest simulated summary distance",
        }
    )
    return rows


def posterior_predictive(
    spec: Any,
    posterior: Any,
    obs_summary: np.ndarray,
    *,
    model_index: bool,
    n_draws: int = 50,
    seed: int = 123,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    draws = posterior.sample(obs_summary, n_draws=n_draws, seed=seed)
    reps: list[np.ndarray] = []
    for draw in np.asarray(draws, dtype=float):
        if model_index:
            data = spec.simulate_given_model(int(draw[0]), draw[1:], rng)
        else:
            data = spec.simulate(draw, rng)
        reps.append(np.asarray(spec.summarize(data), dtype=float).reshape(-1))
    rep = np.vstack(reps)
    rows: list[dict[str, object]] = []
    for j, value in enumerate(obs_summary):
        rows.append(
            {
                "diagnostic": f"posterior_predictive_summary_{j}_gap",
                "value": round(float(rep[:, j].mean() - value), 6),
                "notes": "replicated-summary mean minus observed summary",
            }
        )
    return rows


def simulation_based_calibration(posterior: Any, test_bank: SimulationBank, n_draws: int = 200, seed: int = 321) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    cover_flags: list[float] = []
    ranks: list[float] = []
    max_cases = min(25, len(test_bank.summaries))
    for i in range(max_cases):
        summary = test_bank.summaries[i]
        true_theta = np.asarray(test_bank.theta[i], dtype=float).reshape(-1)
        draws = np.asarray(posterior.sample(summary, n_draws=n_draws, seed=int(rng.integers(1, 10_000_000))), dtype=float)
        if test_bank.model_index is not None and draws.shape[1] == true_theta.size + 1:
            draws_theta = draws[:, 1:]
        else:
            draws_theta = draws
        lo = np.quantile(draws_theta, 0.05, axis=0)
        hi = np.quantile(draws_theta, 0.95, axis=0)
        cover_flags.append(float(np.all((true_theta >= lo) & (true_theta <= hi))))
        ranks.append(float((draws_theta[:, 0] <= true_theta[0]).mean()))
    return [
        {
            "diagnostic": "sbc_90pct_joint_coverage",
            "value": round(float(np.mean(cover_flags)) if cover_flags else 0.0, 6),
            "notes": "held-out simulated cases; small smoke diagnostic",
        },
        {
            "diagnostic": "sbc_rank_mean_theta0",
            "value": round(float(np.mean(ranks)) if ranks else 0.0, 6),
            "notes": "mean posterior rank of true theta_0; target roughly 0.5 in calibrated runs",
        },
    ]


def validate_project(project_dir: str | Path, observed_data_path: str | Path | None = None) -> dict[str, Path]:
    project = Path(project_dir)
    config = load_config(project / "config.yaml")
    spec = load_simulator(config, extra_path=project)
    posterior = load_posterior(project / "posterior_model.npz")
    train = load_bank(project / "simulated_train.npz")
    valid = load_bank(project / "simulated_valid.npz")
    diagnostics: list[dict[str, object]] = []
    warnings = [
        "Passing diagnostics does not prove the simulator is true.",
        "Diagnostics are conditional on the supplied summaries and mechanism menu.",
    ]
    if observed_data_path is not None:
        obs = observed_summary(spec, observed_data_path)
        if config.validation.prior_predictive_support:
            diagnostics.extend(prior_predictive_support(train, obs))
        if config.validation.posterior_predictive:
            diagnostics.extend(
                posterior_predictive(
                    spec,
                    posterior,
                    obs,
                    model_index=bool(train.model_index is not None),
                    n_draws=min(50, len(train.summaries)),
                    seed=config.seed + 17,
                )
            )
    if config.validation.simulation_based_calibration:
        diagnostics.extend(simulation_based_calibration(posterior, valid, seed=config.seed + 23))
    csv_path = write_csv(project / "validation_diagnostics.csv", diagnostics)
    report_path = write_validation_report(project / "validation_report.md", diagnostics, warnings)
    return {"validation_diagnostics": csv_path, "validation_report": report_path}
