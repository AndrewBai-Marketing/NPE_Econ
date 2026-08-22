"""Run a bounded NFXP-versus-NPE Rust Monte Carlo smoke.

This is Iskhakov-style rather than a reproduction of the paper's full table:
transition probabilities are held fixed at their declared DGP values, MPEC is
not implemented, and the default budget is intentionally small.  The frozen
six-beta/250-run design is available in ``full_campaign.json`` but cannot be
started accidentally.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from structnpe import fit

from replication.rust_1987.model import simulate_choice_summary_batch
from replication.rust_1987.nfxp import estimate_nfxp
from replication.rust_1987.structnpe_model import build_model

TRUTH = np.asarray([11.7257, 2.4569], dtype=float)
TRANSITION_PROBABILITIES = np.asarray([0.0937, 0.4475, 0.4459, 0.0127, 0.0002])
NUM_STATES = 175
BUSES = 50
PERIODS = 120
COST_SCALE = 0.001
PRIOR_BOUNDS = ((7.0, 17.0), (0.8, 4.5))
FULL_BETAS = (0.975, 0.985, 0.995, 0.999, 0.9995, 0.9999)


def _metrics(estimates: np.ndarray, truth: np.ndarray) -> dict[str, list[float]]:
    error = estimates - truth[None, :]
    return {
        "bias": np.mean(error, axis=0).tolist(),
        "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
    }


def run_beta(
    beta: float,
    *,
    repetitions: int,
    simulations: int,
    epochs: int,
    draws: int,
    seed: int,
    progress: bool,
) -> dict[str, object]:
    model = build_model(
        transition_probabilities=TRANSITION_PROBABILITIES,
        num_states=NUM_STATES,
        beta=beta,
        scale=COST_SCALE,
        buses=BUSES,
        periods=PERIODS,
        prior_bounds=PRIOR_BOUNDS,
    )
    training_started = time.perf_counter()
    estimator = fit(
        model,
        simulations=simulations,
        seed=seed,
        hidden_dim=64,
        depth=2,
        components=5,
        epochs=epochs,
        batch_size=min(128, simulations),
        patience=max(2, min(10, epochs)),
        device="cpu",
        progress=progress,
        support_reference_size=min(256, max(2, simulations // 2)),
    )
    training_wall = time.perf_counter() - training_started
    rng = np.random.default_rng(seed + 10_000)
    truth_batch = np.repeat(TRUTH[None, :], repetitions, axis=0)
    panels = simulate_choice_summary_batch(
        truth_batch,
        rng,
        TRANSITION_PROBABILITIES,
        num_states=NUM_STATES,
        beta=beta,
        scale=COST_SCALE,
        buses=BUSES,
        periods=PERIODS,
    )
    nfxp_estimates = np.empty((repetitions, 2), dtype=float)
    nfxp_started = time.perf_counter()
    nfxp_diagnostics: list[dict[str, object]] = []
    for index, panel in enumerate(panels):
        visits = panel[:NUM_STATES]
        replacement = panel[NUM_STATES:]
        result = estimate_nfxp(
            visits - replacement,
            replacement,
            TRANSITION_PROBABILITIES,
            num_states=NUM_STATES,
            beta=beta,
            scale=COST_SCALE,
            start=TRUTH,
            bounds=PRIOR_BOUNDS,
        )
        nfxp_estimates[index] = (result.replacement_cost, result.maintenance_slope)
        nfxp_diagnostics.append(
            {
                "success": result.success,
                "bellman_residual": result.bellman_residual,
                "objective_evaluations": result.objective_evaluations,
            }
        )
    nfxp_seconds = time.perf_counter() - nfxp_started
    inference_started = time.perf_counter()
    inference = estimator.infer(list(panels), draws=draws, seed=seed + 20_000, batch=True)
    inference_seconds = time.perf_counter() - inference_started
    posterior = np.asarray(inference.draws, dtype=float)
    posterior_mean = posterior.mean(axis=1)
    lower = np.quantile(posterior, 0.025, axis=1)
    upper = np.quantile(posterior, 0.975, axis=1)
    coverage = np.mean((TRUTH[None, :] >= lower) & (TRUTH[None, :] <= upper), axis=0)
    correlations = np.asarray(
        [np.corrcoef(draw.T)[0, 1] for draw in posterior], dtype=float
    )
    nfxp_per_panel = nfxp_seconds / repetitions
    npe_per_panel = inference_seconds / repetitions
    if nfxp_per_panel > npe_per_panel:
        break_even: float | None = training_wall / (nfxp_per_panel - npe_per_panel)
        break_even_reason = "finite"
    else:
        break_even = None
        break_even_reason = "No finite break-even because measured NPE inference was not faster than NFXP."
    return {
        "beta": beta,
        "truth": TRUTH.tolist(),
        "repetitions": repetitions,
        "nfxp": {
            **_metrics(nfxp_estimates, TRUTH),
            "estimates": nfxp_estimates.tolist(),
            "total_seconds": nfxp_seconds,
            "seconds_per_panel": nfxp_per_panel,
            "diagnostics": nfxp_diagnostics,
        },
        "npe": {
            **_metrics(posterior_mean, TRUTH),
            "posterior_means": posterior_mean.tolist(),
            "coverage_95": coverage.tolist(),
            "posterior_correlation_mean": float(np.mean(correlations)),
            "posterior_correlations": correlations.tolist(),
            "training_wall_seconds": training_wall,
            "simulation_seconds": estimator.training_metadata["simulation_seconds"],
            "training_seconds_excluding_simulation": estimator.training_metadata["training_seconds"],
            "inference_total_seconds": inference_seconds,
            "inference_seconds_per_panel": npe_per_panel,
            "training_simulations": simulations,
            "epochs_requested": epochs,
            "best_epoch": estimator.training_metadata["best_epoch"],
            "draws_per_panel": draws,
        },
        "amortization": {
            "break_even_panels": break_even,
            "interpretation": break_even_reason,
            "formula": "training_wall / (NFXP_seconds_per_panel - NPE_seconds_per_panel)",
        },
        "transition_treatment": "fixed at the declared DGP values for NFXP and NPE",
        "mpec": {
            "supported": False,
            "reason": "This independent implementation currently provides NFXP and NPE only.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.975])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--draws", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2016)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--confirm-full-campaign", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1 or args.simulations < 10 or args.epochs < 1 or args.draws < 20:
        raise SystemExit("Invalid bounded-run budget.")
    full_scale = tuple(args.betas) == FULL_BETAS and args.repetitions >= 250
    if full_scale and not args.confirm_full_campaign:
        raise SystemExit(
            "The six-beta/250-run campaign requires --confirm-full-campaign. Its configuration "
            "does not imply it has been run."
        )
    results = [
        run_beta(
            beta,
            repetitions=args.repetitions,
            simulations=args.simulations,
            epochs=args.epochs,
            draws=args.draws,
            seed=args.seed + index * 100_000,
            progress=not args.quiet,
        )
        for index, beta in enumerate(args.betas)
    ]
    payload = {
        "schema_version": 1,
        "status": "completed_bounded_run" if not full_scale else "completed_requested_full_scale_run",
        "design": {
            "num_states": NUM_STATES,
            "buses": BUSES,
            "periods": PERIODS,
            "cost_scale": COST_SCALE,
            "transition_probabilities": TRANSITION_PROBABILITIES.tolist(),
        },
        "results": results,
        "scope": (
            "Iskhakov-style bounded comparison; not a reproduction of the paper's MPEC/NFXP "
            "table because MPEC is absent and transition probabilities are fixed."
        ),
        "full_campaign_configured": "replication/iskhakov_2016/full_campaign.json",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
