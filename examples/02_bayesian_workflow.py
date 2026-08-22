"""Use named, aligned joint posterior draws for ordinary Bayesian analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from structnpe import load_estimator

from _replacement_demo import OBSERVED_COUNTS, build_model


DEFAULT_ARTIFACT = Path(__file__).resolve().parent / "assets" / "demo_estimator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--predictive-replications", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = build_model()
    estimator = load_estimator(args.artifact, model=model)
    result = estimator.infer(OBSERVED_COUNTS, draws=args.draws, seed=123)

    # README:posterior:start
    summary = result.summary()
    draws = result.to_dataframe()
    names = list(estimator.parameter_names)
    covariance = pd.DataFrame(result.covariance(), index=names, columns=names)
    correlation = pd.DataFrame(result.correlation(), index=names, columns=names)

    probability = float((draws["maintenance_cost"] > 0.4).mean())
    cost_ratio = draws["replacement_cost"] / draws["maintenance_cost"]
    ratio_interval = cost_ratio.quantile([0.025, 0.5, 0.975])
    # README:posterior:end

    predictive = result.predictive_check(
        replications=args.predictive_replications,
        seed=456,
    )

    print(summary.to_string(index=False))
    print("\nPosterior covariance:\n", covariance.to_string())
    print("\nPosterior correlation:\n", correlation.to_string())
    print(f"\nP(maintenance_cost > 0.4 | data) = {probability:.4f}")
    print("Replacement/maintenance cost ratio quantiles:")
    print(ratio_interval.to_string())
    print("\nPosterior predictive representation summary:")
    print(predictive.to_dataframe().head(3).to_string(index=False))
    print(predictive.warning)
    print("BAYESIAN_WORKFLOW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

