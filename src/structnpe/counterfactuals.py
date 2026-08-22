"""Counterfactual posterior pushforwards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .inference import draw_posterior, load_simulator
from .reports import write_counterfactual_report, write_csv


def load_policy(path_or_value: str | Path | dict[str, Any]) -> Any:
    if isinstance(path_or_value, dict):
        return path_or_value
    path = Path(path_or_value)
    if path.exists():
        raw = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(raw)
        return _parse_policy_yaml(raw)
    return {"name": str(path_or_value)}


def run_counterfactuals(
    project_dir: str | Path,
    observed_data_path: str | Path,
    policies: list[Any],
    n_draws: int = 1000,
    seed: int = 123,
) -> dict[str, Path]:
    project = Path(project_dir)
    config = load_config(project / "config.yaml")
    spec = load_simulator(config, extra_path=project)
    draws = draw_posterior(project, observed_data_path, n_draws=n_draws, seed=seed)
    rng = np.random.default_rng(seed + 991)
    draw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for policy_ref in policies:
        policy = load_policy(policy_ref)
        policy_name = str(policy.get("name", policy_ref)) if isinstance(policy, dict) else str(policy)
        values: list[float] = []
        for i, draw in enumerate(np.asarray(draws, dtype=float)):
            try:
                if config.posterior.type == "model_index":
                    value = spec.counterfactual(int(draw[0]), draw[1:], policy, rng)
                else:
                    value = spec.counterfactual(draw, policy, rng)
            except NotImplementedError:
                value = float("nan")
            value_float = float(np.asarray(value, dtype=float).reshape(-1)[0])
            values.append(value_float)
            draw_rows.append({"policy": policy_name, "draw": i, "delta": value_float})
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            summary_rows.append(
                {
                    "policy": policy_name,
                    "mean": float(finite.mean()),
                    "q05": float(np.quantile(finite, 0.05)),
                    "q50": float(np.quantile(finite, 0.50)),
                    "q95": float(np.quantile(finite, 0.95)),
                }
            )
        else:
            summary_rows.append({"policy": policy_name, "mean": "nan", "q05": "nan", "q50": "nan", "q95": "nan"})

    draws_path = write_csv(project / "counterfactual_draws.csv", draw_rows)
    summary_path = write_csv(project / "counterfactual_summary.csv", summary_rows)
    report_path = write_counterfactual_report(project / "counterfactual_report.md", summary_rows)
    return {"counterfactual_draws": draws_path, "counterfactual_summary": summary_path, "report": report_path}


def _parse_policy_yaml(raw: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out
