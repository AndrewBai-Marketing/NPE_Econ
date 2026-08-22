"""Inference utilities for saved ``structnpe`` projects."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import StructNPEConfig, load_config
from .posterior import load_posterior, posterior_summary
from .reports import read_csv, write_csv, write_inference_report
from .simulator import ensure_summary_array


def load_simulator(config: StructNPEConfig, extra_path: str | Path | None = None) -> Any:
    """Import and instantiate the configured simulator class."""

    if extra_path is not None:
        sys.path.insert(0, str(Path(extra_path).resolve()))
    project_dir = Path(config.output_dir)
    if project_dir.exists():
        sys.path.insert(0, str(project_dir.resolve()))
    importlib.invalidate_caches()
    if extra_path is not None and config.simulator_module in sys.modules:
        del sys.modules[config.simulator_module]
    try:
        module = importlib.import_module(config.simulator_module)
    except ImportError as exc:
        raise ImportError(
            f"Could not import simulator_module={config.simulator_module!r}. "
            "Run from the project directory or ensure the simulator file is beside config.yaml."
        ) from exc
    try:
        cls = getattr(module, config.simulator_class)
    except AttributeError as exc:
        raise AttributeError(
            f"Simulator class {config.simulator_class!r} was not found in module "
            f"{config.simulator_module!r}."
        ) from exc
    return cls()


def observed_summary(spec: Any, observed_data_path: str | Path) -> np.ndarray:
    """Summarize observed data using summary columns or the simulator."""

    path = Path(observed_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Observed data file not found: {path}")
    rows = read_csv(path)
    n_summary = _count_summary_columns(rows[0]) if rows else 0
    if rows and n_summary > 0:
        cols = [f"summary_{j}" for j in range(n_summary)]
        return np.asarray([float(rows[0][col]) for col in cols], dtype=float)
    return ensure_summary_array(spec.summarize(rows))


def draw_posterior(project_dir: str | Path, observed_data_path: str | Path, n_draws: int = 1000, seed: int = 123) -> np.ndarray:
    project = Path(project_dir)
    config = load_config(project / "config.yaml")
    spec = load_simulator(config, extra_path=project)
    summary = observed_summary(spec, observed_data_path)
    posterior = load_posterior(project / "posterior_model.npz")
    return posterior.sample(summary, n_draws=n_draws, seed=seed)


def infer(project_dir: str | Path, observed_data_path: str | Path, n_draws: int = 1000, seed: int = 123) -> dict[str, Path]:
    """Run inference on observed data and write posterior draws/summary."""

    project = Path(project_dir)
    draws = draw_posterior(project, observed_data_path, n_draws=n_draws, seed=seed)
    draw_rows = _draw_rows(draws)
    summary_rows = posterior_summary(draws)
    draws_path = write_csv(project / "posterior_draws.csv", draw_rows)
    summary_path = write_csv(project / "posterior_summary.csv", summary_rows)
    report_path = write_inference_report(
        project / "inference_report.md",
        {"posterior_draws": str(draws_path), "posterior_summary": str(summary_path)},
    )
    return {"posterior_draws": draws_path, "posterior_summary": summary_path, "report": report_path}


def _draw_rows(draws: np.ndarray) -> list[dict[str, object]]:
    arr = np.asarray(draws, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    rows: list[dict[str, object]] = []
    for i, draw in enumerate(arr):
        row: dict[str, object] = {"draw": i}
        for j, value in enumerate(draw):
            row[f"theta_{j}"] = float(value)
        rows.append(row)
    return rows


def _count_summary_columns(row: dict[str, str]) -> int:
    j = 0
    while f"summary_{j}" in row:
        j += 1
    return j
