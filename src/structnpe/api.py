"""Public Python API for ``structnpe``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PosteriorConfig, StructNPEConfig, ValidationConfig, save_config
from .inference import infer
from .posterior import fit_estimator
from .reports import write_training_report
from .simulation_bank import simulate_bank
from .validation import validate_project


def run_training(
    model: Any,
    n_train: int = 50_000,
    n_valid: int = 5_000,
    posterior: str = "gaussian",
    output_dir: str | Path = "runs/structnpe_project",
    seed: int = 123,
) -> dict[str, Path]:
    """Simulate banks and train a posterior map for a user simulator."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = StructNPEConfig(
        project_name=out.name,
        output_dir=out,
        simulator_module=model.__class__.__module__,
        simulator_class=model.__class__.__name__,
        n_train=n_train,
        n_valid=n_valid,
        seed=seed,
        posterior=PosteriorConfig(type=posterior),
        validation=ValidationConfig(),
    )
    save_config(config, out / "config.yaml")
    train = simulate_bank(model, n_train, seed, out / "simulated_train.npz")
    valid = simulate_bank(model, n_valid, seed + 1, out / "simulated_valid.npz")
    estimator = fit_estimator(train, valid, config.posterior)
    model_path = estimator.save(out / "posterior_model.npz")
    report_path = write_training_report(
        out / "training_report.md",
        config.to_dict(),
        {"simulated_train": str(out / "simulated_train.npz"), "simulated_valid": str(out / "simulated_valid.npz"), "posterior_model": str(model_path)},
    )
    return {
        "config": out / "config.yaml",
        "simulated_train": out / "simulated_train.npz",
        "simulated_valid": out / "simulated_valid.npz",
        "posterior_model": model_path,
        "training_report": report_path,
    }


def validate(project_dir: str | Path, observed_data: str | Path | None = None) -> dict[str, Path]:
    """Run validation diagnostics for a saved project."""

    return validate_project(project_dir, observed_data)


__all__ = ["StructNPEConfig", "run_training", "infer", "validate"]
