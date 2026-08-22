"""Command-line interface for ``structnpe``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .config import load_config, save_config
from .counterfactuals import run_counterfactuals
from .inference import infer, load_simulator
from .posterior import fit_estimator
from .reports import write_training_report
from .simulation_bank import load_bank, simulate_bank
from .validation import validate_project


TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"
TEMPLATE_ALIASES = {"tiny_ddc": "finite_ddc"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="structnpe",
        description=(
            "Public-beta simulator-to-posterior toolkit with a trusted-local legacy project workflow. Users provide a "
            "prior, simulator, summaries, and optional counterfactual function."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Create a new project from a bundled template.")
    p_init.add_argument(
        "--template",
        default="custom_simulator",
        choices=["custom_simulator", "tiny_ddc", "finite_ddc", "model_index_ddc"],
        help="Template to copy. tiny_ddc is the official smoke-test demo.",
    )
    p_init.add_argument("--name", required=True, help="Project directory to create.")

    p_sim = sub.add_parser("simulate", help="Simulate train/validation banks from a config.")
    p_sim.add_argument("--config", type=Path, required=True, help="Path to config.yaml or config.json.")

    p_train = sub.add_parser("train", help="Train a posterior estimator from simulated banks.")
    p_train.add_argument("--config", type=Path, required=True, help="Path to config.yaml or config.json.")

    p_infer = sub.add_parser("infer", help="Infer posterior draws for observed summaries/data.")
    p_infer.add_argument("--config", type=Path, required=True, help="Path to config.yaml or config.json.")
    p_infer.add_argument("--observed", type=Path, required=True, help="Observed CSV file.")
    p_infer.add_argument("--n-draws", type=int, default=1000, help="Number of posterior draws.")

    p_val = sub.add_parser("validate", help="Run validation diagnostics.")
    p_val.add_argument("--config", type=Path, required=True, help="Path to config.yaml or config.json.")
    p_val.add_argument("--observed", type=Path, default=None, help="Optional observed CSV file.")

    p_cf = sub.add_parser("counterfactual", help="Run counterfactual pushforwards.")
    p_cf.add_argument("--config", type=Path, required=True, help="Path to config.yaml or config.json.")
    p_cf.add_argument("--observed", type=Path, required=True, help="Observed CSV file.")
    p_cf.add_argument("--policy", action="append", default=[], help="Policy YAML/JSON path. May be repeated.")
    p_cf.add_argument("--n-draws", type=int, default=1000, help="Number of posterior draws.")

    p_report = sub.add_parser("report", help="Print available project reports.")
    p_report.add_argument("--project", type=Path, required=True, help="Project output directory, usually runs/latest.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        if args.command == "init":
            return _cmd_init(args)
        if args.command == "simulate":
            return _cmd_simulate(args)
        if args.command == "train":
            return _cmd_train(args)
        if args.command == "infer":
            config = load_config(args.config)
            infer(config.output_dir, args.observed, n_draws=args.n_draws, seed=config.seed)
            return 0
        if args.command == "validate":
            config = load_config(args.config)
            validate_project(config.output_dir, args.observed)
            return 0
        if args.command == "counterfactual":
            config = load_config(args.config)
            policies = args.policy or config.counterfactuals.policies
            if not policies:
                print("Error: no policies supplied. Use --policy policy.yaml or set counterfactuals.policies.", file=sys.stderr)
                return 2
            run_counterfactuals(config.output_dir, args.observed, policies, n_draws=args.n_draws, seed=config.seed)
            return 0
        if args.command == "report":
            return _cmd_report(args)
    except (FileNotFoundError, ValueError, AttributeError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    raise ValueError(args.command)


def _cmd_init(args: argparse.Namespace) -> int:
    template = TEMPLATE_ALIASES.get(args.template, args.template)
    src = TEMPLATE_ROOT / template
    dst = Path(args.name)
    if dst.exists():
        print(f"Project path already exists: {dst}", file=sys.stderr)
        return 2
    shutil.copytree(src, dst)
    config_path = dst / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        config.project_name = dst.name
        config.output_dir = dst / "runs" / "latest"
        save_config(config, config_path)
    print(f"Created {args.template} project at {dst}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    spec = load_simulator(config, extra_path=args.config.parent)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, config.output_dir / "config.yaml")
    _copy_local_simulator(args.config.parent, config.output_dir)
    simulate_bank(spec, config.n_train, config.seed, config.output_dir / "simulated_train.npz")
    simulate_bank(spec, config.n_valid, config.seed + 1, config.output_dir / "simulated_valid.npz")
    print(f"Wrote simulation banks to {config.output_dir}")
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    train = load_bank(config.output_dir / "simulated_train.npz")
    valid = load_bank(config.output_dir / "simulated_valid.npz")
    est = fit_estimator(train, valid, config.posterior)
    model_path = est.save(config.output_dir / "posterior_model.npz")
    report_path = write_training_report(
        config.output_dir / "training_report.md",
        config.to_dict(),
        {"posterior_model": str(model_path), "simulated_train": str(config.output_dir / "simulated_train.npz")},
    )
    print(f"Wrote posterior model to {model_path}")
    print(f"Wrote training report to {report_path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    names = ["training_report.md", "inference_report.md", "validation_report.md", "counterfactual_report.md"]
    found = [args.project / name for name in names if (args.project / name).exists()]
    if not found:
        print(f"No reports found under {args.project}", file=sys.stderr)
        return 1
    for path in found:
        print(path)
    return 0


def _copy_local_simulator(config_dir: Path, output_dir: Path) -> None:
    model_path = config_dir / "model.py"
    if model_path.exists() and model_path.resolve() != (output_dir / "model.py").resolve():
        shutil.copy2(model_path, output_dir / "model.py")


if __name__ == "__main__":
    raise SystemExit(main())
