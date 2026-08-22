"""Configuration parsing for ``structnpe`` projects."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PosteriorConfig:
    type: str = "gaussian"
    hidden_dim: int = 64
    n_components: int = 3
    epochs: int = 100
    batch_size: int = 256


@dataclass
class ValidationConfig:
    posterior_predictive: bool = True
    prior_predictive_support: bool = True
    simulation_based_calibration: bool = True
    ensemble_seeds: list[int] = field(default_factory=list)


@dataclass
class CounterfactualConfig:
    policies: list[Any] = field(default_factory=list)


@dataclass
class StructNPEConfig:
    project_name: str
    output_dir: Path
    simulator_module: str
    simulator_class: str
    n_train: int = 1000
    n_valid: int = 200
    n_test: int = 0
    seed: int = 123
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    counterfactuals: CounterfactualConfig = field(default_factory=CounterfactualConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_name": self.project_name,
            "output_dir": str(self.output_dir),
            "simulator_module": self.simulator_module,
            "simulator_class": self.simulator_class,
            "n_train": self.n_train,
            "n_valid": self.n_valid,
            "n_test": self.n_test,
            "seed": self.seed,
            "posterior": self.posterior.__dict__,
            "validation": self.validation.__dict__,
            "counterfactuals": self.counterfactuals.__dict__,
        }


def load_config(path: str | Path) -> StructNPEConfig:
    """Load a JSON or small YAML-style config file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        try:
            data = _load_yaml_like(raw)
        except Exception as exc:
            raise ValueError(f"Could not parse config file {path}: {exc}") from exc
    return config_from_dict(data)


def save_config(config: StructNPEConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    else:
        path.write_text(_dump_yaml_like(config.to_dict()), encoding="utf-8")


def config_from_dict(data: dict[str, Any]) -> StructNPEConfig:
    required = ["project_name", "output_dir", "simulator_module", "simulator_class"]
    missing = [field for field in required if not data.get(field)]
    if missing:
        raise ValueError(f"Config is missing required fields: {', '.join(missing)}")
    posterior_data = dict(data.get("posterior") or {})
    validation_data = dict(data.get("validation") or {})
    counterfactual_data = dict(data.get("counterfactuals") or {})
    posterior_type = str(posterior_data.get("type", "gaussian"))
    if posterior_type not in {"finite_grid", "gaussian", "mdn", "model_index"}:
        raise ValueError(
            "posterior.type must be one of finite_grid, gaussian, mdn, model_index; "
            f"got {posterior_type!r}"
        )
    return StructNPEConfig(
        project_name=str(data["project_name"]),
        output_dir=Path(str(data["output_dir"])),
        simulator_module=str(data["simulator_module"]),
        simulator_class=str(data["simulator_class"]),
        n_train=int(data.get("n_train", 1000)),
        n_valid=int(data.get("n_valid", 200)),
        n_test=int(data.get("n_test", 0)),
        seed=int(data.get("seed", 123)),
        posterior=PosteriorConfig(
            type=posterior_type,
            hidden_dim=int(posterior_data.get("hidden_dim", 64)),
            n_components=int(posterior_data.get("n_components", 3)),
            epochs=int(posterior_data.get("epochs", 100)),
            batch_size=int(posterior_data.get("batch_size", 256)),
        ),
        validation=ValidationConfig(
            posterior_predictive=bool(validation_data.get("posterior_predictive", True)),
            prior_predictive_support=bool(validation_data.get("prior_predictive_support", True)),
            simulation_based_calibration=bool(validation_data.get("simulation_based_calibration", True)),
            ensemble_seeds=[int(seed) for seed in validation_data.get("ensemble_seeds", [])],
        ),
        counterfactuals=CounterfactualConfig(policies=list(counterfactual_data.get("policies", []))),
    )


def _parse_scalar(text: str) -> Any:
    text = text.strip()
    if text in {"", "null", "None"}:
        return None
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_yaml_like(raw: str) -> dict[str, Any]:
    """Parse the small YAML subset used by the templates.

    If PyYAML is installed, use it. Otherwise support nested dictionaries by
    indentation and scalar/list values.
    """

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(raw) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config root must be a mapping.")
        return loaded
    except ModuleNotFoundError:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in raw.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line}")
        key, value = line.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(value.strip())
    return root


def _dump_yaml_like(data: dict[str, Any], indent: int = 0) -> str:
    lines: list[str] = []
    pad = " " * indent
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines.append(_dump_yaml_like(value, indent + 2).rstrip())
        else:
            lines.append(f"{pad}{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return "[" + ", ".join(_format_scalar(v) for v in value) + "]"
    return str(value)
