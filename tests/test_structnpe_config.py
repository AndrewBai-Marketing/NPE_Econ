from pathlib import Path

import pytest

from structnpe.config import StructNPEConfig, load_config, save_config


def test_config_yaml_round_trip(tmp_path: Path) -> None:
    cfg = StructNPEConfig(
        project_name="toy",
        output_dir=tmp_path / "runs" / "toy",
        simulator_module="model",
        simulator_class="Toy",
        n_train=20,
        n_valid=5,
    )
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded.project_name == "toy"
    assert loaded.output_dir == tmp_path / "runs" / "toy"
    assert loaded.posterior.type == "gaussian"


def test_config_validation_rejects_unknown_posterior(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "\n".join(
            [
                "project_name: bad",
                "output_dir: runs/bad",
                "simulator_module: model",
                "simulator_class: Toy",
                "posterior:",
                "  type: magic",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="posterior.type"):
        load_config(path)
