from pathlib import Path

from structnpe.cli import main


def test_structnpe_cli_help(capsys) -> None:
    assert main([]) == 0
    assert "structnpe" in capsys.readouterr().out


def test_structnpe_cli_custom_template_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--template", "tiny_ddc", "--name", "toy_project"]) == 0
    config = tmp_path / "toy_project" / "config.yaml"
    observed = tmp_path / "toy_project" / "observed_example.csv"
    policy = tmp_path / "toy_project" / "policy.yaml"
    assert main(["simulate", "--config", str(config)]) == 0
    assert main(["train", "--config", str(config)]) == 0
    assert main(["infer", "--config", str(config), "--observed", str(observed), "--n-draws", "20"]) == 0
    assert main(["validate", "--config", str(config), "--observed", str(observed)]) == 0
    assert main(["counterfactual", "--config", str(config), "--observed", str(observed), "--policy", str(policy), "--n-draws", "20"]) == 0
    out = tmp_path / "toy_project" / "runs" / "latest"
    assert (out / "posterior_draws.csv").exists()
    assert (out / "validation_report.md").exists()
    assert (out / "counterfactual_report.md").exists()
