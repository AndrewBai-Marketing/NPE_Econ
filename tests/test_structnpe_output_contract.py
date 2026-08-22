import csv
from pathlib import Path

from structnpe.cli import main


def _header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def test_tiny_ddc_output_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--template", "tiny_ddc", "--name", "demo_tiny_ddc"]) == 0
    config = tmp_path / "demo_tiny_ddc" / "config.yaml"
    observed = tmp_path / "demo_tiny_ddc" / "observed_example.csv"
    assert main(["simulate", "--config", str(config)]) == 0
    assert main(["train", "--config", str(config)]) == 0
    assert main(["infer", "--config", str(config), "--observed", str(observed), "--n-draws", "30"]) == 0
    assert main(["validate", "--config", str(config), "--observed", str(observed)]) == 0
    assert main(["report", "--project", str(tmp_path / "demo_tiny_ddc" / "runs" / "latest")]) == 0

    out = tmp_path / "demo_tiny_ddc" / "runs" / "latest"
    for name in [
        "posterior_summary.csv",
        "posterior_draws.csv",
        "validation_report.md",
        "training_report.md",
        "inference_report.md",
    ]:
        assert (out / name).exists(), name

    assert {"parameter", "mean", "sd", "q05", "q50", "q95"}.issubset(set(_header(out / "posterior_summary.csv")))
    assert {"draw", "theta_0"}.issubset(set(_header(out / "posterior_draws.csv")))
    assert {"diagnostic", "value", "notes"}.issubset(set(_header(out / "validation_diagnostics.csv")))
