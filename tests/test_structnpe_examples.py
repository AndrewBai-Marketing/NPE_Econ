import shutil
from pathlib import Path

from structnpe.cli import main


ROOT = Path(__file__).resolve().parents[1]


def test_tiny_ddc_example_runs_from_copy(tmp_path: Path, monkeypatch) -> None:
    src = ROOT / "examples" / "tiny_ddc"
    dst = tmp_path / "tiny_ddc"
    shutil.copytree(src, dst)
    monkeypatch.chdir(dst)
    assert main(["simulate", "--config", "config.yaml"]) == 0
    assert main(["train", "--config", "config.yaml"]) == 0
    assert main(["infer", "--config", "config.yaml", "--observed", "observed_example.csv", "--n-draws", "20"]) == 0
    assert (dst / "runs" / "tiny_ddc" / "posterior_summary.csv").exists()
