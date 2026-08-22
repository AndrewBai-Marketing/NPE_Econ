"""Install one built artifact in a temporary environment and verify import."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--kind", choices=("wheel", "sdist"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def main() -> int:
    args = parse_args()
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    with tempfile.TemporaryDirectory(prefix=f"structnpe-{args.kind}-install-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        empty_cwd = root / "empty-cwd"
        empty_cwd.mkdir()
        venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=True).create(environment)
        scripts = "Scripts" if sys.platform == "win32" else "bin"
        python = environment / scripts / ("python.exe" if sys.platform == "win32" else "python")
        install = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps"]
        if args.kind == "sdist":
            install.append("--no-build-isolation")
        install_result = _run([*install, str(artifact)], empty_cwd)
        code = (
            "import importlib.metadata, json, pathlib, structnpe; "
            "p=pathlib.Path(structnpe.__file__).resolve(); "
            "print(json.dumps({'version': importlib.metadata.version('structnpe'), "
            "'origin_name': p.name, 'origin_in_venv': 'site-packages' in p.parts}))"
        )
        import_result = _run([str(python), "-c", code], empty_cwd)
        import_payload = json.loads(import_result.stdout.strip())
        if import_payload["version"] != "0.1.0b1" or not import_payload["origin_in_venv"]:
            raise RuntimeError(f"unexpected installed package: {import_payload}")
        cli_result = _run([str(python), "-m", "structnpe.cli", "--help"], empty_cwd)
        if "structnpe" not in cli_result.stdout or "Public-beta" not in cli_result.stdout:
            raise RuntimeError("installed CLI help check failed")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "artifact_kind": args.kind,
        "artifact_name": artifact.name,
        "package_version": import_payload["version"],
        "package_origin": "temporary virtual environment site-packages",
        "system_site_packages": True,
        "dependency_scope": "The prepared replication environment supplied dependencies; only the target artifact was installed.",
        "cli_help": "PASS",
        "pip_output": "Full installer output is retained in the stage log, not embedded in this shareable JSON.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "kind": args.kind, "artifact": artifact.name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
