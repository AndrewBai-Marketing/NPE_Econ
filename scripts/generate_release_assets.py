#!/usr/bin/env python3
"""Create a checksummed GitHub-release bundle without publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_metadata() -> tuple[str, str, str]:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = payload["project"]
    return str(project["name"]), str(project["version"]), str(project["requires-python"])


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _single(dist_dir: Path, pattern: str) -> Path:
    matches = sorted(path for path in dist_dir.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {pattern!r} artifact, found {matches}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--commit", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    name, version, requires_python = _project_metadata()
    expected_tag = f"v{version}"
    if args.tag != expected_tag:
        raise ValueError(f"tag/version mismatch: expected {expected_tag!r}, received {args.tag!r}")

    commit = args.commit or _git_commit()
    if COMMIT_PATTERN.fullmatch(commit) is None:
        raise ValueError("release commit must be a complete 40-character lowercase Git SHA")

    dist_dir = args.dist_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not dist_dir.is_dir():
        raise ValueError(f"distribution directory does not exist: {dist_dir}")
    if output_dir == dist_dir:
        raise ValueError("output directory must differ from the distribution directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be absent or empty: {output_dir}")

    wheel = _single(dist_dir, f"{name}-{version}-*.whl")
    sdist = _single(dist_dir, f"{name}-{version}.tar.gz")
    unexpected = sorted(
        path.name
        for path in dist_dir.iterdir()
        if path.is_file() and path.suffix in {".whl", ".gz"} and path not in {wheel, sdist}
    )
    if unexpected:
        raise ValueError(f"unexpected distribution artifacts: {unexpected}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for source in (wheel, sdist):
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)

    artifacts = [
        {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(copied, key=lambda item: item.name)
    ]
    manifest = {
        "schema_version": "structnpe.github-release-manifest.v1",
        "package": {"name": name, "version": version, "requires_python": requires_python},
        "source": {"tag": args.tag, "commit": commit},
        "release": {
            "channel": "github-prerelease",
            "pypi_publication": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "build_environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "artifacts": artifacts,
    }
    (output_dir / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['filename']}\n" for item in artifacts),
        encoding="utf-8",
    )
    print(f"release_tag={args.tag}")
    print(f"release_commit={commit}")
    for item in artifacts:
        print(f"sha256={item['sha256']}  {item['filename']}")
    print(f"release_bundle={output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, tomllib.TOMLDecodeError) as error:
        print(f"release asset generation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error

