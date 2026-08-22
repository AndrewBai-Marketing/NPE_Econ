#!/usr/bin/env python3
"""Extract checked documentation snippets from executable public examples."""

from __future__ import annotations

import argparse
import difflib
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SNIPPET_ROOT = ROOT / "docs" / "snippets"
README = ROOT / "README.md"
QUICKSTART_OUTPUT = SNIPPET_ROOT / "quickstart_output.txt"


@dataclass(frozen=True)
class Snippet:
    name: str
    source: Path
    destination: Path


SNIPPETS = (
    Snippet(
        "quickstart",
        ROOT / "examples" / "01_quickstart_saved_estimator.py",
        SNIPPET_ROOT / "quickstart.py",
    ),
    Snippet(
        "posterior",
        ROOT / "examples" / "02_bayesian_workflow.py",
        SNIPPET_ROOT / "posterior.py",
    ),
    Snippet(
        "custom",
        ROOT / "examples" / "03_custom_simulator.py",
        SNIPPET_ROOT / "custom_model.py",
    ),
)


def _render(snippet: Snippet) -> str:
    lines = snippet.source.read_text(encoding="utf-8").splitlines()
    start_marker = f"# README:{snippet.name}:start"
    end_marker = f"# README:{snippet.name}:end"
    starts = [index for index, line in enumerate(lines) if line.strip() == start_marker]
    ends = [index for index, line in enumerate(lines) if line.strip() == end_marker]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError(
            f"{snippet.source}: expected one ordered {start_marker!r}/{end_marker!r} pair"
        )
    body = textwrap.dedent("\n".join(lines[starts[0] + 1 : ends[0]])).rstrip()
    if not body:
        raise ValueError(f"{snippet.source}: extracted snippet is empty")
    relative = snippet.source.relative_to(ROOT).as_posix()
    return (
        f"# Generated from {relative} by scripts/render_readme_examples.py.\n"
        "# Edit the executable example, then rerun this script with --write.\n\n"
        f"{body}\n"
    )


def _generated_body(expected: str) -> str:
    lines = expected.splitlines()
    return "\n".join(lines[3:]).strip()


def _check(snippet: Snippet, expected: str) -> bool:
    if not snippet.destination.is_file():
        print(f"missing generated snippet: {snippet.destination.relative_to(ROOT)}", file=sys.stderr)
        return False
    current = snippet.destination.read_text(encoding="utf-8")
    valid = True
    if current != expected:
        diff = difflib.unified_diff(
            current.splitlines(),
            expected.splitlines(),
            fromfile=str(snippet.destination.relative_to(ROOT)),
            tofile=f"generated from {snippet.source.relative_to(ROOT)}",
            lineterm="",
        )
        print("\n".join(diff), file=sys.stderr)
        valid = False
    readme = README.read_text(encoding="utf-8")
    if _generated_body(expected) not in readme:
        print(
            f"README.md does not contain the generated {snippet.name!r} example body",
            file=sys.stderr,
        )
        valid = False
    return valid


def _render_quickstart_output() -> str:
    environment = os.environ.copy()
    source = str(ROOT / "src")
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source if not existing else os.pathsep.join((source, existing))
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "01_quickstart_saved_estimator.py"),
            "--draws",
            "10000",
            "--seed",
            "123",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    lines = completed.stdout.splitlines()
    end = next(
        (index for index, line in enumerate(lines) if line.startswith("joint_draw_table_shape=")),
        None,
    )
    if end is None or end < 2:
        raise ValueError("quickstart output did not contain its declared table and shape marker")
    return "\n".join(lines[:end]).rstrip() + "\n"


def _check_quickstart_output(expected: str) -> bool:
    valid = True
    if not QUICKSTART_OUTPUT.is_file() or QUICKSTART_OUTPUT.read_text(encoding="utf-8") != expected:
        print("docs/snippets/quickstart_output.txt is stale", file=sys.stderr)
        valid = False
    if expected.strip() not in README.read_text(encoding="utf-8"):
        print("README.md does not contain the generated quickstart output", file=sys.stderr)
        valid = False
    return valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail if checked snippets are stale")
    mode.add_argument("--write", action="store_true", help="regenerate the checked snippets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = [(snippet, _render(snippet)) for snippet in SNIPPETS]
    quickstart_output = _render_quickstart_output()
    if args.write:
        SNIPPET_ROOT.mkdir(parents=True, exist_ok=True)
        for snippet, content in rendered:
            snippet.destination.write_text(content, encoding="utf-8")
            print(f"wrote {snippet.destination.relative_to(ROOT)}")
        QUICKSTART_OUTPUT.write_text(quickstart_output, encoding="utf-8")
        print(f"wrote {QUICKSTART_OUTPUT.relative_to(ROOT)}")
        return 0
    snippets_valid = all(_check(snippet, content) for snippet, content in rendered)
    output_valid = _check_quickstart_output(quickstart_output)
    return 0 if snippets_valid and output_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
