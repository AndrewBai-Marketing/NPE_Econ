"""Markdown and CSV report helpers for ``structnpe``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def write_csv(path: str | Path, rows: Iterable[dict[str, object]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    if not materialized:
        fieldnames = ["status"]
        materialized = [{"status": "empty"}]
    else:
        fieldnames = list(materialized[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)
    return path


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_training_report(path: str | Path, config: dict[str, object], outputs: dict[str, str]) -> Path:
    return _write_report(
        path,
        "Training Report",
        [
            "This report describes a simulator-defined training run.",
            "The learned object is simulator-relative and summary-conditional.",
            _dict_section("Config", config),
            _dict_section("Outputs", outputs),
        ],
    )


def write_inference_report(path: str | Path, outputs: dict[str, str], warnings: list[str] | None = None) -> Path:
    warnings = warnings or []
    return _write_report(
        path,
        "Inference Report",
        [
            "Posterior draws were generated from the saved posterior map.",
            _dict_section("Outputs", outputs),
            _list_section("Warnings", warnings or ["No additional warnings recorded."]),
        ],
    )


def write_validation_report(path: str | Path, diagnostics: list[dict[str, object]], warnings: list[str] | None = None) -> Path:
    lines = [
        "Passing diagnostics does not prove the simulator is true.",
        "Failing diagnostics means counterfactuals should not be trusted without investigation.",
        "",
        "| diagnostic | value | notes |",
        "| --- | ---: | --- |",
    ]
    for row in diagnostics:
        lines.append(f"| {row.get('diagnostic')} | {row.get('value')} | {row.get('notes', '')} |")
    lines.append("")
    lines.append(_list_section("Warnings", warnings or ["Validation is simulator-relative and summary-conditional."]))
    return _write_report(path, "Validation Report", lines)


def write_counterfactual_report(path: str | Path, rows: list[dict[str, object]], warnings: list[str] | None = None) -> Path:
    lines = [
        "Counterfactual draws are posterior pushforwards through the user-supplied policy simulator.",
        "",
        "| policy | mean | q05 | q50 | q95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('policy')} | {row.get('mean')} | {row.get('q05')} | {row.get('q50')} | {row.get('q95')} |"
        )
    lines.append("")
    lines.append(_list_section("Warnings", warnings or ["Counterfactuals inherit simulator and posterior approximation limits."]))
    return _write_report(path, "Counterfactual Report", lines)


def _write_report(path: str | Path, title: str, sections: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# " + title + "\n\n" + "\n\n".join(sections).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _dict_section(title: str, data: dict[str, object]) -> str:
    lines = [f"## {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in data.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _list_section(title: str, items: list[str]) -> str:
    lines = [f"## {title}"]
    lines.extend(f"- {item}" for item in items)
    return "\n".join(lines)
