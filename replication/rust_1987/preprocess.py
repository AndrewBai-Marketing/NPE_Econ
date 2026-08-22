"""Safe, NumPy-only preprocessing for Rust's group-4 ASCII data.

The default ``historical_ruspy`` convention reproduces the group-4 inputs used
by the public ruspy replication.  It uses floor-discretized states and codes
the first post-replacement usage increment as one.  ``state_difference`` uses
the literal reset-state difference instead, exposing rather than hiding the
known preprocessing discrepancy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from .constants import (
        BUS_COUNT,
        CANONICAL_TRANSITION_COUNTS,
        CHOICE_OBSERVATION_COUNT,
        MILEAGE_BIN_SIZE,
        OBSERVATION_COUNT,
        PERIOD_COUNT,
        RAW_BYTES,
        RAW_FILENAME,
        RAW_SHA256,
        RAW_VALUES,
        SOURCE_COMMIT,
        SOURCE_REPOSITORY,
        SOURCE_TAG,
    )
except ImportError:  # pragma: no cover - direct script execution
    from constants import (
        BUS_COUNT,
        CANONICAL_TRANSITION_COUNTS,
        CHOICE_OBSERVATION_COUNT,
        MILEAGE_BIN_SIZE,
        OBSERVATION_COUNT,
        PERIOD_COUNT,
        RAW_BYTES,
        RAW_FILENAME,
        RAW_SHA256,
        RAW_VALUES,
        SOURCE_COMMIT,
        SOURCE_REPOSITORY,
        SOURCE_TAG,
    )

CONVENTIONS = {"historical_ruspy", "state_difference"}
CSV_COLUMNS = ("bus_id", "period", "state", "mileage", "usage", "decision")


@dataclass(frozen=True)
class PanelData:
    bus_id: np.ndarray
    period: np.ndarray
    state: np.ndarray
    mileage: np.ndarray
    usage: np.ndarray
    decision: np.ndarray
    convention: str

    def __post_init__(self) -> None:
        arrays = (
            self.bus_id,
            self.period,
            self.state,
            self.mileage,
            self.usage,
            self.decision,
        )
        if any(np.asarray(item).shape != (OBSERVATION_COUNT,) for item in arrays):
            raise ValueError(f"Every panel column must have {OBSERVATION_COUNT} rows.")
        if self.convention not in CONVENTIONS:
            raise ValueError(f"Unknown preprocessing convention {self.convention!r}.")

    @property
    def transition_counts(self) -> np.ndarray:
        valid = np.isfinite(self.usage)
        usage = np.asarray(self.usage[valid], dtype=int)
        if usage.size != CHOICE_OBSERVATION_COUNT or np.any((usage < 0) | (usage > 2)):
            raise ValueError("Processed transition increments are outside the expected support.")
        return np.bincount(usage, minlength=3)

    @property
    def transition_probabilities(self) -> np.ndarray:
        counts = self.transition_counts
        return counts / counts.sum()

    def likelihood_rows(self) -> tuple[np.ndarray, np.ndarray]:
        mask = self.period > 0
        return self.state[mask].astype(int), self.decision[mask].astype(int)

    def choice_counts(self, num_states: int) -> tuple[np.ndarray, np.ndarray]:
        state, decision = self.likelihood_rows()
        if np.any((state < 0) | (state >= int(num_states))):
            raise ValueError("A likelihood state lies outside the declared state grid.")
        total = np.bincount(state, minlength=int(num_states)).astype(float)
        replacement = np.bincount(state, weights=decision, minlength=int(num_states)).astype(float)
        return total - replacement, replacement

    def observation_summary(self, num_states: int) -> np.ndarray:
        keep, replacement = self.choice_counts(num_states)
        return np.concatenate((keep + replacement, replacement))


def _authenticate_raw(path: str | Path, *, verify_checksum: bool = True) -> bytes:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Raw input must be a regular non-symlink file.")
    payload = source.read_bytes()
    if len(payload) > 1_000_000:
        raise ValueError("Refusing an unexpectedly large raw-data file.")
    if verify_checksum:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != RAW_SHA256 or len(payload) != RAW_BYTES:
            raise ValueError("Raw-data bytes do not match the pinned official file.")
    try:
        payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Raw input must be ASCII.") from exc
    return payload


def load_raw_matrix(path: str | Path, *, verify_checksum: bool = True) -> np.ndarray:
    """Load the vectorized 128-by-37 ASCII matrix without pickle."""

    payload = _authenticate_raw(path, verify_checksum=verify_checksum)
    tokens = payload.decode("ascii").split()
    if len(tokens) != RAW_VALUES:
        raise ValueError(f"Expected exactly {RAW_VALUES} numeric ASCII fields.")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Raw input contains a nonnumeric field.") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError("Raw input contains a non-finite numeric field.")
    return values.reshape((128, BUS_COUNT), order="F")


def preprocess_matrix(matrix: np.ndarray, *, convention: str = "historical_ruspy") -> PanelData:
    """Convert an authenticated raw matrix to a deterministic rectangular panel."""

    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {sorted(CONVENTIONS)}.")
    raw = np.asarray(matrix, dtype=np.float64)
    if raw.shape != (128, BUS_COUNT) or not np.all(np.isfinite(raw)):
        raise ValueError(f"Raw group-4 matrix must have shape {(128, BUS_COUNT)}.")
    bus_ids = raw[0].astype(np.int64)
    if len(np.unique(bus_ids)) != BUS_COUNT:
        raise ValueError("Raw group-4 bus identifiers are not unique.")
    replacements_completed = np.zeros(BUS_COUNT, dtype=np.int8)
    records: list[tuple[int, int, int, int, int]] = []
    for raw_row in range(11, 128):
        mileage = raw[raw_row].copy()
        mileage[replacements_completed == 1] -= raw[5, replacements_completed == 1]
        mileage[replacements_completed == 2] -= raw[8, replacements_completed == 2]
        if np.any(mileage < 0):
            raise ValueError("Replacement-adjusted mileage became negative.")
        decision = np.zeros(BUS_COUNT, dtype=np.int8)
        if raw_row < 127:
            first = (
                (raw[raw_row + 1] > raw[5])
                & (raw[5] != 0)
                & (replacements_completed == 0)
            )
            decision[first] = 1
            replacements_completed[first] += 1
            second = (
                (raw[raw_row + 1] > raw[8])
                & (raw[8] != 0)
                & (replacements_completed == 1)
            )
            decision[second] = 1
            replacements_completed[second] += 1
        state = np.floor(mileage / MILEAGE_BIN_SIZE).astype(np.int64)
        for column in range(BUS_COUNT):
            records.append(
                (
                    int(bus_ids[column]),
                    raw_row - 11,
                    int(state[column]),
                    int(round(mileage[column])),
                    int(decision[column]),
                )
            )
    records.sort(key=lambda row: (row[0], row[1]))
    values = np.asarray(records, dtype=np.int64)
    bus_id, period, state, mileage, decision = values.T
    usage = np.full(OBSERVATION_COUNT, np.nan, dtype=np.float64)
    for index in range(1, OBSERVATION_COUNT):
        if bus_id[index] != bus_id[index - 1]:
            continue
        if decision[index - 1] == 1:
            increment = int(state[index])
            if convention == "historical_ruspy":
                # This historically coded minimum-one increment is necessary
                # for Rust Table IX / the public ruspy group-4 convention.
                increment = max(1, increment)
        else:
            increment = int(state[index] - state[index - 1])
        usage[index] = float(increment)
    panel = PanelData(bus_id, period, state, mileage, usage, decision, convention)
    if convention == "historical_ruspy" and tuple(panel.transition_counts) != CANONICAL_TRANSITION_COUNTS:
        raise ValueError(
            "Preprocessing did not reproduce canonical transition counts "
            f"{CANONICAL_TRANSITION_COUNTS}; got {tuple(panel.transition_counts)}."
        )
    return panel


def preprocess_file(path: str | Path, *, convention: str = "historical_ruspy") -> PanelData:
    return preprocess_matrix(load_raw_matrix(path), convention=convention)


def _csv_rows(panel: PanelData) -> Iterator[list[str]]:
    yield list(CSV_COLUMNS)
    for i in range(OBSERVATION_COUNT):
        yield [
            str(int(panel.bus_id[i])),
            str(int(panel.period[i])),
            str(int(panel.state[i])),
            str(int(panel.mileage[i])),
            "" if not np.isfinite(panel.usage[i]) else str(int(panel.usage[i])),
            str(int(panel.decision[i])),
        ]


def write_processed_csv(panel: PanelData, path: str | Path) -> tuple[Path, str]:
    """Atomically write a deterministic, non-pickle processed representation."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerows(_csv_rows(panel))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return target, digest


def read_processed_csv(path: str | Path, *, convention: str = "historical_ruspy") -> PanelData:
    source = Path(path)
    if not source.is_file() or source.is_symlink() or source.stat().st_size > 2_000_000:
        raise ValueError("Processed input must be a small regular non-symlink CSV file.")
    with source.open("r", encoding="ascii", newline="") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader, ())) != CSV_COLUMNS:
            raise ValueError("Processed CSV header does not match the frozen schema.")
        rows = list(reader)
    if len(rows) != OBSERVATION_COUNT or any(len(row) != len(CSV_COLUMNS) for row in rows):
        raise ValueError(f"Processed CSV must contain exactly {OBSERVATION_COUNT} data rows.")
    try:
        columns = list(zip(*rows, strict=True))
        bus_id = np.asarray(columns[0], dtype=np.int64)
        period = np.asarray(columns[1], dtype=np.int64)
        state = np.asarray(columns[2], dtype=np.int64)
        mileage = np.asarray(columns[3], dtype=np.int64)
        usage = np.asarray([np.nan if value == "" else int(value) for value in columns[4]], dtype=float)
        decision = np.asarray(columns[5], dtype=np.int8)
    except (TypeError, ValueError) as exc:
        raise ValueError("Processed CSV contains an invalid field.") from exc
    return PanelData(bus_id, period, state, mileage, usage, decision, convention)


def write_metadata(panel: PanelData, csv_path: Path, csv_sha256: str, path: str | Path) -> Path:
    metadata = {
        "schema_version": 1,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "tag": SOURCE_TAG,
            "commit": SOURCE_COMMIT,
            "filename": RAW_FILENAME,
            "sha256": RAW_SHA256,
        },
        "preprocessing": {
            "implementation": "structnpe.replication.rust_1987.preprocess",
            "convention": panel.convention,
            "mileage_bin_size": MILEAGE_BIN_SIZE,
            "initial_choice_rows_excluded_from_likelihood": True,
        },
        "processed": {
            "filename": csv_path.name,
            "sha256": csv_sha256,
            "observations": OBSERVATION_COUNT,
            "choice_observations": CHOICE_OBSERVATION_COUNT,
            "buses": BUS_COUNT,
            "periods_per_bus": PERIOD_COUNT,
            "replacements": int(panel.decision.sum()),
            "transition_counts": panel.transition_counts.astype(int).tolist(),
            "transition_probabilities": panel.transition_probabilities.tolist(),
        },
        "nonclaim": "This metadata records a preprocessing convention; it is not an inference result.",
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent / "data"
    parser.add_argument("--input", type=Path, default=root / "raw" / RAW_FILENAME)
    parser.add_argument("--output", type=Path, default=root / "processed" / "group4.csv")
    parser.add_argument("--metadata", type=Path, default=root / "processed" / "group4.metadata.json")
    parser.add_argument("--convention", choices=sorted(CONVENTIONS), default="historical_ruspy")
    args = parser.parse_args()
    panel = preprocess_file(args.input, convention=args.convention)
    output, digest = write_processed_csv(panel, args.output)
    write_metadata(panel, output, digest, args.metadata)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "observations": OBSERVATION_COUNT,
                "transition_counts": panel.transition_counts.tolist(),
                "transition_probabilities": panel.transition_probabilities.tolist(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

