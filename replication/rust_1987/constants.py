"""Frozen public-data and model constants for the Rust replication."""

from __future__ import annotations

SOURCE_COMMIT = "43d37bd1d24a0a6c6e73e747545ad490d1f7f95b"
SOURCE_TAG = "v0.14"
SOURCE_REPOSITORY = "https://github.com/OpenSourceEconomics/zurcher-data"
RAW_FILENAME = "a530875.asc"
RAW_URL = (
    "https://raw.githubusercontent.com/OpenSourceEconomics/zurcher-data/"
    f"{SOURCE_COMMIT}/data/original_data/{RAW_FILENAME}"
)
RAW_SHA256 = "5e85a1c33c11632effbec3ffb213c8e4c92501a49dfe388ad28a203f8c732387"
RAW_BYTES = 42_624
RAW_VALUES = 128 * 37

BUS_COUNT = 37
PERIOD_COUNT = 117
OBSERVATION_COUNT = BUS_COUNT * PERIOD_COUNT
CHOICE_OBSERVATION_COUNT = BUS_COUNT * (PERIOD_COUNT - 1)
MILEAGE_BIN_SIZE = 5_000
NUM_STATES = 90
DISCOUNT_FACTOR = 0.9999
COST_SCALE = 0.001

CANONICAL_TRANSITION_COUNTS = (1_682, 2_555, 55)
CANONICAL_TRANSITION_PROBABILITIES = tuple(
    count / CHOICE_OBSERVATION_COUNT for count in CANONICAL_TRANSITION_COUNTS
)
CANONICAL_NFXP = (10.07494, 2.29309)

PRIOR_BOUNDS = ((4.0, 18.0), (0.2, 6.0))

