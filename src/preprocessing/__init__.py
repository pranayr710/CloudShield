"""
UNSW-NB15 preprocessing package — single source of truth for all three tracks.
23CSE301 ML Capstone.  Owner: Person A.

Design rules enforced across this package (rubric B1/B2 and guidelines 7.1):
  * Every scaler / encoder / clipper / level-whitelist is fitted on the TRAIN
    split ONLY. The fit and apply steps are separate functions so that a fit
    can only ever see training rows.
  * One stratified 80/20 split (random_state=42) is reused by all three tracks,
    so metric comparisons are made on identical rows.
  * Duplicate flow records are removed BEFORE splitting, because 42.6% of test
    rows would otherwise have an exact twin in train (measured, see
    DATA_AUDIT.md finding 3).

Module layout
-------------
config.py    every threshold, column list and target name, in one place
cleaning.py  load + clean (B1), then engineer features (B3)
pipeline.py  split, fold rare levels, clip, encode, scale (B2)

Usage
-----
    from src.preprocessing import get_dataset

    d = get_dataset("classification")
    X_train, y_train = d["X_train"], d["y_train"]

    # regression, with the five leakage columns removed and a log1p target
    d = get_dataset("regression", drop_leaky=True, log_target=True)

    # for the O(n^2) models; the TEST split is never subsampled
    d = get_dataset("classification", subsample=30_000)

Regenerate the processed parquet files with:
    python -m src.preprocessing
"""

from .cleaning import clean, engineer_features, load_raw
from .config import (
    CLUSTERING_FEATURES,
    LEAKY_FOR_SLOSS,
    RANDOM_STATE,
    TARGET_CLF,
    TARGET_REG,
    TEST_SIZE,
)
from .pipeline import export_processed, get_dataset, make_mock_dataset

__all__ = [
    # main entry point
    "get_dataset",
    # stage functions, for the EDA notebook to show its working
    "load_raw",
    "clean",
    "engineer_features",
    "export_processed",
    "make_mock_dataset",
    # constants worth importing into notebooks
    "RANDOM_STATE",
    "TEST_SIZE",
    "TARGET_REG",
    "TARGET_CLF",
    "LEAKY_FOR_SLOSS",
    "CLUSTERING_FEATURES",
]
