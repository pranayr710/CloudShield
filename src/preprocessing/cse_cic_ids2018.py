"""CloudShield Track 1 — CSE-CIC-IDS2018 preprocessing for the regression task.

Target: network byte rate. The spec names it `fl_byt_s`; the CSE-CIC-IDS2018
processed CSVs published by CIC normally use "Flow Byts/s". This module does
NOT pick one. It inspects the columns that are actually present and resolves
the target against a list of known aliases, raising a descriptive error if none
matches, rather than silently substituting a different column.

Nothing in this module runs on assumed schema. Every column list is filtered by
`if c in df.columns` before use.

Owner: shared module (Members 1, 2 and 3 all train regressors on its output).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utils.config import (
    CLIP_HIGH,
    CLIP_LOW,
    EPS,
    RANDOM_STATE,
    RAW_IDS2018,
    REG_DROP_ALWAYS,
    REG_DROP_LEAKY_CANDIDATES,
    REG_TARGET_ALIASES,
    TEST_SIZE,
)


class DatasetNotAvailable(FileNotFoundError):
    """Raised when the CSE-CIC-IDS2018 CSVs are not present locally."""


class TargetColumnNotFound(KeyError):
    """Raised when no known byte-rate alias exists in the loaded data."""


# --------------------------------------------------------------------------- #
# Inspection - always run this before any modelling code
# --------------------------------------------------------------------------- #

def available_files() -> list[Path]:
    return sorted(RAW_IDS2018.glob("*.csv"))


def require_dataset() -> list[Path]:
    files = available_files()
    if not files:
        raise DatasetNotAvailable(
            f"No CSV files found in {RAW_IDS2018}.\n\n"
            "CSE-CIC-IDS2018 has not been downloaded yet. Fetch it with:\n"
            "    python data/download_cse_cic_ids2018.py\n\n"
            "Until then the regression track cannot be run, and no regression "
            "results should be reported."
        )
    return files


def inspect(nrows: int = 5000) -> pd.DataFrame:
    """Report the real schema of every available file.

    Reads only the first `nrows` of each file, so it is cheap on a 6 GB
    dataset. Run this before writing any dataset-specific code.
    """
    rows = []
    for path in require_dataset():
        head = pd.read_csv(path, nrows=nrows, low_memory=False)
        head.columns = [c.strip() for c in head.columns]
        target = next((a for a in REG_TARGET_ALIASES if a in head.columns), None)
        rows.append({
            "file": path.name,
            "size_MB": round(path.stat().st_size / 1_048_576, 1),
            "n_columns": head.shape[1],
            "target_alias_found": target or "NONE",
            "has_Label": "Label" in head.columns,
        })
    return pd.DataFrame(rows)


def resolve_target(columns) -> str:
    """Return the byte-rate column actually present, or raise.

    Never substitutes a different column - spec section 3, DATASET 1.
    """
    cols = [str(c).strip() for c in columns]
    for alias in REG_TARGET_ALIASES:
        if alias in cols:
            return alias
    raise TargetColumnNotFound(
        "None of the known byte-rate aliases were found in the data.\n"
        f"  looked for : {REG_TARGET_ALIASES}\n"
        f"  columns are: {cols}\n\n"
        "Stopping rather than substituting a different column. Inspect the "
        "file and add the correct name to REG_TARGET_ALIASES in "
        "src/utils/config.py if the dataset uses another spelling."
    )


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_raw(files: list[str] | None = None,
             nrows_per_file: int | None = None,
             verbose: bool = True) -> pd.DataFrame:
    """Load and concatenate the selected CSE-CIC-IDS2018 CSVs.

    Parameters
    ----------
    files : list[str] | None
        Filenames to load. None loads every CSV present. Each day's capture is
        a separate file, so which files are used must be documented in the
        notebook (spec step 7).
    nrows_per_file : int | None
        Row cap per file. The full dataset is ~16M rows; if a cap is used it
        MUST be reported in the notebook alongside the reason (spec section 17).
    """
    paths = require_dataset()
    if files:
        wanted = set(files)
        paths = [p for p in paths if p.name in wanted]
        if not paths:
            raise DatasetNotAvailable(f"None of {files} found in {RAW_IDS2018}")

    frames = []
    for p in paths:
        df = pd.read_csv(p, nrows=nrows_per_file, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        # some CIC files repeat the header mid-file; those rows parse as the
        # literal header text and must be removed before numeric coercion
        if "Label" in df.columns:
            df = df[df["Label"] != "Label"]
        frames.append(df)
        if verbose:
            print(f"  loaded {p.name:52s} {df.shape[0]:>9,} rows x {df.shape[1]} cols")

    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  {'COMBINED':52s} {out.shape[0]:>9,} rows x {out.shape[1]} cols")
    return out


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #

def clean(df: pd.DataFrame, target: str, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Drop identifiers, coerce numerics, handle inf/NaN, dedupe.

    CSE-CIC-IDS2018 genuinely contains +/-inf in its rate columns (a division by
    a zero-length flow window), unlike the curated UNSW-NB15 partition files.
    """
    meta: dict = {"n_rows_raw": len(df)}
    df = df.copy()

    dropped = [c for c in REG_DROP_ALWAYS if c in df.columns]
    df = df.drop(columns=dropped)
    meta["identifier_columns_dropped"] = dropped

    # coerce everything except the label to numeric; CIC files often carry
    # numeric columns as object dtype because of the repeated-header rows
    for c in df.columns:
        if c != "Label":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    n_inf = int(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum())
    df = df.replace([np.inf, -np.inf], np.nan)
    n_nan = int(df.isna().sum().sum())
    meta["n_inf"], meta["n_nan"] = n_inf, n_nan

    # rows with a missing TARGET cannot be used for regression at all
    n_before = len(df)
    df = df[df[target].notna()]
    meta["rows_dropped_missing_target"] = n_before - len(df)

    # remaining gaps are imputed rather than dropped, to preserve attack rows
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # impossible values: a negative duration or count is a parsing artefact
    for c in [c for c in ("Flow Duration", "Fwd IAT Tot", "Bwd IAT Tot")
              if c in df.columns]:
        neg = int((df[c] < 0).sum())
        if neg:
            meta[f"negative_{c}"] = neg
            df = df[df[c] >= 0]

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    meta["n_duplicates_dropped"] = n_before - len(df)
    meta["n_rows_clean"] = len(df)

    if verbose:
        for k, v in meta.items():
            print(f"  {k:34s} {v}")
    return df, meta


# --------------------------------------------------------------------------- #
# Feature engineering  (only where source columns actually exist)
# --------------------------------------------------------------------------- #

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add domain features whose source columns are present.

    Spec section 11: a feature is implemented only when its source columns
    exist. Each block below is guarded, and the function reports which features
    it was actually able to build.
    """
    f = df.copy()
    added: list[str] = []

    def have(*cols):
        return all(c in f.columns for c in cols)

    # SYN floods drive half-open connections, exhausting the VM connection table
    if have("SYN Flag Cnt", "FIN Flag Cnt", "ACK Flag Cnt"):
        f["syn_flood_ratio"] = f["SYN Flag Cnt"] / (
            f["SYN Flag Cnt"] + f["FIN Flag Cnt"] + f["ACK Flag Cnt"] + EPS)
        added.append("syn_flood_ratio")

    # uniform large packets (flood) vs mixed sizes (interactive / botnet)
    if have("Pkt Len Max", "Pkt Len Mean"):
        f["burst_index"] = f["Pkt Len Max"] / (f["Pkt Len Mean"] + EPS)
        added.append("burst_index")

    # periodic idle->active cycling is the signature of C&C polling
    if have("Idle Mean", "Active Mean"):
        f["idle_active_ratio"] = f["Idle Mean"] / (f["Active Mean"] + EPS)
        added.append("idle_active_ratio")

    # exfiltration skews download, amplification skews upload, browsing balances
    if have("Tot Fwd Pkts", "Tot Bwd Pkts"):
        f["dir_asymmetry"] = ((f["Tot Fwd Pkts"] - f["Tot Bwd Pkts"]).abs()
                              / (f["Tot Fwd Pkts"] + f["Tot Bwd Pkts"] + EPS))
        added.append("dir_asymmetry")

    return f, added


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_regression_data(
    files: list[str] | None = None,
    nrows_per_file: int | None = None,
    log_target: bool = False,
    subsample: int | None = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> dict:
    """Return the preprocessed CSE-CIC-IDS2018 regression dataset.

    Raises DatasetNotAvailable if the CSVs are absent, and
    TargetColumnNotFound if no known byte-rate alias is present. Neither is
    worked around silently.
    """
    if verbose:
        print("Loading CSE-CIC-IDS2018 ...")
    df = load_raw(files=files, nrows_per_file=nrows_per_file, verbose=verbose)

    target = resolve_target(df.columns)
    if verbose:
        print(f"\nresolved regression target -> '{target}'")

    df, meta = clean(df, target=target, verbose=verbose)
    df, engineered = engineer_features(df)
    meta["engineered_features"] = engineered
    meta["target"] = target
    if verbose:
        print(f"\nengineered features built: {engineered or 'none - source columns absent'}")

    y_all = df[target].to_numpy(dtype=float)
    if log_target:
        y_all = np.log1p(np.clip(y_all, 0, None))

    # the target itself and any near-duplicate rate measure leave X
    drop = [target, "Label"]
    drop += [c for c in REG_DROP_LEAKY_CANDIDATES if c in df.columns]
    X_df = df.drop(columns=[c for c in drop if c in df.columns])
    meta["leakage_columns_removed"] = [c for c in drop if c in df.columns]

    strata = df["Label"] if "Label" in df.columns else None

    idx = np.arange(len(df))
    tr_idx, te_idx, y_train, y_test = train_test_split(
        idx, y_all, test_size=TEST_SIZE, stratify=strata,
        random_state=random_state)

    X_train_df = X_df.iloc[tr_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[te_idx].reset_index(drop=True)

    num_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train_df.columns if c not in num_cols]

    # clip bounds from TRAIN only
    bounds = pd.DataFrame({"lo": X_train_df[num_cols].quantile(CLIP_LOW),
                           "hi": X_train_df[num_cols].quantile(CLIP_HIGH)})
    for c in num_cols:
        lo, hi = bounds.loc[c, "lo"], bounds.loc[c, "hi"]
        if lo == hi:          # constant column: clipping would destroy it
            continue
        X_train_df[c] = X_train_df[c].clip(lo, hi)
        X_test_df[c] = X_test_df[c].clip(lo, hi)

    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                               drop="first"), cat_cols)],
        remainder="drop", verbose_feature_names_out=False)
    X_train = np.asarray(pre.fit_transform(X_train_df), dtype=np.float64)
    X_test = np.asarray(pre.transform(X_test_df), dtype=np.float64)
    feature_names = list(pre.get_feature_names_out())

    meta["subsampled"] = False
    if subsample is not None and subsample < X_train.shape[0]:
        keep = np.random.default_rng(random_state).choice(
            X_train.shape[0], subsample, replace=False)
        X_train, y_train = X_train[keep], y_train[keep]
        meta["subsampled"] = True
        meta["subsample_n"] = int(subsample)

    assert target not in feature_names, "regression target leaked into X"
    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert len(feature_names) == X_train.shape[1]

    meta.update({"n_features": X_train.shape[1], "n_train": X_train.shape[0],
                 "n_test": X_test.shape[0], "log_target": log_target,
                 "random_state": random_state})

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names, "target": target,
        "preprocessor": pre, "meta": meta,
    }


if __name__ == "__main__":
    try:
        print(inspect().to_string(index=False))
    except DatasetNotAvailable as exc:
        print(exc)
