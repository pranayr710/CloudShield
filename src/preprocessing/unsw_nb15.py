"""CloudShield Track 2 — UNSW-NB15 preprocessing for the classification task.

Shared by all classification models (Members 1 and 2), so every classifier is
trained on identical rows with identical encoding. This is what makes the
consolidated 10-model table in spec section 9 a fair comparison.

Leakage rules enforced here (spec section 8):
  * `attack_cat` and `label` never appear in X. `label` is a deterministic
    function of `attack_cat`, so leaving it in would hand any model the
    benign/malicious boundary for free.
  * Every transform that learns a parameter - the rare-level whitelist, the
    outlier clip bounds, the one-hot categories, the scaler - is fitted on the
    TRAIN split only and merely applied to the test split.
  * Duplicate flow records are removed BEFORE the split.

Owner: shared module. Do not fork it per member.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from ..utils.config import (
    CLF_CATEGORICAL,
    CLF_DROP_ALWAYS,
    CLF_DROP_REDUNDANT,
    CLF_LABEL_COLS,
    CLF_NO_CLIP,
    CLF_TARGET,
    CLIP_HIGH,
    CLIP_LOW,
    EPS,
    PROTO_TOP_N,
    RANDOM_STATE,
    RARE_LEVEL_MIN,
    RAW_UNSW,
    TEST_SIZE,
    UNSW_FILES,
)


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load_raw() -> pd.DataFrame:
    """Concatenate both partition files.

    The official distribution ships the two files with their names swapped:
    UNSW_NB15_training-set.csv holds 82,332 rows and the testing file holds
    175,341, the reverse of the split published in Moustafa & Slay (2015). We
    therefore ignore the supplied partition and build our own stratified split,
    which also removes the distribution shift baked into the official one.
    """
    missing = [f for f in UNSW_FILES if not (RAW_UNSW / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing UNSW-NB15 partition files in {RAW_UNSW}: {missing}\n"
            "Run: python data/download_unsw_nb15.py"
        )
    frames = [pd.read_csv(RAW_UNSW / f) for f in UNSW_FILES]
    df = pd.concat(frames, ignore_index=True)
    if df.shape[1] != 45:
        warnings.warn(
            f"Expected 45 columns, found {df.shape[1]}. You may have the "
            "49-column raw dump rather than the partition files."
        )
    return df


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #

def clean(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """Drop identifiers and redundant columns, handle missing values, dedupe."""
    meta: dict = {"n_rows_raw": len(df)}
    df = df.copy()

    df = df.drop(columns=[c for c in CLF_DROP_ALWAYS + CLF_DROP_REDUNDANT
                          if c in df.columns])

    n_nan = int(df.isna().sum().sum())
    numeric = df.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(numeric.to_numpy()).sum())
    meta["n_nan"], meta["n_inf"] = n_nan, n_inf

    # Stated strategy, applied only if the data ever changes: inf can only come
    # from a rate division, so treat it as missing, then median/mode impute.
    # The curated partition files contain neither, which the notebook verifies
    # rather than assumes.
    if n_nan or n_inf:
        df = df.replace([np.inf, -np.inf], np.nan)
        for c in df.columns:
            if df[c].isna().any():
                fill = (df[c].median() if pd.api.types.is_numeric_dtype(df[c])
                        else df[c].mode()[0])
                df[c] = df[c].fillna(fill)
        warnings.warn(f"Imputed {n_nan} NaN and {n_inf} inf values.")

    for c in CLF_CATEGORICAL + [CLF_TARGET]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip().replace("", "Normal")

    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    meta["n_duplicates_dropped"] = n_before - len(df)
    meta["n_rows_clean"] = len(df)

    if verbose:
        pct = meta["n_duplicates_dropped"] / n_before * 100
        print(f"raw rows               : {meta['n_rows_raw']:,}")
        print(f"NaN / inf found        : {n_nan} / {n_inf}")
        print(f"duplicate rows dropped : {meta['n_duplicates_dropped']:,} ({pct:.1f}%)")
        print(f"rows after cleaning    : {meta['n_rows_clean']:,}")
    return df, meta


# --------------------------------------------------------------------------- #
# Feature engineering  (spec section 11 - source columns verified to exist)
# --------------------------------------------------------------------------- #

ENGINEERED = [
    "dst_loss_ratio", "ttl_diff", "jitter_ratio", "iat_ratio",
    "no_dst_response", "conn_fanout", "pkt_dir_ratio", "byte_dir_asymmetry",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Eight row-wise features, each with a cybersecurity interpretation.

    All are functions of a single row, so they can be computed before the split
    without leaking any cross-row statistic.

    Deliberately excluded, because both are degenerate on this dataset:
      * sbytes / spkts       -- already present as `smean`, identical in 100%
                                of rows.
      * (synack+ackdat)/tcprtt -- identically 1.0, since tcprtt is *defined* as
                                synack + ackdat.
    """
    f = df.copy()

    # loss as a fraction of volume, so small and large flows are comparable
    f["dst_loss_ratio"] = f["dloss"] / (f["dpkts"] + EPS)
    # spoofed/crafted packets carry TTLs inconsistent with the return path
    f["ttl_diff"] = f["sttl"] - f["dttl"]
    # one-sided jitter indicates directional congestion or queue exhaustion
    f["jitter_ratio"] = f["sjit"] / (f["djit"] + EPS)
    # automated tooling emits on a fixed cadence; victims reply irregularly
    f["iat_ratio"] = f["sinpkt"] / (f["dinpkt"] + EPS)
    # ~19% of flows never get a reply: a discontinuity, not a continuum
    f["no_dst_response"] = (f["dpkts"] == 0).astype(int)
    # recon touches many services on one host; a flood repeats one service
    f["conn_fanout"] = f["ct_srv_src"] / (f["ct_dst_ltm"] + EPS)
    # directional imbalance: exfiltration skews down, amplification skews up
    f["pkt_dir_ratio"] = f["spkts"] / (f["dpkts"] + EPS)
    f["byte_dir_asymmetry"] = ((f["sbytes"] - f["dbytes"]).abs()
                               / (f["sbytes"] + f["dbytes"] + EPS))
    return f


# --------------------------------------------------------------------------- #
# Fit-on-train-only transforms
# --------------------------------------------------------------------------- #

def _fit_level_whitelist(train_col, top_n=None, min_count=RARE_LEVEL_MIN):
    counts = train_col.value_counts()
    keep = counts[counts >= min_count]
    if top_n is not None:
        keep = keep.head(top_n)
    return keep.index.tolist()


def _apply_level_whitelist(col, keep):
    return col.where(col.isin(keep), other="other")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_classification_data(
    binary: bool = False,
    subsample: int | None = None,
    random_state: int = RANDOM_STATE,
    return_frames: bool = False,
    verbose: bool = False,
) -> dict:
    """Return the preprocessed UNSW-NB15 classification dataset.

    Parameters
    ----------
    binary : bool
        False (default) -> multiclass on attack_cat, the primary task.
        True            -> optional secondary Normal vs Malicious experiment.
    subsample : int | None
        Stratified subsample of the TRAIN split only, for the O(n^2) models
        (SVM). The TEST split is never subsampled, so every model is scored on
        identical rows and the comparison stays fair.
    return_frames : bool
        Additionally return the pre-encoding train/test DataFrames as
        X_train_df / X_test_df. Needed for analyses that require the ORIGINAL
        feature scale, such as demonstrating what happens to a distance-based
        model without standardisation.

    Returns
    -------
    dict with X_train, X_test, y_train, y_test, feature_names, label_encoder,
    preprocessor, meta.
    """
    df, meta = clean(load_raw(), verbose=verbose)
    df = engineer_features(df)

    strata = df[CLF_TARGET]

    if binary:
        y_all = df["label"].to_numpy()
        label_encoder = None
        class_names = ["Normal", "Malicious"]
    else:
        label_encoder = LabelEncoder().fit(df[CLF_TARGET])
        y_all = label_encoder.transform(df[CLF_TARGET])
        class_names = list(label_encoder.classes_)

    X_df = df.drop(columns=[c for c in CLF_LABEL_COLS if c in df.columns])

    # ---- stratified split -------------------------------------------------
    idx = np.arange(len(df))
    tr_idx, te_idx, y_train, y_test = train_test_split(
        idx, y_all, test_size=TEST_SIZE, stratify=strata,
        random_state=random_state,
    )
    X_train_df = X_df.iloc[tr_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[te_idx].reset_index(drop=True)

    # ---- rare categorical levels, whitelist from TRAIN only ---------------
    kept_levels = {}
    for col in CLF_CATEGORICAL:
        if col not in X_train_df.columns:
            continue
        keep = _fit_level_whitelist(
            X_train_df[col], top_n=PROTO_TOP_N if col == "proto" else None)
        X_train_df[col] = _apply_level_whitelist(X_train_df[col], keep)
        X_test_df[col] = _apply_level_whitelist(X_test_df[col], keep)
        kept_levels[col] = keep
    meta["categorical_levels_kept"] = kept_levels

    # ---- outlier clipping, bounds from TRAIN only -------------------------
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    clip_cols = [c for c in num_cols if c not in CLF_NO_CLIP]
    bounds = pd.DataFrame({
        "lo": X_train_df[clip_cols].quantile(CLIP_LOW),
        "hi": X_train_df[clip_cols].quantile(CLIP_HIGH),
    })
    for c in clip_cols:
        X_train_df[c] = X_train_df[c].clip(bounds.loc[c, "lo"], bounds.loc[c, "hi"])
        X_test_df[c] = X_test_df[c].clip(bounds.loc[c, "lo"], bounds.loc[c, "hi"])
    meta["n_clipped_cols"] = len(clip_cols)
    meta["clip_bounds"] = bounds

    # ---- encode + scale, fitted on TRAIN only -----------------------------
    cat_cols = [c for c in CLF_CATEGORICAL if c in X_train_df.columns]
    pre = ColumnTransformer(
        [("num", StandardScaler(), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                               drop="first"), cat_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )
    X_train = np.asarray(pre.fit_transform(X_train_df), dtype=np.float64)
    X_test = np.asarray(pre.transform(X_test_df), dtype=np.float64)
    feature_names = list(pre.get_feature_names_out())

    # ---- optional TRAIN subsample, applied AFTER fit/transform ------------
    # so the feature space and the test matrix are byte-identical to a full run
    meta["subsampled"] = False
    if subsample is not None and subsample < X_train.shape[0]:
        strat_train = strata.iloc[tr_idx].reset_index(drop=True)
        keep_idx, _ = train_test_split(
            np.arange(X_train.shape[0]), train_size=subsample,
            stratify=strat_train, random_state=random_state)
        X_train, y_train = X_train[keep_idx], y_train[keep_idx]
        meta["subsampled"] = True
        meta["subsample_n"] = int(subsample)

    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isnan(X_test).any(), "NaN in X_test"
    assert len(feature_names) == X_train.shape[1]
    assert not any(c in feature_names for c in CLF_LABEL_COLS), "label leaked"

    meta.update({
        "task": "binary" if binary else "multiclass",
        "n_features": X_train.shape[1],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "random_state": random_state,
    })

    if verbose:
        print(f"task                   : {meta['task']}")
        print(f"X_train / X_test       : {X_train.shape} / {X_test.shape}")
        print(f"features after encoding: {len(feature_names)}")

    out = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "preprocessor": pre,
        "meta": meta,
    }
    if return_frames:
        out["X_train_df"] = X_train_df
        out["X_test_df"] = X_test_df
        out["numeric_columns"] = num_cols
    return out


if __name__ == "__main__":
    d = get_classification_data(verbose=True)
    print("classes:", d["class_names"])
