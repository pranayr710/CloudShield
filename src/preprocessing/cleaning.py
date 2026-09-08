"""Stage 1 and 2 of the pipeline: load + clean, then engineer features.

Nothing in this module looks at the train/test split, because every operation
here is either a whole-dataset structural decision (dropping identifiers,
removing duplicate rows) or a strictly row-wise transform (the engineered
features). Anything that has to be *fitted* lives in pipeline.py instead.

Rubric: B1 (cleaning) and B3 (feature engineering).
Owner: Person A.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from .config import (
    CATEGORICAL_COLS,
    EXACT_DUPLICATE_COLS,
    ID_COLS,
    PARTITION_FILES,
    RAW_DIR_CANDIDATES,
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _raw_dir() -> str:
    for d in RAW_DIR_CANDIDATES:
        if all(os.path.exists(os.path.join(d, f)) for f in PARTITION_FILES):
            return d
    raise FileNotFoundError(
        "Could not find the UNSW-NB15 partition CSVs. Expected both of\n"
        f"  {PARTITION_FILES}\nin one of:\n  "
        + "\n  ".join(RAW_DIR_CANDIDATES)
    )


def load_raw() -> pd.DataFrame:
    """Concatenate both partition files into one frame.

    The official distribution ships the two files with their names swapped:
    UNSW_NB15_training-set.csv holds 82,332 rows and
    UNSW_NB15_testing-set.csv holds 175,341 rows, the reverse of the partition
    described in Moustafa & Slay (2015). We therefore ignore the supplied
    partition entirely and build our own stratified split, which also removes
    the distribution shift baked into the official one.
    """
    d = _raw_dir()
    frames = [pd.read_csv(os.path.join(d, f)) for f in PARTITION_FILES]
    df = pd.concat(frames, ignore_index=True)
    if df.shape != (257673, 45):
        warnings.warn(
            f"Expected combined shape (257673, 45), got {df.shape}. "
            "You may have a different variant of UNSW-NB15."
        )
    return df


# --------------------------------------------------------------------------- #
# Cleaning  (rubric B1)
# --------------------------------------------------------------------------- #

def clean(df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, dict]:
    """Drop identifiers, redundant columns and duplicate flow records.

    No missing-value imputation is required: the partition files contain zero
    NaN and zero inf values in all 45 columns (unlike the raw UNSW-NB15_1..4
    dumps, which do). This is asserted rather than assumed.
    """
    meta: dict = {"n_rows_raw": len(df)}
    df = df.copy()

    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])
    df = df.drop(columns=[c for c in EXACT_DUPLICATE_COLS if c in df.columns])

    # --- missing / non-finite: assert the audit finding still holds ---
    n_nan = int(df.isna().sum().sum())
    num = df.select_dtypes(include=[np.number])
    n_inf = int(np.isinf(num.to_numpy()).sum())
    meta["n_nan"] = n_nan
    meta["n_inf"] = n_inf
    if n_nan or n_inf:
        # Justified strategy, applied only if the data ever changes: inf is a
        # rate-computation artefact -> treat as missing, then median-impute.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.median(numeric_only=True))
        warnings.warn(f"Imputed {n_nan} NaN and {n_inf} inf values.")

    # --- trim whitespace on categoricals (raw dumps have padded values) ---
    for c in CATEGORICAL_COLS + ["attack_cat"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = df[c].str.strip()

    # --- duplicates: the single most consequential cleaning decision ---
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    meta["n_duplicates_dropped"] = n_before - len(df)
    meta["n_rows_clean"] = len(df)

    if verbose:
        print(f"raw rows                : {meta['n_rows_raw']:,}")
        print(f"NaN / inf found         : {n_nan} / {n_inf}")
        print(f"duplicate rows dropped  : {meta['n_duplicates_dropped']:,} "
              f"({meta['n_duplicates_dropped'] / n_before * 100:.1f}%)")
        print(f"rows after cleaning     : {meta['n_rows_clean']:,}")

    return df, meta


# --------------------------------------------------------------------------- #
# Feature engineering  (rubric B3)
# --------------------------------------------------------------------------- #

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add six domain-motivated features.

    All are row-wise functions of existing columns, so they can be computed
    before the split without leaking any cross-row statistic.

    Deliberately NOT included, because both turn out to be degenerate on this
    dataset (see DATA_AUDIT.md finding 4):
      * bytes-per-source-packet -- already present as `smean`, exactly.
      * (synack + ackdat) / tcprtt -- identically 1.0, since tcprtt is defined
        as synack + ackdat and matches in 100% of rows.
    """
    df = df.copy()
    eps = 1e-6

    # 1. Destination-side loss fraction. Normalises loss by flow volume so a
    #    10-packet flow and a 10,000-packet flow are comparable. Destination
    #    side only -- the source-side equivalent would divide the regression
    #    target by one of its own predictors.
    df["dst_loss_ratio"] = df["dloss"] / (df["dpkts"] + eps)

    # 2. TTL asymmetry. Crafted and spoofed packets (Fuzzers, Exploits,
    #    Shellcode) carry TTLs inconsistent with the return path, so the
    #    forward/reverse difference is a classic IDS heuristic. sttl has only
    #    13 distinct values and dttl 9, so the difference is a compact
    #    categorical-like signal.
    df["ttl_diff"] = df["sttl"] - df["dttl"]

    # 3. Jitter asymmetry. One-sided jitter indicates congestion or queue
    #    exhaustion in a single direction -- the direct mechanism behind QoS
    #    degradation, which is what the regression track predicts.
    df["jitter_ratio"] = df["sjit"] / (df["djit"] + eps)

    # 4. Inter-arrival asymmetry. Automated tooling emits packets on a fixed
    #    cadence while the victim replies irregularly, so the ratio separates
    #    machine-generated from interactive traffic.
    df["iat_ratio"] = df["sinpkt"] / (df["dinpkt"] + eps)

    # 5. No-response flag. 19.4% of flows have dpkts == 0 -- a structurally
    #    distinct population (unanswered probes, scans, backscatter) rather
    #    than a continuum. Making it explicit lets linear models express the
    #    discontinuity that trees would otherwise have to spend splits on.
    df["no_dst_response"] = (df["dpkts"] == 0).astype(int)

    # 6. Connection-fan-out intensity. Reconnaissance touches many services on
    #    one host; a flood repeats one service. Ratio of same-service count to
    #    same-destination count separates the two shapes.
    df["conn_fanout"] = df["ct_srv_src"] / (df["ct_dst_ltm"] + eps)

    # --- two more that are informative for classification but are algebraic
    #     descendants of the leaky columns, so drop_leaky removes them too ---
    df["pkt_dir_ratio"] = df["spkts"] / (df["dpkts"] + eps)
    df["byte_dir_asymmetry"] = (
        (df["sbytes"] - df["dbytes"]).abs() / (df["sbytes"] + df["dbytes"] + eps)
    )

    return df


ENGINEERED_COLS = [
    "dst_loss_ratio", "ttl_diff", "jitter_ratio", "iat_ratio",
    "no_dst_response", "conn_fanout", "pkt_dir_ratio", "byte_dir_asymmetry",
]
