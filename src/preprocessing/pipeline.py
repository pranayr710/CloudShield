"""Stage 3: the fitted pipeline — split, fold rare levels, clip, encode, scale.

Everything in this module that learns a parameter is split into a `_fit_*`
function and an `_apply_*` function. The fit functions only ever receive the
training split, which makes the no-leakage guarantee structural rather than a
matter of remembering to do it right.

`get_dataset()` is the single entry point used by all three track notebooks, so
regression, classification and clustering physically cannot end up with
different splits.

Rubric: B2 (encoding, scaling, stratified split).
Owner: Person A.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .cleaning import clean, engineer_features, load_raw
from .config import (
    CATEGORICAL_COLS,
    CLIP_LOWER_Q,
    CLIP_UPPER_Q,
    CLUSTERING_FEATURES,
    LABEL_COLS,
    LEAKY_ENGINEERED,
    LEAKY_FOR_SLOSS,
    NO_CLIP_COLS,
    PROCESSED_DIR,
    PROTO_TOP_N,
    RANDOM_STATE,
    RARE_LEVEL_MIN_COUNT,
    TARGET_CLF,
    TARGET_REG,
    TEST_SIZE,
)


# --------------------------------------------------------------------------- #
# Fit-on-train-only transforms
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Fit-on-train-only transforms
# --------------------------------------------------------------------------- #

def _fit_level_whitelist(train_col: pd.Series, top_n: int | None = None,
                         min_count: int = RARE_LEVEL_MIN_COUNT) -> list:
    """Levels to keep for one categorical column, learned from TRAIN only.

    Everything else is folded into "other" so that no unseen level ever reaches
    the encoder at transform time.
    """
    vc = train_col.value_counts()
    keep = vc[vc >= min_count]
    if top_n is not None:
        keep = keep.head(top_n)
    return keep.index.tolist()


def _apply_level_whitelist(s: pd.Series, keep: list) -> pd.Series:
    return s.where(s.isin(keep), other="other")


def _fit_clip_bounds(X_train: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Percentile bounds from TRAIN only."""
    return pd.DataFrame({
        "lo": X_train[cols].quantile(CLIP_LOWER_Q),
        "hi": X_train[cols].quantile(CLIP_UPPER_Q),
    })


def _apply_clip(X: pd.DataFrame, bounds: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in bounds.index:
        X[c] = X[c].clip(bounds.loc[c, "lo"], bounds.loc[c, "hi"])
    return X


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_dataset(
    task: str,
    drop_leaky: bool = False,
    log_target: bool = False,
    subsample: int | None = None,
    random_state: int = RANDOM_STATE,
    verbose: bool = False,
) -> dict:
    """Return the fully preprocessed dataset for one track.

    Parameters
    ----------
    task : {"regression", "classification", "clustering"}
        "regression"     -> y = sloss (continuous count)
        "classification" -> y = attack_cat, label-encoded to 0..9
        "clustering"     -> y = None, labels removed from X entirely
    drop_leaky : bool
        Regression only. Drops the five columns algebraically tied to sloss
        (sbytes, spkts, smean, sload, rate) plus the two engineered features
        derived from them. Ignored for the other tasks.
    log_target : bool
        Regression only. Returns y = log1p(sloss) instead of sloss.
        RECOMMENDED as the primary framing: sloss is so heavy-tailed that
        just 10 test rows carry 62.7% of its total variance and 100 rows carry
        98.7%, so R2 on the raw target saturates near 0.999 for every
        tree-based model and the required "ranked by R2" comparison becomes
        uninformative. On log1p the same comparison separates model families
        cleanly (Linear 0.86 vs Random Forest 0.997). Report MAE back on the
        original packet scale with np.expm1(pred) for interpretability.
    subsample : int | None
        Stratified subsample size for the TRAIN split only, for the O(n^2)
        models (SVR, SVC, AgglomerativeClustering). The TEST split is never
        subsampled, so every model is scored on identical rows.
    random_state : int
    verbose : bool
        Print the cleaning and shape report.

    Returns
    -------
    dict with keys
      X_train, X_test : np.ndarray   encoded + scaled, no NaN
      y_train, y_test : np.ndarray | None
      feature_names   : list[str]    len == X_train.shape[1]
      label_encoder   : LabelEncoder | None
      preprocessor    : fitted ColumnTransformer  (for the Streamlit app)
      meta            : dict
    """
    if task not in {"regression", "classification", "clustering"}:
        raise ValueError(f"unknown task {task!r}")

    df, meta = clean(load_raw(), verbose=verbose)
    df = engineer_features(df)

    strata = df[TARGET_CLF]

    # ---------------- select y ----------------
    label_encoder = None
    if task == "regression":
        y = df[TARGET_REG].to_numpy(dtype=float)
        if log_target:
            y = np.log1p(y)
    elif task == "classification":
        label_encoder = LabelEncoder().fit(df[TARGET_CLF])
        y = label_encoder.transform(df[TARGET_CLF])
    else:
        y = None

    # ---------------- select X ----------------
    drop = list(LABEL_COLS)
    if task == "regression":
        drop.append(TARGET_REG)
        if drop_leaky:
            drop += LEAKY_FOR_SLOSS + LEAKY_ENGINEERED
    elif task == "clustering":
        # flow-behaviour statistics only; engineered ratios kept as they are
        # behavioural too, but volume/label columns excluded
        keep = [c for c in CLUSTERING_FEATURES if c in df.columns]
        keep += ["dst_loss_ratio", "ttl_diff", "jitter_ratio", "iat_ratio",
                 "no_dst_response", "conn_fanout"]
        X_df = df[keep].copy()

    if task != "clustering":
        X_df = df.drop(columns=[c for c in drop if c in df.columns])

    # ---------------- stratified split (identical rows for every track) -----
    idx = np.arange(len(df))
    if y is None:
        tr_idx, te_idx = train_test_split(
            idx, test_size=TEST_SIZE, stratify=strata, random_state=random_state
        )
        y_train = y_test = None
    else:
        tr_idx, te_idx, y_train, y_test = train_test_split(
            idx, y, test_size=TEST_SIZE, stratify=strata,
            random_state=random_state,
        )

    X_train_df = X_df.iloc[tr_idx].reset_index(drop=True)
    X_test_df = X_df.iloc[te_idx].reset_index(drop=True)

    # ---------------- rare-level folding, fitted on TRAIN only -------------
    # NOTE: this runs on the FULL train split, before any subsampling, so the
    # feature space is identical whether or not `subsample` is used.
    kept_levels = {}
    for col in CATEGORICAL_COLS:
        if col not in X_train_df.columns:
            continue
        top_n = PROTO_TOP_N if col == "proto" else None
        keep = _fit_level_whitelist(X_train_df[col], top_n=top_n)
        X_train_df[col] = _apply_level_whitelist(X_train_df[col], keep)
        X_test_df[col] = _apply_level_whitelist(X_test_df[col], keep)
        kept_levels[col] = keep
    meta["categorical_levels_kept"] = kept_levels

    # ---------------- outlier clipping, bounds from TRAIN only -------------
    num_cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()
    clip_cols = [c for c in num_cols if c not in NO_CLIP_COLS]
    bounds = _fit_clip_bounds(X_train_df, clip_cols)
    X_train_df = _apply_clip(X_train_df, bounds)
    X_test_df = _apply_clip(X_test_df, bounds)
    meta["n_clipped_cols"] = len(clip_cols)

    # ---------------- encode + scale, fit on TRAIN only --------------------
    cat_cols = [c for c in CATEGORICAL_COLS if c in X_train_df.columns]
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                  drop="first"), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    X_train = pre.fit_transform(X_train_df)      # fit on TRAIN only
    X_test = pre.transform(X_test_df)            # transform only
    feature_names = list(pre.get_feature_names_out())

    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)

    # ---------------- optional stratified TRAIN subsample ------------------
    # Applied AFTER fitting and transforming, so the feature space and the
    # test-set transform are byte-identical to a full run. Only the number of
    # training rows changes -- which is what the SVR/SVC footnote documents.
    meta["subsampled"] = False
    if subsample is not None and subsample < X_train.shape[0]:
        strat_train = strata.iloc[tr_idx].reset_index(drop=True)
        keep_idx, _ = train_test_split(
            np.arange(X_train.shape[0]),
            train_size=subsample,
            stratify=strat_train,
            random_state=random_state,
        )
        X_train = X_train[keep_idx]
        if y_train is not None:
            y_train = y_train[keep_idx]
        meta["subsampled"] = True
        meta["subsample_n"] = int(subsample)

    assert not np.isnan(X_train).any(), "NaN in X_train"
    assert not np.isnan(X_test).any(), "NaN in X_test"
    assert np.isfinite(X_train).all(), "non-finite in X_train"
    assert len(feature_names) == X_train.shape[1]

    meta.update({
        "task": task,
        "drop_leaky": drop_leaky,
        "log_target": log_target,
        "n_features": X_train.shape[1],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "random_state": random_state,
    })

    if verbose:
        print(f"task                    : {task} "
              f"(drop_leaky={drop_leaky}, log_target={log_target})")
        print(f"X_train / X_test        : {X_train.shape} / {X_test.shape}")
        print(f"features after encoding : {len(feature_names)}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "preprocessor": pre,
        "meta": meta,
    }


def export_processed() -> None:
    """Write the cleaned + engineered frame (pre-encoding) to data/processed/.

    Lets the EDA notebook and the other tracks inspect real, human-readable
    values without re-running the pipeline.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df, meta = clean(load_raw(), verbose=True)
    df = engineer_features(df)

    strata = df[TARGET_CLF]
    tr_idx, te_idx = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, stratify=strata,
        random_state=RANDOM_STATE,
    )
    df.iloc[tr_idx].reset_index(drop=True).to_parquet(
        os.path.join(PROCESSED_DIR, "train.parquet"), index=False)
    df.iloc[te_idx].reset_index(drop=True).to_parquet(
        os.path.join(PROCESSED_DIR, "test.parquet"), index=False)
    print(f"wrote train.parquet ({len(tr_idx):,} rows) and "
          f"test.parquet ({len(te_idx):,} rows) to data/processed/")


def make_mock_dataset(task, n=2000, n_features=40, random_state=RANDOM_STATE):
    """Fake data with the same output shape as get_dataset().

    Kept for reference now that get_dataset() is implemented; Person B and
    Person C should set USE_MOCK = False.
    """
    rng = np.random.default_rng(random_state)
    n_tr = int(n * (1 - TEST_SIZE))
    X_train = rng.normal(size=(n_tr, n_features))
    X_test = rng.normal(size=(n - n_tr, n_features))
    names = [f"f{i}" for i in range(n_features)]
    if task == "regression":
        y_train = np.abs(X_train[:, 0] * 5 + rng.normal(size=n_tr))
        y_test = np.abs(X_test[:, 0] * 5 + rng.normal(size=n - n_tr))
        le = None
    elif task == "classification":
        y_train = rng.integers(0, 10, n_tr)
        y_test = rng.integers(0, 10, n - n_tr)
        le = LabelEncoder().fit(
            ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
             "Normal", "Reconnaissance", "Shellcode", "Worms"])
    else:
        y_train = y_test = le = None
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train,
            "y_test": y_test, "feature_names": names, "label_encoder": le,
            "preprocessor": None, "meta": {"mock": True}}

