"""Main preprocessing pipeline - single source of truth for all tracks.

This module provides get_dataset() which returns preprocessed data for
regression, classification, or clustering tracks. All tracks use the SAME
train/test split and preprocessing to ensure consistency.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import RANDOM_STATE, TEST_SIZE, LEAKY_COLS, LABEL_COLS, CAT_COLS, EPS
from .cleaning import load_raw, drop_columns, handle_missing, remove_duplicates, clip_outliers
from .features import engineer_features
from .split_scale import preprocess_categorical, create_split, encode_and_scale


def get_dataset(
    task: str,
    drop_leaky: bool = False,
    log_target: bool = False,
    subsample: int = None,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Return the fully preprocessed dataset for one track.
    
    Parameters
    ----------
    task : {"regression", "classification", "clustering"}
        Which track to prepare data for.
    drop_leaky : bool
        Regression only - drop columns algebraically tied to sloss.
    log_target : bool
        Regression only - use log1p(sloss) as target instead of raw sloss.
    subsample : int | None
        Stratified subsample of the TRAIN set (for SVR/SVC/Agglomerative).
    random_state : int
        
    Returns
    -------
    dict with keys:
        X_train, X_test : np.ndarray, encoded + scaled (scaler fit on train ONLY)
        y_train, y_test : np.ndarray | None (None for clustering)
        feature_names   : list[str], len == X_train.shape[1]
        label_encoder   : sklearn LabelEncoder | None
        meta            : dict with n_rows_before_clean, n_dropped, subsampled, etc.
    """
    assert task in ("regression", "classification", "clustering"), \
        f"task must be 'regression', 'classification', or 'clustering', got '{task}'"
    
    # === 1. Load raw data ===
    df = load_raw()
    n_rows_before = len(df)
    
    # === 2. Drop id and ct_ftp_cmd ===
    df = drop_columns(df)
    
    # === 3. Handle missing values ===
    df, missing_report = handle_missing(df)
    
    # === 4. Remove duplicates ===
    df, dup_report = remove_duplicates(df)
    
    # === 5. Normalize attack_cat ===
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    df["attack_cat"] = df["attack_cat"].replace("", "Normal")
    
    # === 6. Feature engineering ===
    df = engineer_features(df, drop_leaky=drop_leaky)
    
    # === 7. Drop leaky columns if requested ===
    if drop_leaky:
        df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns], errors="ignore")
    
    # === 8. Preprocess categoricals (fold rare levels) ===
    df = preprocess_categorical(df, CAT_COLS)
    
    # === 9. Create stratified split (BEFORE dropping labels) ===
    train_df, test_df = create_split(df, stratify_col="attack_cat")
    
    # === 10. Clip outliers (using train set bounds) ===
    train_df, clip_report_train = clip_outliers(train_df, list(range(len(train_df))))
    test_df, clip_report_test = clip_outliers(test_df, list(range(len(test_df))))
    
    # === 11. Prepare targets and drop label columns ===
    label_encoder = None
    
    if task == "regression":
        y_train_raw = train_df["sloss"].values.astype(float)
        y_test_raw = test_df["sloss"].values.astype(float)
        
        if log_target:
            y_train = np.log1p(y_train_raw)
            y_test = np.log1p(y_test_raw)
        else:
            y_train = y_train_raw
            y_test = y_test_raw
        
        # Drop label columns AND regression targets from features
        drop_cols = LABEL_COLS + ["sloss", "dloss"]
        train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")
        
    elif task == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(train_df["attack_cat"])
        y_test = le.transform(test_df["attack_cat"])
        label_encoder = le
        
        # Drop label columns from features
        train_df = train_df.drop(columns=[c for c in LABEL_COLS if c in train_df.columns], errors="ignore")
        test_df = test_df.drop(columns=[c for c in LABEL_COLS if c in test_df.columns], errors="ignore")
        
    else:  # clustering
        y_train = None
        y_test = None
        # Drop label columns from features
        train_df = train_df.drop(columns=[c for c in LABEL_COLS if c in train_df.columns], errors="ignore")
        test_df = test_df.drop(columns=[c for c in LABEL_COLS if c in test_df.columns], errors="ignore")
    
    # === 12. Encode and scale ===
    cat_cols_present = [c for c in CAT_COLS if c in train_df.columns]
    num_cols = [c for c in train_df.columns if c not in cat_cols_present]
    
    X_train, X_test, feature_names, scaler, encoder = encode_and_scale(
        train_df, test_df, cat_cols_present, num_cols
    )
    
    # === 13. Subsample if requested ===
    subsampled = False
    if subsample is not None and subsample < len(X_train):
        if task == "clustering" or y_train is None:
            # Random subsample for clustering
            rng = np.random.default_rng(random_state)
            sub_idx = rng.choice(len(X_train), size=subsample, replace=False)
        elif task == "classification":
            # Stratified subsample for classification
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - subsample/len(X_train), 
                                          random_state=random_state)
            sub_idx, _ = next(sss.split(X_train, y_train))
        else:
            # Regression: bin the target for stratification, then subsample
            from sklearn.model_selection import StratifiedShuffleSplit
            # Create bins for stratification (quantile-based)
            n_bins = min(10, len(np.unique(y_train)))
            y_binned = pd.qcut(y_train, q=n_bins, labels=False, duplicates='drop')
            sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - subsample/len(X_train), 
                                          random_state=random_state)
            sub_idx, _ = next(sss.split(X_train, y_binned))
        
        X_train = X_train[sub_idx]
        if y_train is not None:
            y_train = y_train[sub_idx]
        subsampled = True
    
    # === 14. Build metadata ===
    meta = {
        "task": task,
        "n_rows_before_clean": n_rows_before,
        "n_rows_after_clean": len(df),
        "duplicates_removed": dup_report["duplicates_removed"],
        "n_features": X_train.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "drop_leaky": drop_leaky,
        "log_target": log_target,
        "subsampled": subsampled,
        "missing_report": missing_report,
        "duplicate_report": dup_report,
    }
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "meta": meta,
    }


def make_mock_dataset(task: str, n: int = 2000, n_features: int = 40, 
                      random_state: int = RANDOM_STATE) -> dict:
    """Fake data with the SAME output shape as get_dataset().
    
    For B and C to develop model code before M2. Delete usages after M2.
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
            ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
             "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"]
        )
    else:
        y_train = y_test = le = None
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": names,
        "label_encoder": le,
        "meta": {"mock": True},
    }
