"""Data cleaning: load, deduplicate, handle missing values, clip outliers."""
import numpy as np
import pandas as pd
from pathlib import Path

from .config import DROP_ALWAYS, LABEL_COLS, RANDOM_STATE

DATA_RAW = Path(__file__).parent.parent.parent / "data" / "raw"


def load_raw() -> pd.DataFrame:
    """Load and concatenate both UNSW-NB15 partition files.
    
    Note: The official partition filenames are swapped relative to the published
    split. We concatenate both and build our own stratified split.
    """
    train_file = DATA_RAW / "UNSW_NB15_training-set.csv"
    test_file = DATA_RAW / "UNSW_NB15_testing-set.csv"
    
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found. Run 'python data/download_data.py' first.\n"
            f"Expected: {train_file} and {test_file}"
        )
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that carry no signal or are redundant."""
    cols_to_drop = [c for c in DROP_ALWAYS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def handle_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Handle missing values and infinities.
    
    Returns cleaned DataFrame and a report dict.
    """
    report = {}
    
    # Replace infinities with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols])
    n_inf = inf_mask.sum().sum()
    if n_inf > 0:
        df = df.replace([np.inf, -np.inf], np.nan)
    report["infinities_replaced"] = int(n_inf)
    
    # Count NaN
    nan_counts = df.isnull().sum()
    total_nan = nan_counts.sum()
    report["total_nan"] = int(total_nan)
    report["nan_by_column"] = nan_counts[nan_counts > 0].to_dict()
    
    # Drop rows with NaN if < 1% affected, else median-impute
    n_rows = len(df)
    nan_rows = df.isnull().any(axis=1).sum()
    
    if nan_rows / n_rows < 0.01:
        df = df.dropna()
        report["action"] = f"dropped {nan_rows} rows with NaN"
    else:
        # Median impute numeric, mode impute categorical
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        report["action"] = "median/mode imputed"
    
    return df, report


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows.
    
    UNSW-NB15 has ~36.8% duplicates which cause 42.6% test-set contamination
    if not removed before splitting.
    """
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df)
    
    report = {
        "rows_before": n_before,
        "rows_after": n_after,
        "duplicates_removed": n_before - n_after,
        "duplicate_pct": round((n_before - n_after) / n_before * 100, 1)
    }
    return df, report


def clip_outliers(df: pd.DataFrame, train_idx: list) -> tuple[pd.DataFrame, dict]:
    """Clip continuous features at 1st/99th percentile computed on training set only.
    
    We clip rather than delete because extreme values ARE the attack signal.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Don't clip label-like columns
    exclude = ["label", "sloss", "dloss"]  # targets should not be clipped
    clip_cols = [c for c in numeric_cols if c not in exclude]
    
    train_df = df.loc[train_idx]
    lower = train_df[clip_cols].quantile(0.01)
    upper = train_df[clip_cols].quantile(0.99)
    
    n_clipped = 0
    for col in clip_cols:
        clipped = ((df[col] < lower[col]) | (df[col] > upper[col])).sum()
        n_clipped += clipped
        df[col] = df[col].clip(lower=lower[col], upper=upper[col])
    
    report = {
        "columns_clipped": len(clip_cols),
        "total_values_clipped": int(n_clipped),
        "percentile_range": "1st-99th (train set)"
    }
    return df, report


def clean(df: pd.DataFrame, train_idx: list = None) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pipeline.
    
    Args:
        df: Raw concatenated DataFrame
        train_idx: Training indices for computing clip bounds
        
    Returns:
        Cleaned DataFrame and report dict
    """
    report = {}
    
    # Drop unnecessary columns
    df = drop_columns(df)
    
    # Handle missing values
    df, missing_report = handle_missing(df)
    report["missing"] = missing_report
    
    # Remove duplicates
    df, dup_report = remove_duplicates(df)
    report["duplicates"] = dup_report
    
    # Clip outliers
    if train_idx is not None:
        df, clip_report = clip_outliers(df, train_idx)
        report["outliers"] = clip_report
    
    return df, report
