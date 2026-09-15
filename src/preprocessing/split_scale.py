"""Train/test split, encoding, and scaling."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from .config import RANDOM_STATE, TEST_SIZE, CAT_COLS, RARE_THRESHOLD, LABEL_COLS, LEAKY_COLS


def fold_rare_levels(df: pd.DataFrame, col: str, threshold: int) -> pd.Series:
    """Fold rare categorical levels into 'other'.
    
    Whitelist is determined from the full dataset here, but in production
    should be fit on training data only.
    """
    counts = df[col].value_counts()
    keep = counts[counts >= threshold].index.tolist()
    return df[col].apply(lambda x: x if x in keep else "other")


def preprocess_categorical(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    """Fold rare levels in categorical columns."""
    for col in cat_cols:
        if col in df.columns:
            df[col] = fold_rare_levels(df, col, RARE_THRESHOLD)
    return df


def create_split(df: pd.DataFrame, stratify_col: str = "attack_cat") -> tuple:
    """Create stratified train/test split.
    
    Returns train and test DataFrames with the same stratification.
    """
    # Create a temporary encoding for stratification
    le = LabelEncoder()
    stratify_labels = le.fit_transform(df[stratify_col].astype(str))
    
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=stratify_labels, random_state=RANDOM_STATE
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def encode_and_scale(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                     cat_cols: list, num_cols: list) -> tuple:
    """One-hot encode categoricals and scale numerics.
    
    IMPORTANT: Fit on train only, transform both. This prevents data leakage.
    """
    # Separate categorical and numerical
    train_cat = train_df[cat_cols].copy()
    test_cat = test_df[cat_cols].copy()
    train_num = train_df[num_cols].copy().astype(float)
    test_num = test_df[num_cols].copy().astype(float)
    
    # OneHotEncoder fit on train only
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")
    encoder.fit(train_cat)
    
    train_cat_encoded = encoder.transform(train_cat)
    test_cat_encoded = encoder.transform(test_cat)
    
    # StandardScaler fit on train only
    scaler = StandardScaler()
    scaler.fit(train_num)
    
    train_num_scaled = scaler.transform(train_num)
    test_num_scaled = scaler.transform(test_num)
    
    # Combine
    X_train = np.hstack([train_num_scaled, train_cat_encoded])
    X_test = np.hstack([test_num_scaled, test_cat_encoded])
    
    # Feature names
    cat_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols + cat_feature_names
    
    return X_train, X_test, feature_names, scaler, encoder
