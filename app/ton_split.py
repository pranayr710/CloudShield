"""The TON_IoT classification split and feature matrix of notebooks/03_classification_1.ipynb.

Shared by app/build_models.py and cloudshield-site/scripts/export_ton_iot.py so both see exactly
the training and test rows the notebook used. Not needed by the deployed app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pipelines as P

RS = 42


def load(path) -> pd.DataFrame:
    """Read, de-duplicate, clean and shuffle, as the notebook does before splitting."""
    df = pd.read_csv(path).drop_duplicates(ignore_index=True)
    df = P.ton_cls_clean(df)
    return df.sample(frac=1, random_state=RS, ignore_index=True)


def split(df: pd.DataFrame):
    """First fold of StratifiedGroupKFold(5) grouped by source IP."""
    return next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(df, df["type"], groups=df["src_ip"]))


def matrices(df: pd.DataFrame, tr, te, processor=None, fit=None):
    """Feature matrices; fits the statistics and the processor on the training rows unless given."""
    fit = fit or P.ton_cls_fit(df.iloc[tr])
    X_train, X_test = P.ton_cls_build(df.iloc[tr], fit), P.ton_cls_build(df.iloc[te], fit)
    if processor is None:
        num = X_train.select_dtypes(include=np.number).columns.tolist()
        cat = X_train.select_dtypes(include="object").columns.tolist()
        processor = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat),
        ]).fit(X_train)
    dense = lambda M: M.toarray() if hasattr(M, "toarray") else M
    return dense(processor.transform(X_train)), dense(processor.transform(X_test)), processor, fit
