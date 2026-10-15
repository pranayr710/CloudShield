"""The TON_IoT classification split and feature matrix of notebooks/03_classification_1.ipynb.

Shared by app/build_models.py and cloudshield-site/scripts/export_ton_iot.py so both see exactly
the training and test rows the notebook used. Not needed by the deployed app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import pipelines as P

RS = 42
SAMPLE_N = 45000


def load(path) -> pd.DataFrame:
    """Read, de-duplicate, take the 45k stratified sample, clean and shuffle, as the notebook does."""
    df = pd.read_csv(path).drop_duplicates(ignore_index=True)
    df, _ = train_test_split(df, train_size=SAMPLE_N, stratify=df["type"], random_state=RS)
    df = P.ton_cls_clean(df.reset_index(drop=True))
    return df.sample(frac=1, random_state=RS, ignore_index=True)


def split(df: pd.DataFrame):
    """Machine-grouped split: the notebook's six test machines on one side, everything else on the other."""
    te_mask = df["src_ip"].isin(P.TON_TEST_MACHINES).to_numpy()
    return np.where(~te_mask)[0], np.where(te_mask)[0]


def matrices(df: pd.DataFrame, tr, te, processor=None, fit=None):
    """Feature frames; fits the statistics and the processor on the training rows unless given."""
    fit = fit or P.ton_cls_fit(df.iloc[tr])
    X_train, X_test = P.ton_cls_build(df.iloc[tr], fit), P.ton_cls_build(df.iloc[te], fit)
    if processor is None:
        num = X_train.select_dtypes(include=np.number).columns.tolist()
        cat = X_train.select_dtypes(exclude=np.number).columns.tolist()
        processor = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat),
        ]).fit(X_train)
    names = processor.get_feature_names_out()
    frame = lambda M, X: pd.DataFrame(M.toarray() if hasattr(M, "toarray") else M, columns=names, index=X.index)
    return frame(processor.transform(X_train), X_train), frame(processor.transform(X_test), X_test), processor, fit
