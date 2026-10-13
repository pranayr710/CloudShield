"""Build the deployable model bundles for the Gradio app (and the data for the website's tuning game).

    py -3.13 app/build_models.py          # from the project root

Reproduces each notebook's pipeline with the same seeds, checks the test scores against the
numbers the notebooks report, and writes:

    app/models/classification.joblib   GBC (tuned), TON_IoT, split grouped by src_ip
    app/models/clustering.joblib       K-Means (k from the elbow), TON_IoT
    app/models/regression.joblib       Random Forest, UNSW-NB15 (the saved notebook model, recompressed)
    results/tuning_grid_ton_iot.json   cross-validated scores for the website's "tune it yourself" game
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

pd.set_option("future.infer_string", False)  # object dtype for text, as in the notebooks' pandas
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, r2_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import pipelines as P  # noqa: E402

OUT = ROOT / "app" / "models"
OUT.mkdir(parents=True, exist_ok=True)
TON = ROOT / "data/raw/TON_IoT/train_test_network.csv"
RS = 42
T0 = time.time()
ONLY = set(sys.argv[1:])  # e.g. "classification tuning" to rebuild only some parts


def tick(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def wanted(part):
    return not ONLY or part in ONLY


# ---------------------------------------------------------------- shared TON_IoT split (classification)
def ton_classification_data():
    df = pd.read_csv(TON).drop_duplicates(ignore_index=True)
    df = P.ton_cls_clean(df)
    df = df.sample(frac=1, random_state=RS, ignore_index=True)
    tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS).split(df, df["type"], groups=df["src_ip"]))
    fit = P.ton_cls_fit(df.iloc[tr])
    X_train, X_test = P.ton_cls_build(df.iloc[tr], fit), P.ton_cls_build(df.iloc[te], fit)
    num = X_train.select_dtypes(include=np.number).columns.tolist()
    cat = X_train.select_dtypes(include="object").columns.tolist()
    processor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), cat),
    ])
    A = processor.fit_transform(X_train)
    B = processor.transform(X_test)
    A = A.toarray() if hasattr(A, "toarray") else A
    B = B.toarray() if hasattr(B, "toarray") else B
    return df, tr, te, fit, processor, A, B, df["type"].values[tr], df["type"].values[te]


if wanted("classification") or wanted("tuning"):
    df, tr, te, fit, processor, A, B, y_tr, y_te = ton_classification_data()
    tick(f"TON_IoT classification matrix: train {A.shape}, test {B.shape} (notebook: 123473 x 165 / 67001 x 165)")

if wanted("classification"):
    PARAMS = dict(n_estimators=100, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=RS)  # GBC (tuned), cell 43
    model = GradientBoostingClassifier(**PARAMS).fit(A, y_tr)
    pred = model.predict(B)
    scores = {"acc": accuracy_score(y_te, pred), "f1w": f1_score(y_te, pred, average="weighted"), "f1m": f1_score(y_te, pred, average="macro")}
    tick(f"GBC (tuned) on the grouped test split: {scores}  (notebook: 0.836525 / 0.874187 / 0.534530)")
    joblib.dump({"model": model, "processor": processor, "fit": fit, "params": PARAMS, "classes": list(model.classes_),
                 "test_scores": scores, "train_rows": int(len(tr)), "test_rows": int(len(te))},
                OUT / "classification.joblib", compress=3)
    tick("saved app/models/classification.joblib")

if wanted("tuning"):
    # the website's game: every setting scored by 3-fold CV on a training sample, then on the grouped test split
    sub, _ = train_test_split(np.arange(len(y_tr)), train_size=8000, stratify=y_tr, random_state=RS)
    grid = []
    for n in (50, 100, 200):
        for lr in (0.05, 0.1, 0.2):
            for depth in (3, 5):
                est = GradientBoostingClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=0.8, random_state=RS)
                cv = cross_val_score(est, A[sub], y_tr[sub], cv=StratifiedKFold(3, shuffle=True, random_state=RS), scoring="f1_weighted", n_jobs=3)
                p = est.fit(A[sub], y_tr[sub]).predict(B)
                grid.append({"n_estimators": n, "learning_rate": lr, "max_depth": depth, "cv": round(float(cv.mean()), 4),
                             "cvStd": round(float(cv.std()), 4), "test": round(float(f1_score(y_te, p, average="weighted")), 4),
                             "testAcc": round(float(accuracy_score(y_te, p)), 4)})
                tick(f"grid n={n} lr={lr} depth={depth}: CV {cv.mean():.4f}  test {grid[-1]['test']:.4f}")
    (ROOT / "results" / "tuning_grid_ton_iot.json").write_text(json.dumps({"sampleRows": 8000, "grid": grid}, indent=1))
    tick("saved results/tuning_grid_ton_iot.json")

if wanted("clustering"):
    # same steps and seeds as notebooks/clustering.ipynb
    raw = pd.read_csv(TON, low_memory=False).drop_duplicates(ignore_index=True)
    sample = raw.sample(n=30_000, random_state=RS).reset_index(drop=True)
    top_ports = sample["dst_port"].value_counts().head(10).index
    F = P.clu_build(sample, top_ports)
    num = [c for c in F.columns if pd.api.types.is_numeric_dtype(F[c])]
    cat = [c for c in F.columns if c not in num]
    cpre = ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cat)])
    X = cpre.fit_transform(F)
    inert = np.array([KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X).inertia_ for k in range(2, 13)])
    ks = np.arange(2, 13)
    K = int(ks[np.argmax(np.interp(ks, [ks[0], ks[-1]], [inert[0], inert[-1]]) - inert)])
    km = KMeans(n_clusters=K, n_init=20, random_state=RS).fit(X)
    mix = pd.crosstab(km.labels_, sample["type"].values, normalize="index")
    profiles = {int(c): {"size": int((km.labels_ == c).sum()),
                         "mix": {k: round(float(v), 3) for k, v in mix.loc[c].sort_values(ascending=False).items() if v >= 0.02}}
                for c in range(K)}
    sil = silhouette_score(X, km.labels_, sample_size=5000, random_state=RS)
    tick(f"K-Means k={K}, sizes {np.bincount(km.labels_).tolist()}, silhouette {sil:.4f} (notebook: k=6, 0.4741 on its 6,000-flow subset)")
    joblib.dump({"model": km, "processor": cpre, "top_ports": list(top_ports), "k": K, "profiles": profiles},
                OUT / "clustering.joblib", compress=3)
    tick("saved app/models/clustering.joblib")

if wanted("regression"):
    # preprocessing notebook, as it was run to build processed_data/ (82,332 training rows, no de-duplication)
    tr_raw = pd.read_csv(ROOT / "data/raw/UNSW_NB15_training-set.csv")
    te_raw = pd.read_csv(ROOT / "data/raw/UNSW_NB15_testing-set.csv")
    tr_df, protos = P.unsw_reg_prepare(tr_raw)
    te_df, _ = P.unsw_reg_prepare(te_raw, protos)
    y_log = np.log1p(tr_df[P.REG_TARGET])
    X_full = pd.get_dummies(tr_df.drop(columns=[P.REG_TARGET]), columns=P.REG_CATEGORICAL, drop_first=False)
    bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    X_tr, _, _, _ = train_test_split(X_full, y_log, test_size=0.20, random_state=RS, stratify=bins)
    bounds = {}
    for col in [c for c in P.REG_SKEWED if c in X_tr.columns]:
        q1, q3 = X_tr[col].quantile([0.25, 0.75])
        bounds[col] = (q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))
    X_cap = X_tr.copy()
    for col, (lo, hi) in bounds.items():
        X_cap[col] = X_cap[col].clip(lo, hi)
    X_eng = P._reg_engineer(X_cap)
    scaled = [c for c in X_eng.columns if X_eng[c].dtype != "bool" and X_eng[c].nunique() > 2]
    scaler = StandardScaler().fit(X_eng[scaled])

    rf = joblib.load(ROOT / "models/regression_best.joblib")
    processed = pd.read_csv(ROOT / "processed_data/reg_test_processed.csv")
    features = [c for c in processed.columns if c not in ("dbytes", "dbytes_log")]
    art = {"columns": list(X_full.columns), "bounds": bounds, "scaled": scaled, "scaler": scaler, "features": features}

    X_te = P.unsw_reg_transform(te_df, art)
    gap = float(np.abs(X_te.values.astype(float) - processed[features].values.astype(float)).max())
    r2 = r2_score(np.log1p(te_df[P.REG_TARGET]), rf.predict(X_te.values))
    tick(f"regression: max difference from processed_data {gap:.2e}; Random Forest test R² {r2:.4f} (notebook: 0.9994)")

    # template for the form: median numeric values and most common categories of the training rows
    template = {c: (tr_df[c].mode()[0] if tr_df[c].dtype == object else float(tr_df[c].median()))
                for c in tr_df.columns if c != P.REG_TARGET}
    joblib.dump({"model": rf, "artifacts": art, "protos": protos, "template": template, "test_r2": r2},
                OUT / "regression.joblib", compress=3)
    tick("saved app/models/regression.joblib")

for f in sorted(OUT.glob("*.joblib")):
    print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")
