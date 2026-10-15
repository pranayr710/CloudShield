"""Build the deployable model bundles for the Gradio app (and the data for the website's tuning game).

    py -3.13 app/build_models.py          # from the project root

Reproduces each notebook's pipeline with the same seeds, checks the test scores against the
numbers the notebooks report, and writes:

    app/models/classification.joblib   Bagging (tuned), TON_IoT, split grouped by src_ip
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
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, r2_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import pipelines as P  # noqa: E402
import ton_split  # noqa: E402

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
    df = ton_split.load(TON)
    tr, te = ton_split.split(df)
    A, B, processor, fit = ton_split.matrices(df, tr, te)
    return df, tr, te, fit, processor, A, B, df["type"].values[tr].astype(object), df["type"].values[te].astype(object)


def bagging(**kw):
    """Bagging of class-balanced decision trees, as in 03_classification_1.ipynb."""
    base = dict(n_estimators=100, max_samples=1.0, max_features=0.3, bootstrap=True, random_state=RS, n_jobs=-1)
    depth = kw.pop("max_depth", None)
    return BaggingClassifier(estimator=DecisionTreeClassifier(class_weight="balanced", max_depth=depth, random_state=RS), **(base | kw))


FORM = ["proto", "service", "conn_state", "dst_port", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_port"]


def presets(df, te, y_te, model, processor, fit):
    """Per class, the real test flow that the form rebuilds best: the model must still get the rebuilt row right."""
    test_df = df.iloc[te].reset_index(drop=True)
    cand = np.concatenate([np.random.RandomState(RS).permutation(np.where(y_te == c)[0])[:300] for c in sorted(set(y_te))])
    vals = test_df.iloc[cand][FORM].to_numpy(dtype=object)
    internal = test_df.iloc[cand]["dst_ip"].astype(str).str.startswith("192.168.1.").to_numpy()
    rows = pd.concat([P.ton_row(*v[:9], src_port=v[9], dst_internal=bool(n)) for v, n in zip(vals, internal)], ignore_index=True)
    X = processor.transform(P.ton_cls_build(P.ton_cls_clean(rows), fit))
    pr = model.predict_proba(X.toarray() if hasattr(X, "toarray") else X)
    top = np.asarray(model.classes_)[pr.argmax(1)]
    out = {}
    for c in sorted(set(y_te)):
        ok = np.where((y_te[cand] == c) & (top == c))[0]
        if len(ok):
            i = ok[np.argmax(pr[ok].max(1))]
            out[c] = {"values": [v.item() if hasattr(v, "item") else v for v in vals[i]], "dst_internal": bool(internal[i]),
                      "p": float(pr[i].max())}
    return out


if wanted("classification") or wanted("tuning"):
    df, tr, te, fit, processor, A, B, y_tr, y_te = ton_classification_data()
    tick(f"TON_IoT classification matrix: train {A.shape}, test {B.shape} (notebook: 33912 / 11088 rows, 83 features)")

if wanted("classification"):
    PARAMS = dict(n_estimators=200, max_features=0.3, max_depth=None)  # Bagging (tuned), notebook tuning cell
    model = bagging(**PARAMS).fit(A.values, y_tr)
    pred = model.predict(B.values)
    scores = {"acc": accuracy_score(y_te, pred), "f1w": f1_score(y_te, pred, average="weighted"), "f1m": f1_score(y_te, pred, average="macro")}
    tick(f"Bagging (tuned) on the grouped test split: {scores}  (notebook: 0.921717 / 0.924889 / 0.835102)")
    pre = presets(df, te, y_te, model, processor, fit)
    tick(f"presets: { {k: round(v['p'], 3) for k, v in pre.items()} }")
    joblib.dump({"model": model, "processor": processor, "fit": fit, "params": PARAMS, "classes": list(model.classes_),
                 "test_scores": scores, "train_rows": int(len(tr)), "test_rows": int(len(te)), "presets": pre},
                OUT / "classification.joblib", compress=3)
    tick("saved app/models/classification.joblib")

if wanted("tuning"):
    # the website's game: every setting scored by 3-fold stratified CV on a training sample (machines shared
    # between folds), then on the machine-grouped test split
    sub, _ = train_test_split(np.arange(len(y_tr)), train_size=8000, stratify=y_tr, random_state=RS)
    grid = []
    for n in (25, 100, 200):
        for mf in (0.1, 0.3, 0.8):
            for depth in (None, 8):
                est = bagging(n_estimators=n, max_features=mf, max_depth=depth, n_jobs=1)
                cv = cross_val_score(est, A.iloc[sub], y_tr[sub], cv=StratifiedKFold(3, shuffle=True, random_state=RS),
                                     scoring="f1_weighted", n_jobs=3)
                p = est.set_params(n_jobs=-1).fit(A.iloc[sub], y_tr[sub]).predict(B)
                grid.append({"n_estimators": n, "max_features": mf, "max_depth": depth, "cv": round(float(cv.mean()), 4),
                             "cvStd": round(float(cv.std()), 4), "test": round(float(f1_score(y_te, p, average="weighted")), 4),
                             "testAcc": round(float(accuracy_score(y_te, p)), 4)})
                tick(f"grid n={n} max_features={mf} depth={depth}: CV {cv.mean():.4f}  test {grid[-1]['test']:.4f}")
    (ROOT / "results" / "tuning_grid_ton_iot.json").write_text(json.dumps({"model": "Bagging", "sampleRows": 8000, "grid": grid}, indent=1))
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
