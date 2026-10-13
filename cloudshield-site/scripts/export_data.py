"""Export every number the CloudShield site shows into src/data/cloudshield.json.

Run from the site folder after the notebooks have produced their reports and models:

    py -3.13 scripts/export_data.py

Use the Python whose scikit-learn version saved the models (1.8 here), or joblib
will refuse to load them. To switch dataset for a later review, change the CONFIG
block below; nothing in the React code reads the raw data directly.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    silhouette_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- CONFIG
ROOT = Path(__file__).resolve().parents[2]  # the ML project folder
OUT = Path(__file__).resolve().parents[1] / "src" / "data" / "cloudshield.json"

DATASET = {
    "name": "UNSW-NB15",
    "source": "University of New South Wales, Canberra (IXIA PerfectStorm testbed)",
    "citation": "Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. MilCIS.",
    "url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
}
RAW_FILES = [ROOT / "data/raw/UNSW_NB15_training-set.csv", ROOT / "data/raw/UNSW_NB15_testing-set.csv"]
RAW_TEST = ROOT / "data/raw/UNSW_NB15_testing-set.csv"  # row-aligned with the processed test files
CLS_TRAIN = ROOT / "processed_data/train_processed.csv"
CLS_TEST = ROOT / "processed_data/test_processed.csv"
REG_TEST = ROOT / "processed_data/reg_test_processed.csv"
REG_VAL = ROOT / "processed_data/reg_val_processed.csv"
CLS_MODEL = ROOT / "models/classification_best.joblib"
REG_MODEL = ROOT / "models/regression_best.joblib"
REPORTS = ROOT / "reports"

LABEL = "attack_cat"
NORMAL = "Normal"
REG_TARGET = "dbytes"          # raw bytes column
REG_TARGET_LOG = "dbytes_log"  # what the model predicts (log1p)
WHATIF_FEATURE = "dpkts"       # the slider in the forecaster
LEAKY = ["dmean", "dload"]     # columns dropped from regression because they are formulas of the target
DROP_ID = ["id"]

# Numbers that only exist in the notebook output (no CSV was saved for them).
REG_CV = [  # 02_regression_modeling.ipynb, 5-fold CV on the training set
    {"model": "Random Forest (tuned)", "mean": 0.9995, "std": 0.0001},
    {"model": "Gradient Boosting (tuned)", "mean": 0.9992, "std": 0.0001},
    {"model": "Ridge", "mean": 0.9802, "std": 0.0033},
    {"model": "Lasso", "mean": 0.8176, "std": 0.0020},
]
REG_BASELINE_RMSE = 144422  # mean-prediction baseline, censored byte scale (02_regression_modeling)
CONTAMINATION = 0.426  # share of test rows identical to a training row when duplicates are kept (01_eda)

SEED = 42
rng = np.random.RandomState(SEED)


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def tick(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- raw data audit
raw = pd.concat([pd.read_csv(f) for f in RAW_FILES], ignore_index=True)
raw[LABEL] = raw[LABEL].astype(str).str.strip().replace({"": NORMAL, "nan": NORMAL})
no_id = raw.drop(columns=DROP_ID)
dups = int(no_id.duplicated().sum())
counts = raw[LABEL].value_counts()
feature_cols = [c for c in raw.columns if c not in DROP_ID + [LABEL, "label"]]
tick(f"raw: {len(raw)} rows")

leak = {c: r(np.corrcoef(raw[c], raw[REG_TARGET])[0, 1]) for c in LEAKY}
# dmean is bytes per packet, so dmean * dpkts rebuilds the target almost exactly
rebuilt = raw["dmean"] * raw["dpkts"]
leak_rebuild = r(np.corrcoef(rebuilt, raw[REG_TARGET])[0, 1], 5)

# the models use the official train/test files; how many test flows also occur in the training file?
train_raw = pd.read_csv(RAW_FILES[0]).drop(columns=DROP_ID)
test_raw = pd.read_csv(RAW_TEST).drop(columns=DROP_ID)
seen = set(map(tuple, train_raw[feature_cols].astype(str).values))
official_overlap = float(test_raw[feature_cols].astype(str).apply(tuple, axis=1).isin(seen).mean())

overview = {
    "dataset": DATASET,
    "rows": int(len(raw)),
    "features": int(len(feature_cols)),
    "classes": int(counts.size),
    "attackShare": r(1 - counts[NORMAL] / len(raw)),
    "duplicates": dups,
    "duplicateShare": r(dups / len(raw)),
    "contamination": CONTAMINATION,
    "officialOverlap": r(official_overlap),
    "imbalance": int(round(counts.max() / counts.min())),
    "classCounts": [{"name": k, "count": int(v)} for k, v in counts.items()],
    "protocols": int(raw["proto"].nunique()),
    "services": int(raw["service"].nunique()),
}

# ---------------------------------------------------------------- classification
bundle = joblib.load(CLS_MODEL)
knn, scaler, feats, classes = bundle["model"], bundle["scaler"], bundle["feature_names"], list(bundle["classes"])
cls_test = pd.read_csv(CLS_TEST)
raw_test = pd.read_csv(RAW_TEST)
raw_test[LABEL] = raw_test[LABEL].astype(str).str.strip().replace({"": NORMAL, "nan": NORMAL})
assert (cls_test[LABEL].values == raw_test[LABEL].values).all(), "processed and raw test rows are not aligned"

Xt = scaler.transform(cls_test[feats].values)
proba = knn.predict_proba(Xt)
pred = np.array(knn.classes_)[proba.argmax(1)]
y = cls_test[LABEL].values
tick("knn predicted the full test set")

order = [c for c in counts.index if c in classes]  # most to least frequent
cm = confusion_matrix(y, pred, labels=order)
cm_norm = cm / cm.sum(1, keepdims=True)

res = pd.read_csv(REPORTS / "classification_results.csv")
cls_models = [
    {"model": row["Model"], "accVal": r(row["Accuracy val"]), "acc": r(row["Accuracy test"]),
     "f1w": r(row["F1w test"]), "f1m": r(row["F1m test"]), "fit": r(row["fit (s)"], 2)}
    for _, row in res.iterrows()
]
cv = pd.read_csv(REPORTS / "classification_cv.csv")
per = pd.read_csv(REPORTS / "classification_per_class.csv")

# flows for the classifier lab: per class, confident hits plus one honest miss
show_cols = ["proto", "service", "state", "dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "ct_srv_src"]


def flow_record(i):
    row = raw_test.iloc[i]
    top = np.argsort(-proba[i])[:4]
    return {
        "id": int(i),
        "label": y[i],
        "pred": pred[i],
        "proba": [{"name": knn.classes_[j], "p": r(proba[i, j], 3)} for j in top if proba[i, j] > 0],
        "f": {c: (row[c] if isinstance(row[c], str) else r(row[c], 4)) for c in show_cols},
    }


lab = []
for c in order:
    idx = np.where(y == c)[0]
    hits = idx[(pred[idx] == c)]
    miss = idx[(pred[idx] != c)]
    conf = proba[hits, list(knn.classes_).index(c)] if len(hits) else np.array([])
    pick = list(hits[np.argsort(-conf)][:40])
    rng.shuffle(pick)
    chosen = pick[:2] + (list(rng.choice(miss, 1)) if len(miss) else [])
    lab += [flow_record(int(i)) for i in chosen]

stream_idx = rng.choice(len(y), 80, replace=False)
stream = [flow_record(int(i)) for i in stream_idx]

classification = {
    "best": bundle.get("name", "K-Nearest Neighbors"),
    "k": int(knn.n_neighbors),
    "order": order,
    "models": cls_models,
    "baseline": {"acc": r((y == NORMAL).mean()), "note": "always predict Normal"},
    "cv": [{"model": row["Model"], "mean": r(row["CV F1w mean"]), "std": r(row["CV F1w std"])} for _, row in cv.iterrows()],
    "perClass": [{"name": row["class"], "precision": r(row["precision"]), "recall": r(row["recall"]), "f1": r(row["f1"])} for _, row in per.iterrows()],
    "confusion": {"labels": order, "norm": [[r(v, 4) for v in row] for row in cm_norm], "counts": cm.tolist()},
    "testRows": int(len(y)),
    "trainRows": int(sum(1 for _ in open(CLS_TRAIN)) - 1),
    "lab": lab,
    "stream": stream,
}
tick("classification exported")

# ---------------------------------------------------------------- regression
rf = joblib.load(REG_MODEL)
reg_test = pd.read_csv(REG_TEST)
reg_feats = [c for c in reg_test.columns if c not in (REG_TARGET, REG_TARGET_LOG)]
assert len(reg_feats) == rf.n_features_in_, (len(reg_feats), rf.n_features_in_)
Xr = reg_test[reg_feats].values

reg_res = pd.read_csv(REPORTS / "regression_results.csv")
tune = pd.read_csv(REPORTS / "regression_tuning.csv")
imp = sorted(zip(reg_feats, rf.feature_importances_), key=lambda t: -t[1])[:12]

# predicted vs actual scatter (log scale)
sc_idx = rng.choice(len(Xr), 1600, replace=False)
sc_pred = rf.predict(Xr[sc_idx])
scatter = [[r(a, 3), r(p, 3)] for a, p in zip(reg_test[REG_TARGET_LOG].values[sc_idx], sc_pred)]


def tree_preds(X):
    return np.stack([t.predict(X) for t in rf.estimators_])  # (trees, rows), log scale


# forecaster: flows across the volume range, each with a what-if sweep over one feature
wi_col = reg_feats.index(WHATIF_FEATURE)
proc_vals = reg_test[WHATIF_FEATURE].values
raw_vals = raw_test[WHATIF_FEATURE].values
# log-spaced packet counts, each mapped to the processed value of a real row with that count
targets = [0, 1, 2, 3, 4, 6, 8, 10, 14, 20, 30, 44, 64, 100, 150, 220, 330, 500, 750, 1100, 1700, 2500, 4000]
sweep_proc, sweep_raw = [], []
for t in targets:
    j = int(np.abs(raw_vals - t).argmin())
    if proc_vals[j] not in sweep_proc:
        sweep_proc.append(proc_vals[j])
        sweep_raw.append(int(raw_vals[j]))
sweep_proc = np.array(sweep_proc)

y_bytes = raw_test[REG_TARGET].values
bands = [(0, 0), (1, 500), (500, 5_000), (5_000, 50_000), (50_000, 500_000), (500_000, 10**12)]
fc_idx = []
for lo, hi in bands:
    pool = np.where((y_bytes >= lo) & (y_bytes <= hi))[0]
    fc_idx += list(rng.choice(pool, 2 if hi else 1, replace=False))

forecaster = []
for i in fc_idx:
    i = int(i)
    tp = tree_preds(Xr[i : i + 1])[:, 0]
    X_sweep = np.repeat(Xr[i : i + 1], len(sweep_proc), axis=0)
    X_sweep[:, wi_col] = sweep_proc
    sp = tree_preds(X_sweep)
    rec = flow_record(i)
    forecaster.append({
        "id": i,
        "label": y[i],
        "f": rec["f"],
        "actual": int(y_bytes[i]),
        "pred": r(np.expm1(tp.mean()), 1),
        "lo": r(np.expm1(np.percentile(tp, 10)), 1),
        "hi": r(np.expm1(np.percentile(tp, 90)), 1),
        "whatIfAt": int(np.abs(np.array(sweep_raw) - raw_vals[i]).argmin()),
        "sweep": [[r(np.expm1(sp[:, k].mean()), 1), r(np.expm1(np.percentile(sp[:, k], 10)), 1),
                   r(np.expm1(np.percentile(sp[:, k], 90)), 1)] for k in range(len(sweep_proc))],
    })
tick("regression exported")

regression = {
    "target": REG_TARGET,
    "best": "Random Forest",
    "models": [
        {"model": row["Model"], "r2Val": r(row["R2 val"]), "r2": r(row["R2 test"]),
         "rmseLog": r(row["RMSE test (log)"]), "maeLog": r(row["MAE test (log)"]),
         "rmse": r(row["RMSE test (bytes)"], 0), "mae": r(row["MAE test (bytes)"], 0), "fit": r(row["fit (s)"], 2)}
        for _, row in reg_res.iterrows()
    ],
    "tuning": [{"model": row["Model"], "r2Before": r(row["R2 before"]), "r2After": r(row["R2 after"]),
                "rmseBefore": r(row["RMSE bytes before"], 0), "rmseAfter": r(row["RMSE bytes after"], 0)}
               for _, row in tune.iterrows()],
    "cv": REG_CV,
    "baselineRmse": REG_BASELINE_RMSE,
    "importance": [{"name": n, "value": r(v)} for n, v in imp],
    "scatter": scatter,
    "leak": {"columns": LEAKY, "corr": leak, "rebuild": leak_rebuild},
    "whatIf": {"feature": WHATIF_FEATURE, "values": sweep_raw},
    "flows": forecaster,
    "trainRows": int(sum(1 for _ in open(REG_TEST.with_name("reg_train_processed.csv"))) - 1),
    "features": len(reg_feats),
}

# ---------------------------------------------------------------- clustering (unsupervised, labels only used afterwards)
cls_train = pd.read_csv(CLS_TRAIN)
sample = cls_train.sample(n=6000, random_state=SEED)
Xs = scaler.transform(sample[feats].values)
ys = sample[LABEL].values

elbow = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
    elbow.append({"k": k, "inertia": r(km.inertia_, 1), "silhouette": r(silhouette_score(Xs, km.labels_, sample_size=3000, random_state=SEED))})
# elbow: the k whose inertia sits furthest below the straight line from k=2 to k=10
ks = np.array([e["k"] for e in elbow], float)
inert = np.array([e["inertia"] for e in elbow], float)
line = inert[0] + (inert[-1] - inert[0]) * (ks - ks[0]) / (ks[-1] - ks[0])
best_k = int(ks[np.argmax(line - inert)])
km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xs)
agg_idx = rng.choice(len(Xs), 3000, replace=False)
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(Xs[agg_idx])


def scores(X, labels):
    return {"silhouette": r(silhouette_score(X, labels)), "daviesBouldin": r(davies_bouldin_score(X, labels)),
            "calinskiHarabasz": r(calinski_harabasz_score(X, labels), 1)}


pca = PCA(n_components=2, random_state=SEED).fit(Xs)
P = pca.transform(Xs)
# heavy-tailed flow statistics give a few extreme PCA scores; compress them for display and
# leave out the outermost 1% instead of piling them up on the plot's edges
Pc = np.arcsinh(P / np.median(np.abs(P), axis=0))
lo, hi = np.percentile(Pc, 0.5, axis=0), np.percentile(Pc, 99.5, axis=0)
Pn = (Pc - lo) / (hi - lo)
inside = np.where(((Pn >= 0) & (Pn <= 1)).all(1))[0]
pt_idx = rng.choice(inside, 2400, replace=False)

comp = pd.crosstab(km.labels_, ys, normalize="index")
clusters = []
for c in range(best_k):
    share = comp.loc[c].sort_values(ascending=False)
    clusters.append({"id": c, "size": int((km.labels_ == c).sum()),
                     "mix": [{"name": n, "share": r(s, 3)} for n, s in share.items() if s >= 0.02]})

clustering = {
    "note": "Computed by scripts/export_data.py on a 6,000-flow sample of the training set; replace with the Review 2 notebook output when it exists.",
    "sample": int(len(Xs)),
    "k": best_k,
    "elbow": elbow,
    "kmeans": scores(Xs, km.labels_),
    "agglomerative": scores(Xs[agg_idx], agg.labels_),
    "pcaVariance": [r(v) for v in pca.explained_variance_ratio_],
    "clusters": clusters,
    "points": [[r(Pn[i, 0], 3), r(Pn[i, 1], 3), int(km.labels_[i]), order.index(ys[i])] for i in pt_idx],
}
tick("clustering exported")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"overview": overview, "classification": classification, "regression": regression,
                           "clustering": clustering}, separators=(",", ":"), default=str), encoding="utf-8")
tick(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
