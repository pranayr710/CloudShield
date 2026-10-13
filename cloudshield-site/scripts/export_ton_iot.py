"""Export the TON_IoT classification and clustering results into src/data/toniot.json.

Run from the site folder after the notebooks have run and the deployable models are built:

    py -3.13 ../app/build_models.py        # once, from the project root (writes app/models/*.joblib)
    py -3.13 scripts/export_ton_iot.py

Classification comes from notebooks/03_classification_1.ipynb: its result tables, its tuning
output, and the final model rebuilt by app/build_models.py on exactly the same grouped split.
Clustering repeats notebooks/clustering.ipynb with the same seeds.
"""
from __future__ import annotations

import json
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

pd.set_option("future.infer_string", False)
warnings.filterwarnings("ignore")

from sklearn.cluster import AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,  # noqa: E402
                             roc_auc_score, silhouette_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app"))
import pipelines as P  # noqa: E402
import ton_split  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "src" / "data" / "toniot.json"
DATA = ROOT / "data/raw/TON_IoT/train_test_network.csv"
RES = ROOT / "results"
NB_CLS = ROOT / "notebooks/03_classification_1.ipynb"
NB_CLU = ROOT / "notebooks/clustering.ipynb"
RS = 42
T0 = time.time()

# Settings of each model as written in 03_classification_1.ipynb
PART_A = ["Logistic Regression", "K-Nearest Neighbors", "Gaussian NB", "Decision Tree", "SVC (RBF)"]
PARAMS = {
    "Logistic Regression": "C=1.0, class_weight=balanced",
    "K-Nearest Neighbors": "k=5, distance-weighted",
    "Gaussian NB": "no hyperparameters",
    "Decision Tree": "max_depth=None, class_weight=balanced",
    "SVC (RBF)": "C=1.0, RBF, class_weight=balanced, 12k-row sample",
    "Random Forest": "100 trees",
    "AdaBoost": "100 rounds of depth-3 trees, learning_rate=1.0",
    "GradientBoostingClassifier": "100 trees, depth 3, learning_rate=0.1",
    "GBC (tuned)": "100 trees, depth 5, learning_rate=0.05, subsample=0.8",
    "Bagging": "100 trees, max_features=0.3",
    "Bagging (tuned)": "200 trees of depth 30, max_samples=0.7, max_features=0.2",
    "Neural Network": "MLP (100, 50), ReLU, Adam",
}
DISPLAY = {"GradientBoostingClassifier": "Gradient Boosting", "Neural Network": "MLP (neural network)"}


def tick(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def notebook_text(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
                     for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))


# ---------------------------------------------------------------- data overview
raw = pd.read_csv(DATA, low_memory=False)
dedup = raw.drop_duplicates(ignore_index=True)
counts = dedup["type"].value_counts()
machines = dedup.groupby("type")["src_ip"].nunique()
overview = {
    "name": "TON_IoT", "file": "train_test_network.csv", "source": "UNSW Canberra Cyber, IoT/IIoT testbed",
    "citation": "Moustafa, N. (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. Sustainable Cities and Society, 72, 102994.",
    "url": "https://research.unsw.edu.au/projects/toniot-datasets",
    "rows": int(len(raw)), "rowsClean": int(len(dedup)), "duplicates": int(len(raw) - len(dedup)), "columns": int(raw.shape[1]),
    "classes": [{"name": k, "count": int(v), "machines": int(machines[k])} for k, v in counts.items()],
    "machines": int(dedup["src_ip"].nunique()), "attackShare": r(1 - counts["normal"] / len(dedup)),
}
tick("overview")

# ---------------------------------------------------------------- classification (team notebook + deployed model)
bundle = joblib.load(ROOT / "app/models/classification.joblib")
model = bundle["model"]
df = ton_split.load(DATA)
tr, te = ton_split.split(df)
_, B, _, _ = ton_split.matrices(df, tr, te, processor=bundle["processor"], fit=bundle["fit"])
y_te = df["type"].values[te]
proba = model.predict_proba(B)
classes = list(model.classes_)
pred = np.array(classes)[proba.argmax(1)]
test_classes = [c for c in counts.index if c in set(y_te)]
pred_classes = [c for c in counts.index if c in set(pred) | set(y_te)]
cm = confusion_matrix(y_te, pred, labels=pred_classes)[[pred_classes.index(c) for c in test_classes]]
present = [classes.index(c) for c in test_classes]
final = {
    "acc": r((pred == y_te).mean()), "f1w": r(f1_score(y_te, pred, average="weighted")), "f1m": r(f1_score(y_te, pred, average="macro")),
    "precision": r(precision_score(y_te, pred, average="weighted", zero_division=0)),
    "recall": r(recall_score(y_te, pred, average="weighted", zero_division=0)),
    # one-vs-rest AUC over the classes that occur in the test fold
    "auc": r(np.mean([roc_auc_score(y_te == c, proba[:, classes.index(c)]) for c in test_classes])),
}
tick(f"final model re-scored: {final}")

board = pd.read_csv(RES / "classification_12model_ton_iot.csv")
models = [{
    "model": DISPLAY.get(row["Model"], row["Model"]), "key": row["Model"],
    "part": "A" if row["Model"] in PART_A else "B", "tuned": "(tuned)" in row["Model"],
    "acc": r(row["Accuracy test"]), "f1w": r(row["F1w test"]), "f1m": r(row["F1m test"]), "params": PARAMS.get(row["Model"], ""),
} for _, row in board.sort_values("F1w test", ascending=False).iterrows()]
pc_p = precision_score(y_te, pred, labels=test_classes, average=None, zero_division=0)
pc_r = recall_score(y_te, pred, labels=test_classes, average=None, zero_division=0)
pc_f = f1_score(y_te, pred, labels=test_classes, average=None, zero_division=0)

text = notebook_text(NB_CLS)
searches = []
for name, block in zip(["GBC (tuned)", "Bagging (tuned)"], re.findall(r"Best params : (\{[^\n]*\})\s*\nBest CV F1w : ([\d.]+)", text)):
    test = next(m for m in models if m["key"] == name)
    searches.append({"model": name, "params": block[0], "cv": float(block[1]), "test": test["f1w"]})

show = ["proto", "service", "conn_state", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "dst_port"]
test_df = df.iloc[te].reset_index(drop=True)


def flow(i):
    row = test_df.iloc[i]
    top = np.argsort(-proba[i])[:4]
    return {"id": int(te[i]), "label": y_te[i], "pred": pred[i],
            "proba": [{"name": classes[j], "p": r(proba[i, j], 3)} for j in top if proba[i, j] >= 0.005],
            "f": {c: (row[c] if isinstance(row[c], str) else r(row[c], 4)) for c in show}}


rng = np.random.RandomState(RS)
lab = []
for c in test_classes:
    idx = np.where(y_te == c)[0]
    hits, miss = idx[pred[idx] == c], idx[pred[idx] != c]
    pick = list(rng.choice(hits, min(2, len(hits)), replace=False)) + (list(rng.choice(miss, 1)) if len(miss) else [])
    lab += [flow(int(i)) for i in pick]
stream = [flow(int(i)) for i in rng.choice(len(y_te), 120, replace=False)]

p_attack = 1 - proba[:, classes.index("normal")]
unsure = np.where((p_attack > 0.02) & (p_attack < 0.98))[0]
if len(unsure) / len(y_te) < 0.2:
    hard = rng.choice(unsure, min(260, len(unsure)), replace=False)
    easy = rng.choice(np.setdiff1d(np.arange(len(y_te)), unsure), 440, replace=False)
    fw_idx = rng.permutation(np.concatenate([hard, easy]))
else:
    hard = np.intersect1d(fw_idx := rng.choice(len(y_te), 700, replace=False), unsure)
order_all = list(counts.index)
firewall = {"flows": [[r(p_attack[i], 3), order_all.index(y_te[i])] for i in fw_idx],
            "hardShare": r(len(hard) / len(fw_idx), 3), "unsureShareTest": r(len(unsure) / len(y_te), 4),
            "oversampled": bool(len(unsure) / len(y_te) < 0.2)}

SHOW_CV5 = False  # 5-fold CV results are left off the site for now; set True to show them
cv_path = RES / "classification_cv5_top2.csv"
cv5 = [{"model": row["Model"], "mean": r(row["CV F1w mean"]), "std": r(row["CV F1w std"]), "f1m": r(row["CV F1m mean"]),
        "acc": r(row["CV Acc mean"]), "folds": [float(x) for x in str(row["F1w per fold"]).strip("[]").split(",")]}
       for _, row in pd.read_csv(cv_path).iterrows()] if SHOW_CV5 and cv_path.exists() else []
grid_path = RES / "tuning_grid_ton_iot.json"
tuning = json.loads(grid_path.read_text()) if grid_path.exists() else None

classification = {
    "best": "GBC (tuned)",
    "bestDisplay": "Gradient Boosting (tuned)",
    "split": "StratifiedGroupKFold (5 folds) grouped by src_ip; first fold as the test set",
    "order": test_classes,
    "allClasses": order_all,
    "missingFromTest": [c for c in order_all if c not in test_classes],
    "models": models,
    "final": final,
    "finalParams": bundle["params"],
    "searches": searches,
    "cv5": cv5,
    "perClass": [{"name": c, "precision": r(pc_p[i]), "recall": r(pc_r[i]), "f1": r(pc_f[i])} for i, c in enumerate(test_classes)],
    "confusion": {"rows": test_classes, "cols": pred_classes, "counts": cm.tolist(),
                  "norm": [[r(v, 4) for v in row] for row in cm / cm.sum(1, keepdims=True)]},
    "recall": {c: r(cm[i, pred_classes.index(c)] / cm[i].sum()) for i, c in enumerate(test_classes)},
    "trainRows": int(len(tr)), "testRows": int(len(te)), "features": int(B.shape[1]),
    "trainMachines": int(df.iloc[tr]["src_ip"].nunique()), "testMachines": int(df.iloc[te]["src_ip"].nunique()),
    "lab": lab, "stream": stream, "firewall": firewall, "tuning": tuning,
}
tick("classification exported")

# ---------------------------------------------------------------- clustering (same code and seeds as clustering.ipynb)
sample = dedup.sample(n=30_000, random_state=RS).reset_index(drop=True)
truth = sample["type"].values
top_ports = sample["dst_port"].value_counts().head(10).index
F = P.clu_build(sample, top_ports)
num_cols = [c for c in F.columns if pd.api.types.is_numeric_dtype(F[c])]
cat_cols = [c for c in F.columns if c not in num_cols]
cpre = ColumnTransformer([("num", StandardScaler(), num_cols),
                          ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cat_cols)])
X = cpre.fit_transform(F)
agg_idx = np.random.RandomState(RS).choice(len(X), 6000, replace=False)

elbow = []
for k in range(2, 13):
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
    elbow.append({"k": k, "inertia": r(km.inertia_, 1), "silhouette": r(silhouette_score(X, km.labels_, sample_size=5000, random_state=RS))})
ks = np.array([e["k"] for e in elbow]); inert = np.array([e["inertia"] for e in elbow])
K = int(ks[np.argmax(np.interp(ks, [ks[0], ks[-1]], [inert[0], inert[-1]]) - inert)])
lab_km = KMeans(n_clusters=K, n_init=20, random_state=RS).fit(X).labels_
tick(f"k-means k={K}")

links = []
for name in ["ward", "complete", "average", "single"]:
    la = AgglomerativeClustering(n_clusters=K, linkage=name).fit_predict(X[agg_idx])
    links.append({"linkage": name, "silhouette": r(silhouette_score(X[agg_idx], la)), "largest": r(np.bincount(la).max() / len(la))})

metrics = pd.read_csv(RES / "clustering_metrics_ton_iot.csv", index_col=0)
stab = pd.read_csv(RES / "clustering_stability_ton_iot.csv", index_col=0)
ari = re.search(r"ARI ([\d.]+) ± ([\d.]+)", notebook_text(NB_CLU))

pca = PCA(n_components=2, random_state=RS).fit(X)
pts = pca.transform(X[agg_idx])
lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
Pn = (pts - lo) / (hi - lo)
keep = np.where(((Pn >= 0) & (Pn <= 1)).all(1))[0][:2600]
comp = pd.crosstab(lab_km, truth, normalize="index")
clusters = [{"id": c, "size": int((lab_km == c).sum()),
             "mix": [{"name": n, "share": r(s, 3)} for n, s in comp.loc[c].sort_values(ascending=False).items() if s >= 0.02]}
            for c in range(K)]

clustering = {
    "sample": int(len(X)), "k": K, "elbow": elbow, "linkages": links,
    "metrics": [{"name": idx, "silhouette": r(row.iloc[0]), "daviesBouldin": r(row.iloc[1]), "calinskiHarabasz": r(row.iloc[2], 1)}
                for idx, row in metrics.iterrows()],
    "stability": {"silhouette": [r(stab.loc["mean", "Silhouette"]), r(stab.loc["std", "Silhouette"])],
                  "ari": [float(ari.group(1)), float(ari.group(2))] if ari else None},
    "pcaVariance": [r(v) for v in pca.explained_variance_ratio_],
    "clusters": clusters,
    "points": [[r(Pn[i, 0], 3), r(Pn[i, 1], 3), int(lab_km[agg_idx][i]), order_all.index(truth[agg_idx][i])] for i in keep],
}
tick("clustering exported")

OUT.write_text(json.dumps({"overview": overview, "classification": classification, "clustering": clustering},
                          separators=(",", ":"), default=str), encoding="utf-8")
tick(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
