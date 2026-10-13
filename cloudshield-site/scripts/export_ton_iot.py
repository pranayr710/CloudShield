"""Export the TON_IoT classification and clustering results into src/data/toniot.json.

Run from the site folder after notebooks/classification.ipynb and notebooks/clustering.ipynb
have been executed (they write results/*.csv and results/classification_best_ton_iot.joblib):

    py -3.10 scripts/export_ton_iot.py

Use the Python whose scikit-learn version ran the notebooks, or joblib will refuse the model.
The split, features and clustering below repeat the notebooks' code with the same seeds, so the
numbers match the notebooks exactly.
"""
from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "src" / "data" / "toniot.json"
DATA = ROOT / "data/raw/TON_IoT/train_test_network.csv"
RES = ROOT / "results"
NB_CLS = ROOT / "notebooks/classification.ipynb"
NB_CLU = ROOT / "notebooks/clustering.ipynb"
RANDOM_STATE = 42
T0 = time.time()


def tick(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def r(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# ---------------------------------------------------------------- same features as the notebooks
LOG_COLS = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "dst_pkts", "src_ip_bytes",
            "dst_ip_bytes", "http_request_body_len", "http_response_body_len"]
CAT_COLS = ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_version",
            "ssl_cipher", "ssl_resumed", "ssl_established", "http_method", "weird_name", "weird_notice",
            "dns_qtype", "dns_rcode", "dns_qclass", "http_status_code", "dst_port_bucket"]


def build_features(d, top_ports):
    X = pd.DataFrame(index=d.index)
    for c in LOG_COLS:
        X[c] = np.log1p(d[c].clip(lower=0))
    X["bytes_per_src_pkt"] = np.log1p(d["src_bytes"] / d["src_pkts"].replace(0, np.nan)).fillna(0)
    X["bytes_per_dst_pkt"] = np.log1p(d["dst_bytes"] / d["dst_pkts"].replace(0, np.nan)).fillna(0)
    X["has_reply"] = (d["dst_pkts"] > 0).astype(int)
    port = d["dst_port"].where(d["dst_port"].isin(top_ports))
    X["dst_port_bucket"] = port.map(lambda p: "other" if pd.isna(p) else str(int(p)))
    for c in CAT_COLS:
        if c != "dst_port_bucket":
            X[c] = d[c].astype(str)
    return X


def notebook_text(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
                     for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))


# ---------------------------------------------------------------- data
raw = pd.read_csv(DATA, low_memory=False)
df = raw.drop_duplicates(ignore_index=True)
counts = df["type"].value_counts()
machines = df.groupby("type")["src_ip"].nunique()
tick(f"data {raw.shape} -> {df.shape}")

overview = {
    "name": "TON_IoT",
    "file": "train_test_network.csv",
    "source": "UNSW Canberra Cyber, IoT/IIoT testbed",
    "citation": "Moustafa, N. (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. Sustainable Cities and Society, 72.",
    "url": "https://research.unsw.edu.au/projects/toniot-datasets",
    "rows": int(len(raw)),
    "rowsClean": int(len(df)),
    "duplicates": int(len(raw) - len(df)),
    "columns": int(raw.shape[1]),
    "classes": [{"name": k, "count": int(v), "machines": int(machines[k])} for k, v in counts.items()],
    "machines": int(df["src_ip"].nunique()),
    "attackShare": r(1 - counts["normal"] / len(df)),
}

# ---------------------------------------------------------------- classification
bundle = joblib.load(RES / "classification_best_ton_iot.joblib")
model, pre, le, classes = bundle["model"], bundle["preprocessor"], bundle["label_encoder"], list(bundle["classes"])
y = le.transform(df["type"].values)
idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.20, stratify=y, random_state=RANDOM_STATE)
assert list(bundle["top_ports"]) == list(df.iloc[idx_train]["dst_port"].value_counts().head(10).index), "split does not match the notebook"
X_test = pre.transform(build_features(df.iloc[idx_test], bundle["top_ports"]))
y_test = y[idx_test]
proba = model.predict_proba(X_test)
pred = proba.argmax(1)
acc = float((pred == y_test).mean())
tick(f"final model re-scored: accuracy {acc:.4f}")

order = [c for c in counts.index]                      # most to least frequent
oi = [classes.index(c) for c in order]
cm = confusion_matrix(y_test, pred, labels=oi)

res = pd.read_csv(RES / "classification_results_ton_iot.csv", index_col=0)
PART_A = ["Logistic Regression", "K-Nearest Neighbors", "Gaussian NB", "Decision Tree", "SVC"]
base_rows = [m for m in res.index if "(" not in m]
models = [{
    "model": m, "part": "A" if m in PART_A else "B",
    "acc": r(res.loc[m, "Accuracy"]), "precision": r(res.loc[m, "Precision (w)"]), "recall": r(res.loc[m, "Recall (w)"]),
    "f1w": r(res.loc[m, "F1 (w)"]), "f1m": r(res.loc[m, "F1 (macro)"]), "auc": r(res.loc[m, "ROC-AUC (OvR)"]),
    "fit": r(res.loc[m, "Fit (s)"], 1), "trainRows": int(res.loc[m, "Train rows"]),
} for m in base_rows]
tuning = pd.read_csv(RES / "classification_tuning_ton_iot.csv")
tuned = {row["Model"]: {"best": row["best"], "cv": r(row["CV F1 (w)"]), "tried": int(row["configs tried"])} for _, row in tuning.iterrows()}
for m in models:
    m.update({"params": tuned.get(m["model"], {}).get("best", "library defaults"), "cvTune": tuned.get(m["model"], {}).get("cv"),
              "tried": tuned.get(m["model"], {}).get("tried", 0)})
cv = pd.read_csv(RES / "classification_cv_ton_iot.csv")
stages = pd.read_csv(RES / "classification_final_tuning_ton_iot.csv", index_col=0)
best_name = res.drop(index=[m for m in res.index if "(" in m]).sort_values("F1 (w)", ascending=False).index[0]

text = notebook_text(NB_CLS)
m = re.search(r"accuracy ([\d.]+)\s+weighted F1 ([\d.]+)\s+macro F1 \(present classes\) ([\d.]+)", text)
present = re.search(r"classes in the unseen-machine test fold: \[([^\]]*)\]", text)
stress = {"acc": float(m.group(1)), "f1w": float(m.group(2)), "f1m": float(m.group(3)),
          "classes": [c.strip(" '\"") for c in present.group(1).split(",")]}

show = ["proto", "service", "conn_state", "duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "dst_port"]
test_df = df.iloc[idx_test].reset_index(drop=True)


def flow(i):
    row = test_df.iloc[i]
    top = np.argsort(-proba[i])[:4]
    return {"id": int(idx_test[i]), "label": classes[y_test[i]], "pred": classes[pred[i]],
            "proba": [{"name": classes[j], "p": r(proba[i, j], 3)} for j in top if proba[i, j] >= 0.005],
            "f": {c: (row[c] if isinstance(row[c], str) else r(row[c], 4)) for c in show}}


rng = np.random.RandomState(RANDOM_STATE)
lab = []
for c in order:
    k = classes.index(c)
    idx = np.where(y_test == k)[0]
    hits, miss = idx[pred[idx] == k], idx[pred[idx] != k]
    pick = list(rng.choice(hits, min(2, len(hits)), replace=False)) + (list(rng.choice(miss, 1)) if len(miss) else [])
    lab += [flow(int(i)) for i in pick]
stream = [flow(int(i)) for i in rng.choice(len(y_test), 120, replace=False)]

# games: a quiz pool (5 flows per class, including the model's misses) and a firewall stream
quiz = []
for c in order:
    k = classes.index(c)
    idx = np.where(y_test == k)[0]
    hits, miss = idx[pred[idx] == k], idx[pred[idx] != k]
    n_miss = min(2, len(miss))
    pick = list(rng.choice(hits, 5 - n_miss, replace=False)) + (list(rng.choice(miss, n_miss, replace=False)) if n_miss else [])
    quiz += [flow(int(i)) for i in pick]
p_attack = 1 - proba[:, classes.index("normal")]
unsure = np.where((p_attack > 0.02) & (p_attack < 0.98))[0]
hard = rng.choice(unsure, min(260, len(unsure)), replace=False)
easy = rng.choice(np.setdiff1d(np.arange(len(y_test)), unsure), 440, replace=False)
fw_idx = rng.permutation(np.concatenate([hard, easy]))
firewall = {
    "flows": [[r(p_attack[i], 3), order.index(classes[y_test[i]])] for i in fw_idx],
    "hardShare": r(len(hard) / len(fw_idx), 3),
    "unsureShareTest": r(len(unsure) / len(y_test), 4),
}

classification = {
    "best": best_name,
    "order": order,
    "models": sorted(models, key=lambda d: -d["f1w"]),
    "final": {"name": f"{best_name} (tuned)", "acc": acc,
              "stages": [{"stage": s, "acc": r(stages.loc[s, "Accuracy"]), "f1w": r(stages.loc[s, "F1 (w)"]),
                          "f1m": r(stages.loc[s, "F1 (macro)"]), "auc": r(stages.loc[s, "ROC-AUC (OvR)"])} for s in stages.index]},
    "cv": [{"model": row["Model"], "mean": r(row["CV F1 (w) mean"]), "std": r(row["std"]),
            "folds": [float(x) for x in str(row["folds"]).strip("[]").split(",")]} for _, row in cv.iterrows()],
    "confusion": {"labels": order, "counts": cm.tolist(), "norm": [[r(v, 4) for v in row] for row in cm / cm.sum(1, keepdims=True)]},
    "recall": {c: r(cm[i, i] / cm[i].sum()) for i, c in enumerate(order)},
    "stress": stress,
    "trainRows": int(len(idx_train)),
    "testRows": int(len(idx_test)),
    "features": int(X_test.shape[1]),
    "lab": lab,
    "stream": stream,
    "quiz": quiz,
    "firewall": firewall,
}
tick("classification exported")

# ---------------------------------------------------------------- clustering (same code and seeds as clustering.ipynb)
sample = df.sample(n=30_000, random_state=RANDOM_STATE).reset_index(drop=True)
truth = sample["type"].values
top_ports = sample["dst_port"].value_counts().head(10).index
F = build_features(sample, top_ports)
num_cols = [c for c in F.columns if pd.api.types.is_numeric_dtype(F[c])]
cat_cols = [c for c in F.columns if c not in num_cols]
cpre = ColumnTransformer([("num", StandardScaler(), num_cols),
                          ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False), cat_cols)])
X = cpre.fit_transform(F)
agg_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), 6000, replace=False)

elbow = []
for k in range(2, 13):
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    elbow.append({"k": k, "inertia": r(km.inertia_, 1), "silhouette": r(silhouette_score(X, km.labels_, sample_size=5000, random_state=RANDOM_STATE))})
ks = np.array([e["k"] for e in elbow]); inert = np.array([e["inertia"] for e in elbow])
K = int(ks[np.argmax(np.interp(ks, [ks[0], ks[-1]], [inert[0], inert[-1]]) - inert)])
kmeans = KMeans(n_clusters=K, n_init=20, random_state=RANDOM_STATE).fit(X)
lab_km = kmeans.labels_
tick(f"k-means k={K}")

links = []
for name in ["ward", "complete", "average", "single"]:
    la = AgglomerativeClustering(n_clusters=K, linkage=name).fit_predict(X[agg_idx])
    links.append({"linkage": name, "silhouette": r(silhouette_score(X[agg_idx], la)), "largest": r(np.bincount(la).max() / len(la))})

metrics = pd.read_csv(RES / "clustering_metrics_ton_iot.csv", index_col=0)
stab = pd.read_csv(RES / "clustering_stability_ton_iot.csv", index_col=0)
ctext = notebook_text(NB_CLU)
ari = re.search(r"ARI ([\d.]+) ± ([\d.]+)", ctext)

P = PCA(n_components=2, random_state=RANDOM_STATE).fit(X)
pts = P.transform(X[agg_idx])
lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
Pn = (pts - lo) / (hi - lo)
keep = np.where(((Pn >= 0) & (Pn <= 1)).all(1))[0][:2600]
comp = pd.crosstab(lab_km, truth, normalize="index")
clusters = []
for c in range(K):
    share = comp.loc[c].sort_values(ascending=False)
    clusters.append({"id": c, "size": int((lab_km == c).sum()),
                     "mix": [{"name": n, "share": r(s, 3)} for n, s in share.items() if s >= 0.02]})

clustering = {
    "sample": int(len(X)),
    "k": K,
    "elbow": elbow,
    "linkages": links,
    "metrics": [{"name": idx, "silhouette": r(row.iloc[0]), "daviesBouldin": r(row.iloc[1]), "calinskiHarabasz": r(row.iloc[2], 1)}
                for idx, row in metrics.iterrows()],
    "stability": {"silhouette": [r(stab.loc["mean", "Silhouette"]), r(stab.loc["std", "Silhouette"])],
                  "ari": [float(ari.group(1)), float(ari.group(2))] if ari else None},
    "pcaVariance": [r(v) for v in P.explained_variance_ratio_],
    "clusters": clusters,
    "points": [[r(Pn[i, 0], 3), r(Pn[i, 1], 3), int(lab_km[agg_idx][i]), order.index(truth[agg_idx][i])] for i in keep],
}
tick("clustering exported")

OUT.write_text(json.dumps({"overview": overview, "classification": classification, "clustering": clustering},
                          separators=(",", ":"), default=str), encoding="utf-8")
tick(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
