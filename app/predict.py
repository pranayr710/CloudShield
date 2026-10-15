"""Load the three CloudShield models and turn form inputs into predictions.

Shared by the Gradio app (app.py) and the Streamlit app (streamlit/streamlit_app.py).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import pipelines as P

try:  # text columns as object dtype, matching how the encoders were fitted
    pd.set_option("future.infer_string", False)
except Exception:
    pass

MODELS = Path(__file__).resolve().parent / "models"
CLS = joblib.load(MODELS / "classification.joblib")
CLS["model"].set_params(n_jobs=1)  # one row at a time: worker processes would only add start-up time
CLU = joblib.load(MODELS / "clustering.joblib")
REG = joblib.load(MODELS / "regression.joblib")

LABEL = {"ddos": "DDoS", "dos": "DoS", "xss": "XSS", "mitm": "MITM"}
NOTE = {
    "normal": "Ordinary IoT and IT traffic.",
    "scanning": "Probing hosts and ports to map the network.",
    "ddos": "Many machines flooding one target.",
    "dos": "One machine flooding a target to exhaust it.",
    "injection": "Smuggling commands or SQL into a trusted input.",
    "password": "Guessing or cracking credentials.",
    "backdoor": "A hidden channel left behind for remote access.",
    "xss": "Injecting script into a web page other users load.",
    "ransomware": "Malware that encrypts files and demands payment.",
    "mitm": "Secretly relaying or altering traffic between two parties.",
}


def name(c: str) -> str:
    return LABEL.get(c, c.capitalize())


def classify(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts,
             src_port=49152, dst_internal=True) -> dict:
    row = P.ton_row(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts,
                    src_port=src_port, dst_internal=dst_internal)
    X = CLS["processor"].transform(P.ton_cls_build(P.ton_cls_clean(row), CLS["fit"]))
    X = X.toarray() if hasattr(X, "toarray") else X
    proba = CLS["model"].predict_proba(X)[0]
    probs = dict(sorted(((c, float(p)) for c, p in zip(CLS["classes"], proba)), key=lambda kv: -kv[1]))
    top = next(iter(probs))
    return {"prediction": top, "label": name(top), "confidence": probs[top], "note": NOTE[top],
            "probabilities": probs, "model": "Bagging (tuned)", "dataset": "TON_IoT"}


def cluster(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts, *_unused) -> dict:
    row = P.ton_row(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts)
    X = CLU["processor"].transform(P.clu_build(row, CLU["top_ports"]))
    km = CLU["model"]
    c = int(km.predict(X)[0])
    dist = np.linalg.norm(km.cluster_centers_ - X[0], axis=1)
    prof = CLU["profiles"][c]
    return {"cluster": c + 1, "k": CLU["k"], "distance": float(dist[c]), "distances": [float(d) for d in dist],
            "size": prof["size"], "mix": prof["mix"], "model": "K-Means", "dataset": "TON_IoT"}


def regress(proto, service, state, dur, spkts, dpkts, sbytes, sttl, dttl, dloss) -> dict:
    row = dict(REG["template"])
    row.update(proto=proto, service=service, state=state, dur=max(0.0, float(dur)), spkts=int(spkts), dpkts=int(dpkts),
               sbytes=int(sbytes), sttl=int(sttl), dttl=int(dttl), dloss=int(dloss))
    df, _ = P.unsw_reg_prepare(pd.DataFrame([row]), REG["protos"])
    X = P.unsw_reg_transform(df, REG["artifacts"]).values
    trees = np.array([t.predict(X)[0] for t in REG["model"].estimators_])
    return {"bytes": float(np.expm1(trees.mean())), "low": float(np.expm1(np.percentile(trees, 10))),
            "high": float(np.expm1(np.percentile(trees, 90))), "model": "Random Forest", "dataset": "UNSW-NB15"}


CLS_SCORES = CLS["test_scores"]
CLS_PRESETS = CLS.get("presets", {})
