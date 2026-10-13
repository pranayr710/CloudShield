"""Preprocessing for the three CloudShield tracks, copied from the project notebooks.

Each block reproduces one notebook exactly, so a deployed model sees its inputs the same way
it saw them in training:

- Classification, TON_IoT ........ notebooks/03_classification_1.ipynb
- Clustering, TON_IoT ............ notebooks/clustering.ipynb
- Regression, UNSW-NB15 .......... preprocessing/preprocessing_regression_corrected_executed.ipynb
                                   + notebooks/02_regression_modeling.ipynb

The functions at the bottom turn the few values a user types into a full one-row frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ====================================================================== classification (TON_IoT)
TON_LOG = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes"]
TON_CAT = ["proto", "service", "conn_state"]
TON_DNS_S = ["dns_AA", "dns_RD", "dns_RA", "dns_rejected"]
TON_DROP_AFT = TON_DNS_S + ["dns_qclass", "missed_bytes", "src_ip"]
TON_DROP = ["http_user_agent", "http_orig_mime_types", "http_resp_mime_types", "weird_name", "weird_addl",
            "weird_notice", "http_trans_depth", "http_method", "http_uri", "http_version", "ssl_subject", "ssl_issuer"]
TON_SSL_KEEP = ["ssl_version", "ssl_cipher", "ssl_resumed", "ssl_established"]


def ton_cls_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Column clean-up done before the split in 03_classification_1.ipynb."""
    df = df.copy()
    df[TON_SSL_KEEP] = df[TON_SSL_KEEP].replace("-", "Unknown")
    return df.drop(columns=[c for c in TON_DROP if c in df.columns])


def ton_cls_fit(train: pd.DataFrame) -> dict:
    """Statistics learned from the training rows only (same expressions as the notebook)."""
    return {
        "dur_hi": train["duration"].quantile(0.999),
        # as in the notebook: the set of the ten largest port *counts*
        "ports": set(train["dst_port"].value_counts().head(10)),
        "cats": {c: pd.Categorical(train[c]).categories for c in TON_CAT},
    }


def ton_cls_build(d: pd.DataFrame, fit: dict) -> pd.DataFrame:
    df = d.copy()
    df["dns_present"] = d[TON_DNS_S].ne("-").any(axis=1).astype("int8")
    df["dns_query_parsed"] = d["service"].eq("dns").astype("int8")
    df["has_missed"] = df["missed_bytes"].gt(0).astype("int8")
    df["duration"] = df["duration"].clip(upper=fit["dur_hi"])
    df[TON_LOG] = np.log1p(df[TON_LOG].to_numpy())
    df["dst_port"] = np.where(df["dst_port"].isin(fit["ports"]), df["dst_port"], -1).astype("int16")
    for c in TON_CAT:
        df[c] = pd.Categorical(df[c], categories=fit["cats"][c]).codes.astype("int16")
    return df.drop(columns=[c for c in TON_DROP_AFT + ["type", "label"] if c in df.columns])


# ====================================================================== clustering (TON_IoT)
CLU_LOG = ["duration", "src_bytes", "dst_bytes", "missed_bytes", "src_pkts", "dst_pkts", "src_ip_bytes",
           "dst_ip_bytes", "http_request_body_len", "http_response_body_len"]
CLU_CAT = ["proto", "service", "conn_state", "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_version",
           "ssl_cipher", "ssl_resumed", "ssl_established", "http_method", "weird_name", "weird_notice",
           "dns_qtype", "dns_rcode", "dns_qclass", "http_status_code", "dst_port_bucket"]


def clu_build(d: pd.DataFrame, top_ports) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    for c in CLU_LOG:
        X[c] = np.log1p(d[c].clip(lower=0))
    X["bytes_per_src_pkt"] = np.log1p(d["src_bytes"] / d["src_pkts"].replace(0, np.nan)).fillna(0)
    X["bytes_per_dst_pkt"] = np.log1p(d["dst_bytes"] / d["dst_pkts"].replace(0, np.nan)).fillna(0)
    X["has_reply"] = (d["dst_pkts"] > 0).astype(int)
    port = d["dst_port"].where(d["dst_port"].isin(top_ports))
    X["dst_port_bucket"] = port.map(lambda p: "other" if pd.isna(p) else str(int(p)))
    for c in CLU_CAT:
        if c != "dst_port_bucket":
            X[c] = d[c].astype(str)
    return X


# ====================================================================== regression (UNSW-NB15)
REG_TARGET = "dbytes"
REG_DROP = ["dmean", "dload", "attack_cat", "label", "id"]
REG_SKEWED = ["dur", "sbytes", "rate", "sload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit"]
REG_CATEGORICAL = ["proto", "service", "state"]


def unsw_reg_prepare(raw: pd.DataFrame, common_protos=None):
    """Steps 1-2 of the preprocessing notebook: service '-', dropped columns, rare protocols."""
    df = raw.copy()
    df["service"] = df["service"].replace("-", "no_service")
    df = df.drop(columns=[c for c in REG_DROP if c in df.columns])
    if common_protos is None:
        freq = df["proto"].value_counts(normalize=True)
        common_protos = freq[freq >= 0.01].index.tolist()
    df["proto"] = df["proto"].where(df["proto"].isin(common_protos), "other")
    return df, common_protos


def _reg_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_dur"] = np.log1p(df["dur"].clip(lower=0))
    df["src_activity_score"] = df["ct_srv_src"] + df["ct_src_ltm"] + df["ct_src_dport_ltm"]
    df["is_tcp"] = df["proto_tcp"] if "proto_tcp" in df.columns else 0
    return df


def unsw_reg_transform(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """One-hot, align, IQR-cap, engineer, scale: the notebook's steps 3-5 with saved artefacts."""
    X = df.drop(columns=[REG_TARGET], errors="ignore")
    X = pd.get_dummies(X, columns=REG_CATEGORICAL, drop_first=False).reindex(columns=art["columns"], fill_value=0)
    for col, (lo, hi) in art["bounds"].items():
        X[col] = X[col].clip(lo, hi)
    X = _reg_engineer(X)
    X[art["scaled"]] = art["scaler"].transform(X[art["scaled"]])
    return X[art["features"]]


# ====================================================================== form inputs -> one raw row
TON_DEFAULTS = {
    "src_ip": "0.0.0.0", "src_port": 49152, "dst_ip": "0.0.0.0", "dst_port": 80, "proto": "tcp", "service": "-",
    "duration": 0.0, "src_bytes": 0, "dst_bytes": 0, "conn_state": "SF", "missed_bytes": 0, "src_pkts": 1,
    "src_ip_bytes": 0, "dst_pkts": 0, "dst_ip_bytes": 0, "dns_query": "-", "dns_qclass": 0, "dns_qtype": 0,
    "dns_rcode": 0, "dns_AA": "-", "dns_RD": "-", "dns_RA": "-", "dns_rejected": "-", "ssl_version": "-",
    "ssl_cipher": "-", "ssl_resumed": "-", "ssl_established": "-", "ssl_subject": "-", "ssl_issuer": "-",
    "http_trans_depth": "-", "http_method": "-", "http_uri": "-", "http_version": "-", "http_request_body_len": 0,
    "http_response_body_len": 0, "http_status_code": 0, "http_user_agent": "-", "http_orig_mime_types": "-",
    "http_resp_mime_types": "-", "weird_name": "-", "weird_addl": "-", "weird_notice": "-",
}


def ton_row(proto, service, conn_state, dst_port, duration, src_bytes, dst_bytes, src_pkts, dst_pkts) -> pd.DataFrame:
    """A TON_IoT flow from the nine form fields; IP-level byte counts add a 40-byte header per packet."""
    row = dict(TON_DEFAULTS)
    row.update(proto=proto, service=service, conn_state=conn_state, dst_port=int(dst_port), duration=max(0.0, float(duration)),
               src_bytes=max(0, int(src_bytes)), dst_bytes=max(0, int(dst_bytes)), src_pkts=max(0, int(src_pkts)),
               dst_pkts=max(0, int(dst_pkts)))
    row["src_ip_bytes"] = row["src_bytes"] + 40 * row["src_pkts"]
    row["dst_ip_bytes"] = row["dst_bytes"] + 40 * row["dst_pkts"]
    if service == "dns":
        row.update(dns_qclass=1, dns_qtype=1, dns_AA="F", dns_RD="T", dns_RA="F", dns_rejected="F")
    return pd.DataFrame([row])
