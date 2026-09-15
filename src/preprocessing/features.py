"""Feature engineering for UNSW-NB15.

All features are justified in DATA_AUDIT.md. Each has a domain reason, not just a formula.
"""
import numpy as np
import pandas as pd
from .config import EPS


def engineer_features(df: pd.DataFrame, drop_leaky: bool = False) -> pd.DataFrame:
    """Add engineered features to the dataset.
    
    Features valid in every framing:
        - dst_loss_ratio: Destination-side loss fraction
        - ttl_diff: Source minus destination TTL
        - jitter_ratio: Source over destination jitter
        - iat_ratio: Source over destination inter-arrival time
        - no_dst_response: Binary flag for flows with no destination reply
        - conn_fanout: Connection fan-out ratio
    
    Features dropped under drop_leaky=True (algebraic descendants of leaky columns):
        - pkt_dir_ratio: Source over destination packet count
        - byte_dir_asymmetry: Absolute byte difference over total
    """
    # === Valid in all framings ===
    
    # Destination-side loss fraction
    # Normalises loss by volume so flows of different sizes are comparable
    df["dst_loss_ratio"] = df["dloss"] / (df["dpkts"] + EPS)
    
    # TTL difference - crafted/spoofed packets have inconsistent TTLs
    df["ttl_diff"] = df["sttl"] - df["dttl"]
    
    # One-sided jitter ratio - indicates directional congestion
    df["jitter_ratio"] = df["sjit"] / (df["djit"] + EPS)
    
    # Inter-arrival time ratio - separates machine-generated from interactive traffic
    df["iat_ratio"] = df["sinpkt"] / (df["dinpkt"] + EPS)
    
    # No destination reply flag - 19.4% of flows have dpkts == 0
    df["no_dst_response"] = (df["dpkts"] == 0).astype(int)
    
    # Connection fan-out - reconnaissance touches many services on one host
    df["conn_fanout"] = df["ct_srv_src"] / (df["ct_dst_ltm"] + EPS)
    
    # === Only for classification/clustering (dropped when drop_leaky=True) ===
    
    # Packet direction ratio - descendant of spkts/dpkts
    df["pkt_dir_ratio"] = df["spkts"] / (df["dpkts"] + EPS)
    
    # Byte direction asymmetry - descendant of sbytes/dbytes
    df["byte_dir_asymmetry"] = np.abs(df["sbytes"] - df["dbytes"]) / (df["sbytes"] + df["dbytes"] + EPS)
    
    if drop_leaky:
        df = df.drop(columns=["pkt_dir_ratio", "byte_dir_asymmetry"], errors="ignore")
    
    return df
