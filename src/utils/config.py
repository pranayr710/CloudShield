"""CloudShield — shared project configuration.

Every constant that must be identical across the three tracks and the three
team members lives here. Import from this module rather than redefining values
locally, so that all results remain directly comparable.

CloudShield: Multi-Dataset ML-Driven Cloud Threat Detection and
Resource Abuse Prediction
23CSE301 Machine Learning Capstone, 2026-27
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility  (spec section 18)
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

RAW_IDS2018 = RAW / "CSE-CIC-IDS2018"
RAW_UNSW = RAW / "UNSW-NB15"
RAW_TONIOT = RAW / "TON_IoT"

PROC_REGRESSION = PROCESSED / "cse_cic_ids2018_regression"
PROC_CLASSIFICATION = PROCESSED / "unsw_nb15_classification"
PROC_CLUSTERING = PROCESSED / "ton_iot_clustering"

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

for _p in (
    RAW_IDS2018, RAW_UNSW, RAW_TONIOT,
    PROC_REGRESSION, PROC_CLASSIFICATION, PROC_CLUSTERING,
    MODELS / "regression", MODELS / "classification", MODELS / "clustering",
    RESULTS / "regression", RESULTS / "classification", RESULTS / "clustering",
    FIGURES,
):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Track 1 — CSE-CIC-IDS2018, regression
# --------------------------------------------------------------------------- #

# The spec names `fl_byt_s` as the regression target. That is the abbreviated
# column name used by some CIC-IDS2017 redistributions; the CSE-CIC-IDS2018
# processed CSVs published by CIC normally use "Flow Byts/s".
#
# We therefore do NOT hard-code a single name. resolve_target() in
# src/preprocessing/cse_cic_ids2018.py searches these aliases against the
# columns actually present and raises if none is found, rather than silently
# substituting a different column (spec section 3, DATASET 1).
REG_TARGET_ALIASES = [
    "fl_byt_s",
    "Flow Byts/s",
    "Flow Bytes/s",
    "flow_byts_s",
    "Flow_Bytes/s",
]

# Identifier and leakage columns to remove when they exist (spec step 18).
# Flow Pkts/s is excluded because byte rate and packet rate are near-duplicate
# rate measures over the same window; keeping it would make the target close to
# trivially recoverable. This is verified empirically in the notebook, not
# assumed here.
REG_DROP_ALWAYS = [
    "Flow ID", "FlowID", "Src IP", "Source IP", "Dst IP", "Destination IP",
    "Src Port", "Source Port", "Timestamp", "Unnamed: 0",
]
REG_DROP_LEAKY_CANDIDATES = [
    "Flow Pkts/s", "Flow Packets/s", "fl_pkt_s",
]

# --------------------------------------------------------------------------- #
# Track 2 — UNSW-NB15, classification
# --------------------------------------------------------------------------- #

UNSW_FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]

CLF_TARGET = "attack_cat"      # multiclass, the primary task
CLF_BINARY = "label"           # optional secondary binary experiment

CLF_DROP_ALWAYS = ["id"]                 # not unique once the files are joined
CLF_DROP_REDUNDANT = ["ct_ftp_cmd"]      # duplicate of is_ftp_login (verified)
CLF_LABEL_COLS = ["attack_cat", "label"] # neither may ever appear in X

CLF_CATEGORICAL = ["proto", "service", "state"]

PROTO_TOP_N = 10            # proto has 133 levels; the top 10 cover ~97%
RARE_LEVEL_MIN = 20         # rarer categorical levels fold into "other"
CLIP_LOW, CLIP_HIGH = 0.01, 0.99   # outlier clip bounds, fitted on TRAIN only

# Bounded, discrete or binary columns. Percentile clipping would collapse these
# to a constant (both percentiles are 0 for a 99%-zero column), so they are
# exempt.
CLF_NO_CLIP = {
    "sttl", "dttl", "swin", "dwin", "ct_state_ttl", "is_ftp_login",
    "is_sm_ips_ports", "trans_depth", "ct_flw_http_mthd", "no_dst_response",
}

EPS = 1e-6   # divide-by-zero guard in engineered features

# --------------------------------------------------------------------------- #
# Track 3 — TON_IoT, clustering  (Member 3)
# --------------------------------------------------------------------------- #

CLUSTER_K_RANGE = range(2, 13)

# --------------------------------------------------------------------------- #
# Plot style  (spec section 10)
# --------------------------------------------------------------------------- #

PALETTE = "colorblind"
FIG_DPI = 150
