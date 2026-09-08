"""Preprocessing constants for the UNSW-NB15 capstone pipeline.

Every threshold, column list and target name used anywhere in the pipeline is
declared here and nowhere else, so a decision can be reviewed or changed in one
place. The justification for each value is in DATA_AUDIT.md.

Owner: Person A.
"""

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))   # repo root

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
TEST_SIZE = 0.20

# The CSVs live in whichever of these exists. Both filenames are read and
# concatenated, so the well-known train/test filename swap in the official
# distribution does not matter (see DATA_AUDIT.md finding 1).
RAW_DIR_CANDIDATES = [
    os.path.join(_ROOT, "data", "raw"),
    os.path.join(_ROOT, "dataset"),
]
PARTITION_FILES = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]

PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")

TARGET_REG = "sloss"          # source packets retransmitted or dropped
TARGET_CLF = "attack_cat"     # 10 classes: Normal + 9 attack families

# Identifier: overlapping 1..175341 ranges across the two files, so it is not
# unique after concatenation and carries no signal. Always dropped.
ID_COLS = ["id"]

# ct_ftp_cmd is byte-for-byte identical to is_ftp_login (100% of rows).
# Keeping both would double-count one signal and inflate its apparent
# importance in tree models.
EXACT_DUPLICATE_COLS = ["ct_ftp_cmd"]

# `label` is a deterministic function of `attack_cat` (Normal <-> 0, everything
# else <-> 1). Leaving it in X would let any classifier reach ~100% on the
# benign/attack boundary for free.
LABEL_COLS = ["attack_cat", "label"]

# Columns that are algebraically tied to the regression target sloss.
# Verified empirically (DATA_AUDIT.md finding 4):
#   sbytes  r = 0.9967 with sloss
#   spkts   r = 0.9738 with sloss, and sloss <= spkts by definition
#   smean   == sbytes / spkts       exactly, 100% of rows
#   sload   ~= sbytes * 8 / dur     r = 0.998
#   rate    ~= (spkts + dpkts - 1) / dur, matches 99.7% of rows
LEAKY_FOR_SLOSS = ["sbytes", "spkts", "smean", "sload", "rate"]

# Engineered features built from the leaky columns above; dropped alongside them.
LEAKY_ENGINEERED = ["pkt_dir_ratio", "byte_dir_asymmetry"]

CATEGORICAL_COLS = ["proto", "service", "state"]

# proto has 133 levels but the top 10 cover 96.8% of rows. One-hot on all 133
# would add ~130 sparse columns that dominate the feature space for the
# distance-based models (KNN, SVR/SVC) without adding signal.
PROTO_TOP_N = 10

# `state` has levels appearing once or twice in the whole dataset ('no', 'CLO',
# 'PAR', 'URN', 'ACC'). A stratified split on attack_cat gives no guarantee
# those land in train, so they arrive at transform time as unknown categories.
# Any level below this count is folded into "other", fitted on TRAIN only.
RARE_LEVEL_MIN_COUNT = 20

# Percentile clip bounds for heavy-tailed continuous features. Bounds are
# computed on the training split only and applied to both splits.
CLIP_LOWER_Q = 0.01
CLIP_UPPER_Q = 0.99

# Features left unclipped: bounded, discrete, or intentionally spiky.
NO_CLIP_COLS = {
    "sttl", "dttl", "swin", "dwin", "ct_state_ttl", "is_ftp_login",
    "is_sm_ips_ports", "trans_depth", "ct_flw_http_mthd", "no_dst_response",
    "tcp_window_active",
}

# Clustering track feature subset (flow-behaviour statistics only).
CLUSTERING_FEATURES = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
    "sinpkt", "dinpkt", "sjit", "djit", "smean", "dmean",
    "ct_srv_src", "ct_state_ttl",
]
