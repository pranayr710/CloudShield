"""Configuration constants for the preprocessing pipeline."""
import numpy as np

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Columns that are algebraically tied to sloss (regression target)
# - sbytes: r = 0.9967 with sloss
# - spkts: r = 0.9738, and sloss <= spkts by definition
# - smean: == sbytes / spkts exactly (100% of rows)
# - sload: ~= sbytes * 8 / dur (r = 0.998)
# - rate: ~= (spkts + dpkts - 1) / dur (99.7% exact)
LEAKY_COLS = ["sbytes", "spkts", "smean", "sload", "rate"]

# Columns to drop permanently (redundant with is_ftp_login)
DROP_ALWAYS = ["id", "ct_ftp_cmd"]

# Columns that are labels, not features
LABEL_COLS = ["attack_cat", "label"]

# Categorical columns
CAT_COLS = ["proto", "service", "state"]

# Rare level threshold for categorical folding
RARE_THRESHOLD = 20

# Small constant to avoid division by zero
EPS = 1e-6
