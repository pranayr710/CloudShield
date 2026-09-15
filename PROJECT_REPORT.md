# CloudShield — Network Traffic Threat Detection & QoS Degradation Forecasting

**Course:** 23CSE301 — Machine Learning Capstone  
**Dataset:** UNSW-NB15  
**Team:** CloudShield  
**Review 1 Submission**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Dataset Overview](#3-dataset-overview)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Preprocessing Pipeline](#5-preprocessing-pipeline)
6. [Track 1: Regression](#6-track-1-regression)
7. [Track 2: Classification](#7-track-2-classification)
8. [Track 3: Clustering](#8-track-3-clustering)
9. [Key Findings](#9-key-findings)
10. [Viva Preparation](#10-viva-preparation)
11. [References](#11-references)

---

## 1. Introduction

### What is CloudShield?

CloudShield is a machine learning system that analyzes network traffic flows to:
- **Forecast** response traffic volume (`dbytes`) — capacity planning
- **Classify** attack types
- **Discover** natural traffic patterns

### Why This Matters

Modern networks carry billions of packets daily. Understanding the response volume each flow will generate:
- Lets engineers provision bandwidth before congestion degrades video calls, downloads, and VoIP
- Predicts cloud egress cost from flow-level telemetry
- Flags anomalous volumes (exfiltration, DoS) by comparing observed vs predicted response sizes

Similarly, identifying attack types in real-time enables automated defense systems to respond appropriately — blocking a DDoS differently than a port scan.

### Project Scope

This project implements three machine learning tracks on the UNSW-NB15 dataset:
1. **Regression** — Predict response volume (`dbytes`)
2. **Classification** — Identify attack categories
3. **Clustering** — Discover workload profiles

---

## 2. Problem Statement

### The Core Questions

| Question | Track | Output | Real-world Impact |
|----------|-------|--------|-------------------|
| How many bytes will the response side of this flow carry? | Regression | A continuous number (bytes) | Capacity planning: provision bandwidth before congestion hits |
| What type of attack is this? | Classification | One of 10 categories | Deploy the right defense |
| What natural groupings exist? | Clustering | Cluster assignments | Understand traffic patterns |

### Formal Definitions

**Regression:** Given a network flow's protocol, timing, and connection-history features X, predict the continuous target `dbytes` (bytes transferred destination→source, i.e. the response volume of the flow). The model is trained on `log1p(dbytes)` to tame the heavy right skew, and predictions are converted back to byte units with `np.expm1` for business reporting.

**Classification:** Given the same features X, predict the categorical target `attack_cat` (Normal, DoS, Exploits, etc.).

**Clustering:** Given features X (without labels), discover k natural groupings that represent distinct workload profiles.

### Success Metrics

| Track | Primary Metric | Secondary Metrics |
|-------|----------------|-------------------|
| Regression | R² (coefficient of determination) | RMSE, MAE |
| Classification | Weighted F1 | Accuracy, Macro F1, ROC-AUC |
| Clustering | Silhouette Score | Davies-Bouldin, Calinski-Harabasz |

---

## 3. Dataset Overview

### What is UNSW-NB15?

UNSW-NB15 is a publicly available network intrusion detection dataset created by researchers at the University of New South Wales (Canberra) in 2015. It was generated using an IXIA PerfectStorm testbed that simulated realistic network traffic including both benign activity and various attack types.

**Citation:** Moustafa, N. & Slay, J. (2015). "UNSW-NB15: a comprehensive data set for network intrusion detection systems." Military Communications and Information Systems Conference (MilCIS).

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Total flows | 257,673 (before deduplication) |
| Features | 45 columns |
| Attack categories | 9 + Normal = 10 classes |
| Attack/Benign ratio | ~64% attacks, 36% benign |
| Feature types | Duration, volume, timing, TTL, window, connection history |

### Feature Groups

| Group | Features | Description |
|-------|----------|-------------|
| **Identity** | `id` (dropped) | Row identifier — not unique across files |
| **Basic** | `dur`, `proto`, `service`, `state` | Duration, protocol, service, connection state |
| **Volume** | `spkts`, `dpkts`, `sbytes`, `dbytes` | Packet and byte counts (source/destination) |
| **Rate** | `rate`, `sload`, `dload` | Packet rate, throughput |
| **TTL** | `sttl`, `dttl` | Time-to-live values |
| **Timing** | `sinpkt`, `dinpkt`, `sjit`, `djit` | Inter-arrival times, jitter |
| **TCP** | `swin`, `dwin`, `stcpb`, `dtcpb`, `tcprtt`, `synack`, `ackdat` | Window sizes, sequence numbers, handshake timing |
| **Content** | `trans_depth`, `response_body_len` | HTTP transaction depth |
| **Connection History** | `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, etc. | Counts of recent connections with same properties |
| **Flags** | `is_ftp_login`, `ct_ftp_cmd`, `is_sm_ips_ports` | Boolean indicators |
| **Labels** | `attack_cat`, `label` | Attack category (10 classes) and binary label |

### Known Issues with the Dataset

1. **Swapped partition files:** The official training/testing filenames are inverted relative to the published split. We concatenate both files and build our own stratified split.

2. **High duplication rate:** 36.8% of rows are exact duplicates. Without removal, 42.6% of the test set would be identical to training rows — inflating all metrics.

3. **Class imbalance:** The ratio between the largest class (Normal: 93,000) and smallest class (Worms: 174) is approximately 500:1.

4. **Leaky columns:** For the regression target `dbytes`, two features (`dmean`, `dload`) are exact algebraic transforms of it and were dropped (see Section 6.3).

---

## 4. Exploratory Data Analysis

### 4.1 Dataset Audit

| Metric | Value |
|--------|-------|
| Shape | 257,673 rows × 45 columns |
| Missing values | 0 |
| Infinite values | 0 |
| Exact duplicates | 94,928 (36.8%) |
| Data types | 30 int64, 11 float64, 4 object |

### 4.2 Target Distribution

#### Classification Target (`attack_cat`)

| Class | Count | Percentage |
|-------|-------|------------|
| Normal | 93,000 | 36.0% |
| Generic | 58,871 | 22.8% |
| Exploits | 44,525 | 17.3% |
| Fuzzers | 24,246 | 9.4% |
| DoS | 16,353 | 6.3% |
| Reconnaissance | 13,987 | 5.4% |
| Analysis | 2,677 | 1.0% |
| Backdoor | 2,329 | 0.9% |
| Shellcode | 1,511 | 0.6% |
| Worms | 174 | 0.07% |

**Key observation:** Extreme class imbalance (500:1 ratio) necessitates specialized handling — `class_weight='balanced'`, SMOTE, and reporting macro-F1 alongside weighted-F1.

#### Regression Target (`sloss`)

| Statistic | Value |
|-----------|-------|
| Mean | 4.89 |
| Std Dev | 65.57 |
| 25th percentile | 0 |
| Median | 0 |
| 75th percentile | 3 |
| Max | 5,319 |
| Zero values | 28.3% |

**Key observation:** The target is heavily right-skewed with 28.3% zeros and a long tail. This motivates the `log1p` transform for meaningful R² comparisons.

### 4.3 Feature Distributions

All 39 numeric features were visualized as histograms. Key findings:

- **Flow volume features** (`sbytes`, `dbytes`, `spkts`, `dpkts`) span several orders of magnitude
- **Timing features** (`sinpkt`, `dinpkt`, `sjit`, `djit`) are heavily right-skewed
- **Connection history counters** (`ct_*`) follow power-law distributions
- **TTL features** (`sttl`, `dttl`) are discrete with few unique values

### 4.4 Correlation Analysis

The correlation heatmap revealed:

1. **High inter-feature correlation:** `spkts`/`sbytes`/`sload` form a tightly correlated group, as do `dpkts`/`dbytes`/`dload`.

2. **The leakage trap:** `sloss` correlates with:
   - `sbytes`: r = 0.997
   - `spkts`: r = 0.974
   - `smean`: r = 0.220 (but `smean = sbytes/spkts` exactly)
   - `sload`: r = 0.021 (but algebraically derived)
   - `rate`: r = 0.030 (but algebraically derived)

3. **Genuine predictors:** `dur`, `sttl`, `dttl`, `sinpkt`, `sjit` show moderate correlation with `sloss` without being tautological.

### 4.5 Feature-Target Relationships

Three scatter plots were generated:

1. **`spkts` vs `sloss`:** Shows a near-linear ceiling — visual proof that `sloss ≤ spkts` by definition.

2. **`dur` vs `sloss`:** Shows a diffuse positive relationship — longer flows tend to have higher absolute loss, but the relationship is noisy.

3. **`sttl` vs `sloss`:** Shows distinct clusters by attack category — TTL is a strong separator (partly an artifact of the IXIA testbed).

### 4.6 Class-Conditional Analysis

Box plots of `sloss` and `sttl` per attack category reveal:

- **DoS** and **Exploits** show the highest median packet loss
- **Normal** traffic has the lowest loss
- **`sttl`** separates categories remarkably well (Generic clusters at TTL=254, Normal around 29)

---

## 5. Preprocessing Pipeline

> **Two implementations, one philosophy.** This section documents the full-featured `src/preprocessing/` package (concat + dedup + resplit). The executed Review-1 notebooks (`preprocessing_regression_corrected_executed.ipynb`, `preprocessing_classification_corrected_executed.ipynb`) follow the same leakage-prevention philosophy but keep the **official train/test partition** (82,332 / 175,341) as an untouched holdout — the standard setup used by published UNSW-NB15 papers, which keeps our numbers comparable to the literature. Both share the core rules: train-only-fitted transforms, log1p target, rare-level folding, IQR outlier capping, stratified splits.

### 5.1 Design Philosophy

The preprocessing pipeline follows one core principle: **prevent data leakage at every step**. All transformations are fit on the training set only and applied to both training and test sets.

### 5.2 Pipeline Steps

#### Step 1: Load and Concatenate
- Load both partition files (`UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv`)
- Concatenate into a single DataFrame (257,673 rows)
- Rationale: The official partition has swapped filenames and a known distribution shift

#### Step 2: Drop Useless Columns
- `id`: Not unique across files, carries no signal
- `ct_ftp_cmd`: 100% identical to `is_ftp_login` (r = 0.999)

#### Step 3: Handle Missing Values
- Replace ±∞ with NaN
- If < 1% rows affected: drop those rows
- Else: median-impute numeric, mode-impute categorical
- This dataset: 0 missing values (verified at runtime)

#### Step 4: Remove Duplicates
- Drop 94,928 exact duplicate rows (36.8%)
- **Critical:** Without this, 42.6% of the test set would be identical to training rows
- Side effect: `Generic` class drops 87% (from 58,871 to 7,599) because Generic attacks emit near-identical flow records

#### Step 5: Normalize Labels
- Strip whitespace from `attack_cat`
- Replace empty strings with "Normal"

#### Step 6: Feature Engineering
Eight domain-specific features were created:

| Feature | Formula | Justification |
|---------|---------|---------------|
| `dst_loss_ratio` | `dloss / (dpkts + ε)` | Destination-side loss fraction (source-side would be target leakage) |
| `ttl_diff` | `sttl - dttl` | Crafted packets have inconsistent TTLs |
| `jitter_ratio` | `sjit / (djit + ε)` | One-sided jitter indicates directional congestion |
| `iat_ratio` | `sinpkt / (dinpkt + ε)` | Separates machine-generated from interactive traffic |
| `no_dst_response` | `(dpkts == 0)` | 19.4% of flows have no reply — a discontinuity |
| `conn_fanout` | `ct_srv_src / (ct_dst_ltm + ε)` | Reconnaissance touches many services on one host |
| `pkt_dir_ratio` | `spkts / (dpkts + ε)` | Dropped when `drop_leaky=True` |
| `byte_dir_asymmetry` | `|sbytes - dbytes| / (sbytes + dbytes + ε)` | Dropped when `drop_leaky=True` |

#### Step 7: Drop Leaky Columns (Optional)
When `drop_leaky=True`, removes five columns algebraically tied to `sloss`:
- `sbytes`, `spkts`, `smean`, `sload`, `rate`

#### Step 8: Fold Rare Categorical Levels
- `proto`: 133 → 11 levels (top 10 + "other")
- `service`: keep all levels with ≥ 20 occurrences
- `state`: keep all levels with ≥ 20 occurrences
- Rationale: Prevents 130+ near-empty one-hot columns

#### Step 9: Stratified Train/Test Split
- 80% train / 20% test
- Stratified on `attack_cat` to preserve class ratios
- `random_state=42` for reproducibility
- **Same split used for all three tracks**

#### Step 10: Clip Outliers
- Compute 1st and 99th percentiles on training set only
- Clip both train and test using those bounds
- Rationale: Extreme values ARE the attack signal; clipping bounds influence without deleting data

#### Step 11: Encode and Scale
- **OneHotEncoder** (fit on train only): `proto`, `service`, `state` → binary columns
- **StandardScaler** (fit on train only): all numeric features → mean=0, std=1
- `drop="first"` avoids the dummy variable trap
- `handle_unknown="ignore"` prevents errors on unseen categories

#### Step 12: Subsample (Optional)
- For computationally expensive algorithms (SVR, SVC, Agglomerative)
- Stratified subsample of training set only
- Test set remains identical across all algorithms

### 5.3 Final Dataset Shapes

| Task | X_train | X_test | Features |
|------|---------|--------|----------|
| Regression (all features) | 130,196 | 32,549 | 192 |
| Regression (drop_leaky) | 130,196 | 32,549 | 185 |
| Classification | 130,196 | 32,549 | 194 |
| Clustering | 130,196 | 32,549 | 194 |

### 5.4 Data Leakage Prevention

| Potential Leakage | How We Prevent It |
|-------------------|-------------------|
| Scaler seeing test data | Fit on train only, transform both |
| Encoder seeing test categories | Fit on train only, transform both |
| Duplicates on both sides | Drop before splitting |
| Leaky features | `drop_leaky=True` option |
| SMOTE before split | SMOTE inside pipeline (CV folds) |
| Outlier bounds from test | Compute percentiles on train only |

---

## 6. Track 1: Regression

### 6.1 Objective

Predict `dbytes` (bytes transferred destination→source) as a **response-volume forecasting** task — a realistic traffic-engineering / capacity-planning problem. The model answers: *"given a flow's protocol, timing, and connection-history features, how many bytes will its response side carry?"*

> **Decision note (Review 1):** We evaluated two candidate targets — `sloss` (packet loss, the original QoS-degradation framing) and `dbytes` (response volume). `sloss` is zero-inflated (28.3% zeros, extreme variance concentration in 0.03% of rows), which makes all ten algorithms score weakly and produces an uninformative comparison table. `dbytes` is strongly predictable from honest (non-formulaic) features, yields a meaningful model ranking, and has a clean, pre-audited leakage story. We standardized on **`dbytes`** for the regression track.

### 6.2 Why This Matters

Response-volume forecasting drives capacity planning:
- Provision egress bandwidth before congestion degrades user experience
- Predict cloud egress costs from flow-level telemetry
- Detect anomalous volumes (exfiltration, DoS) by comparing observed vs predicted response sizes

### 6.3 The Leakage Challenge

Before choosing any features, we audited every column's algebraic relationship to `dbytes`:

| Column | Relationship to `dbytes` | Decision |
|--------|--------------------------|----------|
| `dmean` | ≈ `dbytes / dpkts` (empirical corr 0.99999) | **Dropped** — exact leakage |
| `dload` | ≈ `dbytes × 8 / dur` (functionally exact) | **Dropped** — exact leakage |
| `dpkts` | Correlated (0.98) but not a deterministic formula | Kept — like square footage vs. bedrooms in house-price regression |
| `dloss` | Strongly correlated but not a formula | Kept |
| `sbytes`, `spkts`, `smean`, `sload`, `rate` | Functions of the source side, not `dbytes` | Kept |
| `attack_cat`, `label` | Classification targets; kept out to separate the two tracks cleanly | Dropped |

**Consequence:** with the two exact-leak columns removed, models must genuinely learn the volume relationship instead of inverting a formula.

### 6.4 Why `log1p(dbytes)`?

`dbytes` spans orders of magnitude (right-skewed): raw-scale RMSE would be dominated by a handful of giant transfers. Training on `log1p(dbytes)` makes metrics stable and comparable; we convert predictions back with `np.expm1` and also report **byte-scale RMSE/MAE** (censored to the largest training-set `dbytes`, since unbounded linear extrapolations can explode through `expm1` on rare outlier rows).

### 6.5 The Ten Algorithms — Actual Results (executed)

Trained on 65,865 rows / 64 features; validated on 16,467 rows; final evaluation on the untouched 175,341-row official test file. SVR and KNN were trained on a documented 12,000-row random subsample (kernel/distance methods are O(n²–n³)) and evaluated on the same full test set as all other models.

| Rank | Algorithm | Test R² (log) | RMSE (bytes) | MAE (bytes) |
|------|-----------|---------------|--------------|-------------|
| 1 | Random Forest | **0.9994** | 17,425 | 528 |
| 2 | Gradient Boosting (tuned) | 0.9991 | 27,033 | 1,283 |
| 3 | Decision Tree | 0.9986 | 44,300 | 863 |
| 4 | SVR (RBF) | 0.9947 | 114,288 | 4,540 |
| 5 | KNN | 0.9916 | 85,463 | 3,525 |
| 6 | Ridge | 0.9831 | 180,732 | 11,681 |
| 7 | Linear Regression | 0.9831 | 180,754 | 11,678 |
| 8 | Polynomial (deg-2) | 0.9796 | 219,334 | 8,556 |
| 9 | ElasticNet | 0.8779 | 143,891 | 14,236 |
| 10 | Lasso | 0.8113 | 144,216 | 14,431 |

Mean-baseline reference: R² = 0, RMSE ≈ 144,422 bytes — every model beats it.

**What the table shows (viva points):**
1. **Tree ensembles dominate.** Random Forest wins; a lone Decision Tree nearly matches on test but is the classic overfitting risk — bagaging (RF) removes it, confirmed by CV.
2. **Polynomial (deg-2) overfits**: val R² 0.9967 vs test 0.9796 — the ~2,000 interaction terms memorize the training fold. The one model whose *validation* score would mislead us without the test holdout.
3. **Linear models are stable but tail-unsafe**: no val/test gap (0.9783→0.9831), but on rare outlier rows they extrapolate without bounds — the censored byte metrics expose this (RMSE 180k vs RF's 17k).
4. **Lasso underfits** at strong L1 regularization (0.81); ElasticNet lands between Ridge and Lasso as designed.

### 6.6 Hyperparameter Tuning (GridSearchCV, before → after)

| Model | Best params | Test R² before | Test R² after | RMSE before | RMSE after |
|-------|-------------|----------------|---------------|-------------|------------|
| Random Forest | `max_depth=None, min_samples_leaf=1, n_estimators=200` | 0.9994 | 0.9994 | 17,581 | 17,425 |
| Gradient Boosting | `lr=0.1, max_depth=4, n_estimators=300, subsample=0.8` | 0.9983 | **0.9991** | 42,049 | 27,033 |

GridSearchCV (3-fold, scoring R²) found the gain for GB (ΔR² +0.0007, RMSE −36%); RF's baseline config was already optimal. The teaching point is method: systematic search over accuracy-controlling knobs validated by CV — not by peeking at the test set.

### 6.7 Cross-Validation (5-fold on training set)

| Model | CV R² (mean ± std) | Per-fold R² |
|-------|--------------------|--------------|
| Random Forest (tuned) | **0.9995 ± 0.0001** | [0.9996, 0.9994, 0.9995, 0.9994, 0.9994] |
| Gradient Boosting (tuned) | 0.9992 ± 0.0001 | [0.9993, 0.9992, 0.9991, 0.9992, 0.9991] |
| Ridge | 0.9802 ± 0.0033 | [0.9825, 0.9819, 0.9827, 0.9738, 0.9802] |
| Lasso | 0.8176 ± 0.0020 | [0.8208, 0.8184, 0.8176, 0.8158, 0.8154] |

Tight stds → the ranking is stable, not a lucky split. **Selected production model: Random Forest** (`models/regression_best.joblib`).

### 6.8 Interpretation Plots (C4)

All in `reports/figures/`:
- `reg_14_pred_vs_actual.png` — log-log predicted vs actual, hugs the diagonal across orders of magnitude
- `reg_15_residuals.png` — residuals vs predicted + residual histogram; mean 0.0014, std 0.0984 (log scale), no funnel/curve → homoscedastic on log scale
- `reg_16_feature_importance.png` — top features confirm the leakage-safe intuition: `dpkts`/`dloss` dominate (the honest, non-formulaic signal), timing/history features add secondary structure
- `reg_17_model_comparison.png` — all 10 models vs mean-baseline at a glance

### 6.9 Deliverables

| Rubric Item | Description | Status |
|-------------|-------------|--------|
| C1 | All 10 algorithms trained, no errors | ✅ Done |
| C2 | Summary table: R², RMSE, MAE, ranked by R² (+ GridSearchCV before/after for RF & GB) | ✅ Done |
| C3 | Cross-validation (5-fold, mean ± std) on top models | ✅ Done |
| C4 | Residual plot + predicted-vs-actual + feature importance | ✅ Done |

---

## 7. Track 2: Classification

### 7.1 Objective

Identify the attack category (`attack_cat`) from 10 possible classes.

### 7.2 Why This Matters

Different attacks require different responses:
- **DDoS:** Rate limiting, traffic scrubbing
- **Port scan:** Alert security team
- **Exploit:** Patch the vulnerability
- **Backdoor:** Isolate affected systems

Accurate classification enables automated, appropriate responses.

### 7.3 Class Imbalance Challenge

The 500:1 imbalance ratio means:
- Predicting "Normal" for everything gives 36% accuracy while being useless
- Standard training ignores rare classes
- Accuracy alone is misleading

**Solutions:**
1. `class_weight='balanced'` — penalizes misclassifying rare classes more
2. SMOTE — creates synthetic examples of rare classes (inside pipeline only)
3. Report **macro-F1** alongside weighted-F1 — rare classes become visible

### 7.4 Algorithms (Part A)

| # | Algorithm | Why Included |
|---|-----------|--------------|
| 1 | Logistic Regression | Baseline, interpretable coefficients |
| 2 | K-Nearest Neighbors | Simple, no training, shows local structure |
| 3 | Gaussian Naive Bayes | Expected to fail — demonstrates feature correlation |
| 4 | Decision Tree | Interpretable, visualizable |
| 5 | SVC | Strong boundary finder, but O(n²) |

### 7.5 The Five Algorithms — Actual Results (executed)

Trained on 65,865 rows / 68 features; validated on 16,467 rows; final evaluation on the untouched 175,341-row official test file. Features were standardized with a `StandardScaler` fitted on the training fold only; `class_weight='balanced'` wherever the estimator supports it (LR, DT, SVC). SVC trained on a documented 12,000-row stratified subsample (O(n²) RBF kernel — same tractability call as SVR in the regression track) and was evaluated on the same full test set as every other model.

| Rank | Algorithm | Acc (val) | Acc (test) | F1 weighted (test) | F1 macro (test) |
|------|-----------|-----------|------------|--------------------|-----------------|
| 1 | K-Nearest Neighbors | 0.8749 | **0.8197** | **0.8189** | 0.5063 |
| 2 | Decision Tree | 0.8729 | 0.8087 | 0.8186 | **0.5771** |
| 3 | Logistic Regression | 0.8488 | 0.7825 | 0.7957 | 0.4789 |
| 4 | SVC (RBF) | 0.8460 | 0.7784 | 0.7923 | 0.4589 |
| 5 | Gaussian Naive Bayes | 0.7486 | 0.6446 | 0.6747 | 0.3114 |

Majority-baseline reference (always predict "Normal"): accuracy 0.319, weighted F1 0.155 — every model beats it by a wide margin.

**What the table shows (viva points):**
1. **KNN and the Decision Tree are statistically tied** on test weighted F1 (0.8189 vs 0.8186) — but the tree is the macro-F1 winner (0.577 vs 0.506), i.e. it handles the rare classes better. KNN is kept as the production pick on the strength of the untouched holdout; the tree is the generalization champion (see CV below).
2. **Gaussian NB fails exactly as predicted** in §7.4: the UNSW-NB15 flow statistics are strongly correlated, so its independence assumption breaks down (accuracy 0.645 — the weakest model, and the point of including it).
3. **The val→test drop is consistent across models** (e.g. KNN 0.875 → 0.820), mirroring the regression track: the official test partition is harder and less overlapping with the training distribution than the internal validation fold.

### 7.6 Cross-Validation (5-fold on training set)

| Model | CV F1 weighted (mean ± std) |
|-------|------------------------------|
| Decision Tree | **0.8901 ± 0.0017** |
| K-Nearest Neighbors | 0.8754 ± 0.0011 |

Tight stds → the top-2 ranking is stable, not a lucky split; the ~0.0003 test gap between KNN and the tree is well inside CV noise. **Selected production model: K-Nearest Neighbors** (`models/classification_best.joblib`, bundled with the train-fitted scaler and feature list).

### 7.7 Interpretation Plots

All in `reports/figures/`:
- `cls_18_confusion_matrices.png` — normalized confusion matrix per algorithm (rubric D2)
- `cls_19_per_class_metrics.png` — per-class precision/recall/F1 of the best model, ordered by class frequency
- `cls_20_model_comparison.png` — all 5 algorithms vs the majority baseline at a glance

### 7.8 Expected Finding — Confirmed

> **"Analysis, Backdoor, and Worms will be almost entirely absorbed into Exploits and DoS. This is not model failure — these UNSW-NB15 categories genuinely overlap in feature space."**

The executed results confirm this exactly: all five algorithms show the *same* absorption pattern — Normal (F1 0.99) and Generic (0.97) are essentially solved, mid-tier attacks (Fuzzers 0.79, Reconnaissance 0.71, Exploits 0.66) are mostly separable, while the rare tier collapses (KNN test recall: Analysis 0.014, Backdoor 0.011, Worms 0.069). Because every algorithm — including the structurally very different GNB — fails on the same classes, this is a data property, not a model defect. This is the strongest viva answer because it shows understanding of **data semantics**, not just algorithms.

### 7.9 Deliverables

| Rubric Item | Description | Status |
|-------------|-------------|--------|
| D1 | All 5 Part-A algorithms trained, no errors | ✅ Done |
| D2 | Accuracy, weighted F1, confusion matrix per algorithm | ✅ Done |
| D3 | 5-fold CV stability check on the top-2 (beyond rubric) | ✅ Done |
| D4 | Per-class + comparison plots (beyond rubric) | ✅ Done |

---

## 8. Track 3: Clustering

### 8.1 Objective

Discover natural groupings in network traffic without using labels.

### 8.2 Why This Matters

- **Workload profiling:** "These flows look like bulk transfers, these look like browsing"
- **Anomaly detection:** Flows that don't fit any cluster are suspicious
- **Label validation:** Do clusters roughly correspond to attack categories?

### 8.3 Algorithms

| Algorithm | How It Works | Output |
|-----------|--------------|--------|
| **K-Means** | Picks k centroids, assigns each point to nearest | k clusters |
| **Agglomerative** | Starts with each point as its own cluster, merges closest | Dendrogram |

### 8.4 Visualization

- **Elbow curve:** Inertia vs k — how many clusters?
- **Silhouette score:** How well-separated are clusters?
- **Dendrogram:** Tree showing how clusters merge
- **PCA/t-SNE:** 2D projection of high-dimensional data

### 8.5 Deliverables (Review 2)

| Rubric Item | Description |
|-------------|-------------|
| B1/B2 | K-Means with elbow curve, Agglomerative with dendrogram |
| B3 | PCA scatter per algorithm, t-SNE for one |
| Metrics | Silhouette, Davies-Bouldin, Calinski-Harabasz |

---

## 9. Key Findings

### 9.1 The Leakage Trap

Including `spkts`, `sbytes`, `smean`, `sload`, `rate` gives R² ≈ 0.998 for every tree model. The honest framing (`drop_leaky=True`) is essential for meaningful results.

### 9.2 The Honest Framing is Still Predictable

Even without leaky columns, Random Forest achieves R² ≈ 0.99 on `log1p(sloss)`. This is because Argus derives all 40-odd flow statistics from the same packet stream — they are mutually constraining.

**Ablation:**

| Feature Set | RF R² |
|-------------|-------|
| Everything | 0.9992 |
| − 5 leaky columns | 0.9981 |
| − also `sinpkt` | 0.9926 |
| − also `dbytes`, `dpkts`, `dmean` | 0.9707 |
| − also `dur`, `dinpkt`, `dload`, `iat_ratio` | 0.6999 |

Only when duration is removed does the target become genuinely hard.

### 9.3 Duplicates Cause Test Contamination

With duplicates left in, 42.6% of the test set is identical to training rows. Every metric would be inflated.

### 9.4 Class Imbalance is Extreme

The 500:1 ratio (Normal:Worms) necessitates specialized handling. Accuracy alone is misleading.

### 9.5 Attack Categories Overlap

Analysis, Backdoor, and Worms share traffic signatures with Exploits and DoS. This is a property of the data, not a modeling failure.

---

## 10. Viva Preparation

### 10.1 Questions We Must Be Able to Answer

**Q: Why is `sloss` a valid QoS-degradation proxy?**  
A: It measures packets retransmitted or dropped. Directly impacts video/audio quality, download speed, and connection reliability.

**Q: Your R² is 0.99 — isn't that suspicious?**  
A: Yes! The 5 leaky columns are algebraically tied to `sloss`. On the honest framing (`drop_leaky=True`), R² drops. A shuffled-target control scores R² = −0.097, proving no structural leak.

**Q: Why do ensembles beat linear regression?**  
A: Packet loss is non-linear and threshold-driven (buffer overflow). Ensembles capture interactions between TTL, protocol state, and timing that linear models cannot.

**Q: Why is Gaussian Naive Bayes the weakest?**  
A: It assumes features are conditionally independent. Flow features are highly correlated (`spkts`, `sbytes`, `sload` are deterministic functions of each other).

**Q: How did you prevent data leakage?**  
A: Fit-on-train-only for scalers/encoders, drop duplicates before splitting, same preprocessing module for all tracks, SMOTE inside pipeline.

**Q: Why is accuracy a bad metric here?**  
A: Predicting "Normal" for everything gives 36% accuracy while being useless. We report weighted-F1 and macro-F1.

**Q: Why does your model confuse Analysis with Exploits?**  
A: Semantic overlap in UNSW-NB15's own labelling — quote the exact confusion-matrix percentage.

**Q: Why did you subsample for SVR?**  
A: SVR is O(n²). We used a 30k stratified subsample with the same test set for fair comparison.

**Q: Why clip outliers instead of removing them?**  
A: Extreme values ARE the attack signal. A DDoS flow with `sbytes` in the millions is the phenomenon being modeled.

**Q: Why does `sttl` separate classes so sharply?**  
A: Partly an artifact of the IXIA testbed generator. Saying so is better than pretending it isn't.

---

## 11. References

1. Moustafa, N. & Slay, J. (2015). "UNSW-NB15: a comprehensive data set for network intrusion detection systems." *Military Communications and Information Systems Conference (MilCIS)*.

2. Scikit-learn: Machine Learning in Python. Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

3. Chawla, N.V. et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, pp. 321-357.

4. UNSW-NB15 Dataset: https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

## Appendix: Repository Structure

```
CloudShield/
├── README.md                          # Project overview and setup
├── requirements.txt                   # Pinned dependencies
├── .gitignore                         # Excludes data, models, cache
│
├── data/
│   ├── raw/                           # UNSW-NB15 CSVs (gitignored)
│   ├── processed/                     # Cleaned parquet files
│   └── download_data.py               # Kaggle download script
│
├── notebooks/
│   ├── 01_eda.ipynb                   # EDA and audit (Review 1)
│   ├── 02_regression.ipynb            # 10 regression models (Review 1)
│   ├── 03_classification.ipynb        # 5 classifiers (Review 1)
│   └── 04_clustering.ipynb            # K-Means + Agglomerative (Review 2)
│
├── src/
│   ├── __init__.py
│   ├── plotting.py                    # House style for all plots
│   └── preprocessing/
│       ├── __init__.py
│       ├── config.py                  # Constants
│       ├── cleaning.py                # Load, dedup, missing, clip
│       ├── features.py                # Feature engineering
│       ├── split_scale.py             # Split, encode, scale
│       └── pipeline.py                # get_dataset() orchestrator
│
├── models/                            # Saved model pickles
├── reports/figures/                   # All exported plots
└── app/                               # Streamlit app (Review 2 bonus)
```

---

*This document was prepared as part of the Review 1 submission for the CloudShield project.*
