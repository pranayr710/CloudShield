# ML Capstone — Master Plan (23CSE301, AY 2026–27)

**Project:** Network Traffic Threat Detection & QoS Degradation Forecasting
**Scope of this plan:** full project, with **Review 1 (Full Regression + Classification Part A)** specified to implementation level.

---

## 0. CRITICAL — Dataset conflict to resolve before any code

Two different projects are described in the material provided:

| Source | Dataset | Regression target | Classification target |
|---|---|---|---|
| **Your objective text** (the brief you pasted) | implies **UNSW-NB15** | **packet loss count** (`sloss` / `dloss`) | **9 attack types + Normal** = 10 classes |
| `cloudshield_..._blueprint.html` | **CSE-CIC-IDS2018** | `fl_byt_s` (flow byte rate) | 7 attacks + Benign = 8 classes |

CIC-IDS2018 has **no packet-loss column** and **7 attack families**, not 9. UNSW-NB15 has **exactly** `sloss`/`dloss` (packets retransmitted or dropped) and **exactly 9** attack categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms) + Normal.

**Recommendation: build on UNSW-NB15.** It is the only dataset that satisfies the stated objective verbatim, and it is far more tractable (175k train / 82k test rows, 45 columns) than IDS2018's 16M×80 which will not fit comfortably in a laptop notebook for 10 algorithms including SVR and KNN.

The rest of this plan assumes **UNSW-NB15**. If you must use IDS2018, only §3 (targets/features) changes; the pipeline structure is identical.

---

## 1. What the rubric actually demands (from the guidelines PDF)

### Review 1 — 25 marks
| Section | Item | Marks |
|---|---|---|
| A1 | Dataset loading & audit: shape, dtypes, missing counts, target distribution | 1 |
| A2 | EDA: distribution plots per feature, correlation heatmap, target distribution, ≥2 feature–target scatter plots | 2 |
| A3 | A Markdown observation under **every** major visualisation | 1 |
| B1 | Cleaning: missing values (justified), duplicates, outliers | 1 |
| B2 | Encoding + scaler **fitted on train only** + **stratified** split | 1 |
| B3 | ≥1 engineered feature **with written justification** | 1 |
| C1 | All **10** regression algorithms trained, no errors | 4 |
| C2 | One summary table: R², RMSE, MAE, ranked by R² | 2 |
| C3 | GridSearchCV/RandomizedSearchCV on ≥2 models, before/after metric | 2 |
| C4 | Residual plot + predicted-vs-actual for best model + feature-importance plot | 1 |
| D1 | All **5** Part-A classifiers trained, no errors | 2 |
| D2 | Accuracy, weighted F1, confusion matrix per algorithm + preliminary table | 1 |
| E1 | Presentation quality | 1 |
| — | Viva | 5 |

### Non-negotiable global rules (marks are deducted otherwise)
- `random_state=42` everywhere.
- Scalers/encoders **fit on train only** — fitting on full data is flagged as data leakage.
- Same train/test split reused across all 10 algorithms in a track.
- 5-fold CV for the **top 2** models per track.
- All results in a **single Pandas summary table**, not scattered prints.
- Every plot: title, axis labels, legend, `plt.tight_layout()`, colourblind palette (`tab10`/`Set2`/`colorblind`).
- Multiple commits with meaningful messages — a single bulk upload = **0** for the GitHub criterion.
- Cite AI assistance in the README.

### Review 2 — 25 (+2 bonus)
Classification Part B (5 algos, 6 marks) · full 10-algo table · final model selection + tuning · K-Means + Agglomerative (8 marks: Elbow, Dendrogram, Silhouette/DB/CH, PCA scatter per algorithm, t-SNE for one) · code quality + README + GitHub (3) · presentation & viva (8) · Streamlit GUI +1 · public deployment +1.

---

## 2. Repository structure (required by §8 of the guidelines)

```
ml/
├── README.md                 # written at Review 2, drafted at Review 1
├── requirements.txt
├── PLAN.md                   # this file
├── data/
│   ├── raw/                  # UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
│   ├── processed/            # train/test .parquet after cleaning
│   └── download_data.py      # script so the repo stays lightweight
├── notebooks/
│   ├── 01_eda.ipynb          # Review 1 — Sections A + B
│   ├── 02_regression.ipynb   # Review 1 — Section C
│   ├── 03_classification.ipynb # Review 1 Part A, extended in Review 2 Part B
│   └── 04_clustering.ipynb   # Review 2
├── src/
│   ├── `src/preprocessing/`          # load, clean, engineer, split, scale (single source of truth)
│   ├── metrics.py            # regression + classification metric table builders
│   └── plotting.py           # styled plot helpers
├── models/                   # joblib .pkl of best models
├── reports/figures/          # every saved PNG (for slides)
└── app/                      # Streamlit app (Review 2 bonus)
```

`src/preprocessing/` is the key anti-mark-loss device: both the regression and classification notebooks import the **same** cleaning + split function, guaranteeing an identical preprocessed dataset and a defensible "no leakage" story in the viva.

---

## 3. Dataset & track targets (UNSW-NB15)

**Files:** `UNSW_NB15_training-set.csv` (175,341 rows) and `UNSW_NB15_testing-set.csv` (82,332 rows), 45 columns.
**Where to get it** (verified Sept 2026 — the widely-cited CloudStor link is dead, AARNet decommissioned it):
- **Official:** <https://research.unsw.edu.au/projects/unsw-nb15-dataset> → the "download" link goes to a UNSW SharePoint folder. Inside: `CSV Files/Training and Testing Sets/`. Free for academic use in perpetuity; cite Moustafa & Slay (2015).
- **Kaggle mirrors (faster, no registration wall):** `dhoogla/unswnb15`, `mrwellsdavid/unsw-nb15`, `harshwardhanbhangale/unsw-complete-dataset`, `ucimachinelearning/unsw-nb15-dataset`.
- Take **only** `UNSW_NB15_training-set.csv` (175,341 × 45) and `UNSW_NB15_testing-set.csv` (82,332 × 45). Ignore `UNSW-NB15_1..4.csv` (the 2.54M-row raw dump, 49 cols, no header row) unless you deliberately want the full set.

**Columns of interest**
- Categorical: `proto`, `service`, `state`
- Flow volume: `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sload`, `dload`
- **Loss:** `sloss`, `dloss` ← regression target
- TTL/window: `sttl`, `dttl`, `swin`, `dwin`, `stcpb`, `dtcpb`
- Timing: `sinpkt`, `dinpkt`, `sjit`, `djit`, `tcprtt`, `synack`, `ackdat`, `dur`
- Content: `smean`, `dmean`, `trans_depth`, `response_body_len`
- Connection-history counters: `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, …
- Labels: `attack_cat` (10 classes), `label` (0/1)
- Drop always: `id`

### Track 1 — Regression (Review 1)
**Target: `sloss`** — source packets retransmitted or dropped = direct QoS degradation signal.

> ⚠️ **Superseded by measurement.** The audit in `DATA_AUDIT.md` (findings 4 and 5) corrected
> three things in this section's original draft. Read that file before coding.

**1. The leaky column list is five columns, not three.** Verified on the real data:

| Column | Relationship to `sloss` |
|---|---|
| `sbytes` | r = 0.9967 |
| `spkts` | r = 0.9738, and `sloss <= spkts` by definition |
| `smean` | `== sbytes / spkts` exactly, 100% of rows |
| `sload` | `~= sbytes * 8 / dur`, r = 0.998 |
| `rate` | `~= (spkts + dpkts - 1) / dur`, 99.7% exact |

Use `get_dataset("regression", drop_leaky=True)`.

**2. Use `log1p(sloss)` as the primary target.** Ten test rows carry **62.7%** of `sloss`'s
total variance and 100 rows carry 98.7%. On the raw scale every tree model scores R² ~ 0.998
regardless of framing, so the rubric's "ranked by R²" table becomes ten identical numbers.
On `log1p` the model families separate properly:

| Framing | Linear | Random Forest |
|---|---|---|
| raw `sloss`, all features | 0.6926 | 0.9985 |
| raw `sloss`, drop_leaky | 0.6818 | 0.9984 |
| **log1p, all features** | 0.9633 | 0.9988 |
| **log1p, drop_leaky** | **0.8591** | **0.9966** |

Pass `log_target=True`. Report R²/RMSE in log space, and MAE back-transformed with
`np.expm1()` so the error stays readable in packets.

**3. Removing the leaky columns does NOT collapse R² — and that is a real finding.**
A shuffled-target control scores R² = −0.097, so there is no structural leak; Argus simply
derives all 40-odd flow statistics from one packet stream, making them mutually constraining.
Present the ablation instead of a single number:

| Feature set | RF R² |
|---|---|
| everything | 0.9992 |
| − 5 leaky columns | 0.9981 |
| − also `sinpkt` | 0.9926 |
| − also `dbytes`, `dpkts`, `dmean` | 0.9707 |
| − also `dur`, `dinpkt`, `dload`, `iat_ratio` | 0.6999 |

Only when duration goes does the target become genuinely hard. This ablation is a much
stronger deliverable than "our R² was 0.99".

**Still run both framings** (all-features vs `drop_leaky`) and name the `sloss <= spkts`
tautology yourself before an examiner does.

### Track 2 — Classification (Part A Review 1, Part B Review 2)
**Target: `attack_cat`** — 10 classes. Severe imbalance (`Worms` ≈ 130 rows, `Generic` ≈ 40k). Handle with `class_weight='balanced'` where supported and **SMOTE fitted on the training fold only**; report macro-F1 alongside the mandatory weighted-F1 so the rare classes are visible.

### Track 3 — Clustering (Review 2)
Drop `attack_cat` and `label` entirely. Cluster on scaled numeric flow statistics; overlay true labels only afterwards for interpretation.

---

## 4. Feature engineering (rubric B3 — needs written justification)

**Implemented in `src/preprocessing/cleaning.py::engineer_features()`. Full justifications and the
empirical evidence they earn their place are in `DATA_AUDIT.md`.**

Six features valid in every framing: `dst_loss_ratio`, `ttl_diff`, `jitter_ratio`,
`iat_ratio`, `no_dst_response`, `conn_fanout`. Two more (`pkt_dir_ratio`,
`byte_dir_asymmetry`) are useful for classification and clustering but are algebraic
descendants of the leaky columns, so `drop_leaky=True` removes them.

> ⚠️ **Two candidates from this plan's first draft were rejected after measurement:**
> - `bytes_per_pkt_src = sbytes / spkts` — **already exists as `smean`**, identical in 100% of
>   rows. Claiming it as engineered would be wrong and easy for an examiner to catch.
> - `handshake_share = (synack + ackdat) / tcprtt` — **identically 1.0 everywhere**, because
>   `tcprtt` is *defined* as `synack + ackdat`. A zero-variance column.
>
> Do not reintroduce either.

**Evidence they work:** `iat_ratio` ranks 4th of 65 RF importances in the honest regression
framing (0.061, ahead of `dur` and every categorical dummy); `dst_loss_ratio` alone reaches
R² = 0.5445 as a single-feature decision tree.

## 5. Preprocessing pipeline (identical for both Review-1 tracks)

1. Load train + test CSVs, `df.drop(columns=['id'])`.
2. **Audit** (A1): `shape`, `dtypes`, `isnull().sum()`, `duplicated().sum()`, `attack_cat.value_counts()`, `sloss.describe()`.
3. **Clean** (B1): replace `±inf` → NaN; drop or median-impute (state the rule); drop exact duplicates; clip continuous features at the 1st/99th percentile computed **on the training set only** — justify as robustness to CICFlowMeter rate artefacts, not as deletion of real attack signal.
4. **Engineer** (B3): add the §4 features.
5. **Split** (B2): `train_test_split(..., test_size=0.2, stratify=y_class, random_state=42)`. Use the *same* split object for both tracks — stratify on `attack_cat` in both so the regression and classification rows match, which strengthens the "one preprocessed dataset" claim in C1.
6. **Encode**: `proto`, `service`, `state` → `OneHotEncoder(handle_unknown='ignore')` **fit on train only**. (If dimensionality bites, frequency-encode `proto` — 130+ levels — and one-hot the other two.)
7. **Scale**: `StandardScaler` fit on train only. Required for SVR/SVM, KNN, MLP, and the regularised linear models; harmless for trees.
8. Persist to `data/processed/` so notebooks 02/03 load in seconds and are provably consistent.

**Sampling note:** SVR and SVM are O(n²)-ish. Train them on a stratified 30–50k subsample and **state this in the notebook and in the comparison table footnote** — an honest, documented subsample is fine; a silently different training set is not.

---

## 6. Review 1 — Track C: Regression (9 marks)

Train all 10 on the same `X_train_scaled`:

| # | Model | Tuning note |
|---|---|---|
| 1 | `LinearRegression` | baseline; interpret coefficients |
| 2 | `Ridge` | tune `alpha` |
| 3 | `Lasso` | show how many coefficients go to zero |
| 4 | `ElasticNet` | tune `l1_ratio` |
| 5 | `PolynomialFeatures(degree=2) → LinearRegression` | on a reduced feature subset or it explodes; compare degree 2 vs 3 |
| 6 | `DecisionTreeRegressor` | tune `max_depth`, plot importance |
| 7 | `RandomForestRegressor` | `n_estimators`, `n_jobs=-1` |
| 8 | `GradientBoostingRegressor` (or XGBoost) | tune `learning_rate` |
| 9 | `SVR` | scaled + subsampled |
| 10 | `KNeighborsRegressor` | tune `k`; discuss scaling sensitivity |

**Deliverables**
- **C2:** one DataFrame — columns `Model, R2, RMSE, MAE, FitTime` — `sort_values('R2', ascending=False)`.
- **C3:** `GridSearchCV` on Random Forest + Gradient Boosting (the two expected winners), 3-fold inner CV to keep runtime sane; print best params and a before/after R² row.
- **C4:** for the best model — residual plot (residual vs predicted, with a y=0 line), predicted-vs-actual scatter with the y=x diagonal, and a horizontal bar chart of the top-15 GBM/RF feature importances.
- **Global rule:** 5-fold `cross_val_score(scoring='r2')` on the top 2.

**Expected outcome:** ensembles (RF/GBM) dominate; linear models fail because loss vs. flow behaviour is non-linear and threshold-driven — that is your headline finding.

---

## 7. Review 1 — Track D: Classification Part A (3 marks)

Five algorithms on the same split, target `attack_cat`:

1. `LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='ovr')`
2. `KNeighborsClassifier` — tune `k`, discuss distance metrics
3. `GaussianNB` — expect it to be the weakest; discuss why the conditional-independence assumption fails on correlated flow features (guaranteed viva question)
4. `DecisionTreeClassifier(class_weight='balanced')` — visualise with `plot_tree(max_depth=3)`
5. `SVC(kernel='rbf', class_weight='balanced')` — on the subsample

**Deliverables:** per algorithm — accuracy, weighted precision/recall/F1, **and** macro-F1, plus a normalised confusion-matrix heatmap (10×10, `normalize='true'`); then one preliminary comparison table. Add OvR ROC-AUC now even though it's only mandatory in Review 2 — it costs nothing and pre-loads Review 2's A2.

**Expected finding:** `Analysis`, `Backdoor` and `Worms` are almost entirely misclassified as `Exploits`/`DoS`. Do not hide this — quantify it from the confusion matrix and explain it as genuine semantic overlap between UNSW-NB15 attack families, not model failure. That is the single strongest thing you can say in the Review 1 viva.

---

## 8. Review 2 outline (build after Review 1 is banked)

- **Part B classifiers:** RandomForest, AdaBoost, GradientBoosting, Bagging(DecisionTree), MLPClassifier — same split, same preprocessing.
- **A2:** one consolidated 10-row table: Accuracy, Precision, Recall, F1, ROC-AUC (OvR).
- **A3:** tune the top model, document metric improvement.
- **Clustering:** feature subset (`dur, spkts, dpkts, sbytes, dbytes, rate, sload, dload, sinpkt, dinpkt, sjit, djit, smean, dmean, ct_srv_src, ct_state_ttl`), StandardScaler, subsample ~30–50k for Agglomerative (O(n²) memory).
  - K-Means: Elbow k=2..12 + Silhouette-vs-k curve.
  - Agglomerative (Ward): dendrogram on a ~2k subsample for readability, compare Ward/complete/average linkage.
  - Metrics for both: Silhouette, Davies-Bouldin, Calinski-Harabasz.
  - PCA-2D scatter for **each** algorithm; t-SNE for one.
  - Post-hoc: crosstab cluster × `attack_cat` and name each cluster a workload profile.
- **Bonus:** Streamlit app — inputs → predicted packet loss (regression), predicted attack type + probability bar (classification), cluster/workload label. Deploy to Streamlit Community Cloud, URL in README.

---

## 9. Execution order & commit plan for Review 1

| Step | Output | Commit message |
|---|---|---|
| 1 | `requirements.txt`, folder skeleton, `.gitignore` | `Initialise project structure and dependencies` |
| 2 | `data/download_data.py`, raw CSVs in place | `Add dataset download script` |
| 3 | `notebooks/01_eda.ipynb` — audit + all EDA plots + commentary | `Add dataset audit and EDA with commentary` |
| 4 | `src/preprocessing/` — clean, engineer, split, scale | `Add shared preprocessing and feature engineering module` |
| 5 | `notebooks/02_regression.ipynb` — 10 models + table | `Train all 10 regression models with comparison table` |
| 6 | tuning + plots | `Add GridSearchCV tuning and diagnostic plots for regression` |
| 7 | `notebooks/03_classification.ipynb` — 5 Part-A models | `Add classification Part A: 5 algorithms with confusion matrices` |
| 8 | `README.md` draft, `reports/figures/` exported | `Add README and export figures for Review 1` |

Aim for ≥8 commits before Review 1. Push after each step, not at the end.

---

## 10. Environment

`pandas`, `scikit-learn`, `seaborn`, `imbalanced-learn`, `xgboost`, `joblib` are **not installed** on this machine (only `numpy`, `scipy`, `matplotlib` are). Step 1 must install them and pin versions into `requirements.txt`.

---

## 11. Viva preparation (start collecting answers now)

- Why is `sloss` a valid QoS-degradation proxy, and what does the `spkts` leakage caveat mean?
- Why do ensembles beat linear regression here?
- Why is Gaussian Naive Bayes weak — are flow features conditionally independent?
- How did you prevent data leakage? (fit-on-train-only, one shared `src.preprocessing` module)
- How did you handle `Worms`/`Backdoor` imbalance, and why is macro-F1 reported next to weighted-F1?
- Why does stratification matter for both tracks?
- What does the Elbow curve tell you about traffic behaviour? (Review 2)

---

## Appendix A — Can UNSW-NB15 and CSE-CIC-IDS2018 be clubbed together?

Analysis only — no decision locked in.

### A.1 The blocker: they are built by different feature extractors

UNSW-NB15 features come from **Argus + Bro/Zeek + 12 custom flow algorithms** (UNSW Canberra, 2015, IXIA PerfectStorm testbed).
CIC-IDS2018 features come from **CICFlowMeter-V3** (UNB/CIC, 2018, AWS testbed).

They are not two samples of one feature space. They are two different measurement instruments pointed at different networks three years apart.

**Feature crosswalk — what actually maps:**

| Concept | UNSW-NB15 | CIC-IDS2018 | Mappable? |
|---|---|---|---|
| Flow duration | `dur` (seconds) | `Flow Duration` (µs) | ✅ after unit conversion |
| Fwd/bwd packet count | `spkts`, `dpkts` | `Tot Fwd Pkts`, `Tot Bwd Pkts` | ✅ |
| Fwd/bwd byte count | `sbytes`, `dbytes` | `TotLen Fwd Pkts`, `TotLen Bwd Pkts` | ✅ |
| Packet rate | `rate` | `Flow Pkts/s` | ✅ |
| Throughput | `sload`, `dload` (bits/s) | `Flow Byts/s` (bytes/s) | ⚠️ partial — IDS2018 has no per-direction split |
| Mean packet size | `smean`, `dmean` | `Fwd/Bwd Pkt Len Mean` | ✅ |
| Inter-arrival time | `sinpkt`, `dinpkt` | `Fwd/Bwd IAT Mean` | ✅ |
| TCP window | `swin`, `dwin` | `Init Fwd/Bwd Win Byts` | ⚠️ different semantics |
| Protocol | `proto`, `service`, `state` | `Protocol`, `Dst Port` | ⚠️ lossy — no `service`/`state` equivalent |
| **Packet loss** | **`sloss`, `dloss`** | **— does not exist —** | ❌ |
| TTL | `sttl`, `dttl` | — does not exist — | ❌ |
| Jitter | `sjit`, `djit` | — does not exist — | ❌ |
| Handshake timing | `tcprtt`, `synack`, `ackdat` | — does not exist — | ❌ |
| Connection history | `ct_*` (11 columns) | — does not exist — | ❌ |
| TCP flag counts | — not present — | `SYN/FIN/RST/PSH/ACK Flag Cnt` | ❌ (reverse direction) |
| Active/idle | — not present — | `Active/Idle Mean/Std/Max/Min` | ❌ |

**Realistic shared feature space: ~9–11 columns out of 45 and 80.** You would discard roughly 75% of both datasets' information to make them line up.

### A.2 The regression track cannot be merged at all

`sloss` / `dloss` — your stated target — **has no counterpart in CIC-IDS2018**. CICFlowMeter never records retransmitted or dropped packets. A merged regression set would be ~65% missing on the target column, which is not an imputation problem, it is a "the measurement was never taken" problem. Any proxy you invent would be fabricated data and indefensible in the viva.

Merging is therefore only ever a question about the **classification** and **clustering** tracks.

### A.3 The label spaces do not align cleanly either

| UNSW-NB15 (9) | Nearest IDS2018 (7 families) |
|---|---|
| DoS | DoS ✅ |
| — | DDoS (UNSW has no DDoS class) |
| Reconnaissance | Infiltration ⚠️ loose |
| Exploits | Web attacks / Heartbleed ⚠️ very loose |
| Backdoor | Botnet ⚠️ loose |
| Fuzzers, Analysis, Shellcode, Worms | — no counterpart — |
| — | Brute-force (UNSW folds this into Exploits) |

A merged classifier can only be trained on a **coarse 4–5 class space** (Benign / DoS-DDoS / Recon-Scan / Exploit-family / Other). That throws away the "9 attack types" your objective is built on — the exact selling point of the project.

### A.4 The fatal statistical problem: the dataset-origin shortcut

Even on the ~10 shared features, the two captures have very different distributions — different network topology, different traffic generators, 2015 vs 2018 protocol mix, different flow-timeout settings in the extractors.

If you stack them, a Random Forest will discover it can separate classes by **which capture a row came from** rather than by attack behaviour. Because attack classes are unevenly distributed across the two sources (DDoS only exists in IDS2018, Fuzzers only in UNSW), "which dataset" is an almost perfect predictor of several labels. You would get a beautiful ~99% accuracy that measures nothing but capture artefacts — and it is precisely the kind of thing a viva examiner probes.

This is the strongest single argument against merging, and it is difficult to fully mitigate.

### A.5 Also: the guidelines may not permit it

Guidelines §1: *"Datasets — Assigned by instructor (one dataset per track per team)."* Datasets appear to be **assigned**, and the phrasing implies **one per track**. A self-chosen two-dataset merge may be out of scope regardless of technical merit. Worth one question to your instructor.

### A.6 Pros and cons

**Merging — pros**
- Larger, more diverse training set; a model that generalises across two testbeds is genuinely more credible than one tuned to a single capture.
- Strong novelty and viva talking point: cross-testbed robustness is a real open problem in IDS research.
- Lets you keep the "Cloud Computing" interdomain claim from the CloudShield blueprint (IDS2018 was deployed on AWS), which UNSW-NB15 alone does not give you.
- Reuses work you have already invested in the blueprint.

**Merging — cons**
- ❌ Regression track impossible — the packet-loss target does not exist in IDS2018.
- ❌ Drops to ~10 usable features, gutting the feature-engineering section (rubric B3) — most of your best engineered features rely on TTL, jitter and handshake columns that exist in only one dataset.
- ❌ Forces a coarse 4–5 class label space, contradicting your "9 attack types" objective.
- ❌ Dataset-origin shortcut inflates every metric and is hard to defend.
- ❌ Roughly doubles the preprocessing work — unit conversions, extractor-semantics reconciliation, label crosswalk, alignment validation — all before a single model is trained, and none of it earns marks directly.
- ❌ IDS2018 is ~6.5 GB / 16M rows; subsampling it correctly is its own sub-project.
- ⚠️ Possibly outside what the instructor assigned.

### A.7 The middle path that captures most of the upside

**Do not merge the training data. Use IDS2018 as a held-out cross-dataset test set, in Review 2 only.**

1. Build the entire project on UNSW-NB15 — all three tracks, all 45 features, all 10 classes, packet-loss regression intact.
2. In Review 2, add one short appendix section: map the ~10 shared features, collapse both label spaces to a coarse 4-class space, and evaluate your *already-trained* UNSW-NB15 classifier on the IDS2018 rows it has never seen.
3. Report the accuracy drop honestly. A drop from ~85% to ~55% is not a failure — it is a **finding**, and it is a better viva answer than any number you can produce from a merged set.

Cost: roughly one extra notebook section, entirely optional, entirely additive. It buys the "generalisation" story and the AWS/cloud angle without risking the eight marks that depend on a clean single-dataset pipeline. This is exactly the pattern the CloudShield blueprint itself suggests when it proposes BETH for cross-dataset validation.

### A.8 Verdict

**Merging into one training table: not recommended.** It costs the regression track outright, guts feature engineering, dilutes the class space, and introduces a confound that inflates results.

**UNSW-NB15 as the single backbone, IDS2018 as an optional Review-2 generalisation appendix: recommended** — if your instructor permits a second dataset at all.
