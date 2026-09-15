# CloudShield

**Network Traffic Threat Detection & QoS Degradation Forecasting**

23CSE301 Machine Learning Capstone · Academic Year 2026–27

An end-to-end machine learning system over UNSW-NB15 network flow records that
forecasts service-quality degradation, classifies intrusions into attack
families, and profiles traffic behaviour without labels.

---

## The three tracks

| Track | Type | Target | Notebook | Status |
|---|---|---|---|---|
| 1 — QoS degradation forecasting | Regression | `log1p(dbytes)` | `notebooks/02_regression_modeling.ipynb` | executed |
| 2 — Attack classification | Classification | `attack_cat`, 10 classes | `notebooks/03_classification.ipynb` | executed |
| 3 — Behaviour profiling | Clustering | none (labels withheld) | pending | Review 2 |

Full methodology, results and discussion are in
[`PROJECT_REPORT.md`](PROJECT_REPORT.md).

---

## Dataset

**UNSW-NB15**, generated with the IXIA PerfectStorm tool at the Australian
Centre for Cyber Security. Flow features are extracted with Argus and Bro/Zeek
plus twelve custom algorithms.

- Two partition files, 45 columns, 257,673 rows combined
- **10 classes**: `Normal` plus nine attack families — Fuzzers, Analysis,
  Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms

> Moustafa, N. and Slay, J. (2015). *UNSW-NB15: a comprehensive data set for
> network intrusion detection systems.* MilCIS. Free for academic use.

### Two dataset properties that change the results

Both are measured, with the full audit in
[`docs/dataset_documentation/unsw_nb15_audit.md`](docs/dataset_documentation/unsw_nb15_audit.md).

1. **The partition filenames are swapped.** `UNSW_NB15_training-set.csv` holds
   82,332 rows and `UNSW_NB15_testing-set.csv` holds 175,341 — the reverse of
   the split described in the paper. Taking the names at face value means
   training on the smaller file and testing on the larger one.
2. **36.8% of rows are exact duplicates.** With a random split, that places
   byte-identical copies of the same flow on both sides. Removing them before
   splitting also shifts the class prior substantially — `Generic` drops by
   87% — so deduplicated class counts differ from those usually quoted in the
   literature.

---

## Repository layout

```
CloudShield/
├── README.md
├── PROJECT_REPORT.md                full methodology, results and discussion
├── requirements.txt
├── .gitignore  /  .gitattributes    (large CSVs tracked via Git LFS)
│
├── data/
│   ├── download_unsw_nb15.py        fetch + verify the partition files
│   ├── download_cse_cic_ids2018.py
│   ├── download_ton_iot.py
│   └── raw/                         UNSW-NB15 CSVs (LFS)
│
├── processed_data/                  train/val/test splits, both tracks (LFS)
│   ├── train_processed.csv          classification
│   ├── val_processed.csv
│   ├── test_processed.csv
│   └── reg_{train,val,test}_processed.csv    regression
│
├── notebooks/
│   ├── 01_eda.ipynb                 dataset audit and EDA
│   ├── 02_regression_modeling.ipynb Track 1 — 10 regressors
│   └── 03_classification.ipynb      Track 2 — 5 classifiers (Part A)
│
├── preprocessing_regression_corrected_executed.ipynb
├── preprocessing_classification_corrected_executed.ipynb
│
├── src/
│   ├── preprocessing/
│   │   ├── config.py                shared constants and column lists
│   │   ├── cleaning.py              missing values, duplicates, outliers
│   │   ├── features.py              engineered features
│   │   ├── split_scale.py           stratified split, encoding, scaling
│   │   └── pipeline.py              orchestration
│   └── plotting.py                  shared figure style
│
├── reports/
│   ├── regression_results.csv       10-model comparison
│   ├── regression_tuning.csv        GridSearchCV before/after
│   ├── classification_results.csv   5-model comparison
│   ├── classification_cv.csv        5-fold CV on the top two
│   ├── classification_per_class.csv per-class precision/recall/F1
│   └── figures/                     19 exported plots
│
├── results/
│   ├── classification/              additional classification result tables
│   └── figures/classification/      class distribution, confusion matrices,
│                                    decision tree, per-class recall
│
├── models/                          saved best estimators (.joblib)
├── docs/
│   ├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf
│   └── dataset_documentation/unsw_nb15_audit.md
└── app/                             dashboard (Review 2 bonus)
```

---

## Setup

This repository uses **Git LFS** for the dataset CSVs. Install it before
cloning, or the data files will arrive as small pointer stubs:

```bash
git lfs install
```

```bash
git clone https://github.com/pranayr710/CloudShield.git
```

Then create the environment:

```bash
python -m venv .venv
```

Activate — `.venv\Scripts\activate` (PowerShell) or `source .venv/Scripts/activate` (Git Bash) — then:

```bash
pip install -r requirements.txt
```

If the raw data is missing, fetch and verify it:

```bash
python data/download_unsw_nb15.py
```

## How to run

Open the notebooks in order:

1. `notebooks/01_eda.ipynb` — dataset audit and exploratory analysis
2. `preprocessing_regression_corrected_executed.ipynb` and
   `preprocessing_classification_corrected_executed.ipynb` — build the
   processed splits into `processed_data/`
3. `notebooks/02_regression_modeling.ipynb` — Track 1
4. `notebooks/03_classification.ipynb` — Track 2

The modelling notebooks read from `processed_data/` and re-anchor their relative
paths to the project root, so they run correctly from inside `notebooks/`.

---

## Conventions

| | |
|---|---|
| Reproducibility | `random_state = 42` everywhere |
| Split | stratified, fixed, shared across all models within a track |
| Scaling | `StandardScaler` fitted on the training split only |
| Regression format | `Model · R² · RMSE · MAE` |
| Classification format | `Model · Accuracy · Precision · Recall · Weighted F1 · ROC-AUC` |
| Clustering format | `Method · K · Silhouette · Davies-Bouldin · Calinski-Harabasz` |

Macro F1 is reported alongside weighted F1 for classification. At a ~500:1 class
imbalance the weighted score is dominated by `Normal` and conceals near-total
failure on the rare attack families, so the two are always shown together.

**Sequencing.** All baselines are completed before any hyperparameter tuning, so
the tuning comparison has an honest reference point.

---

## Results

Full tables and discussion in [`PROJECT_REPORT.md`](PROJECT_REPORT.md); raw CSVs
in `reports/`.

### Track 2 — Classification (Part A, five algorithms)

| Rank | Algorithm | Acc (val) | Acc (test) | F1 weighted (test) | F1 macro (test) |
|---|---|---|---|---|---|
| 1 | K-Nearest Neighbors | 0.8749 | **0.8197** | **0.8189** | 0.5063 |
| 2 | Decision Tree | 0.8729 | 0.8087 | 0.8186 | **0.5771** |
| 3 | Logistic Regression | 0.8488 | 0.7825 | 0.7957 | 0.4789 |
| 4 | SVC (RBF) | 0.8460 | 0.7784 | 0.7923 | 0.4589 |
| 5 | Gaussian Naive Bayes | 0.7486 | 0.6446 | 0.6747 | 0.3114 |

Majority-class baseline (always predict `Normal`): accuracy 0.319, weighted F1
0.155. Every model beats it by a wide margin.

Two observations worth carrying into the viva:

- **KNN and the Decision Tree are effectively tied on weighted F1** (0.8189 vs
  0.8186), but the tree wins clearly on macro F1 (0.577 vs 0.506) — it handles
  the rare classes better, which is what matters for a detection system.
- **Gaussian Naive Bayes fails as predicted.** Its conditional-independence
  assumption is badly violated: UNSW-NB15 flow statistics are all derived from
  the same packet stream and are heavily correlated. Including it demonstrates
  the point rather than wasting a slot.

### Track 1 — Regression

Ten algorithms compared on `log1p(dbytes)`. See `reports/regression_results.csv`
and `reports/regression_tuning.csv`.

### Track 3 — Clustering

Review 2.
