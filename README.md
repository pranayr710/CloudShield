# CloudShield

**Machine learning on network traffic: attack classification, behaviour clustering and response-volume forecasting**

23CSE301 Machine Learning Capstone · Academic Year 2026–27

---

## Problem statement

Every connection on a network leaves a flow record: duration, packets and bytes each way, protocol, service, connection state. CloudShield asks three questions of those records, one per track:

| Track | Question | Method | Dataset | Notebook |
|---|---|---|---|---|
| Classification (Part A + Part B) | Which of ten traffic types is this flow: normal, or one of nine attacks? | 10 supervised classifiers | TON_IoT (network) | [`notebooks/classification.ipynb`](notebooks/classification.ipynb) |
| Clustering | Without labels, what natural groups of behaviour exist? | K-Means, Agglomerative | TON_IoT (network) | [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) |
| Regression | How many bytes will the reply side of a flow carry? | 10 regressors | UNSW-NB15 | 10 regressors evaluated |

---

## Datasets

### TON_IoT, network portion (classification and clustering)

`train_test_network.csv` from the UNSW Canberra IoT/IIoT testbed: 211,043 flows, 44 columns, 10 classes (`normal`, `scanning`, `ddos`, `dos`, `injection`, `password`, `backdoor`, `xss`, `ransomware`, `mitm`).

- 20,569 exact duplicate rows (9.7%) are removed first, leaving 190,474 flows.
- `mitm` is rare: 1,041 flows, 0.55% of the data.
- Every attack type comes from only 2 to 6 source machines, so machine identifiers (`src_ip`, `dst_ip`, `src_port`) are dropped before modelling.
- `label` is 0 exactly when `type` is `normal`, so it is dropped as a target leak.

> Moustafa, N. (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. *Sustainable Cities and Society*, 72, 102994.
> https://research.unsw.edu.au/projects/toniot-datasets

### UNSW-NB15 (regression)

Two partition files, 45 columns, 257,673 flows. The regression target is `dbytes` (bytes from destination to source), modelled as `log1p(dbytes)`. `dmean` and `dload` are formulas of the target and are dropped. Audit notes: [`docs/dataset_documentation/unsw_nb15_audit.md`](docs/dataset_documentation/unsw_nb15_audit.md).

> Moustafa, N. and Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *MilCIS*.

---

## Results

### Classification: all ten algorithms (TON_IoT, test set of 38,095 flows)

One stratified 80:20 split (`random_state=42`) shared by every model; every model's hyperparameters chosen by 3-fold cross-validation on training data only. SVC was trained on a 20,000-row stratified sample because RBF kernels scale quadratically.

| Rank | Model | Part | Accuracy | Precision (w) | Recall (w) | F1 (w) | F1 (macro) | ROC-AUC (OvR) | Tuned settings |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Gradient Boosting** | B | **0.9881** | **0.9882** | **0.9881** | **0.9881** | **0.9663** | **0.9997** | learning_rate 0.05, max_iter 200 |
| 2 | Bagging | B | 0.9870 | 0.9872 | 0.9870 | 0.9871 | 0.9657 | 0.9996 | 100 trees, max_features 0.5 |
| 3 | K-Nearest Neighbors | A | 0.9864 | 0.9865 | 0.9864 | 0.9865 | 0.9637 | 0.9935 | k=5, Manhattan, distance-weighted |
| 4 | Random Forest | B | 0.9864 | 0.9865 | 0.9864 | 0.9864 | 0.9645 | 0.9979 | 100 trees, max_features 0.3 |
| 5 | Decision Tree | A | 0.9854 | 0.9865 | 0.9854 | 0.9858 | 0.9566 | 0.9891 | max_depth 25 |
| 6 | MLP | B | 0.9817 | 0.9822 | 0.9817 | 0.9818 | 0.9537 | 0.9995 | (128, 64), tanh |
| 7 | SVC | A | 0.9767 | 0.9772 | 0.9767 | 0.9768 | 0.9426 | 0.9970 | RBF, C=50, gamma=0.05 |
| 8 | Logistic Regression | A | 0.9536 | 0.9674 | 0.9536 | 0.9590 | 0.8996 | 0.9960 | C=10, balanced |
| 9 | AdaBoost | B | 0.9093 | 0.9162 | 0.9093 | 0.9092 | 0.8583 | 0.9650 | 300 rounds, learning_rate 0.5 |
| 10 | Gaussian Naive Bayes | A | 0.4455 | 0.5749 | 0.4455 | 0.4361 | 0.4245 | 0.9467 | none |

**5-fold cross-validation (weighted F1), top two:** Gradient Boosting 0.9872 ± 0.0006, Bagging 0.9867 ± 0.0008.

**Final model: Gradient Boosting.**

| Gradient Boosting | Weighted F1 | Macro F1 |
|---|---|---|
| Library default settings | 0.9846 | 0.9496 |
| Tuned by cross-validation | 0.9881 | 0.9663 |
| Wider GridSearchCV on the full training set | 0.9881 | 0.9650 |

**Stress test, unseen machines:** retrained on flows from some source machines and tested on machines it never saw (grouped by `src_ip`), the same configuration scores accuracy 0.7361, weighted F1 0.6984. Only 8 of the 10 classes occur in that fold.

### Clustering (TON_IoT, labels withheld)

30,000-flow sample; agglomerative clustering on a 6,000-flow subset because it is O(n²) in memory.

| Method | k | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|---|
| K-Means | 6 | 0.4741 | 0.7426 | 2,953.6 |
| Agglomerative (Ward) | 6 | 0.4622 | 0.7524 | 2,844.8 |

- `k` chosen by the elbow of the inertia curve (k = 2 to 12).
- Linkage compared over Ward, complete, average and single; the last three put 99% or more of the flows into one cluster.
- Stability: K-Means refitted on 5 random samples gives silhouette 0.467 ± 0.011, and the runs agree on the same flows with ARI 0.943 ± 0.043.
- PCA (2 components) scatter for both algorithms, t-SNE for K-Means, and a cluster-by-type composition heatmap are in the notebook.

### Regression (UNSW-NB15, target `log1p(dbytes)`)

| Rank | Model | R² (test) | RMSE (bytes) | MAE (bytes) |
|---|---|---|---|---|
| 1 | **Random Forest** | **0.9994** | 17,425 | 528 |
| 2 | Decision Tree | 0.9986 | 44,300 | 863 |
| 3 | Gradient Boosting | 0.9983 | 42,049 | 2,274 |
| 4 | SVR (RBF) | 0.9947 | 114,288 | 4,540 |
| 5 | KNN | 0.9916 | 85,463 | 3,525 |
| 6 | Ridge | 0.9831 | 180,732 | 11,681 |
| 7 | Linear Regression | 0.9831 | 180,754 | 11,678 |
| 8 | Polynomial (degree 2) | 0.9796 | 219,334 | 8,556 |
| 9 | ElasticNet | 0.8779 | 143,891 | 14,236 |
| 10 | Lasso | 0.8113 | 144,216 | 14,431 |

GridSearchCV: Random Forest RMSE 17,581 → 17,425; Gradient Boosting 42,049 → 27,033 (R² 0.9983 → 0.9991). 5-fold CV R²: Random Forest 0.9995 ± 0.0001, Gradient Boosting 0.9992 ± 0.0001.

All result tables are CSV files in `results/` (TON_IoT) and `reports/` (UNSW-NB15); figures are in `results/figures/` and `reports/figures/`.

---

## Repository layout

```
CloudShield/
├── README.md
├── PROJECT_REPORT.md                    Review 1 report (UNSW-NB15)
├── requirements.txt
├── data/
│   ├── download_ton_iot.py              TON_IoT acquisition instructions
│   ├── download_unsw_nb15.py            UNSW-NB15 fetch and verification
│   ├── download_cse_cic_ids2018.py
│   └── raw/                             datasets (not committed)
├── notebooks/
│   ├── classification.ipynb             Classification Part A + B, TON_IoT
│   ├── clustering.ipynb                 Clustering, TON_IoT
│   ├── 01_eda.ipynb                     EDA, UNSW-NB15
│   ├── 03_classification.ipynb          Review 1 Part A on UNSW-NB15 (superseded)
│   └── 03_classification_1.ipynb        early Part B exploration on TON_IoT (superseded)
├── preprocessing/                       UNSW-NB15 preprocessing notebooks
├── src/preprocessing/                   UNSW-NB15 preprocessing package
├── reports/                             UNSW-NB15 result tables and figures
├── results/                             TON_IoT result tables, figures, final model
├── models/                              saved UNSW-NB15 models
├── cloudshield-site/                    presentation website (React + Vite)
└── docs/                                guidelines and dataset audit
```

---

## Setup

```bash
git clone https://github.com/pranayr710/CloudShield.git
cd CloudShield
python -m venv .venv
.venv\Scripts\activate          # PowerShell;  source .venv/Scripts/activate in Git Bash
pip install -r requirements.txt
```

Datasets are not committed.

- **TON_IoT:** download `train_test_network.csv` (Train_Test_datasets → Train_Test_Network) from the official page and place it at `data/raw/TON_IoT/train_test_network.csv`. `python data/download_ton_iot.py` prints the steps.
- **UNSW-NB15:** `python data/download_unsw_nb15.py`, then run the two notebooks in `preprocessing/` to build `processed_data/`.

## How to run

| Notebook | Needs | Runtime (4-core CPU) |
|---|---|---|
| `notebooks/classification.ipynb` | TON_IoT CSV | about 50 minutes, half of it hyperparameter search |
| `notebooks/clustering.ipynb` | TON_IoT CSV | about 2 minutes |

Open a notebook and run all cells top to bottom. The TON_IoT notebooks look for the CSV in `data/raw/TON_IoT/` locally, or in any dataset attached under `/kaggle/input/` when run on Kaggle, and write their tables, figures and final model to `results/` (or `/kaggle/working` on Kaggle).

### Presentation website

```bash
cd cloudshield-site
npm install
npm run build
npm run preview        # http://localhost:4173
```

The site reads `src/data/toniot.json` and `src/data/cloudshield.json`. To refresh them after rerunning the notebooks: `python scripts/export_ton_iot.py` and `python scripts/export_data.py` from inside `cloudshield-site/` (use the Python environment that ran the notebooks, so the saved models load).

---

## Conventions

| | |
|---|---|
| Reproducibility | `random_state = 42` everywhere |
| Split | one stratified 80:20 split per track, shared by every model |
| Leakage | encoders, scalers and port buckets fitted on the training split only; duplicates removed before splitting |
| Tuning | 3-fold CV on training data; the test set is only used to report results |
| Classification metrics | Accuracy, Precision, Recall, weighted F1, macro F1, ROC-AUC (one-vs-rest), confusion matrix |
| Clustering metrics | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Regression metrics | R², RMSE, MAE (log scale and bytes) |
