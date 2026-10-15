# CloudShield

**Machine learning on network traffic: attack classification, behaviour clustering and response-volume forecasting**

23CSE301 Machine Learning Capstone · Academic Year 2026–27

---

## Problem statement

Every connection on a network leaves a flow record: duration, packets and bytes each way, protocol, service, connection state. CloudShield asks three questions of those records, one per track:

| Track | Question | Method | Dataset | Notebook |
|---|---|---|---|---|
| Classification (Part A + Part B) | Which of ten traffic types is this flow: normal, or one of nine attacks? | 10 supervised classifiers (12 configs evaluated) | TON_IoT (network) | [`notebooks/03_classification_1.ipynb`](notebooks/03_classification_1.ipynb) |
| Clustering | Without labels, what natural groups of behaviour exist? | K-Means, Agglomerative | TON_IoT (network) | [`notebooks/clustering.ipynb`](notebooks/clustering.ipynb) |
| Comparative study | How do the algorithms of each track compare beyond one score? | all classifiers and clusterings side by side | TON_IoT (network) | [`notebooks/04_comparative_study.ipynb`](notebooks/04_comparative_study.ipynb) |
| Regression | How many bytes will the reply side of a flow carry? | 10 regressors | UNSW-NB15 | [`notebooks/02_regression_modeling.ipynb`](notebooks/02_regression_modeling.ipynb) |

---

## Datasets

### TON_IoT, network portion (classification and clustering)

`train_test_network.csv` from the UNSW Canberra IoT/IIoT testbed: 211,043 flows, 44 columns, 10 classes (`normal`, `scanning`, `ddos`, `dos`, `injection`, `password`, `backdoor`, `xss`, `ransomware`, `mitm`).

- 20,569 exact duplicate rows (9.7%) are removed first, leaving 190,474 flows.
- `mitm` is rare: 1,041 flows, 0.55% of the data.
- Every attack type comes from only 2 to 6 source machines, so the classification test set is split by source machine (`src_ip`): no machine appears in both training and test.
- `label` is 0 exactly when `type` is `normal`, so it is dropped as a target leak.

> Moustafa, N. (2021). A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets. *Sustainable Cities and Society*, 72, 102994.
> https://research.unsw.edu.au/projects/toniot-datasets

### UNSW-NB15 (regression)

Two partition files, 45 columns, 257,673 flows. The regression target is `dbytes` (bytes from destination to source), modelled as `log1p(dbytes)`. `dmean` and `dload` are formulas of the target and are dropped. Audit notes: [`docs/dataset_documentation/unsw_nb15_audit.md`](docs/dataset_documentation/unsw_nb15_audit.md).

> Moustafa, N. and Slay, J. (2015). UNSW-NB15: a comprehensive data set for network intrusion detection systems. *MilCIS*.

---

## Results

### Classification: all algorithms & tuned configurations (TON_IoT)

45,000-flow stratified sample of the de-duplicated data, split by source machine: six machines (`192.168.1.31, .33, .36, .39, .180, .195`) form the test set, chosen so that all 10 traffic types appear on both sides. Every flow from a machine is in training or in test, never both, so the test score measures how a model does on machines it has never seen. Training: 33,912 flows; test: 11,088 flows; 83 encoded features.

| Rank | Model | Part | Accuracy | Precision (w) | Recall (w) | F1 (weighted) | F1 (macro) | ROC-AUC (OvR) | Train time (s) | Settings |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Bagging (tuned)** | B | **0.9217** | **0.9386** | **0.9217** | **0.9249** | **0.8351** | **0.9894** | 26.0 | 200 balanced trees, max_features=0.3 |
| 2 | Bagging | B | 0.9103 | 0.9337 | 0.9103 | 0.9134 | 0.8463 | 0.9869 | 62.0 | 100 balanced trees, max_features=0.3 |
| 3 | Gradient Boosting | B | 0.8863 | 0.9257 | 0.8863 | 0.8871 | 0.8344 | 0.9912 | 332.3 | 100 trees, depth 3, learning_rate=0.1, balanced weights |
| 4 | Decision Tree | A | 0.8816 | 0.9230 | 0.8816 | 0.8854 | 0.7799 | 0.9166 | 0.7 | max_depth=None, class_weight=balanced |
| 5 | Gradient Boosting (tuned) | B | 0.8842 | 0.9232 | 0.8842 | 0.8850 | 0.8270 | 0.9923 | 176.0 | 100 trees, depth 4, learning_rate=0.05, subsample=0.8 |
| 6 | Random Forest | B | 0.8844 | 0.9209 | 0.8844 | 0.8833 | 0.8025 | 0.9883 | 5.6 | 200 trees, class_weight=balanced_subsample |
| 7 | K-Nearest Neighbors | A | 0.8821 | 0.9231 | 0.8821 | 0.8831 | 0.8064 | 0.9522 | 0.2 | k=5, distance-weighted |
| 8 | Neural Network (MLP) | B | 0.8776 | 0.9181 | 0.8776 | 0.8795 | 0.7621 | 0.9607 | 462.6 | MLP (100, 50), ReLU, Adam |
| 9 | AdaBoost | B | 0.8508 | 0.8875 | 0.8508 | 0.8540 | 0.7492 | 0.9735 | 63.8 | 100 rounds of depth-3 trees, learning_rate=1.0, balanced weights |
| 10 | SVC (RBF) | A | 0.7771 | 0.8613 | 0.7771 | 0.8024 | 0.6791 | 0.9374 | 8.7 | C=1.0, RBF, class_weight=balanced, 12k-row sample |
| 11 | Logistic Regression | A | 0.7567 | 0.8388 | 0.7567 | 0.7751 | 0.6147 | 0.9637 | 221.6 | C=1.0, class_weight=balanced |
| 12 | Gaussian Naive Bayes | A | 0.3920 | 0.4684 | 0.3920 | 0.3243 | 0.2927 | 0.9159 | 0.3 | no hyperparameters |

ROC-AUC is one-vs-rest, averaged over the 10 classes (probabilities, or the SVC's decision function). Training time is the wall-clock fit time from `notebooks/04_comparative_study.ipynb`.

**Hyperparameter tuning (grid search, 3-fold CV grouped by machine on the training set; up to 8 settings per model):**
- **Bagging:** test F1w 0.9134 → tuned **0.9249**, accuracy 0.9103 → 0.9217; selected `{'n_estimators': 200, 'max_features': 0.3, 'estimator__max_depth': None}` (grouped CV F1w 0.3816).
- **Gradient Boosting:** test F1w 0.8871 → tuned 0.8850, ROC-AUC 0.9912 → 0.9923; selected `{'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.8}` (grouped CV F1w 0.3698).

The grouped tuning folds score far below the test set: most attack types come from one to three training machines, so a validation fold often holds every machine of a type.

**5-fold cross-validation of the top two (training set only, weighted F1):**

| Model | Grouped by source machine | Stratified |
|---|---|---|
| Bagging (tuned) | 0.5203 ± 0.3107 (folds 0.170, 0.243, 0.441, 0.989, 0.758) | 0.9817 ± 0.0009 |
| Bagging | 0.5186 ± 0.3105 (folds 0.171, 0.244, 0.433, 0.990, 0.756) | 0.9827 ± 0.0007 |

Grouped folds keep every machine on one side, as the test split does; a fold that holds all the machines of an attack type never trains on it, which is why some grouped folds score low. Stratified folds share machines between training and validation.

**Final model: Bagging (tuned).** On the 11,088 test flows from 6 unseen machines: accuracy 0.9217, weighted precision 0.9386, weighted recall 0.9217, weighted F1 0.9249, macro F1 0.8351, ROC-AUC (one-vs-rest) 0.9894.

Per-class test performance of the final model:

| Class | Precision | Recall | F1-Score | Test flows |
|---|---|---|---|---|
| `password` | 0.9886 | 0.9714 | 0.9799 | 2,411 |
| `scanning` | 0.9662 | 0.9822 | 0.9741 | 843 |
| `ransomware` | 0.9753 | 0.9724 | 0.9738 | 1,703 |
| `injection` | 0.9563 | 0.9616 | 0.9590 | 2,868 |
| `DoS` | 0.9353 | 0.9383 | 0.9368 | 308 |
| `normal` | 0.9766 | 0.7387 | 0.8412 | 1,864 |
| `DDoS` | 0.6818 | 0.9108 | 0.7798 | 807 |
| `MITM` | 0.6233 | 0.7280 | 0.6716 | 125 |
| `backdoor` | 0.5676 | 0.8077 | 0.6667 | 26 |
| `XSS` | 0.4100 | 0.9248 | 0.5681 | 133 |

Persisted tables: [`results/classification_10model_ton_iot.csv`](results/classification_10model_ton_iot.csv) (the notebook's leaderboard, which also lists three post-hoc variants: a soft-vote blend, tuned thresholds and a two-stage detector), [`results/classification_10model_per_class.csv`](results/classification_10model_per_class.csv), and [`results/comparative/classification_comparison.csv`](results/comparative/classification_comparison.csv).

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

### Comparative study (both tracks)

[`notebooks/04_comparative_study.ipynb`](notebooks/04_comparative_study.ipynb) refits every model with the exact data preparation and settings of the track notebooks and compares them on more than one score. Tables: `results/comparative/`; figures: `results/figures/comparative/`.

**Classification, 10 algorithms (12 configurations):**

| Analysis | What it shows | Output |
|---|---|---|
| Consolidated table | all six metrics plus training and prediction time | `classification_comparison.csv` |
| Rank heatmap | rank of every model on every metric, and the mean rank | `cls_01_rank_heatmap.png`, `classification_ranks.csv` |
| Grouped bars and radar | every metric side by side; top five vs the linear baseline | `cls_02_metrics_bars.png`, `cls_03_radar.png` |
| Per-class F1 heatmap | which algorithm fails on which traffic type | `cls_04_per_class_f1.png`, `classification_per_class_f1.csv` |
| Confusion matrices and ROC curves | all 12 configurations | `cls_05_confusion_all.png`, `cls_06_roc_all.png` |
| Cost against benefit | weighted F1 vs training time, Pareto front | `cls_07_f1_vs_time.png` |
| Algorithm families | linear, distance, probabilistic, kernel, single tree, bagging, boosting, neural network | `cls_08_families.png`, `classification_families.csv` |
| Base vs tuned | change in every metric from tuning | `cls_09_tuning_delta.png`, `classification_tuning_delta.csv` |
| Stability | 5-fold CV of the top two, grouped vs stratified | `cls_10_cv_stability.png`, `classification_cv5_folds.csv` |
| Significance | 95% bootstrap intervals; McNemar's test of the best model against every other | `cls_11_bootstrap_ci.png`, `classification_bootstrap_ci.csv`, `classification_mcnemar.csv` |

| Family | Models | Best F1w | Mean F1w | Mean macro F1 |
|---|---|---|---|---|
| Bagging ensemble | Bagging (tuned), Bagging, Random Forest | 0.9249 | 0.9072 | 0.8280 |
| Boosting ensemble | Gradient Boosting, Gradient Boosting (tuned), AdaBoost | 0.8871 | 0.8754 | 0.8036 |
| Single tree | Decision Tree | 0.8854 | 0.8854 | 0.7799 |
| Distance-based | K-Nearest Neighbors | 0.8831 | 0.8831 | 0.8064 |
| Neural network | Neural Network | 0.8795 | 0.8795 | 0.7621 |
| Kernel | SVC (RBF) | 0.8024 | 0.8024 | 0.6791 |
| Linear | Logistic Regression | 0.7751 | 0.7751 | 0.6147 |
| Probabilistic | Gaussian NB | 0.3243 | 0.3243 | 0.2927 |

Weighted F1 of Bagging (tuned): 0.9249, 95% bootstrap interval 0.9203–0.9297. McNemar's test against it is significant (p < 0.05) for 11 of the other 11 configurations.

**Clustering (same 6,000 flows for every algorithm; k = 6 from the elbow):**

| Algorithm | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | ARI ↑ | NMI ↑ | Purity ↑ | Largest cluster | Fit (s) |
|---|---|---|---|---|---|---|---|---|---|
| K-Means | 6 | 0.4741 | 0.7426 | 2,953.6 | 0.2594 | 0.4216 | 0.4587 | 0.3550 | 1.33 |
| Agglomerative (ward) | 6 | 0.4622 | 0.7524 | 2,844.8 | 0.2487 | 0.4162 | 0.4567 | 0.3707 | 4.91 |
| Agglomerative (complete) | 6 | 0.5813 | 0.2994 | 490.9 | 0.0000 | 0.0086 | 0.2263 | 0.9897 | 4.78 |
| Agglomerative (average) | 6 | 0.6799 | 0.3657 | 386.2 | -0.0005 | 0.0030 | 0.2235 | 0.9982 | 5.08 |
| Agglomerative (single) | 6 | 0.6798 | 0.2671 | 384.7 | -0.0005 | 0.0026 | 0.2233 | 0.9983 | 3.86 |
| MiniBatch K-Means *(extra)* | 6 | 0.4908 | 0.7809 | 1,445.0 | 0.2443 | 0.4325 | 0.4598 | 0.3547 | 0.46 |
| Gaussian Mixture *(extra)* | 6 | 0.4889 | 0.8003 | 1,427.5 | 0.2453 | 0.4337 | 0.4618 | 0.3528 | 7.97 |
| BIRCH *(extra)* | 6 | 0.4066 | 0.5104 | 1,290.1 | 0.0954 | 0.2436 | 0.3167 | 0.7845 | 0.71 |
| DBSCAN *(extra)* | 23 | 0.6241 | 0.5179 | 3,167.4 | 0.3005 | 0.5408 | 0.5659 | 0.2946 | 0.55 |

ARI, NMI and purity compare the clusters with the true traffic type after fitting; the labels are never used to fit. *Extra* algorithms are outside the required list and shown for reference. The notebook also sweeps k = 2 to 10 for four algorithms (`clu_01_k_sweep.png`), shows cluster make-up per algorithm (`clu_02_cluster_vs_type.png`), every clustering on the same PCA axes (`clu_03_pca_all.png`), a shared t-SNE (`clu_04_tsne_compare.png`), stability over five runs (Agglomerative (ward) ARI 0.921, BIRCH ARI 0.980, Gaussian Mixture ARI 0.781, K-Means ARI 0.999; `clu_05_stability.png`), fit time from 1,000 to 10,000 flows (`clu_06_runtime.png`) and a rank heatmap (`clu_07_rank_heatmap.png`).

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

## Live models (deployment)

The three final models are served behind input forms:

| App | File | Where |
|---|---|---|
| Streamlit (main) | `app/streamlit_app.py` | Streamlit Community Cloud, main file `app/streamlit_app.py` |
| Gradio | `app/app.py` | Hugging Face Space `pranayraparla/Ml_Capstone` (API endpoints `/classify`, `/cluster`, `/regress`) |

Both apps load the same bundles from `app/models/`: Bagging (tuned) for TON_IoT classification, K-Means (k = 6) for TON_IoT clustering and the Random Forest for UNSW-NB15 regression. `app/build_models.py` rebuilds the bundles from the notebooks' pipelines and checks their test scores against the notebooks (classification accuracy 0.9217, weighted F1 0.9249; regression R² 0.9994). The classification form's examples are real flows from the held-out test machines, one per traffic type.

The Streamlit app also has pages for the overview, classification, clustering, comparative study and regression results, and a page to download every results table. The website's **Live models** page embeds it.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py      # http://localhost:8501
```

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
│   ├── 01_eda.ipynb                     EDA, UNSW-NB15
│   ├── 02_regression_modeling.ipynb     Regression (10 models), UNSW-NB15
│   ├── 03_classification_1.ipynb        Classification Part A + B (12 configs), TON_IoT
│   ├── 04_comparative_study.ipynb       Comparative study of both tracks, TON_IoT
│   └── clustering.ipynb                 Clustering, TON_IoT
├── preprocessing/                       UNSW-NB15 preprocessing notebooks
├── src/preprocessing/                   UNSW-NB15 preprocessing package
├── reports/                             UNSW-NB15 result tables and figures
├── results/                             TON_IoT result tables, figures (comparative/ holds the comparative study)
├── models/                              saved UNSW-NB15 models
├── app/                                 deployed models: Streamlit and Gradio apps, model bundles
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
| `notebooks/03_classification_1.ipynb` | TON_IoT CSV | about 30–60 minutes, includes the grid tuning and both CVs |
| `notebooks/04_comparative_study.ipynb` | TON_IoT CSV | about 40–60 minutes (refits all 12 configurations) |
| `notebooks/02_regression_modeling.ipynb` | `processed_data/` | about 5 minutes, includes GridSearchCV |
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
| Split | classification: six held-out source machines as the test set, every traffic type on both sides; regression: the official UNSW-NB15 test file; one split per track shared by every model |
| Leakage | encoders, scalers and port buckets fitted on the training split only; duplicates removed before splitting |
| Tuning | cross-validation on training data (grouped grid search for classification, GridSearchCV for regression); the test set only reports results |
| Classification metrics | Accuracy, Precision, Recall, weighted F1, macro F1, ROC-AUC (one-vs-rest), confusion matrix |
| Clustering metrics | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Regression metrics | R², RMSE, MAE (log scale and bytes) |
