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

Evaluated using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by `src_ip`. All flows from a given host machine belong exclusively to either train or test, enforcing zero machine-identifier leakage and evaluating true cross-machine attack generalization on unseen systems.

| Rank | Model | Part | Accuracy (test) | F1 (weighted) | F1 (macro) | Notes / Tuned Settings |
|---|---|---|---|---|---|---|
| 1 | **GBC (tuned)** | B | **0.8365** | **0.8742** | **0.5345** | `learning_rate=0.05, max_depth=5, n_estimators=100, subsample=0.8` |
| 2 | Gradient Boosting (base) | B | 0.7578 | 0.7530 | 0.5207 | `n_estimators=100, learning_rate=0.1, max_depth=3` |
| 3 | Bagging (tuned) | B | 0.7232 | 0.7302 | 0.4882 | `n_estimators=200, max_samples=0.7, max_features=0.2, DT max_depth=30` |
| 4 | Bagging (base) | B | 0.7267 | 0.6921 | 0.4620 | `n_estimators=100, max_features=0.3, DT base` |
| 5 | SVC (RBF) | A | 0.6781 | 0.6399 | 0.4754 | `kernel='rbf', C=1.0, class_weight='balanced'` |
| 6 | Random Forest | B | 0.6668 | 0.6059 | 0.4065 | `n_estimators=100, random_state=42` |
| 7 | Neural Network (MLP) | B | 0.6603 | 0.6213 | 0.4055 | `hidden_layer_sizes=(100, 50), relu, adam, max_iter=300` |
| 8 | Logistic Regression | A | 0.6443 | 0.6135 | 0.3967 | `C=1.0, class_weight='balanced', max_iter=3000` |
| 9 | AdaBoost | B | 0.6411 | 0.6650 | 0.4520 | `n_estimators=100, learning_rate=1.0, depth-3 tree base` |
| 10 | Decision Tree | A | 0.6298 | 0.6239 | 0.4103 | `class_weight='balanced', max_depth=None` |
| 11 | K-Nearest Neighbors | A | 0.5993 | 0.6197 | 0.4061 | `k=5, weights='distance'` |
| 12 | Gaussian Naive Bayes | A | 0.2964 | 0.3959 | 0.2453 | independence assumption violated |

**Hyperparameter tuning (RandomizedSearchCV on 35% training subsample):**
- **Gradient Boosting:** baseline test F1w 0.7530 → tuned test F1w **0.8742** (best CV F1w = 0.9888).
- **Bagging:** baseline test F1w 0.6921 → tuned test F1w **0.7302** (best CV F1w = 0.9846).

**Winning model: Tuned Gradient Boosting (`GBC (tuned)`).** On the 67,001 test flows from 6 unseen machines:
accuracy 0.8365, weighted precision 0.9404, weighted recall 0.8365, weighted F1 0.8742, macro F1 0.5345, ROC-AUC (one-vs-rest over the classes in the test fold) 0.9969.

The test fold holds 6 of the 10 classes; `backdoor`, `xss`, `ransomware`, `mitm` have no flows in it because all their source machines fall in the training split.

Per-class test performance of the tuned Gradient Boosting model:

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| `scanning` | 0.9806 | 0.9584 | 0.9694 |
| `normal` | 0.8718 | 0.9826 | 0.9239 |
| `injection` | 0.8941 | 0.9289 | 0.9112 |
| `password` | 0.8052 | 0.9857 | 0.8864 |
| `DoS` | 0.9994 | 0.7915 | 0.8834 |
| `DDoS` | 0.9841 | 0.6340 | 0.7711 |

Persisted table: [`results/classification_12model_ton_iot.csv`](results/classification_12model_ton_iot.csv). The per-class table above is computed from the tuned Gradient Boosting model itself (`app/build_models.py` rebuilds it on the same split).

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

## Live models (deployment)

The three final models are served behind input forms:

| App | File | Where |
|---|---|---|
| Streamlit (main) | `app/streamlit_app.py` | Streamlit Community Cloud, main file `app/streamlit_app.py` |
| Gradio | `app/app.py` | Hugging Face Space `pranayraparla/Ml_Capstone` (API endpoints `/classify`, `/cluster`, `/regress`) |

Both apps load the same bundles from `app/models/`: Gradient Boosting (tuned) for TON_IoT classification, K-Means (k = 6) for TON_IoT clustering and the Random Forest for UNSW-NB15 regression. `app/build_models.py` rebuilds the bundles from the notebooks' pipelines and checks their test scores against the notebooks (classification 0.8365 / 0.8742 / 0.5345, regression R² 0.9994).

The Streamlit app also has pages for the overview, classification, clustering and regression results, and a page to download every results table. The website's **Live models** page embeds it.

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
│   └── clustering.ipynb                 Clustering, TON_IoT
├── preprocessing/                       UNSW-NB15 preprocessing notebooks
├── src/preprocessing/                   UNSW-NB15 preprocessing package
├── reports/                             UNSW-NB15 result tables and figures
├── results/                             TON_IoT result tables, figures, final model
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
| `notebooks/03_classification_1.ipynb` | TON_IoT CSV | about 20–30 minutes, includes RandomizedSearch tuning |
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
| Split | classification: StratifiedGroupKFold grouped by source machine (first fold as test); regression: the official UNSW-NB15 test file; one split per track shared by every model |
| Leakage | encoders, scalers and port buckets fitted on the training split only; duplicates removed before splitting |
| Tuning | cross-validation on training data (RandomizedSearchCV / GridSearchCV); the test set only reports results |
| Classification metrics | Accuracy, Precision, Recall, weighted F1, macro F1, ROC-AUC (one-vs-rest), confusion matrix |
| Clustering metrics | Silhouette, Davies-Bouldin, Calinski-Harabasz |
| Regression metrics | R², RMSE, MAE (log scale and bytes) |
