# CloudShield

**Multi-Dataset ML-Driven Cloud Threat Detection and Resource Abuse Prediction**

23CSE301 Machine Learning Capstone · Academic Year 2026–27

CloudShield analyses cloud and network security from three complementary angles,
using a different dataset and a different learning paradigm for each. It does not
depend on a single algorithm or a single capture.

---

## Architecture

```
 CSE-CIC-IDS2018            UNSW-NB15                  TON_IoT
        │                       │                         │
        ▼                       ▼                         ▼
   Regression             Classification              Clustering
        │                       │                         │
        ▼                       ▼                         ▼
 Resource-abuse          Attack category           Behaviour
   prediction              prediction               clusters
        │                       │                         │
        └───────────────────────┼─────────────────────────┘
                                ▼
                    CloudShield Decision Layer
                                ▼
                        Security Dashboard
```

| Track | Dataset | Task | Target | Question answered |
|---|---|---|---|---|
| 1 | CSE-CIC-IDS2018 | Regression | network byte rate | How much resource is this workload consuming? |
| 2 | UNSW-NB15 | Classification | `attack_cat`, 10 classes | If traffic is malicious, what kind of attack is it? |
| 3 | TON_IoT | Clustering | none (labels withheld) | What hidden behaviour patterns exist? |

Three datasets rather than one, because no single public capture supports all
three questions well. UNSW-NB15 carries a rich multiclass attack taxonomy;
CSE-CIC-IDS2018 was deployed on AWS and carries the flow-rate features that make
resource forecasting meaningful; TON_IoT contributes heterogeneous IoT/IIoT
telemetry suited to unsupervised behaviour discovery.

---

## Current status

| Track | Dataset present | Work complete | Results |
|---|---|---|---|
| 2 — Classification | ✅ yes | Member 1: classifiers 1–5 | **executed, in `results/classification/`** |
| 1 — Regression | ❌ not downloaded | Member 1: pipeline + notebook built | **none — pending download** |
| 3 — Clustering | ❌ not downloaded | Member 3 | not started |

`notebooks/01_cse_ids2018_regression.ipynb` runs top to bottom without the
dataset present and reports `pending data` in every modelling cell rather than
producing numbers it did not compute.

---

## Repository layout

```
CloudShield/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── download_cse_cic_ids2018.py     fetch + inspect Track 1 data
│   ├── download_unsw_nb15.py           fetch + verify Track 2 data
│   ├── download_ton_iot.py             fetch Track 3 data
│   ├── raw/{CSE-CIC-IDS2018,UNSW-NB15,TON_IoT}/      (gitignored)
│   └── processed/                                     (gitignored)
├── notebooks/
│   ├── 01_cse_ids2018_regression.ipynb      Track 1 — Member 1 (models 1–3)
│   ├── 02_unsw_nb15_classification.ipynb    Track 2 — Member 1 (models 1–5)
│   ├── 03_ton_iot_clustering.ipynb          Track 3 — Member 3
│   └── 04_cloudshield_integration.ipynb     final integration
├── src/
│   ├── preprocessing/
│   │   ├── cse_cic_ids2018.py    Track 1 pipeline (schema-inspecting)
│   │   └── unsw_nb15.py          Track 2 pipeline
│   ├── regression/ classification/ clustering/
│   └── utils/
│       ├── config.py     every shared constant, in one place
│       ├── metrics.py    the three results formats
│       └── plotting.py   shared figure style
├── models/{regression,classification,clustering}/
├── results/{regression,classification,clustering,figures}/
├── app/                  Streamlit dashboard
└── docs/
    ├── dataset_documentation/unsw_nb15_audit.md
    └── project_report/ architecture/ screenshots/
```

---

## Setup

```bash
python -m venv .venv
```

Activate — `.venv\Scripts\activate` (PowerShell) or `source .venv/Scripts/activate` (Git Bash) — then:

```bash
pip install -r requirements.txt
```

Fetch the datasets. Each script verifies what it finds and prints instructions
if the data is missing:

```bash
python data/download_unsw_nb15.py
```

```bash
python data/download_cse_cic_ids2018.py --aws
```

```bash
python data/download_ton_iot.py
```

Then run the notebooks in order: `01` → `02` → `03` → `04`.

---

## Shared conventions

These are what keep three members and three datasets from becoming three
unrelated projects.

**Reproducibility.** `random_state = 42` everywhere. Test size 0.20. Five-fold
cross-validation.

**One pipeline per dataset, shared by every model.** Members import
`get_classification_data()` or `get_regression_data()` rather than writing their
own preprocessing. If each member built their own, the consolidated comparison
tables would be comparing models trained on different data.

**One results format per track**, from `src/utils/metrics.py`:

| Track | Columns |
|---|---|
| Regression | `Model · R² · RMSE · MAE` |
| Classification | `Model · Accuracy · Precision · Recall · Weighted F1 · ROC-AUC` |
| Clustering | `Method · K · Silhouette · Davies-Bouldin · Calinski-Harabasz` |

Macro F1 is reported alongside weighted F1 on the classification track, because
at a 500:1 class imbalance the weighted score is dominated by the majority class
and hides rare-class failure entirely.

**Leakage prevention.** The target never enters `X`. Scalers, encoders, clip
bounds and rare-level whitelists are fitted on the training split only. Both
pipelines assert this, and the notebooks prove it numerically: the scaled
training block is centred to machine precision while the test block is not — an
asymmetry that can only exist if the transform never saw the held-out rows.

**Sequencing.** Baselines for all ten models complete before any tuning begins.

**No assumed schema.** The Track 1 pipeline resolves its target column against a
list of known aliases and raises `TargetColumnNotFound` if none matches, rather
than substituting a different column. The spec names the target `fl_byt_s`,
which is the abbreviated form used by some CIC-IDS2017 redistributions;
CSE-CIC-IDS2018 normally uses `Flow Byts/s`. The pipeline determines which is
actually present.

---

## Team ownership

| | Regression (Track 1) | Classification (Track 2) | Clustering (Track 3) |
|---|---|---|---|
| **Member 1** | Linear, Ridge, Lasso | Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM | — |
| **Member 2** | ElasticNet, Polynomial, Decision Tree | Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP | — |
| **Member 3** | Random Forest, Gradient Boosting, SVR, KNN | — | K-Means, Agglomerative, + final integration |

Members do not implement one another's models. Appending to a shared table is
documented at the end of each notebook.

---

## Datasets

**CSE-CIC-IDS2018** — Canadian Institute for Cybersecurity, captured on AWS
(~16M flows, 80 CICFlowMeter features, 7 attack families).
Sharafaldin, I., Habibi Lashkari, A. and Ghorbani, A. (2018). *Toward Generating
a New Intrusion Detection Dataset and Intrusion Traffic Characterization.*
ICISSP.

**UNSW-NB15** — Australian Centre for Cyber Security, IXIA PerfectStorm testbed
(257,673 flows across both partition files, 45 features, 10 classes).
Moustafa, N. and Slay, J. (2015). *UNSW-NB15: a comprehensive data set for
network intrusion detection systems.* MilCIS.

> Two properties of UNSW-NB15 materially affect Track 2 and are documented in
> [`docs/dataset_documentation/unsw_nb15_audit.md`](docs/dataset_documentation/unsw_nb15_audit.md):
> the two partition files ship with their **names swapped** relative to the
> published split, and **36.8% of rows are exact duplicates** which would
> otherwise contaminate the test set. Removing the duplicates changes the class
> distribution substantially, so our class counts differ from those usually
> quoted in the literature — by design.

**TON_IoT** — UNSW Canberra, IoT/IIoT telemetry plus host and network logs.
Moustafa, N. (2021). *A new distributed architecture for evaluating AI-based
security systems at the edge: Network TON_IoT datasets.* Sustainable Cities and
Society, 72, 102994.

All three are free for academic use. None is committed to this repository; the
download scripts fetch them.

---

## Results

Populated as each track completes. Nothing is listed here that has not been
computed.

| Track | Best model | Metric | Status |
|---|---|---|---|
| Classification | see `results/classification/member1_results.csv` | Weighted F1 / Macro F1 | Member 1 baselines complete |
| Regression | — | R² / RMSE / MAE | pending dataset download |
| Clustering | — | Silhouette / DB / CH | not started |
