# PERSON A — Data Foundation Owner + Clustering Track

> Rename this heading to your actual name.

**You own:** repository, environment, dataset, EDA, preprocessing/feature engineering, plotting helpers, README — and the **Clustering track** in Review 2.

**Marks you are directly responsible for:** Review 1 Sections A (4) + B (3) = **7 marks**. Review 2 Section B clustering = **8 marks**. Total **15 of 50** — the largest single share, because you are the critical path.

**You are the bottleneck for the whole team in Review 1.** B and C cannot produce a single real number until your gate **M2** lands. Everything in Phase 1 and 2 below is urgent. Everything after M2 is polish.

---

## Your dependency summary

| You are blocked by | Nobody. You start immediately. |
|---|---|
| **You block** | **Person B** at their task B-T6, **Person C** at their task C-T6 — both need `src/preprocessing/` + processed data (your gate **M2**, target Day 5). |
| **You wait for** | B and C to finish M3 (Day 8) before you can export final figures (A-T9) and assemble slides. |

**If you slip on M2, the entire project slips by the same number of days.** If you're going to be late, tell B and C the moment you know, so they extend their mock-data phase instead of sitting idle.

---

# PHASE 0 — Day 1 (morning) · Setup · ~2 hours

### ☐ A-T0. Confirm the dataset with the instructor — **do this before anything else**

- [ ] Ask: is the assigned dataset **UNSW-NB15** or **CSE-CIC-IDS2018**?
- [ ] Ask: is a **second dataset** permitted as a cross-validation appendix in Review 2? (Guidelines §1 say datasets are *assigned*, one per track.)
- [ ] Post the answer in the team chat.

**Why this is task zero:** the project brief says *packet loss* + *9 attack types*, which is UNSW-NB15. The CloudShield blueprint says *byte rate* + *7 attack types*, which is CIC-IDS2018. See `PLAN.md` §0 and Appendix A. Everything below assumes **UNSW-NB15**. If the answer is IDS2018, the structure is identical but the target column changes to `Flow Byts/s` and there is no packet-loss track — flag this to the team immediately.

---

### ☐ A-T1. Create the repository skeleton

- [ ] `git init` in `C:\Users\prana\Desktop\sem5\projects\ml`
- [ ] Create the directory tree exactly as guidelines §8 requires:

```
data/raw/  data/processed/  notebooks/  src/  models/  reports/figures/  app/
```

- [ ] Add `.gitkeep` files in `data/raw/`, `data/processed/`, `models/`, `reports/figures/` so empty dirs survive git.
- [ ] Write `.gitignore`:

```gitignore
data/raw/*
data/processed/*
!data/**/.gitkeep
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
env/
*.pkl
!models/best_*.pkl
```

- [ ] Create `src/__init__.py` (empty) so `from src.preprocessing import ...` works.
- [ ] Create a GitHub repo (private or public — ask the instructor), add B and C as collaborators.
- [ ] **Commit:** `Initialise project structure and gitignore`

**Acceptance:** B and C can clone and see the full tree.

---

### ☐ A-T2. Environment + `requirements.txt`

Currently only `numpy`, `scipy` and `matplotlib` are installed on this machine. Everything else is missing.

- [ ] Create a virtual environment (strongly recommended — keeps the team's versions identical):
  ```bash
  python -m venv .venv
  ```
  Activate: `.venv\Scripts\activate` (PowerShell) or `source .venv/Scripts/activate` (Git Bash).
- [ ] Install:
  ```bash
  pip install pandas scikit-learn seaborn matplotlib numpy scipy imbalanced-learn xgboost joblib jupyter pyarrow tqdm
  ```
- [ ] Freeze with pinned versions:
  ```bash
  pip freeze > requirements.txt
  ```
- [ ] Verify every import works:
  ```bash
  python -c "import pandas, sklearn, seaborn, imblearn, xgboost, joblib, pyarrow; print('ok')"
  ```
- [ ] **Commit:** `Add pinned Python dependencies`

**Acceptance:** B and C run `pip install -r requirements.txt` and get an identical environment. Tell them in chat when this lands.

---

### ☐ A-T3. **Freeze and publish the `src.preprocessing` contract** ⚠️ CRITICAL — unblocks B and C on Day 1

Write `src/preprocessing/` as a **stub with the real signature and docstring but a `NotImplementedError` body**, plus a working `make_mock_dataset()` so B and C can develop against fake data immediately.

- [ ] Create the package under `src/preprocessing/`:

```python
"""Shared preprocessing for all three tracks. Owner: Person A.
Do not modify get_dataset()'s signature without telling B and C."""
import numpy as np

RANDOM_STATE = 42
TEST_SIZE = 0.2
LEAKY_COLS = ["spkts", "sbytes", "sload"]   # bounded by / trivially predict sloss


def get_dataset(task, drop_leaky=False, subsample=None, random_state=RANDOM_STATE):
    """Return the fully preprocessed dataset for one track.

    Parameters
    ----------
    task : {"regression", "classification", "clustering"}
    drop_leaky : bool   regression only — drop spkts/sbytes/sload
    subsample : int|None  stratified subsample of the TRAIN set (SVR/SVC/Agglomerative)
    random_state : int

    Returns
    -------
    dict with keys:
      X_train, X_test : np.ndarray, encoded + scaled (scaler fitted on train ONLY)
      y_train, y_test : np.ndarray | None   (None when task == "clustering")
      feature_names   : list[str], len == X_train.shape[1]
      label_encoder   : sklearn LabelEncoder | None
      meta            : dict  n_rows_before_clean, n_dropped, subsampled
    """
    raise NotImplementedError("Person A delivers this at gate M2")


def make_mock_dataset(task, n=2000, n_features=40, random_state=RANDOM_STATE):
    """Fake data with the SAME output shape as get_dataset().
    For B and C to develop model code before M2. Delete usages after M2."""
    rng = np.random.default_rng(random_state)
    n_tr = int(n * (1 - TEST_SIZE))
    X_train = rng.normal(size=(n_tr, n_features))
    X_test = rng.normal(size=(n - n_tr, n_features))
    names = [f"f{i}" for i in range(n_features)]
    if task == "regression":
        y_train = np.abs(X_train[:, 0] * 5 + rng.normal(size=n_tr))
        y_test = np.abs(X_test[:, 0] * 5 + rng.normal(size=n - n_tr))
        le = None
    elif task == "classification":
        y_train = rng.integers(0, 10, n_tr)
        y_test = rng.integers(0, 10, n - n_tr)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(
            ["Normal", "Generic", "Exploits", "Fuzzers", "DoS",
             "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"])
    else:
        y_train = y_test = le = None
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train,
            "y_test": y_test, "feature_names": names, "label_encoder": le,
            "meta": {"mock": True}}
```

- [ ] **Commit:** `Add preprocessing contract and mock dataset generator`
- [ ] **Post in team chat: "M0 done — contract frozen, mock available, start your model code."**

**Acceptance:** `from src.preprocessing import make_mock_dataset; make_mock_dataset("regression")` runs. B and C are now unblocked for a full 4 days of parallel work.

---

# PHASE 1 — Days 1–3 · Get the real data · **Gate M1**

### ☐ A-T4. Download UNSW-NB15

- [ ] Source options (verified Sept 2026 — the CloudStor link in older tutorials is **dead**):
  1. **Kaggle (fastest).** Any of these mirrors: `dhoogla/unswnb15`, `mrwellsdavid/unsw-nb15`, `harshwardhanbhangale/unsw-complete-dataset`, `ucimachinelearning/unsw-nb15-dataset`. Open the **Data** tab and confirm the file list before downloading — mirrors vary.
  2. **Official (authoritative — cite this one in the README).** <https://research.unsw.edu.au/projects/unsw-nb15-dataset> → download link goes to a UNSW SharePoint folder → `CSV Files/Training and Testing Sets/`.
- [ ] You want exactly two files: `UNSW_NB15_training-set.csv` (175,341 × 45) and `UNSW_NB15_testing-set.csv` (82,332 × 45).
- [ ] ⚠️ **Do not grab `UNSW-NB15_1.csv` … `_4.csv`** — that's the 2.54M-row raw dump with **49** columns, **no header row**, different column names, and blank cells for benign `attack_cat`. It needs `NUSW-NB15_features.csv` to name the columns. Everything in this plan assumes the 45-column training/testing partition.
- [ ] Cite in the README: Moustafa, N. & Slay, J. (2015), *UNSW-NB15: a comprehensive data set for network intrusion detection systems*, MilCIS. Free for academic use in perpetuity.
- [ ] Place both CSVs in `data/raw/`. **Do not commit them** — `.gitignore` already excludes them.
- [ ] Write `data/download_data.py` so the repo is reproducible without the CSVs:

```python
"""Download UNSW-NB15 into data/raw/. Run: python data/download_data.py"""
# Kaggle API route (requires ~/.kaggle/kaggle.json):
#   kaggle datasets download -d mrwellsdavid/unsw-nb15 -p data/raw --unzip
# Include a MANUAL FALLBACK note with the official URL and expected filenames,
# and assert the expected shapes after loading so a wrong file fails loudly.
```

- [ ] In the script, assert the shapes: training `(175341, 45)`, testing `(82332, 45)`. If they differ, you have a different variant of the dataset — note it and adjust the plan.
- [ ] **Commit:** `Add dataset download script with shape assertions`

---

### ☐ A-T5. Dataset audit — **rubric A1, 1 mark**

Start `notebooks/01_eda.ipynb`. Markdown title cell, then problem statement, then:

- [ ] Load both CSVs, concatenate into one `df` (you will re-split yourself with a stratified split — do **not** use the provided train/test files as your split, because the official split has a known distribution shift; state this decision in a Markdown cell).
- [ ] `df.drop(columns=['id'])`
- [ ] Report, each in its own cell with printed output:
  - [ ] `df.shape`
  - [ ] `df.dtypes` (or `df.info()`)
  - [ ] `df.isnull().sum()` — full column list, not `.sum().sum()`
  - [ ] `df.duplicated().sum()`
  - [ ] `df['attack_cat'].value_counts()` **and** `value_counts(normalize=True)`
  - [ ] `df['label'].value_counts()`
  - [ ] `df['sloss'].describe()` and `df['dloss'].describe()`
  - [ ] `df.select_dtypes('object').nunique()` — shows `proto` has ~130 levels, `service` ~13, `state` ~11
- [ ] Under each, a **Markdown observation** cell.

**Key observations to write (these earn A3):**
- Class imbalance is extreme: `Normal` ≈ 93k vs `Worms` ≈ 174. Ratio > 500:1.
- `sloss` is a heavily right-skewed count — most flows lose zero packets, a long tail loses thousands. This motivates checking whether a log transform helps regression.
- `proto` has ~130 levels — one-hot encoding it naively would add 130 columns; note that you'll frequency-encode or top-N it.

- [ ] **Commit:** `Add dataset audit with shape, dtypes, missing values and class distribution`
- [ ] **🚩 GATE M1 — post in chat: "Real data is in data/raw/, audit committed. B and C can inspect real distributions."**

---

# PHASE 2 — Days 2–4 · EDA · **rubric A2 + A3, 3 marks**

Read the rubric literally: *"At minimum: distribution plots for **each** feature, correlation heatmap, target distribution, and at least **two** scatter plots showing feature-target relationships."*

### ☐ A-T6. Write `src/plotting.py` first

- [ ] Set the house style once so every plot in every notebook matches:

```python
import matplotlib.pyplot as plt, seaborn as sns
FIGDIR = "../reports/figures"

def set_style():
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                         "axes.titlesize": 12, "axes.labelsize": 10})

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=150, bbox_inches="tight")
```

- [ ] Tell B and C it exists — they use `save()` too, so all figures land in one folder for the slides.
- [ ] **Commit:** `Add shared plotting style helpers`

---

### ☐ A-T7. The EDA plots

Every single one needs a title, labelled axes, a legend where applicable, and a Markdown observation underneath.

**Distributions (rubric: "each feature")**
- [ ] Histogram grid of **all ~39 numeric features** — `df[num_cols].hist(figsize=(22,18), bins=50)`. One figure satisfies "each feature". Note the extreme skew on most.
- [ ] Log-scale histogram of `sloss` and `dloss` specifically (the regression targets) — use `np.log1p`.
- [ ] Count plots for `proto` (top 15 only), `service`, `state`.

**Target distribution**
- [ ] Bar chart of `attack_cat.value_counts()` on a **log y-axis** — otherwise `Worms` is an invisible sliver. Annotate counts.
- [ ] Histogram of `sloss` (regression target), plus one with `log1p`.

**Correlation heatmap**
- [ ] `sns.heatmap(df[num_cols].corr(), cmap='coolwarm', center=0)` — full matrix, then a focused version: the top 15 features by `abs(corr)` with `sloss`.
- [ ] **Observation to write:** `sloss` correlates ~0.9+ with `spkts` and `sbytes`. **This is the leakage trap.** Say it explicitly here — it sets up B's two-model framing and is worth real marks.

**Feature–target scatter (rubric: "at least two")**
- [ ] `spkts` vs `sloss`, coloured by `attack_cat` — expect a near-linear ceiling, which visually proves the leakage.
- [ ] `dur` vs `sloss`, coloured by `attack_cat`.
- [ ] `sttl` vs `sloss` — a third one, since `sttl` is the strongest non-leaky signal.
- [ ] Use `alpha=0.3` and subsample to ~20k points or the scatter turns into a black blob.

**Class-conditional views**
- [ ] Box plot: `sloss` per `attack_cat` (log y-axis) — shows which attacks actually degrade QoS.
- [ ] Box plot: `sttl` per `attack_cat` — a famously sharp separator in UNSW-NB15; note it and be ready for the viva question "is `sttl` too good to be true?"

- [ ] **Commit after each group of plots** — that's 3–4 commits from this task alone.

---

# PHASE 3 — Days 4–5 · Preprocessing + feature engineering · **rubric B1/B2/B3, 3 marks** · **Gate M2**

This is the task the whole team is waiting on. Timebox it.

### ☐ A-T8. Implement `get_dataset()` for real

Replace the `NotImplementedError` body. Keep the signature identical.

**Cleaning (B1, 1 mark)**
- [ ] Replace `±np.inf` with `NaN` across numeric columns.
- [ ] Handle NaN with a **stated rule** — e.g. drop rows if < 1% affected, else median-impute. Print how many rows each rule removed and write the justification in the notebook.
- [ ] `drop_duplicates()` — report the count. UNSW-NB15 has a meaningful number of exact duplicate flows.
- [ ] Outliers: clip continuous features at the **1st/99th percentile computed on the training split only**. Justify in writing as robustness to flow-meter rate artefacts — and explicitly say you clip rather than delete, because extreme values *are* the attack signal here and deleting them would remove the phenomenon you're modelling. (This is a strong viva answer.)
- [ ] Normalise `attack_cat`: strip whitespace, and map the blank/`NaN` entries to `"Normal"` if your copy uses blanks for benign rows.

**Feature engineering (B3, 1 mark)** — implement 4–6 from `PLAN.md` §4:
- [ ] `loss_ratio_dst = dloss / (dpkts + 1e-6)`
- [ ] `bytes_per_pkt_src = sbytes / (spkts + 1e-6)`
- [ ] `dir_asymmetry = abs(sbytes - dbytes) / (sbytes + dbytes + 1e-6)`
- [ ] `jitter_ratio = sjit / (djit + 1e-6)`
- [ ] `ttl_diff = sttl - dttl`
- [ ] `handshake_share = (synack + ackdat) / (tcprtt + 1e-6)`
- [ ] ⚠️ **Do not create a source-side loss ratio** (`sloss / spkts`) — that is the regression target divided by a feature, i.e. direct target leakage. `loss_ratio_dst` is destination-side and therefore safe. Write this distinction down; it's exactly the kind of thing an examiner probes.
- [ ] Each feature gets a Markdown paragraph in `01_eda.ipynb` explaining the *domain reason*, not just the formula.

**Split (B2, 1 mark)**
- [ ] `train_test_split(X, y, test_size=0.2, stratify=df['attack_cat'], random_state=42)`
- [ ] Use the **same split indices for all three tracks** — stratify on `attack_cat` even for the regression track. Compute the split once, reuse it.

**Encoding + scaling (B2)**
- [ ] `state`, `service` → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`, **fit on train only**.
- [ ] `proto` → keep only the top 15 levels, bucket the rest as `"other"`, then one-hot. (130 one-hot columns would swamp SVR and KNN.) Justify in writing.
- [ ] `StandardScaler` fit on train only, transform both.
- [ ] Assemble with `ColumnTransformer` inside a `Pipeline` so leakage is structurally impossible — then say exactly that in the viva.
- [ ] `feature_names` must come from `preprocessor.get_feature_names_out()` so B's feature-importance plot has real labels, not `f0…f83`.

**Task-specific behaviour**
- [ ] `task="regression"` → `y = sloss`; if `drop_leaky=True`, drop `LEAKY_COLS` **and** `bytes_per_pkt_src` (it contains `sbytes/spkts`).
- [ ] `task="classification"` → `y = LabelEncoder().fit_transform(attack_cat)`; return the encoder.
- [ ] `task="clustering"` → drop `attack_cat` and `label` from X entirely; return `y=None`.
- [ ] `subsample=N` → stratified subsample of the **train set only**, never the test set (the test set must stay identical across all algorithms for a fair comparison).

**Export**
- [ ] Save cleaned + engineered (pre-encoding) frames to `data/processed/train.parquet` / `test.parquet` so EDA-style inspection is still possible downstream.
- [ ] Cache the encoded arrays too, so B's and C's notebooks load in seconds instead of re-running the pipeline every time.

**Self-test before you announce the gate:**
- [ ] `get_dataset("regression")["X_train"].shape` — sensible, no NaN: `np.isnan(X).any() == False`
- [ ] `get_dataset("regression", drop_leaky=True)` has fewer columns than without
- [ ] `get_dataset("classification")["label_encoder"].classes_` — exactly 10 classes
- [ ] `get_dataset("clustering")["y_train"] is None`
- [ ] `len(feature_names) == X_train.shape[1]` for every task
- [ ] Calling it twice gives identical arrays (determinism check)

- [ ] **Commit:** `Implement cleaning, feature engineering, stratified split and scaling`
- [ ] **🚩 GATE M2 — post in chat: "`src/preprocessing/` is live. Swap out make_mock_dataset() and run for real."**

---

# PHASE 4 — Days 5–12 · Support, documentation, figures

You are now off the critical path. Your job shifts to unblocking others and to documentation.

### ☐ A-T9. On-call for B and C
- [ ] Expect bug reports against `src/preprocessing/` in the first 24h after M2. Fix fast — they're blocked while you don't.
- [ ] If B or C asks for a signature change, evaluate it once and change it for both at the same time. Never make two separate changes.

### ☐ A-T10. Finish `01_eda.ipynb`
- [ ] Restart kernel, **Run All**, confirm zero errors and all outputs visible (rubric D1 requires this).
- [ ] Every plot has an observation cell. Re-read the notebook as if you were the examiner and find the plots you forgot to comment on.
- [ ] **Commit:** `Finalise EDA notebook with full commentary`

### ☐ A-T11. README draft
Guidelines require: dataset description, problem statement, results summary table, environment setup, how-to-run. Full marks for this are in Review 2, but draft it now.
- [ ] Dataset description: source, licence, rows, columns, 10 classes, what `sloss` measures.
- [ ] Problem statement + the three tracks.
- [ ] Setup: `python -m venv`, `pip install -r requirements.txt`, `python data/download_data.py`.
- [ ] How to run: notebook order `01 → 02 → 03 → 04`.
- [ ] **Cite AI assistance** — guidelines §7.5 requires it if used.
- [ ] Results table: leave placeholders, B and C fill them at M4.
- [ ] **Commit:** `Add README with dataset description and setup instructions`

### ☐ A-T12. Figure export + slide assembly *(needs B and C at M3/M4)*
- [ ] Confirm every figure is in `reports/figures/` with a sane name.
- [ ] Build the deck: problem → data → EDA → preprocessing/FE → regression results → classification Part A results → conclusion. Guidelines want a story arc.
- [ ] You present the data/EDA/preprocessing section.

### ☐ A-T13. Dry run *(Day 13, all three)*
- [ ] All notebooks: restart + Run All, top to bottom, zero errors.
- [ ] Cross-teach session — see `TEAM_OVERVIEW.md`.
- [ ] Verify commit history: `git log --oneline | wc -l` ≥ 24 across the team, spread over time.

---

# PHASE 5 — Review 2 · Your clustering track · **8 marks**

Start only after Review 1 is presented. Full detail in `PLAN.md` §8.

### ☐ A-T14. Prepare the clustering dataset
- [ ] `get_dataset("clustering")` — labels already dropped. Confirm `attack_cat` and `label` are absent from X; if a label leaks in, the entire track is invalid.
- [ ] Feature subset: `dur, spkts, dpkts, sbytes, dbytes, rate, sload, dload, sinpkt, dinpkt, sjit, djit, smean, dmean, ct_srv_src, ct_state_ttl`.
- [ ] Subsample to ~30–50k. **Agglomerative clustering is O(n²) in memory** — 175k rows will exhaust RAM. State the subsample size in the notebook.

### ☐ A-T15. K-Means — **rubric B1/B2**
- [ ] Fit k = 2…12, `random_state=42`, `n_init=10`.
- [ ] **Elbow curve** (inertia vs k) — mandatory.
- [ ] Silhouette-vs-k curve alongside it; pick k from both and justify.
- [ ] Final model at chosen k → cluster labels.

### ☐ A-T16. Agglomerative Hierarchical — **rubric B1/B2**
- [ ] `AgglomerativeClustering(n_clusters=k, linkage='ward')`.
- [ ] **Dendrogram** — mandatory. Use `scipy.cluster.hierarchy.dendrogram` on a ~2000-row subsample or it's unreadable; use `truncate_mode='lastp'`.
- [ ] Compare `ward` / `complete` / `average` linkage and report the metric difference.

### ☐ A-T17. Metrics — **all three mandatory, for both algorithms**
- [ ] Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
- [ ] One comparison table: rows = algorithms (and linkages), columns = the three metrics.

### ☐ A-T18. Visualisation — **rubric B3, 2 marks**
- [ ] PCA to 2 components → scatter coloured by cluster, **one plot per algorithm** (mandatory: "for every algorithm").
- [ ] t-SNE for at least one algorithm (`perplexity=30`, on ≤10k rows or it takes forever).
- [ ] **Post-hoc validation:** `pd.crosstab(cluster_labels, true_attack_cat)` heatmap. Name each cluster a workload profile — "bulk transfer", "scan/recon", "flood", "normal browsing".
- [ ] Write the caveat: the labels were used **only after** fitting, never during. The rubric calls this out specifically.

### ☐ A-T19. Commits
`Add K-Means clustering with elbow curve` · `Add hierarchical clustering with dendrogram and linkage comparison` · `Add clustering evaluation metrics table` · `Add PCA and t-SNE cluster visualisations` · `Add post-hoc cluster-to-attack-category validation`

---

## Your commit checklist (target ≥ 12 by Review 1)

1. `Initialise project structure and gitignore`
2. `Add pinned Python dependencies`
3. `Add preprocessing contract and mock dataset generator`
4. `Add dataset download script with shape assertions`
5. `Add dataset audit with shape, dtypes, missing values and class distribution`
6. `Add shared plotting style helpers`
7. `Add feature distribution plots and correlation heatmap`
8. `Add target distribution and feature-target scatter plots`
9. `Add class-conditional box plots for packet loss and TTL`
10. `Implement cleaning, feature engineering, stratified split and scaling`
11. `Finalise EDA notebook with full commentary`
12. `Add README with dataset description and setup instructions`

---

## Viva questions aimed at you

- Why did you re-split rather than use the official UNSW-NB15 train/test files?
- Where exactly is the scaler fitted, and how do you know there's no leakage?
- Why clip outliers instead of removing them?
- Why is `loss_ratio_dst` safe but `sloss/spkts` would be leakage?
- Why bucket `proto` to the top 15 levels?
- Why does `sttl` separate the classes so sharply — is that a real signal or an artefact of the synthetic testbed? *(Honest answer: partly an artefact of the IXIA generator. Saying so is better than pretending it isn't.)*
- (R2) What does the elbow curve tell you about network traffic behaviour?
- (R2) Your clusters don't map 1:1 to attack categories — why is that expected rather than a failure?
