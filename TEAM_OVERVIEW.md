# Team Work Division — Overview & Dependency Map

**Read this first, then open your own file.**

| Person | File | Track ownership | Review 1 marks owned | Review 2 marks owned |
|---|---|---|---|---|
| **A** | `TEAM_A_data_and_clustering.md` | Data foundation + **Clustering** | Sections A + B = **7** | Clustering = **8** |
| **B** | `TEAM_B_regression.md` | **Regression** | Section C = **9** | Integration + bonus = **3 (+2)** |
| **C** | `TEAM_C_classification.md` | **Classification** | Section D = **3** | Part B = **6** |

Rename "Person A/B/C" to real names in each file before you start.

> Guidelines §7.4: *"Each team member should be the primary owner of at least one track… divide by track, not by step."* This split follows that. Section A gets the data-foundation work in Review 1 because the clustering track doesn't start until Review 2 — otherwise A would be idle for half the semester.

---

## The one hard dependency

**Person A owns `src/preprocessing/`. B and C both consume it. Nobody can produce real numbers until it is frozen.**

To stop B and C sitting idle for a week, A publishes the **function contract on Day 1** (before writing the body). B and C write all their model code against that contract using a fake DataFrame, then swap in the real one at gate **M2**. This is the single most important coordination move in the project — do not skip it.

### The contract (frozen Day 1, body delivered at M2)

```python
# src/preprocessing/config.py
RANDOM_STATE = 42

def get_dataset(
    task: str,                    # "regression" | "classification" | "clustering"
    drop_leaky: bool = False,     # regression only: drops spkts, sbytes, sload
    subsample: int | None = None, # stratified subsample size, for SVR/SVC/Agglomerative
    random_state: int = 42,
) -> dict:
    """
    Returns a dict with keys:
      X_train        np.ndarray, encoded + scaled
      X_test         np.ndarray, encoded + scaled
      y_train        np.ndarray  (sloss floats | encoded class ints | None for clustering)
      y_test         np.ndarray  (same; None for clustering)
      feature_names  list[str], len == X_train.shape[1]
      label_encoder  sklearn LabelEncoder or None
      meta           dict: n_rows_before_clean, n_dropped, subsampled (bool)
    """
```

Once frozen, **A may not change this signature** without telling B and C in the group chat. Changing it silently breaks two notebooks.

---

## Milestone gates

Ordered. "Blocks" means the listed people cannot proceed past a specific task until it lands on `main`.

| Gate | Day | Owner | Deliverable | Blocks |
|---|---|---|---|---|
| **M0** | 1 | A (all present) | Repo + git + `requirements.txt` + contract frozen | B, C — everything |
| **M1** | 3 | A | `data/raw/*.csv` present + `download_data.py` + audit output | B-T4, C-T4 (real data inspection) |
| **M2** | 5 | A | **`src/preprocessing/` frozen + `data/processed/*.parquet`** | **B-T6, C-T6 — the big one** |
| **M3** | 8 | B, C | B: 10 regressors trained. C: 5 classifiers trained | A-T9 (figure export) |
| **M4** | 10 | B, C | B: tuning + diagnostic plots. C: confusion matrices + ROC | — |
| **M5** | 12 | A | EDA notebook final + README draft + `reports/figures/` | all — slides |
| **M6** | 13 | all | Full notebook re-run top-to-bottom, slides, dry-run viva | Review 1 |

Days are **relative**. If you have 3 weeks, multiply by 1.5. If you have 8 days, compress M1→Day 2 and M2→Day 3 and cut the polish, never the freeze.

---

## Parallel-work timeline

```
Day   1    2    3    4    5    6    7    8    9   10   11   12   13
A   [repo][download][EDA──────][preproc  ][support/README──][figures][dry run]
                      ▲M1        ▲M2                          ▲M5      ▲M6
B   [scaffold+mock models────────][real training───][tuning+plots][slides][dry run]
                                  ▲M2 unblocks       ▲M3      ▲M4
C   [scaffold+mock models────────][real training───][CM+ROC+table][slides][dry run]
                                  ▲M2 unblocks       ▲M3      ▲M4
```

Nobody is blocked for more than ~half a day if the contract is honoured.

---

## File ownership — prevents merge conflicts

**Only the owner edits these. If you need a change in someone else's file, message them.**

| File / folder | Owner |
|---|---|
| `src/preprocessing/`, `src/plotting.py` | **A** |
| `notebooks/01_eda.ipynb`, `notebooks/04_clustering.ipynb` | **A** |
| `data/download_data.py`, `data/` | **A** |
| `src/metrics_reg.py` | **B** |
| `notebooks/02_regression.ipynb` | **B** |
| `app/` (Streamlit bonus) | **B** |
| `src/metrics_clf.py` | **C** |
| `notebooks/03_classification.ipynb` | **C** |
| `README.md` | **A** drafts, **B** finalises in R2 |
| `requirements.txt` | **A**, but anyone may append (one dependency per commit) |

Metrics helpers are deliberately split into two files so B and C never touch the same lines.

---

## Git rules (rubric C3 in Review 2 — a single bulk commit scores **0**)

- Branch per person: `feat/a-data`, `feat/b-regression`, `feat/c-classification`. Merge to `main` via PR or plain merge — but **merge often**, at least daily.
- Commit at every task tick in your checklist. Target **≥ 8 commits each** before Review 1.
- Commit messages in the imperative: `Add GridSearchCV tuning for Random Forest regressor` — not `update`, not `final`, not `asdf`.
- **Never commit notebooks with merge conflicts.** If two people ever edit one notebook, `.ipynb` conflicts are unmergeable — that's why the ownership table above exists.
- `data/raw/*.csv` goes in `.gitignore`. The download script is what gets committed.

---

## Shared conventions — agree on Day 1, no exceptions

- `random_state=42` **everywhere** — `train_test_split`, every model, `GridSearchCV`, `KMeans`, `TSNE`.
- `test_size=0.2`, `stratify=attack_cat` — identical split across all three tracks.
- Scalers/encoders **fit on train only**. This lives inside `src/preprocessing/` so nobody can get it wrong individually.
- Every plot: title, axis labels, legend where applicable, `plt.tight_layout()`, palette `'colorblind'` / `'tab10'` / `'Set2'`.
- Every plot saved: `plt.savefig(f'../reports/figures/<track>_<name>.png', dpi=150, bbox_inches='tight')`.
- Every major visualisation gets a **Markdown cell underneath it** with a written observation. This is rubric A3 — 1 full mark — and it is the easiest mark in the project to lose.
- Results go in **one Pandas DataFrame per track**, never scattered `print()` calls.

---

## Viva insurance

Guidelines §7.4: *"the teacher may direct questions to any team member about any part of the work."*

Two days before the review, hold a **60-minute cross-teach session**: each person walks the other two through their notebook line by line. Everyone must be able to answer, for any track:

- What the target is and why it was chosen
- Why the split is stratified
- Where the scaler is fitted, and why fitting on full data would be leakage
- Which model won and *why* that model suits this data
- What the confusion matrix / residual plot / elbow curve actually shows

Also make sure **all three** can explain the two headline findings:
1. The `sloss ≤ spkts` leakage trap and the two-model framing (owner: B, but A and C must know it).
2. Why `Analysis` / `Backdoor` / `Worms` collapse into `Exploits` in the confusion matrix (owner: C, but A and B must know it).
