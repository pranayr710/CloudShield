# PERSON C — Classification Track Owner

> Rename this heading to your actual name.

**You own:** the entire classification track across **both reviews** — 5 algorithms in Part A (Review 1), 5 more in Part B (Review 2), the consolidated 10-algorithm table, final model selection, and `src/metrics_clf.py`.

**Marks you are directly responsible for:** Review 1 Section D = **3 marks**. Review 2 Section A = **6 marks**. Total **9 of 50**.

**Target:** `attack_cat` — 10 classes (Normal + 9 attack types: Fuzzers, Analysis, Backdoor, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms).

> ⚠️ Your Review 1 share is small (3 marks) but your Review 2 share is large (6). **Use the spare capacity in Review 1 to do Part B work early** — see C-T13. Reviewers reward a Part A section that already reports ROC-AUC and a proper imbalance analysis, and you'll walk into Review 2 half-finished.

---

## Your dependency summary

| You are blocked by | **Nothing — `src/preprocessing/` is already delivered and verified (gate M2 met).** Skip the mock phase; set `USE_MOCK = False` from the start. |
|---|---|
| **What A must finish for you to start at all** | **A-T3** (contract + `make_mock_dataset`), target Day 1. Then 4 full days of parallel work on mock data. |
| **What A must finish for you to produce real results** | **A-T8 → gate M2**. Specifically you need `label_encoder` populated with the real 10 classes. |
| **You block** | Person A's figure export (A-T12) and the slides. Hit **M3 by Day 8**. |
| **Useful from A, not blocking** | A-T5's `attack_cat.value_counts()` — you need the real imbalance ratios for your class-weighting rationale. |
| **Coordinate with B** | Nothing technical. You use `metrics_clf.py`, B uses `metrics_reg.py` — separate files, no conflicts. |

**If A slips on M2:** extend the mock phase and start Part B algorithms early. Do **not** write your own preprocessing — a different split from B's would break the "same preprocessed dataset" claim the rubric rests on.

---

# PHASE 0 — Day 1 · Setup · ~1 hour

### ☐ C-T1. Get the repo running
*Blocked by: A-T1, A-T2 (Day 1 morning).*
- [ ] Clone, create branch `feat/c-classification`.
- [ ] `python -m venv .venv` → activate → `pip install -r requirements.txt`.
- [ ] Verify: `python -c "import sklearn, imblearn; print('ok')"`.
- [ ] Read `PLAN.md` §3 (Track 2) and §7. Read `TEAM_OVERVIEW.md`.

### ☐ C-T2. Understand the problem before you model it

Write this as a Markdown cell at the top of your notebook.

- **10 classes, brutally imbalanced.** Measured **after deduplication** (`DATA_AUDIT.md` finding 3), train/test split:

  | Class | Train | Test |
  |---|---|---|
  | Normal | 68,578 | 17,144 |
  | Exploits | 21,947 | 5,487 |
  | Fuzzers | 16,768 | 4,192 |
  | Reconnaissance | 7,993 | 1,998 |
  | Generic | 6,079 | 1,520 |
  | DoS | 4,400 | 1,100 |
  | Analysis | 1,625 | 407 |
  | Backdoor | 1,504 | 376 |
  | Shellcode | 1,165 | 291 |
  | Worms | **137** | **34** |

  A **500:1** ratio between `Normal` and `Worms`.

- ⚠️ **`Generic` is not the second-largest class, contrary to every paper on this dataset.**
  Deduplication cut it by **87%** (58,871 → 7,599) because `Generic` attacks block ciphers and
  emit near-identical flow records by construction. Removing those duplicates was mandatory —
  they caused 42.6% test-set contamination. Be ready to explain this in the viva; it is the
  most likely "your numbers don't match the literature" challenge you will face.
- **Accuracy is a misleading metric here.** Predicting `Normal` for everything gets ~33% accuracy while being completely useless. This is why the rubric mandates F1, and why you will additionally report **macro-F1**.
- **The finding to expect:** `Analysis`, `Backdoor` and `Worms` will be almost entirely absorbed into `Exploits` and `DoS`. This is not your model failing — those UNSW-NB15 categories genuinely overlap in feature space (an "Analysis" flow is often a port scan that also looks like reconnaissance; a "Backdoor" flow looks like an exploit). **Quantify it from the confusion matrix and explain it.** This is the strongest single thing you can say in the Review 1 viva — do not hide it behind a good weighted-F1 number.

---

# PHASE 1 — Days 1–5 · Build everything on mock data ⚡

*Unblocked by A-T3.*

### ☐ C-T3. Write `src/metrics_clf.py`

You own this file — B has a separate `metrics_reg.py`, so you never conflict.

```python
"""Classification metric helpers. Owner: Person C."""
import time, numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize


def evaluate(model, X_train, y_train, X_test, y_test, name, classes):
    """Fit, predict, return one row of the comparison table."""
    t0 = time.time(); model.fit(X_train, y_train); fit_s = time.time() - t0
    pred = model.predict(X_test)
    row = {
        "Model":      name,
        "Accuracy":   accuracy_score(y_test, pred),
        "Precision":  precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall":     recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        "F1_macro":   f1_score(y_test, pred, average="macro", zero_division=0),
        "FitTime_s":  round(fit_s, 2),
    }
    # OvR ROC-AUC — mandatory for multi-class
    try:
        proba = model.predict_proba(X_test)
        y_bin = label_binarize(y_test, classes=np.arange(len(classes)))
        row["ROC_AUC_OvR"] = roc_auc_score(y_bin, proba, average="weighted",
                                           multi_class="ovr")
    except (AttributeError, ValueError):
        row["ROC_AUC_OvR"] = np.nan   # e.g. SVC without probability=True
    row["_pred"] = pred
    return row


def results_table(rows):
    """rows -> DataFrame ranked by weighted F1 (rubric D2 / R2-A2)."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_pred"} for r in rows])
    return df.sort_values("F1_weighted", ascending=False).reset_index(drop=True).round(4)
```

- [ ] Add a `plot_confusion(y_test, pred, classes, name)` helper: `confusion_matrix(..., normalize='true')` → `sns.heatmap(annot=True, fmt='.2f', cmap='Blues')`, rotated tick labels, saved via `src.plotting.save`.
- [ ] Test on mock data.
- [ ] **Commit:** `Add classification metric helpers with OvR ROC-AUC and confusion matrix plot`

> **Why `normalize='true'`:** raw counts on a 500:1 imbalanced problem make every row except `Normal` unreadable. Row-normalising shows per-class recall, which is exactly what you need to demonstrate the `Worms`/`Backdoor` collapse.

### ☐ C-T4. Notebook scaffold — `notebooks/03_classification.ipynb`

Structure it so Part B slots in at Review 2 without restructuring:

- [ ] Headings: `1. Problem & target` · `2. Class imbalance analysis` · `3. Load preprocessed data` · `4. Part A — 5 algorithms` · `5. Part A comparison table (D2)` · `6. Confusion matrices` · `7. Per-class analysis` · `8. [Review 2] Part B — 5 algorithms` · `9. [Review 2] Consolidated 10-algorithm table` · `10. [Review 2] Final model selection & tuning` · `11. Conclusions`
- [ ] Load cell with the one-line toggle:

```python
import sys; sys.path.append("..")
from src.preprocessing import get_dataset
from src.metrics_clf import evaluate, results_table, plot_confusion

USE_MOCK = True   # ← flip to False at gate M2

loader  = make_mock_dataset if USE_MOCK else get_dataset
d       = loader("classification")
classes = d["label_encoder"].classes_
d_sub   = loader("classification", subsample=30000)   # for SVC
```
- [ ] Bright Markdown warning above it: **"MOCK DATA — meaningless until USE_MOCK=False."** Delete after the swap.
- [ ] **Commit:** `Add classification notebook scaffold with mock data loader`

### ☐ C-T5. Implement the 5 Part-A algorithms *(rubric D1, 2 marks)*

All on the same `X_train`, all `random_state=42`. Loop over a `MODELS` dict.

| # | Model | Constructor | Notes for the notebook |
|---|---|---|---|
| A1 | Logistic Regression | `LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1, random_state=42)` | Baseline. Interpret a few coefficients as log-odds. May warn about convergence — raise `max_iter` rather than ignoring it. |
| A2 | K-Nearest Neighbors | `KNeighborsClassifier(n_neighbors=5, n_jobs=-1)` | Tune `k` ∈ {3,5,11,21}. Discuss `euclidean` vs `manhattan` and why scaling was essential. |
| A3 | Gaussian Naive Bayes | `GaussianNB()` | **Expect it to be the worst — that's the point.** Show a correlation figure proving the features are strongly dependent, which violates conditional independence. Guaranteed viva question. |
| A4 | Decision Tree | `DecisionTreeClassifier(max_depth=15, class_weight="balanced", random_state=42)` | Tune `max_depth`. **`plot_tree(clf, max_depth=3, feature_names=..., class_names=...)`** — the rubric says "visualise the tree"; depth 3 only, or it's an unreadable smear. |
| A5 | SVM | `SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)` | ⚠️ **O(n²)** — use `d_sub` (30k). `probability=True` is needed for ROC-AUC but roughly 5× the fit time; budget for it. Evaluate on the **same full test set**. |

- [ ] Wrap the loop in `try/except` printing the model name — one crash must not lose four results.
- [ ] **Commit:** `Add all 5 Part A classification algorithms with unified training loop`

### ☐ C-T6. Class-imbalance strategy — decide and document

- [ ] Use `class_weight='balanced'` where supported (LogReg, DecisionTree, SVC). KNN and GaussianNB don't support it — note that as a limitation.
- [ ] Optionally add a **SMOTE** variant of the best Part-A model:
  ```python
  from imblearn.pipeline import Pipeline as ImbPipeline
  from imblearn.over_sampling import SMOTE
  pipe = ImbPipeline([("smote", SMOTE(random_state=42, k_neighbors=3)),
                      ("clf", DecisionTreeClassifier(random_state=42))])
  ```
  - [ ] ⚠️ **SMOTE must be inside the pipeline**, so it is fitted on the training fold only. Applying SMOTE before the split is a leakage error and will cost marks.
  - [ ] ⚠️ `Worms` has ~174 rows; the default `k_neighbors=5` can fail on the rarest class. Use `k_neighbors=3`.
  - [ ] Report with vs without SMOTE on **macro-F1** — that's where it shows up; weighted-F1 barely moves.
- [ ] Write the reasoning in Markdown either way. Choosing *not* to use SMOTE with a stated justification is also acceptable; silently ignoring imbalance is not.
- [ ] **Commit:** `Add class imbalance handling with balanced weights and SMOTE comparison`

---

# 🚩 GATE M2 — Day 5 · A delivers `src/preprocessing/`

### ☐ C-T7. The swap
- [ ] `git pull`
- [ ] `USE_MOCK = True` → `False`. Delete the mock warning cell.
- [ ] Restart kernel, Run All.
- [ ] Sanity checks before trusting anything:
  - [ ] `len(classes) == 10` and the names read correctly — `Normal, Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms`
  - [ ] `np.bincount(d["y_train"])` — the imbalance is present, and **every class appears in both train and test** (if stratification failed, `Worms` may vanish from the test set entirely — tell A at once)
  - [ ] `np.isnan(d["X_train"]).any()` is `False`
  - [ ] `d["X_train"].shape[1] == len(d["feature_names"])`
- [ ] Any failure → message A. Don't patch it locally.

---

# PHASE 2 — Days 5–8 · Real results · **Gate M3**

### ☐ C-T8. Train all 5 on real data
- [ ] Expect 15–45 minutes, dominated by SVC with `probability=True` and by KNN prediction on 50k test rows.
- [ ] Cache fitted models with `joblib.dump` so a kernel restart doesn't cost you an hour.
- [ ] **Expected ranking:** Decision Tree ≈ SVM > Logistic Regression > KNN >> GaussianNB. Weighted-F1 roughly 0.70–0.85; **macro-F1 much lower, roughly 0.35–0.55** — that gap is your headline.

### ☐ C-T9. Comparison table — **rubric D2, 1 mark**

The rubric asks for accuracy, weighted F1 and a confusion matrix **per algorithm**, plus a preliminary comparison table.

- [ ] `results_table(rows)` — Accuracy, Precision, Recall, F1_weighted, **F1_macro**, ROC_AUC_OvR, FitTime.
- [ ] Style with `.background_gradient(subset=["F1_weighted"], cmap="Greens")`.
- [ ] ROC-AUC is only mandatory in Review 2 — **include it now anyway.** It costs nothing and pre-completes Review 2's A2.
- [ ] Markdown commentary: which won, why, and why GaussianNB collapsed.
- [ ] **Commit:** `Add Part A classification comparison table`

### ☐ C-T10. Confusion matrices — **one per algorithm, required**
- [ ] `plot_confusion` for all 5, `normalize='true'`, 10×10, class names on both axes, rotated labels, saved to `reports/figures/`.
- [ ] Markdown observation under each.
- [ ] **Commit:** `Add normalised confusion matrices for all Part A algorithms`

### ☐ C-T11. Per-class analysis — the part that earns the viva marks
- [ ] `classification_report(y_test, pred, target_names=classes)` for the best model.
- [ ] Build a per-class recall bar chart, sorted ascending. `Worms`, `Backdoor`, `Analysis` will sit near zero.
- [ ] From the confusion matrix, extract the exact numbers: *"87% of `Analysis` flows are predicted as `Exploits`."* Quote real figures, not vague statements.
- [ ] Write the interpretation: these are **semantically overlapping attack families in UNSW-NB15's own labelling**, not a modelling defect. Reconnaissance, Analysis and Backdoor share the same probe-then-exploit traffic signature.
- [ ] Add the practical framing: for a real IDS, coarse detection (attack vs. normal) is often sufficient — compute binary accuracy using the `label` column as a secondary result and note that it is far higher.
- [ ] **Commit:** `Add per-class recall analysis and attack family confusion discussion`
- [ ] **🚩 GATE M3 — post in chat: "5 Part A classifiers done, table committed."**

---

# PHASE 3 — Days 8–13 · Polish and get ahead

### ☐ C-T12. Notebook hygiene *(rubric D1: "no errors")*
- [ ] Restart kernel → **Run All** → zero errors, all outputs visible.
- [ ] Every table and plot has a Markdown observation.
- [ ] Delete scratch cells and the mock warning.
- [ ] Conclusion cell: best Part-A model, its metrics, the imbalance finding, and what Part B will add.
- [ ] **Commit:** `Finalise Part A classification notebook with conclusions`

### ☐ C-T13. 🎯 Get ahead on Part B *(optional in Review 1, 6 marks in Review 2)*

You have spare capacity now and Review 2 is your heavy review. If time allows before the Review 1 dry run, stub in one or two Part-B models. Even one working Random Forest in the Part A notebook is a strong "here's where we're heading" slide.

- [ ] Do **not** let this delay C-T12. Part A being flawless matters more.

### ☐ C-T14. Feed A your numbers
- [ ] Send A the comparison table as Markdown, best-model metrics, and figure filenames for the README and slides.

### ☐ C-T15. Slides + dry run
- [ ] You present the classification section: 10 classes → imbalance → 5 algorithms → table → confusion matrix → the family-overlap finding.
- [ ] **Lead with macro-F1, not accuracy.** Volunteering the weakness before you're asked reads as competence.
- [ ] Cross-teach session — you must also explain A's preprocessing and B's regression.

---

# PHASE 4 — Review 2 · Part B + final selection · **6 marks**

### ☐ C-T16. Part B — 5 more algorithms *(rubric A1, 3 marks)*

Same split, same preprocessing, same `X_train` as Part A. This is explicitly required: *"trained on the same dataset as Review 1 Part A."*

| # | Model | Constructor | Notes |
|---|---|---|---|
| B6 | Random Forest | `RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)` | Expected top-2. Feature importance plot. |
| B7 | AdaBoost | `AdaBoostClassifier(n_estimators=200, random_state=42)` | Tune `n_estimators`, `learning_rate`. Note it struggles with multi-class imbalance. |
| B8 | Gradient Boosting | `GradientBoostingClassifier(random_state=42)` or `XGBClassifier(tree_method="hist", objective="multi:softprob", random_state=42)` | Expected #1. sklearn's GBM is slow on 10 classes × 200k rows — **prefer XGBoost/LightGBM and say which you used.** |
| B9 | Bagging | `BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=100, n_jobs=-1, random_state=42)` | Rubric explicitly requires a Decision Tree base estimator. |
| B10 | MLP | `MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", max_iter=300, early_stopping=True, random_state=42)` | Tune `hidden_layer_sizes` and `activation`. `early_stopping=True` or it runs forever. |

- [ ] **Commit:** `Add all 5 Part B classification algorithms`

### ☐ C-T17. Consolidated 10-algorithm table *(rubric A2, 1 mark)*
- [ ] **One single table**, 10 rows, columns: Accuracy, Precision, Recall, F1, ROC-AUC. This is why you stored Part A's rows — just concatenate.
- [ ] Add a grouped bar chart comparing all 10 on F1.
- [ ] **Commit:** `Add consolidated 10-algorithm classification comparison table`

### ☐ C-T18. Final model selection *(rubric A3, 2 marks)*
- [ ] Pick the best model with a **written justification** — not just the top F1. Consider inference time, interpretability, and rare-class recall.
- [ ] `GridSearchCV` on it. Report `best_params_` and an explicit **before/after** table showing improvement in at least one metric.
- [ ] 5-fold CV on the top 2.
- [ ] `joblib.dump(best, "../models/best_classifier.pkl")` — B needs this for the Streamlit app.
- [ ] **Commit:** `Tune and select final classification model`

---

## Your commit checklist (target ≥ 8 by Review 1)

1. `Add classification metric helpers with OvR ROC-AUC and confusion matrix plot`
2. `Add classification notebook scaffold with mock data loader`
3. `Add all 5 Part A classification algorithms with unified training loop`
4. `Add class imbalance handling with balanced weights and SMOTE comparison`
5. `Train Part A classifiers on preprocessed dataset`
6. `Add Part A classification comparison table`
7. `Add normalised confusion matrices for all Part A algorithms`
8. `Add per-class recall analysis and attack family confusion discussion`
9. `Finalise Part A classification notebook with conclusions`

---

## Viva questions aimed at you

- Why is accuracy a bad metric here, and what did you use instead?
- Why is Gaussian Naive Bayes the weakest? Are CICFlowMeter/Argus flow features conditionally independent? *(No — `spkts`, `sbytes` and `sload` are near-deterministic functions of each other. Have a correlation figure ready.)*
- Why does your model confuse `Analysis` with `Exploits`? *(Semantic overlap in UNSW-NB15's own labelling — quote the exact confusion-matrix percentage.)*
- How did you handle `Worms` with only 174 samples? Why `k_neighbors=3` in SMOTE?
- Why must SMOTE go inside the pipeline rather than before the split?
- What is One-vs-Rest ROC-AUC and why is it needed for multi-class?
- Why did you subsample for SVC, and is the comparison still fair? *(Same test set; documented; SVC is O(n²).)*
- Your weighted-F1 is 0.82 but macro-F1 is 0.45 — what does that gap mean? *(You should be raising this yourself, not waiting for it.)*
- Would you deploy this? *(Honest answer: as a coarse attack/normal detector, yes; as a 10-way classifier for rare families, not yet.)*
