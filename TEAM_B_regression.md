# PERSON B — Regression Track Owner

> Rename this heading to your actual name.

**You own:** the entire regression track — 10 algorithms, comparison table, tuning, diagnostic plots. Plus `src/metrics_reg.py`, and in Review 2 the pipeline-integration marks and the Streamlit bonus.

**Marks you are directly responsible for:** Review 1 Section C = **9 marks** — the single largest block in Review 1. Review 2 Section C (integration/README/GitHub) = **3** + bonus **up to 2**. Total **≈14 of 52**.

**Target:** predict `sloss` — source packets retransmitted or dropped, i.e. QoS degradation.

---

## Your dependency summary

| You are blocked by | **Nothing — `src/preprocessing/` is already delivered and verified (gate M2 met).** Skip the mock phase; set `USE_MOCK = False` from the start. |
|---|---|
| **What A must finish for you to start at all** | **A-T3** (contract + `make_mock_dataset`), target Day 1. After that you have 4 full days of productive parallel work on mock data. |
| **What A must finish for you to produce real results** | **A-T8 → gate M2**. Until then, every number you produce is fake — **label those cells clearly so they don't end up in the submission.** |
| **You block** | Person A's figure export (A-T12) and the slides. Hit **M3 by Day 8**. |
| **Nice to have from A** | A-T7's correlation heatmap — it visually proves the leakage trap your two-model framing is built on. Not blocking. |

**If A slips on M2:** keep extending the mock phase — add more models, write more Markdown, build the tuning grids. Do **not** start writing your own preprocessing. Two preprocessing pipelines = inconsistent splits = lost marks on C2 and a bad viva.

---

# PHASE 0 — Day 1 · Setup · ~1 hour

### ☐ B-T1. Get the repo running
*Blocked by: A-T1, A-T2 (Day 1 morning).*
- [ ] Clone the repo, create branch `feat/b-regression`.
- [ ] `python -m venv .venv` → activate → `pip install -r requirements.txt`.
- [ ] Verify: `python -c "import pandas, sklearn, xgboost; print('ok')"`.
- [ ] Read `PLAN.md` §3 (Track 1) and §6 in full — especially the leakage trap.
- [ ] Read `TEAM_OVERVIEW.md` — the contract and the git rules.

### ☐ B-T2. Understand the target before you model it

**`src/preprocessing/` is already delivered and `DATA_AUDIT.md` documents the target in detail.
Read finding 4 and finding 5 there before you write a line of model code.** Summary:

- `sloss` = source packets retransmitted or dropped. Right-skewed count, 28.3% zeros,
  median 2, 99th percentile 53, max 5,319.
- **Five columns are algebraically tied to it**, not three: `sbytes` (r=0.9967),
  `spkts` (r=0.9738, and `sloss <= spkts` by definition), `smean` (`== sbytes/spkts` exactly),
  `sload`, `rate`. `drop_leaky=True` removes all five plus two engineered descendants.
- **Use `log_target=True` as your primary framing.** Ten test rows carry 62.7% of `sloss`'s
  variance, so raw-scale R² saturates at ~0.998 for every tree model and your required
  "ranked by R²" table would be ten identical numbers. On `log1p`, Linear scores 0.859 and
  Random Forest 0.997 — a table that actually says something.
- Report R²/RMSE in log space; convert MAE back with `np.expm1(pred)` so it reads in packets.

**Your three deliverable framings:**

| Run | Call | Purpose |
|---|---|---|
| 1 | `get_dataset("regression", log_target=True)` | headline: all features |
| 2 | `get_dataset("regression", drop_leaky=True, log_target=True)` | headline: leakage removed |
| 3 | `get_dataset("regression")` | secondary: raw scale, to show why log was needed |

**The finding to lead with:** removing the leaky columns barely moves R² (0.9988 → 0.9966 on
log1p). That is **not** a bug — a shuffled-target control scores R² = −0.097, proving the
pipeline is clean. Argus derives all 40-odd flow statistics from one packet stream, so they are
mutually constraining. Run the ablation in `DATA_AUDIT.md` finding 5 and present *that*:

| Feature set | RF R² |
|---|---|
| everything | 0.9992 |
| − 5 leaky columns | 0.9981 |
| − also `sinpkt` | 0.9926 |
| − also `dbytes`, `dpkts`, `dmean` | 0.9707 |
| − also `dur`, `dinpkt`, `dload`, `iat_ratio` | 0.6999 |

Only when duration is removed does the target get genuinely hard. Reproducing that table is
worth more than any single R² you can quote.

# PHASE 1 — Days 1–5 · Build everything on mock data ⚡ **this is why you're not blocked**

*Unblocked by A-T3. Every model here runs on `make_mock_dataset()`, which returns the exact same dict shape as the real thing. At M2 you change one line.*

### ☐ B-T3. Write `src/metrics_reg.py`

You own this file — C has a separate `metrics_clf.py`, so you never conflict.

```python
"""Regression metric helpers. Owner: Person B."""
import time, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate(model, X_train, y_train, X_test, y_test, name):
    """Fit, predict, return one row of the comparison table."""
    t0 = time.time(); model.fit(X_train, y_train); fit_s = time.time() - t0
    pred = model.predict(X_test)
    return {"Model": name,
            "R2":   r2_score(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "MAE":  mean_absolute_error(y_test, pred),
            "FitTime_s": round(fit_s, 2)}

def results_table(rows):
    """rows -> DataFrame ranked by R2 descending (rubric C2)."""
    return (pd.DataFrame(rows)
              .sort_values("R2", ascending=False)
              .reset_index(drop=True)
              .round(4))
```

- [ ] Sanity-check on mock data.
- [ ] **Commit:** `Add regression metric helpers and comparison table builder`

### ☐ B-T4. Notebook scaffold — `notebooks/02_regression.ipynb`

- [ ] Markdown structure with headings for every rubric item so nothing gets forgotten:
  `1. Problem & target` · `2. Load preprocessed data` · `3. Baseline linear models` · `4. Regularised models` · `5. Polynomial` · `6. Tree-based` · `7. SVR & KNN` · `8. Comparison table (C2)` · `9. Hyperparameter tuning (C3)` · `10. Diagnostics (C4)` · `11. Cross-validation` · `12. Conclusions`
- [ ] Load cell with a single toggle so the swap at M2 is one line:

```python
import sys; sys.path.append("..")
from src.preprocessing import get_dataset
from src.metrics_reg import evaluate, results_table

USE_MOCK = True   # ← flip to False at gate M2

loader = make_mock_dataset if USE_MOCK else get_dataset
d      = loader("regression")                       # Model A: all features
d_hon  = loader("regression", drop_leaky=True)      # Model B: no leaky columns
```
- [ ] ⚠️ Put a bright Markdown warning above it: **"MOCK DATA — results below are meaningless until USE_MOCK=False."** Remove the warning after the swap.
- [ ] **Commit:** `Add regression notebook scaffold with mock data loader`

### ☐ B-T5. Implement all 10 algorithms *(the 4-mark task — C1)*

All on the same `X_train`. All `random_state=42`. Build a `MODELS` dict so you loop instead of copy-pasting.

| # | Model | Constructor | Notes for the notebook |
|---|---|---|---|
| 1 | Linear Regression | `LinearRegression()` | baseline; print the 10 largest coefficients and interpret signs |
| 2 | Ridge | `Ridge(alpha=1.0, random_state=42)` | L2; alpha tuned later |
| 3 | Lasso | `Lasso(alpha=0.01, random_state=42, max_iter=5000)` | **count how many coefficients hit exactly 0** — the rubric asks for feature sparsity |
| 4 | ElasticNet | `ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000)` | tune `l1_ratio` |
| 5 | Polynomial | `Pipeline([("poly", PolynomialFeatures(2, include_bias=False)), ("lin", LinearRegression())])` | ⚠️ 84 features → degree 2 = ~3,500 columns. **Select the top 15 features by mutual information first**, then expand. Compare degree 2 vs 3 and note where it starts overfitting. |
| 6 | Decision Tree | `DecisionTreeRegressor(max_depth=12, random_state=42)` | tune `max_depth`; plot importances |
| 7 | Random Forest | `RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)` | expected top-2 |
| 8 | Gradient Boosting | `GradientBoostingRegressor(random_state=42)` or `XGBRegressor(tree_method="hist", random_state=42)` | expected #1 — use XGBoost if sklearn GBM is too slow, and say which you used |
| 9 | SVR | `SVR(kernel="rbf", C=1.0)` | ⚠️ **O(n²)** — use `subsample=20000`. Document it. |
| 10 | KNN | `KNeighborsRegressor(n_neighbors=5, n_jobs=-1)` | tune `k`; discuss why scaling matters (A already scaled — say so) |

- [ ] Run the loop for **Model A** (all features) → `rows_a`
- [ ] Run the loop for **Model B** (`drop_leaky=True`) → `rows_b`
- [ ] Wrap the loop in `try/except` that prints the model name on failure — a single crash must not lose the other nine results.
- [ ] For SVR: pass the subsampled dataset, and **evaluate on the same full test set** as everything else. Different training size is acceptable and documented; a different *test* set invalidates the comparison.
- [ ] **Commit:** `Add all 10 regression algorithms with unified training loop`

### ☐ B-T6. Prepare the tuning grids (still on mock)

- [ ] Write the `GridSearchCV` calls now so they're ready to fire the moment real data lands:

```python
rf_grid  = {"n_estimators": [100, 300], "max_depth": [10, 20, None],
            "min_samples_leaf": [1, 5]}
gbm_grid = {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]}
# cv=3 (not 5) inside GridSearchCV to keep runtime sane; the mandatory
# 5-fold CV is reported separately on the top-2 final models.
```
- [ ] Time one fit on mock data and extrapolate — if a grid would take > 40 minutes on 140k rows, cut it or switch to `RandomizedSearchCV(n_iter=20)`. Both are allowed by the rubric.
- [ ] **Commit:** `Add hyperparameter search grids for Random Forest and Gradient Boosting`

---

# 🚩 GATE M2 — Day 5 · A delivers `src/preprocessing/`

### ☐ B-T7. The swap
- [ ] `git pull`
- [ ] Change `USE_MOCK = True` → `False`. Delete the mock warning Markdown cell.
- [ ] Restart kernel, Run All.
- [ ] Sanity checks before you trust anything:
  - [ ] `d["X_train"].shape` — roughly (206k, ~84) for the 80/20 split of the combined file
  - [ ] `np.isnan(d["X_train"]).any()` is `False`
  - [ ] `len(d["feature_names"]) == d["X_train"].shape[1]`
  - [ ] `d_hon["X_train"].shape[1] < d["X_train"].shape[1]`
  - [ ] `d["y_train"].min() >= 0` — `sloss` is a count, negatives mean a cleaning bug in `src.preprocessing`
- [ ] If anything fails, **message A immediately** — don't patch it yourself in your notebook.

---

# PHASE 2 — Days 5–8 · Real results · **Gate M3**

### ☐ B-T8. Train all 10 on real data, both framings
- [ ] Run both loops. Expect the full set to take 20–60 minutes; run it once and cache with `joblib.dump`.
- [ ] **Model A expectation:** RF/GBM R² ≈ 0.97–0.99. Linear ≈ 0.85–0.95.
- [ ] **Model B expectation:** everything drops sharply — GBM maybe 0.35–0.65, linear near 0.1–0.3. **This is the correct and interesting result.** Do not panic and do not quietly re-add the leaky features.
- [ ] If Model B R² is near zero for everything, say so honestly and interpret it: packet loss may be largely unpredictable from non-volume features, which is itself a finding about network QoS.

### ☐ B-T9. Comparison table — **rubric C2, 2 marks**
- [ ] `results_table(rows_a)` and `results_table(rows_b)` — R², RMSE, MAE, ranked by R² descending. One table each, plus a merged side-by-side view.
- [ ] Style it: `.style.background_gradient(subset=["R2"], cmap="Greens")`.
- [ ] Markdown commentary: which family won, and **why** — ensembles capture threshold-like non-linear interactions between TTL, protocol state and timing that no linear model can represent.
- [ ] Explain any surprises. If KNN does well, that says the feature space has strong local structure. If Lasso zeroes 60% of coefficients, say which survived.
- [ ] **Commit:** `Add regression comparison table for all 10 algorithms`
- [ ] **🚩 GATE M3 — post in chat: "10 regressors done, table committed."**

---

# PHASE 3 — Days 8–10 · Tuning + diagnostics · **Gate M4**

### ☐ B-T10. Hyperparameter tuning — **rubric C3, 2 marks**

The rubric wants: *"GridSearchCV or RandomizedSearchCV applied to at least 2 models; best parameters **and improvement in metric** reported."*

- [ ] Tune **Random Forest** and **Gradient Boosting** (your top 2).
- [ ] `scoring='r2'`, `cv=3`, `n_jobs=-1`, `random_state=42`.
- [ ] Print `best_params_` for each.
- [ ] Build an explicit **before/after table** — this is the part people forget and lose the mark on:

| Model | R² (default) | R² (tuned) | Δ | Best params |
|---|---|---|---|---|
| Random Forest | … | … | … | … |
| Gradient Boosting | … | … | … | … |

- [ ] If tuning barely improves anything (common with RF), **say so and explain why**: Random Forest is famously insensitive to hyperparameters because bagging already controls variance. That's a better answer than pretending to a 0.001 improvement.
- [ ] **Commit:** `Add GridSearchCV tuning with before/after comparison`

### ☐ B-T11. Cross-validation — **mandatory global rule**
- [ ] `cross_val_score(model, X_train, y_train, cv=5, scoring='r2')` on the **top 2 models only**.
- [ ] Report mean ± std. Commentary: a large std means the model is unstable across folds.
- [ ] **Commit:** `Add 5-fold cross-validation for top two regression models`

### ☐ B-T12. Diagnostic plots — **rubric C4, 1 mark** (three plots required)

- [ ] **Residual plot** — residuals (`y_test - pred`) vs predicted, with a red `y=0` line. Comment on the funnel shape: variance grows with predicted value, i.e. heteroscedasticity, which is expected for a right-skewed count target.
- [ ] **Predicted vs actual** — scatter with the `y=x` diagonal. Use `alpha=0.3`, subsample to 20k. Comment on where it deviates.
- [ ] **Feature importance** — horizontal bar chart, top 15, from RF or GBM, labelled with **real feature names** from `d["feature_names"]`.
  - [ ] For **Model A**, expect `spkts`/`sbytes` to dominate — annotate that as visual proof of the tautology.
  - [ ] For **Model B**, this is the interesting plot: check whether A's engineered features (`ttl_diff`, `jitter_ratio`, `dir_asymmetry`) appear in the top 15. **If they do, screenshot it and send it to A** — it's direct evidence for rubric B3 and worth a mark on their side.
- [ ] All three: titles, labelled axes, `save(fig, "reg_<name>")` into `reports/figures/`.
- [ ] Markdown observation under each.
- [ ] **Commit:** `Add residual, predicted-vs-actual and feature importance plots`
- [ ] **🚩 GATE M4.**

### ☐ B-T13. Save the model
- [ ] `joblib.dump(best_model, "../models/best_regressor.pkl")` — you'll need it for the Streamlit bonus.
- [ ] **Commit:** `Save best regression model`

---

# PHASE 4 — Days 11–13 · Wrap-up

### ☐ B-T14. Notebook hygiene *(rubric D1: "no errors in notebook output")*
- [ ] Restart kernel → **Run All** → confirm zero errors, all outputs visible.
- [ ] Every major output has a Markdown observation.
- [ ] Delete every leftover scratch cell and the mock-data warning.
- [ ] Conclusion cell: best model, its R²/RMSE/MAE, the leakage story, one limitation.
- [ ] **Commit:** `Finalise regression notebook with conclusions`

### ☐ B-T15. Feed A your numbers *(A needs these for the README and slides)*
- [ ] Send A the comparison table as Markdown, the best-model metrics, and the figure filenames.

### ☐ B-T16. Slides + dry run
- [ ] You present the regression section: target → why → the two framings → table → best model → diagnostics.
- [ ] Lead with the leakage story. It is the most impressive thing in Review 1.
- [ ] Cross-teach session (see `TEAM_OVERVIEW.md`) — you must also be able to explain A's preprocessing and C's classifiers.

---

# PHASE 5 — Review 2 · Integration + bonus

You have no new track in Review 2, so you take integration and the bonus marks.

### ☐ B-T17. Rubric C — Pipeline integration & quality (**3 marks**)
- [ ] **C1 Code quality:** all four notebooks run top-to-bottom clean; helpers are in `src/`, not duplicated in cells; key steps commented.
- [ ] **C2 README:** finalise A's draft — fill in the real results tables from all three tracks, environment setup, how-to-run.
- [ ] **C3 GitHub:** audit `git log`. Meaningful messages, spread over time, `requirements.txt` present. Fix any junk messages **before** the review.

### ☐ B-T18. Streamlit app (**+1 bonus**)
- [ ] `app/streamlit_app.py`: input widgets for the main flow features → three outputs: predicted packet loss (your regressor), predicted attack type + probability bar (C's classifier), workload cluster (A's K-Means).
- [ ] Load all three `.pkl` files with `joblib`. **You must also save and load the fitted preprocessor** — raw user input needs the identical encoding/scaling, or predictions are nonsense. Ask A to export it at M2.
- [ ] Sensible defaults so a demo is one click, plus 2–3 preset example flows (one benign, one DoS).

### ☐ B-T19. Deployment (**+1 bonus**)
- [ ] Push to Streamlit Community Cloud or Hugging Face Spaces. Public URL in the README.
- [ ] Test the live URL from a phone before the review.

### ☐ B-T20. Optional — cross-dataset appendix
Only if the instructor permits a second dataset (A confirms this at A-T0). See `PLAN.md` Appendix A.7.

---

## Your commit checklist (target ≥ 10 by Review 1)

1. `Add regression metric helpers and comparison table builder`
2. `Add regression notebook scaffold with mock data loader`
3. `Add all 10 regression algorithms with unified training loop`
4. `Add hyperparameter search grids for Random Forest and Gradient Boosting`
5. `Train all regression models on preprocessed dataset`
6. `Add regression comparison table for all 10 algorithms`
7. `Add GridSearchCV tuning with before/after comparison`
8. `Add 5-fold cross-validation for top two regression models`
9. `Add residual, predicted-vs-actual and feature importance plots`
10. `Save best regression model`
11. `Finalise regression notebook with conclusions`

---

## Viva questions aimed at you

- What is `sloss` and why is it a reasonable QoS-degradation target?
- Your R² is 0.99 — isn't that suspicious? *(Have the leakage answer ready. Volunteer it before you're asked.)*
- Why do ensembles beat linear regression here?
- Why did Lasso zero out those particular coefficients?
- Why did you subsample for SVR, and does that make the comparison unfair? *(Same test set, documented; SVR is O(n²).)*
- What does the funnel shape in your residual plot mean?
- Tuning barely improved Random Forest — why?
- Why 5-fold CV on only the top two and not all ten? *(Compute cost; the guidelines specify top two.)*
- If you had to deploy one model to a live network, which and why? *(Hint: consider inference latency, not just R².)*
