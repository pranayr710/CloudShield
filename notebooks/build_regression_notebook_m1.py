"""Generates notebooks/01_cse_ids2018_regression.ipynb — Member 1's share.

CloudShield Track 1: CSE-CIC-IDS2018 resource-abuse regression.
Member 1 owns regressors 1-3 (Linear, Ridge, Lasso). Models 4-10 belong to
Members 2 and 3 and are not implemented here.

The notebook is written to run top-to-bottom whether or not the dataset is
present. If it is absent, every modelling cell reports "pending data" instead of
raising, so the notebook stays clean while making unambiguous that no results
exist yet.

    python notebooks/build_regression_notebook_m1.py
    python -m jupyter nbconvert --execute --inplace \
        notebooks/01_cse_ids2018_regression.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(t):
    C.append(nbf.v4.new_markdown_cell(t.strip("\n")))


def code(t):
    C.append(nbf.v4.new_code_cell(t.strip("\n")))


def obs(t):
    md("**Interpretation.** " + t.strip())


# ========================================================================== #
md(
    """
# CloudShield — Track 1: Resource-Abuse Regression on CSE-CIC-IDS2018

**Multi-Dataset ML-Driven Cloud Threat Detection and Resource Abuse Prediction**
23CSE301 Machine Learning Capstone · 2026–27

> ## Member 1 — regressors 1 to 3
>
> This notebook implements **only** Linear Regression, Ridge and Lasso.
> ElasticNet, Polynomial and Decision Tree belong to **Member 2**; Random
> Forest, Gradient Boosting, SVR and KNN belong to **Member 3**. All seven use
> the same shared pipeline and append to the same results table.

## Where this sits in CloudShield

| Track | Dataset | Question | Owner |
|---|---|---|---|
| **1 — Regression** | **CSE-CIC-IDS2018** | **How much network resource is this workload consuming?** | **M1 (1–3) / M2 (4–6) / M3 (7–10)** |
| 2 — Classification | UNSW-NB15 | If traffic is malicious, what kind of attack is it? | M1 / M2 |
| 3 — Clustering | TON_IoT | What hidden behaviour patterns exist? | M3 |

## Target

Network **byte rate** — flow bytes per second. High sustained byte rate is the
observable signature of bandwidth-consuming behaviour, which is what the
resource-abuse component of CloudShield forecasts.

**The target column name is resolved from the data, not assumed.** The project
spec names it `fl_byt_s`, which is the abbreviated form used by some CIC-IDS2017
redistributions; the CSE-CIC-IDS2018 processed CSVs published by CIC normally
use `Flow Byts/s`. §2 inspects the real schema and resolves the target against a
list of known aliases, **stopping with a descriptive error if none matches**
rather than silently substituting a different column.

## A note on interpretation

A high predicted byte rate is a **risk indicator**, not proof of an attack.
Legitimate workloads — backups, replication, media delivery — also produce
sustained high throughput. This notebook treats byte rate as a forecasting and
triage signal, and does not claim that a high prediction demonstrates
cryptomining, bandwidth theft or any specific attack.

## Rules this notebook follows

- `random_state = 42` throughout.
- **The target never appears in `X`** (spec §8). Near-duplicate rate measures
  such as `Flow Pkts/s` are removed alongside it, since the two are computed
  over the same window and would make the target close to trivially recoverable.
- Scalers and encoders are fitted on the **training split only**.
- No tuning before the ten-model baseline comparison is complete (spec §5).
- No result is reported that has not actually been computed (spec §22).
"""
)

code(
    """
import sys, warnings
sys.path.append("..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV

from src.utils.config import RANDOM_STATE, RAW_IDS2018, MODELS, RESULTS
from src.utils.metrics import evaluate_regressor, regression_table, save_table
from src.utils.plotting import set_style, save_fig
from src.preprocessing import cse_cic_ids2018 as ids

set_style()
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 180)
warnings.filterwarnings("ignore", category=FutureWarning)

MEMBER = "Member 1"
print(f"CloudShield Track 1 | {MEMBER} | seed = {RANDOM_STATE}")
"""
)

# ---------------------------------------------------------------- availability
md(
    """
---
## 1. Dataset availability — spec step 6

Before anything else, check whether the data is actually present. Every
subsequent cell is guarded on this flag, so the notebook runs cleanly either
way and never reports a number it did not compute.
"""
)

code(
    """
files = ids.available_files()
DATA_AVAILABLE = len(files) > 0

print(f"CSE-CIC-IDS2018 directory : {RAW_IDS2018}")
print(f"CSV files present         : {len(files)}")
if DATA_AVAILABLE:
    total = sum(f.stat().st_size for f in files) / 1_048_576
    for f in files:
        print(f"  {f.name:58s} {f.stat().st_size / 1_048_576:8.1f} MB")
    print(f"  {'TOTAL':58s} {total:8.1f} MB")
else:
    print()
    print("=" * 70)
    print("DATASET NOT PRESENT - no regression results can be produced.")
    print("=" * 70)
    print("Fetch it with:")
    print("    python data/download_cse_cic_ids2018.py --aws")
    print()
    print("Every modelling cell below is guarded on DATA_AVAILABLE and will")
    print("report 'pending data' rather than fabricating output. Re-run this")
    print("notebook once the CSVs are in place.")
"""
)

# ---------------------------------------------------------------- inspection
md(
    """
---
## 2. Schema inspection and target resolution — spec steps 7–8

The spec is explicit: *"Before writing code: INSPECT THE ACTUAL DATASET COLUMNS.
Never invent column names."* This section does that first, and resolves the
byte-rate target against the aliases actually present.
"""
)

code(
    """
if DATA_AVAILABLE:
    schema = ids.inspect()
    display(schema)
    print()
    sample = pd.read_csv(files[0], nrows=5000, low_memory=False)
    sample.columns = [c.strip() for c in sample.columns]
    target = ids.resolve_target(sample.columns)
    print(f"resolved regression target -> '{target}'")
    print(f"total columns in {files[0].name}: {sample.shape[1]}")
    print()
    print("Full column list (the real schema, not an assumed one):")
    for i, c in enumerate(sample.columns, 1):
        print(f"  {i:3d}. {c}")
else:
    print("pending data - schema inspection requires the CSV files")
"""
)

obs(
    """
Two things must be checked here before any modelling, and both are reasons this
section exists rather than being skipped.

**The target alias.** If `target_alias_found` reports `NONE` for a file, the
pipeline stops rather than proceeding. Adding the correct spelling to
`REG_TARGET_ALIASES` in `src/utils/config.py` is the fix — silently substituting
a different column would change what the project is actually predicting while
leaving every label and chart claiming otherwise.

**Which files to use.** CSE-CIC-IDS2018 ships one CSV per capture day, each with
a different attack mix. Using all ten is ~16M rows; using one is a single day's
traffic profile. Whichever is chosen must be stated explicitly, because it
determines what the regression model has actually learned to predict.
"""
)

# ---------------------------------------------------------------- load/clean
md(
    """
---
## 3. Load, clean and engineer — spec steps 9–11, 16–20

The shared pipeline in `src/preprocessing/cse_cic_ids2018.py` performs, in
order: load the selected files → drop identifier columns → coerce numerics →
replace ±inf → drop rows with a missing target → median-impute the rest →
remove impossible values → deduplicate → engineer features → split → clip →
encode → scale.

Two points specific to this dataset, as opposed to UNSW-NB15:

**Infinities are real here.** CICFlowMeter divides by the flow window, so a
zero-length window yields ±inf in the rate columns. These are replaced with NaN
and then imputed — except where the *target* is affected, in which case the row
is dropped, since a row with no target cannot train a regressor.

**Repeated headers.** Several CIC files contain their header row again partway
through, which makes pandas read numeric columns as `object`. The loader filters
those rows before numeric coercion.
"""
)

code(
    """
# Row cap per file. None uses everything. If a cap is applied it MUST be
# reported here and in the report, with the reason (spec section 17).
NROWS_PER_FILE = None
FILES_USED = None            # None = every CSV present

if DATA_AVAILABLE:
    reg = ids.get_regression_data(
        files=FILES_USED,
        nrows_per_file=NROWS_PER_FILE,
        log_target=False,
        random_state=RANDOM_STATE,
        verbose=True,
    )
    X_train, X_test = reg["X_train"], reg["X_test"]
    y_train, y_test = reg["y_train"], reg["y_test"]
    features, target = reg["feature_names"], reg["target"]
    print()
    print(f"X_train {X_train.shape}   X_test {X_test.shape}")
    print(f"target : {target}")
    print(f"leakage columns removed: {reg['meta']['leakage_columns_removed']}")
    print(f"engineered features    : {reg['meta']['engineered_features']}")
else:
    reg = None
    print("pending data - loading requires the CSV files")
"""
)

code(
    """
if DATA_AVAILABLE:
    print("LEAKAGE CHECKS")
    print(f"  target in feature matrix : "
          f"{'FAIL' if target in features else 'NONE  [PASS]'}")
    tr_mean = np.abs(X_train.mean(axis=0)).max()
    te_mean = np.abs(X_test.mean(axis=0)).max()
    print(f"  scaler fit on train only : TRAIN max|mean| = {tr_mean:.3e}, "
          f"TEST max|mean| = {te_mean:.6f}")
    print(f"  verdict                  : "
          f"{'PASS' if tr_mean < 1e-8 else 'CHECK'}")
    print()
    print("TARGET DISTRIBUTION (spec step 11)")
    s = pd.Series(y_train)
    print(s.describe().to_string())
    print(f"  skew    : {s.skew():.2f}")
    print(f"  zeros   : {(s == 0).mean() * 100:.1f}%")
else:
    print("pending data")
"""
)

code(
    """
if DATA_AVAILABLE:
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].hist(y_train, bins=80, color="#4878CF", edgecolor="none")
    ax[0].set_yscale("log")
    ax[0].set_title(f"{target} — original scale")
    ax[0].set_xlabel(target); ax[0].set_ylabel("count (log scale)")
    ax[1].hist(np.log1p(np.clip(y_train, 0, None)), bins=60,
               color="#4878CF", edgecolor="none")
    ax[1].set_title(f"log1p({target})")
    ax[1].set_xlabel(f"log1p({target})"); ax[1].set_ylabel("count")
    fig.suptitle("Regression target distribution — CSE-CIC-IDS2018", fontsize=13)
    save_fig(fig, "reg_01_target_distribution", subdir="regression")
    plt.show()
else:
    print("pending data - no target distribution to plot")
"""
)

# ---------------------------------------------------------------- models
md(
    """
---
## 4. Member 1's three regressors — spec step 24

All three are linear models differing only in how they penalise coefficient
magnitude. Comparing them isolates the effect of regularisation on this feature
space, which is informative in its own right: flow features are heavily
correlated, so how a model handles collinearity matters.

### 4.1 — Linear Regression

**What it is.** Ordinary least squares: the coefficient vector minimising the
squared error, with no penalty term.

**Why it is here.** The mandatory baseline. Every model in the ten-model
comparison must justify itself against a plain linear fit.

**Intuition and weakness.** It treats the target as a weighted sum of features.
With correlated predictors — and CICFlowMeter features are heavily correlated,
since they are all derived from the same packet stream — the individual
coefficients become unstable: the model can inflate one and offset it with
another at almost no cost to the loss. Predictions stay reasonable; the
coefficients stop being interpretable. That instability is exactly what Ridge
addresses.

### 4.2 — Ridge Regression (L2)

**What it is.** Least squares plus a penalty of `alpha` times the sum of squared
coefficients.

**Why it suits this data.** L2 shrinkage is the standard remedy for
multicollinearity. Rather than letting two correlated features take large
opposing coefficients, Ridge shares the effect between them and shrinks both,
which stabilises the solution and usually improves generalisation.

**Intuition.** Coefficients are shrunk toward zero but never reach it, so every
feature is retained with reduced influence. `alpha` controls the trade-off:
zero recovers OLS, large values shrink everything toward the mean.

### 4.3 — Lasso Regression (L1)

**What it is.** Least squares plus a penalty of `alpha` times the sum of
absolute coefficients.

**Why it suits this data.** The L1 penalty drives coefficients to **exactly
zero**, performing feature selection as part of fitting. On a redundant feature
space this is directly useful: the count of surviving coefficients is a
measurement of how much of the schema is actually non-redundant.

**Intuition.** The absolute-value penalty has a corner at zero, so the optimum
frequently lands exactly there — unlike L2, whose smooth penalty only
approaches it.

**Scaling matters for both.** Ridge and Lasso penalise coefficient *magnitude*,
so a feature measured in bytes would be penalised differently from the same
feature measured in kilobytes. The shared pipeline standardises all numeric
features before fitting, which is what makes the penalty comparable across
features.
"""
)

code(
    """
rows, fitted = [], {}

if DATA_AVAILABLE:
    specs = [
        ("Linear Regression", LinearRegression(), ""),
        ("Ridge Regression", Ridge(alpha=1.0, random_state=RANDOM_STATE),
         "alpha=1.0 (baseline; tuned in step 28)"),
        ("Lasso Regression", Lasso(alpha=0.001, max_iter=5000,
                                   random_state=RANDOM_STATE),
         "alpha=0.001 (baseline; tuned in step 28)"),
    ]
    for name, est, note in specs:
        try:
            r = evaluate_regressor(est, X_train, y_train, X_test, y_test,
                                   name, note=note)
            rows.append(r); fitted[name] = r["_model"]
            print(f"  {name:20s} R2={r['R2']:8.4f}  RMSE={r['RMSE']:12.3f}  "
                  f"MAE={r['MAE']:12.3f}  ({r['FitTime_s']}s)")
        except Exception as exc:
            print(f"  {name:20s} FAILED: {type(exc).__name__}: {exc}")
else:
    print("pending data - no models trained, no metrics to report")
"""
)

code(
    """
if DATA_AVAILABLE and rows:
    # regularisation path: what does the penalty actually do here?
    path = []
    for a in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]:
        rg = Ridge(alpha=a, random_state=RANDOM_STATE).fit(X_train, y_train)
        ls = Lasso(alpha=a, max_iter=5000,
                   random_state=RANDOM_STATE).fit(X_train, y_train)
        from sklearn.metrics import r2_score
        path.append({
            "alpha": a,
            "Ridge_R2": r2_score(y_test, rg.predict(X_test)),
            "Lasso_R2": r2_score(y_test, ls.predict(X_test)),
            "Lasso_zero_coefs": int((ls.coef_ == 0).sum()),
            "Lasso_kept": int((ls.coef_ != 0).sum()),
        })
    print(f"Regularisation path ({len(features)} features total)")
    print(pd.DataFrame(path).round(4).to_string(index=False))
else:
    print("pending data")
"""
)

obs(
    """
The regularisation path is the substantive result from Member 1's three models,
and it should be read for the Lasso sparsity column rather than for R².

As `alpha` rises, the number of coefficients driven to exactly zero climbs while
R² initially holds and then collapses. The point just before the collapse is a
direct measurement of how redundant the CICFlowMeter feature space is: if a
large fraction of features can be eliminated with negligible loss, the schema is
carrying substantially duplicated information — which is expected, since many of
its columns are arithmetic transformations of the same underlying packet
counts. Ridge, by contrast, never zeroes anything; it only shrinks. That
contrast is the practical difference between L1 and L2 made concrete on this
dataset.

*Numbers pending — this cell reports `pending data` until CSE-CIC-IDS2018 is
downloaded.*
"""
)

# ---------------------------------------------------------------- table
md(
    """
---
## 5. Member 1 results table — shared format

Spec §9 format, so Members 2 and 3 append without reformatting:
`| Model | R² | RMSE | MAE |`
"""
)

code(
    """
if DATA_AVAILABLE and rows:
    table = regression_table(rows)
    display(table.style.background_gradient(subset=["R2"], cmap="Greens")
            .format({c: "{:.4f}" for c in ["R2", "RMSE", "MAE"]})
            .hide(axis="index"))
    save_table(table, RESULTS / "regression" / "member1_results.csv",
               "Member 1 regression results")
else:
    print("pending data - no results table")
    print()
    print("Expected format once the dataset is available:")
    display(pd.DataFrame(columns=["Rank", "Model", "R2", "RMSE", "MAE",
                                  "FitTime_s", "Note"]))
"""
)

code(
    """
if DATA_AVAILABLE and rows:
    best = max(rows, key=lambda r: r["R2"])
    pred = best["_pred"]
    resid = y_test - pred
    idx = np.random.default_rng(RANDOM_STATE).choice(
        len(pred), min(15_000, len(pred)), replace=False)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].scatter(pred[idx], resid[idx], s=6, alpha=0.25, color="#4878CF")
    ax[0].axhline(0, color="#C44E52", lw=1.5, label="zero error")
    ax[0].set_xlabel("predicted"); ax[0].set_ylabel("residual")
    ax[0].set_title(f"Residuals — {best['Model']}"); ax[0].legend()

    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax[1].scatter(y_test[idx], pred[idx], s=6, alpha=0.25, color="#6ACC64")
    ax[1].plot(lim, lim, color="#C44E52", lw=1.5, label="perfect (y = x)")
    ax[1].set_xlabel("actual"); ax[1].set_ylabel("predicted")
    ax[1].set_title("Predicted vs actual"); ax[1].legend()

    save_fig(fig, "reg_02_m1_diagnostics", subdir="regression")
    plt.show()

    for name, est in fitted.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(est, MODELS / "regression" / f"m1_{slug}.joblib")
    print(f"saved {len(fitted)} models -> models/regression/")
else:
    print("pending data - no diagnostics to plot")
"""
)

# ---------------------------------------------------------------- handoff
md(
    """
---
## 6. Status and handoff

### Current status

Run the cell below for the authoritative statement of what has and has not been
produced. Nothing in this notebook reports a metric it did not compute.
"""
)

code(
    """
status = pd.DataFrame([
    {"item": "Member", "value": "1 (regressors 1-3)"},
    {"item": "Dataset", "value": "CSE-CIC-IDS2018 (CloudShield Track 1)"},
    {"item": "Dataset present", "value": "YES" if DATA_AVAILABLE else "NO"},
    {"item": "Target column",
     "value": (reg["target"] if DATA_AVAILABLE and reg else
               "unresolved - requires the data")},
    {"item": "Models trained", "value": f"{len(rows)} / 3"},
    {"item": "Results available",
     "value": "yes" if rows else "NO - pending download"},
    {"item": "Tuning performed", "value": "no - deferred to spec step 28"},
])
display(status)

if not DATA_AVAILABLE:
    print("ACTION REQUIRED")
    print("  python data/download_cse_cic_ids2018.py --aws")
    print("  then re-run this notebook top to bottom.")
"""
)

md(
    """
### For Members 2 and 3 — how to append regressors 4–10

Use the identical data call and helpers. Do not rebuild the preprocessing.

```python
from src.preprocessing.cse_cic_ids2018 import get_regression_data
from src.utils.metrics import evaluate_regressor, regression_table

reg = get_regression_data()               # identical split, seed 42
rows = [evaluate_regressor(ElasticNet(random_state=42),
                           reg["X_train"], reg["y_train"],
                           reg["X_test"], reg["y_test"], "ElasticNet")]
```

Merge for the consolidated ten-model table (spec step 27):

```python
m1 = pd.read_csv("../results/regression/member1_results.csv")
consolidated = pd.concat([m1, regression_table(rows)]).sort_values(
    "R2", ascending=False)
```

Three things to carry over rather than rediscover:

- **Byte rate is extremely right-skewed** on this dataset family. If every
  tree-based model lands at an indistinguishable R², check whether a handful of
  rows dominate the variance; `log_target=True` on `get_regression_data` gives a
  comparison that discriminates between model families.
- **SVR is O(n²).** Use the `subsample` argument and state the size in the
  table footnote (spec §17) rather than sampling silently.
- **Do not tune before the ten-model baseline is complete** (spec §5).

### Sequencing

Steps 28–30 — tuning, 5-fold cross-validation and final regression analysis —
run once all ten baselines exist. They are not started here.
"""
)

nb["cells"] = C
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
})
out = "notebooks/01_cse_ids2018_regression.ipynb"
nbf.write(nb, out)
print(f"wrote {out}: {len(C)} cells "
      f"({sum(1 for c in C if c.cell_type == 'markdown')} md, "
      f"{sum(1 for c in C if c.cell_type == 'code')} code)")
