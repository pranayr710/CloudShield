"""CloudShield — shared evaluation framework.

Spec section 9 fixes one results format per track, for all three team members:

    Regression      | Model | R2 | RMSE | MAE |
    Classification  | Model | Accuracy | Precision | Recall | Weighted F1 | ROC-AUC |
    Clustering      | Method | K | Silhouette | Davies-Bouldin | Calinski-Harabasz |

Use these helpers rather than writing per-member metric code, so that every
member's rows drop into the same consolidated table without reformatting.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import label_binarize

# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #

def evaluate_regressor(model, X_train, y_train, X_test, y_test, name, note=""):
    """Fit one regressor and return a row in the shared regression format.

    Returns a dict with the public columns plus private keys (leading
    underscore) carrying the fitted model and predictions for later plotting.
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_test)
    predict_s = time.perf_counter() - t0

    return {
        "Model": name,
        "R2": r2_score(y_test, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "MAE": mean_absolute_error(y_test, pred),
        "FitTime_s": round(fit_s, 2),
        "PredictTime_s": round(predict_s, 3),
        "n_train": len(y_train),
        "Note": note,
        "_pred": pred,
        "_model": model,
    }


def regression_table(rows, sort_by="R2"):
    """Consolidated regression table, ranked (spec step 27)."""
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    df = pd.DataFrame(public).sort_values(sort_by, ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.reset_index(drop=True).round(4)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def evaluate_classifier(model, X_train, y_train, X_test, y_test, name,
                        classes, note=""):
    """Fit one classifier and return a row in the shared classification format.

    ROC-AUC uses One-vs-Rest for the multiclass task, as the spec requires.
    Models without predict_proba fall back to decision_function; if neither is
    available the cell is left as NaN rather than being silently omitted.
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = model.predict(X_test)
    predict_s = time.perf_counter() - t0

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, average="weighted",
                                     zero_division=0),
        "Recall": recall_score(y_test, pred, average="weighted",
                               zero_division=0),
        "Weighted F1": f1_score(y_test, pred, average="weighted",
                                zero_division=0),
    }

    # Macro F1 is not in the mandated format but is reported alongside it:
    # on a 500:1 imbalance the weighted score is dominated by the majority
    # class and hides rare-class failure entirely (spec section 34).
    row["Macro F1"] = f1_score(y_test, pred, average="macro", zero_division=0)

    row["ROC-AUC"] = _ovr_roc_auc(model, X_test, y_test, len(classes))

    row["FitTime_s"] = round(fit_s, 2)
    row["PredictTime_s"] = round(predict_s, 3)
    row["n_train"] = len(y_train)
    row["Note"] = note
    row["_pred"] = pred
    row["_model"] = model
    return row


def _ovr_roc_auc(model, X_test, y_test, n_classes):
    """ROC-AUC. One-vs-Rest weighted for multiclass, standard for binary.

    label_binarize returns a single column for a 2-class problem while
    predict_proba returns two, so the binary case must be handled separately
    rather than folded into the OvR path - otherwise it silently returns NaN.
    """
    try:
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        else:
            return np.nan

        if n_classes == 2:
            # positive-class score only
            if scores.ndim == 2:
                scores = scores[:, 1]
            return roc_auc_score(y_test, scores)

        if scores.ndim == 1:
            return np.nan
        y_bin = label_binarize(y_test, classes=np.arange(n_classes))
        return roc_auc_score(y_bin, scores, average="weighted",
                             multi_class="ovr")
    except (ValueError, AttributeError, IndexError):
        return np.nan


def classification_table(rows, sort_by="Weighted F1"):
    """Consolidated classification table, ranked (spec section 9)."""
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    df = pd.DataFrame(public).sort_values(sort_by, ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.reset_index(drop=True).round(4)


# --------------------------------------------------------------------------- #
# Clustering  (Member 3)
# --------------------------------------------------------------------------- #

def evaluate_clustering(X, labels, method, k):
    """One row in the shared clustering format (spec section 9)."""
    return {
        "Method": method,
        "K": k,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
    }


def clustering_table(rows):
    return pd.DataFrame(rows).round(4)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def cv_report(model, X, y, cv=5, scoring="r2", name=""):
    """Cross-validated score with its spread (spec steps 29, 35)."""
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "Model": name,
        "Scoring": scoring,
        f"CV{cv}_mean": scores.mean(),
        f"CV{cv}_std": scores.std(),
        f"CV{cv}_min": scores.min(),
        f"CV{cv}_max": scores.max(),
        "folds": np.round(scores, 4).tolist(),
    }


def tuning_table(entries, metric="R2"):
    """Before/after tuning comparison (spec steps 28, 35)."""
    df = pd.DataFrame(entries)
    df[f"Delta_{metric}"] = (df[f"{metric}_tuned"] - df[f"{metric}_default"]).round(5)
    cols = ["Model", f"{metric}_default", f"{metric}_tuned",
            f"Delta_{metric}", "best_params"]
    return df[[c for c in cols if c in df.columns]].round(5)


def save_table(df, path, label=""):
    """Persist a results table as CSV so other members can merge it."""
    path = str(path)
    df.to_csv(path, index=False)
    print(f"saved {label or 'table'} -> {path}")
    return df
