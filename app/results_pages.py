"""Results pages of the Streamlit app: overview, classification, clustering, regression and downloads.

Every number is read from the project's result files (results/, reports/) and from the export the
website uses (cloudshield-site/src/data/*.json); nothing is computed here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RES, REP = ROOT / "results", ROOT / "reports"
SITE = ROOT / "cloudshield-site" / "src" / "data"
LABEL = {"ddos": "DDoS", "dos": "DoS", "xss": "XSS", "mitm": "MITM"}
name = lambda c: LABEL.get(c, c.capitalize())  # noqa: E731
SHOW_CV5 = True  # set False to leave the 5-fold CV results out
PART_A = {"Logistic Regression", "K-Nearest Neighbors", "Gaussian NB", "Decision Tree", "SVC (RBF)"}

PAGES = ["Live predictions", "Overview", "Classification", "Clustering", "Comparative study", "Regression", "Reports and downloads"]


@st.cache_data
def ton() -> dict:
    return json.loads((SITE / "toniot.json").read_text(encoding="utf-8"))


@st.cache_data
def unsw() -> dict:
    return json.loads((SITE / "cloudshield.json").read_text(encoding="utf-8"))


def csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


# ---------------------------------------------------------------- overview
def overview() -> None:
    T, D = ton(), unsw()
    C = T["classification"]
    st.header("Overview")
    st.write(
        "Every connection on a network leaves a flow record: how long it lasted, how many packets and bytes went each way, "
        "which protocol and service it used. CloudShield asks three questions of those records."
    )
    a, b, c = st.columns(3)
    a.metric("Classification · TON_IoT", f"{C['final']['f1w']:.3f}", "weighted F1 on unseen machines", delta_color="off")
    b.metric("Clustering · TON_IoT", f"k = {T['clustering']['k']}", f"silhouette {T['clustering']['metrics'][0]['silhouette']:.3f}", delta_color="off")
    c.metric("Regression · UNSW-NB15", f"R² {D['regression']['models'][0]['r2']:.4f}", D["regression"]["best"], delta_color="off")
    st.subheader("Datasets")
    O = T["overview"]
    st.dataframe(pd.DataFrame([
        {"Dataset": "TON_IoT (network)", "Used for": "classification, clustering", "Flows": O["rows"], "After de-duplication": O["rowsClean"],
         "Classes": len(O["classes"]), "Source machines": O["machines"]},
        {"Dataset": "UNSW-NB15", "Used for": "regression", "Flows": D["overview"]["rows"], "After de-duplication": None,
         "Classes": D["overview"]["classes"], "Source machines": None},
    ]), hide_index=True, use_container_width=True)
    st.subheader("TON_IoT traffic types")
    st.bar_chart(pd.DataFrame({"flows": [c["count"] for c in O["classes"]]}, index=[name(c["name"]) for c in O["classes"]]), horizontal=True)


# ---------------------------------------------------------------- classification
def classification() -> None:
    C = ton()["classification"]
    st.header("Classification · TON_IoT")
    st.write(
        f"Ten classifiers (five Part A, five Part B) plus the two tuned configurations, trained on {C['trainRows']:,} flows from "
        f"{C['trainMachines']} machines and tested on {C['testRows']:,} flows from {C['testMachines']} machines never seen in training. "
        f"All {len(C['order'])} traffic types appear on both sides of the split."
    )
    board = pd.DataFrame(C["models"]).rename(columns={"model": "Model", "acc": "Accuracy", "f1w": "Weighted F1", "f1m": "Macro F1", "params": "Settings",
                                                      "precision": "Precision (w)", "recall": "Recall (w)", "auc": "ROC-AUC (OvR)"})
    board.insert(1, "Part", ["A" if k in PART_A else "B" for k in board["key"]])
    st.subheader("All models on the same test set")
    metric_cols = [c for c in ("Accuracy", "Precision (w)", "Recall (w)", "Weighted F1", "Macro F1", "ROC-AUC (OvR)") if c in board and board[c].notna().all()]
    st.dataframe(board[["Model", "Part", *metric_cols, "Settings"]], hide_index=True, use_container_width=True,
                 column_config={k: st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0) for k in metric_cols})

    st.subheader(f"Final model: {C['bestDisplay']}")
    f = C["final"]
    cols = st.columns(6)
    for col, (k, v) in zip(cols, [("Accuracy", f["acc"]), ("Precision (w)", f["precision"]), ("Recall (w)", f["recall"]),
                                  ("F1 (w)", f["f1w"]), ("F1 (macro)", f["f1m"]), ("ROC-AUC (OvR)", f["auc"])]):
        col.metric(k, f"{v:.4f}")
    st.caption("ROC-AUC is one-vs-rest, averaged over the 10 classes.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Hyperparameter tuning (grouped 3-fold CV on the training machines)**")
        by = {m["key"]: m for m in C["models"]}
        st.dataframe(pd.DataFrame([
            {"Model": "Bagging", "Baseline F1": by["Bagging"]["f1w"], "Tuned F1": by["Bagging (tuned)"]["f1w"]},
            {"Model": "Gradient Boosting", "Baseline F1": by["Gradient Boosting"]["f1w"], "Tuned F1": by["Gradient Boosting (tuned)"]["f1w"]},
        ]).assign(Change=lambda d: d["Tuned F1"] - d["Baseline F1"]), hide_index=True, use_container_width=True)
        for s in C["searches"]:
            st.caption(f"{s['model']}: best {s['params']}; grouped CV {s['cv']:.4f} on training machines vs {s['test']:.4f} on unseen machines.")
    with right:
        if SHOW_CV5 and C["cv5"]:
            st.markdown("**5-fold cross-validation of the top two (weighted F1)**")
            st.dataframe(pd.DataFrame([{"CV": "grouped by machine" if c["kind"] == "grouped" else "stratified", "Model": c["model"],
                                        "Mean": c["mean"], "Std": c["std"], "Folds": ", ".join(f"{x:.3f}" for x in c["folds"])}
                                       for c in C["cv5"]]), hide_index=True, use_container_width=True)
            st.caption("Grouped folds keep each machine on one side. A fold that holds every machine of an attack type "
                       "never trains on it, so that fold's score collapses; stratified folds share machines and score higher.")

    st.subheader("Where the final model goes wrong")
    cm = C["confusion"]
    norm = pd.DataFrame(cm["norm"], index=[name(r) for r in cm["rows"]], columns=[name(c) for c in cm["cols"]])
    st.dataframe(norm.style.format("{:.2f}").background_gradient(cmap="Greens", axis=None), use_container_width=True)
    st.caption("Rows: true class; columns: predicted class; each row sums to 1.")
    st.dataframe(pd.DataFrame(C["perClass"]).assign(name=lambda d: d["name"].map(name)).rename(columns={"name": "Class"}),
                 hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    with a:
        image(RES / "figures/toniot_partB/cls_roc_top2.png", "ROC curves of the top two models")
    with b:
        image(RES / "figures/toniot_partB/cls_cm_top2.png", "Confusion matrices of the top two models")


# ---------------------------------------------------------------- comparative study
def comparative() -> None:
    CMP, FIG = RES / "comparative", RES / "figures/comparative"
    st.header("Comparative study")
    st.write("All algorithms of both Review 2 tracks compared side by side, from `notebooks/04_comparative_study.ipynb`.")
    tab_c, tab_k = st.tabs(["Classification · 10 algorithms", "Clustering"])
    with tab_c:
        t = csv(CMP / "classification_comparison.csv")
        if t is None:
            st.info("Run notebooks/04_comparative_study.ipynb to produce the comparison.")
        else:
            m = ["Accuracy", "Precision (w)", "Recall (w)", "F1 (weighted)", "F1 (macro)", "ROC-AUC (OvR)"]
            st.dataframe(t, hide_index=True, use_container_width=True,
                         column_config={k: st.column_config.ProgressColumn(format="%.4f", min_value=0.0, max_value=1.0) for k in m} |
                         {"Train time (s)": st.column_config.NumberColumn(format="%.1f"),
                          "Predict time (s)": st.column_config.NumberColumn(format="%.2f")})
            a, b = st.columns(2)
            with a:
                image(FIG / "cls_01_rank_heatmap.png", "Rank of each model on each metric")
            with b:
                image(FIG / "cls_07_f1_vs_time.png", "Weighted F1 against training time")
            image(FIG / "cls_04_per_class_f1.png", "Per-class F1 of every configuration")
            a, b = st.columns(2)
            with a:
                image(FIG / "cls_08_families.png", "Algorithm families")
            with b:
                image(FIG / "cls_09_tuning_delta.png", "Tuned minus base, per metric")
            fam = csv(CMP / "classification_families.csv")
            if fam is not None:
                st.dataframe(fam, hide_index=True, use_container_width=True)
            a, b = st.columns(2)
            with a:
                image(FIG / "cls_10_cv_stability.png", "5-fold CV of the top two: grouped vs stratified")
            with b:
                image(FIG / "cls_11_bootstrap_ci.png", "95% bootstrap intervals of weighted F1")
            mc = csv(CMP / "classification_mcnemar.csv")
            if mc is not None:
                st.markdown("**McNemar's test: the best model against each other model (same test flows)**")
                st.dataframe(mc, hide_index=True, use_container_width=True)
            image(FIG / "cls_05_confusion_all.png", "Confusion matrices of all configurations")
    with tab_k:
        k = csv(CMP / "clustering_comparison.csv")
        if k is None:
            st.info("Run notebooks/04_comparative_study.ipynb to produce the comparison.")
        else:
            st.dataframe(k, hide_index=True, use_container_width=True)
            st.caption("Required: K-Means and Agglomerative (four linkages). Extra, for reference: MiniBatch K-Means, Gaussian Mixture, "
                       "BIRCH, DBSCAN. ARI, NMI, homogeneity, completeness and purity use the true traffic type after fitting only.")
            image(FIG / "clu_01_k_sweep.png", "Every metric across k")
            a, b = st.columns(2)
            with a:
                image(FIG / "clu_07_rank_heatmap.png", "Rank of each clustering on each metric")
            with b:
                image(FIG / "clu_05_stability.png", "Agreement between runs")
            image(FIG / "clu_03_pca_all.png", "Every clustering on the same PCA axes")
            image(FIG / "clu_02_cluster_vs_type.png", "Traffic types inside each cluster")
            a, b = st.columns(2)
            with a:
                image(FIG / "clu_04_tsne_compare.png", "t-SNE: K-Means vs Agglomerative vs true type")
            with b:
                image(FIG / "clu_06_runtime.png", "Fit time against data size")


# ---------------------------------------------------------------- clustering
def clustering() -> None:
    K = ton()["clustering"]
    st.header("Clustering · TON_IoT")
    st.write(f"K-Means and agglomerative clustering on {K['sample']:,} flows with the labels withheld; the labels are used only afterwards to describe the clusters.")
    st.dataframe(pd.DataFrame(K["metrics"]).rename(columns={"name": "Method", "silhouette": "Silhouette ↑", "daviesBouldin": "Davies-Bouldin ↓",
                                                          "calinskiHarabasz": "Calinski-Harabasz ↑"}), hide_index=True, use_container_width=True)
    if K["stability"]["ari"]:
        st.caption(f"Stability over 5 resamples: silhouette {K['stability']['silhouette'][0]:.3f} ± {K['stability']['silhouette'][1]:.3f}; "
                   f"agreement between runs ARI {K['stability']['ari'][0]:.3f}.")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Elbow curve** (chosen k = {K['k']})")
        st.line_chart(pd.DataFrame(K["elbow"]).set_index("k")[["inertia"]])
    with b:
        st.markdown("**Linkage strategies (agglomerative)**")
        st.dataframe(pd.DataFrame(K["linkages"]).rename(columns={"largest": "share in largest cluster"}), hide_index=True, use_container_width=True)
    st.markdown("**Flows on the first two principal components, coloured by K-Means cluster**")
    pts = pd.DataFrame(K["points"], columns=["PC1", "PC2", "cluster", "type"])
    pts["cluster"] = "Cluster " + (pts["cluster"] + 1).astype(str)
    st.scatter_chart(pts, x="PC1", y="PC2", color="cluster", height=420)
    st.markdown("**What each cluster contains**")
    st.dataframe(pd.DataFrame([{"Cluster": c["id"] + 1, "Flows": c["size"],
                                "Make-up": ", ".join(f"{name(m['name'])} {m['share']:.0%}" for m in c["mix"][:4])} for c in K["clusters"]]),
                 hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    with a:
        image(RES / "figures/clu_02_dendrogram.png", "Dendrogram (Ward linkage)")
    with b:
        image(RES / "figures/clu_04_tsne.png", "t-SNE, K-Means clusters vs true type")


# ---------------------------------------------------------------- regression
def regression() -> None:
    R = unsw()["regression"]
    st.header("Regression · UNSW-NB15")
    st.write("Ten regressors predict log1p(dbytes), the bytes a flow sends back, from the other flow statistics.")
    st.dataframe(pd.DataFrame(R["models"]).rename(columns={"model": "Model", "r2Val": "R² (val)", "r2": "R² (test)", "rmse": "RMSE (bytes)", "mae": "MAE (bytes)"})
                 [["Model", "R² (val)", "R² (test)", "RMSE (bytes)", "MAE (bytes)"]], hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.markdown("**GridSearchCV, before and after**")
        st.dataframe(pd.DataFrame(R["tuning"]), hide_index=True, use_container_width=True)
    with b:
        st.markdown("**5-fold CV R² (training set)**")
        st.dataframe(pd.DataFrame(R["cv"]), hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    with a:
        image(REP / "figures/reg_14_pred_vs_actual.png", "Predicted vs actual")
    with b:
        image(REP / "figures/reg_16_feature_importance.png", "Random Forest feature importance")


# ---------------------------------------------------------------- downloads
def reports() -> None:
    st.header("Reports and downloads")
    files = [
        ("Classification · leaderboard of 03_classification_1.ipynb (TON_IoT)", RES / "classification_10model_ton_iot.csv"),
        ("Classification · per-class report of the final model", RES / "classification_10model_per_class.csv"),
        *([("Classification · 5-fold CV, grouped by machine", RES / "classification_cv5_top2.csv"),
           ("Classification · 5-fold CV, stratified", RES / "classification_cv5_top2_stratified.csv")] if SHOW_CV5 else []),
        ("Comparative study · classification, all metrics and times", RES / "comparative/classification_comparison.csv"),
        ("Comparative study · per-class F1 of every model", RES / "comparative/classification_per_class_f1.csv"),
        ("Comparative study · bootstrap intervals", RES / "comparative/classification_bootstrap_ci.csv"),
        ("Comparative study · McNemar tests", RES / "comparative/classification_mcnemar.csv"),
        ("Comparative study · clustering, all metrics", RES / "comparative/clustering_comparison.csv"),
        ("Comparative study · clustering across k", RES / "comparative/clustering_k_sweep.csv"),
        ("Comparative study · clustering stability", RES / "comparative/clustering_stability_ari.csv"),
        ("Clustering · metrics", RES / "clustering_metrics_ton_iot.csv"),
        ("Clustering · stability over 5 runs", RES / "clustering_stability_ton_iot.csv"),
        ("Regression · 10 models (UNSW-NB15)", REP / "regression_results.csv"),
        ("Regression · tuning before and after", REP / "regression_tuning.csv"),
    ]
    for label, path in files:
        if path.exists():
            c1, c2 = st.columns([4, 1])
            c1.write(f"**{label}** · `{path.name}`")
            c2.download_button("Download CSV", path.read_bytes(), file_name=path.name, mime="text/csv", key=path.name)
    C = ton()["classification"]
    per = pd.DataFrame(C["perClass"]).to_csv(index=False).encode()
    c1, c2 = st.columns([4, 1])
    c1.write("**Classification · per-class precision, recall, F1 of the final model** · `final_model_per_class.csv`")
    c2.download_button("Download CSV", per, file_name="final_model_per_class.csv", mime="text/csv", key="per_class")


def render(page: str) -> None:
    {"Overview": overview, "Classification": classification, "Clustering": clustering, "Comparative study": comparative,
     "Regression": regression, "Reports and downloads": reports}[page]()
