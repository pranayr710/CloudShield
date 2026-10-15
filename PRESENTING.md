# Presenting CloudShield (Review 2)

## Links

| What | Where |
|---|---|
| Live app (Streamlit, all three models) | https://cloudshield-live.streamlit.app/ |
| Code | https://github.com/pranayr710/CloudShield |
| Gradio app (backup, needs the Hugging Face CPU quota) | https://huggingface.co/spaces/pranayraparla/Ml_Capstone |

Open the live app a few minutes before the review: a sleeping Streamlit app shows a "get this app back up" button and takes about a minute to wake.

## Start the website

```bash
cd cloudshield-site
npm install            # first time only
npm run build
npm run preview        # http://localhost:4173
```

The videos and data are local; the fonts and the live-models page need internet. If there is no internet, present the website's own sections and run the app locally instead:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py     # http://localhost:8501
```

## Suggested order

1. **Website hero → three track cards**: the problem and the three questions.
2. **01 Data, 02 Traps**: TON_IoT, duplicates, the leaking `label` column, splitting by machine.
3. **03 Classifier lab → 04 Ten models**: the 10-algorithm comparison (switch the metric tabs), tuning baseline → tuned, 5-fold CV grouped vs stratified.
4. **05 Errors**: confusion matrix, per-class results, the unseen-machines explanation.
5. **06 Play**: "Tune it yourself" (cross-validation vs unseen machines), then the firewall threshold game.
6. **07 Clusters**: elbow, linkage comparison, PCA map, cluster make-up.
7. **08 Compare**: the comparative study. Sort the scoreboard by each metric, hover the per-class F1 grid, the cost-vs-benefit chart and the McNemar list; then the clustering table, the k sweep and stability/cost.
8. **09–10 Regression (UNSW-NB15)** and **11 Findings**.
9. **Live models ↗** (top-right of the website): pick a real test flow for any traffic type, or type your own, and get a prediction from the deployed model. Then show the Streamlit app's results, comparative-study and downloads pages.

## Where each rubric item is (for the viva)

| Rubric item | File and place |
|---|---|
| A1 Part B models (RF, AdaBoost, GB, Bagging, MLP) | `notebooks/03_classification_1.ipynb`, sections 6 to 10 |
| A2 One table of all 10 algorithms (Accuracy, Precision, Recall, F1, ROC-AUC) | same notebook, "Final leaderboard"; `notebooks/04_comparative_study.ipynb` section 1.3; `results/comparative/classification_comparison.csv` |
| A3 Final model, tuning, improvement | same notebook, tuning cell (Bagging and Gradient Boosting), per-class report, ROC and confusion-matrix figures; comparative study section 1.9 (base vs tuned on every metric) |
| 5-fold CV of the top two | same notebook, "5-fold cross-validation of the top two models" (grouped and stratified) |
| B1–B3 Clustering: K-Means, Agglomerative, metrics, elbow, dendrogram, PCA, t-SNE | `notebooks/clustering.ipynb` |
| D3 Comparative results table (algorithms vs metrics), both tracks | `notebooks/04_comparative_study.ipynb`; `results/comparative/` |
| C1 Clean notebooks | all notebooks run top to bottom with outputs |
| C2 README | `README.md` |
| C3 Commit history, requirements | GitHub history; `requirements.txt`, `app/requirements.txt` |
| Bonus: GUI + public deployment | `app/streamlit_app.py` at https://cloudshield-live.streamlit.app/ |
| Regression (Review 1 track) | `Class/02_regression_modeling.ipynb`, `reports/` |

## Before the review

- [ ] Fill the "Observation (write in your own words)" boxes in `notebooks/clustering.ipynb` and `notebooks/04_comparative_study.ipynb`.
- [ ] Each member reads through the track they will explain.
- [ ] Wake the live app and open the website on the presenting laptop.
