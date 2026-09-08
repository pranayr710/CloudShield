# UNSW-NB15 — Data Audit & Preprocessing Decisions

Every number below was measured on the actual files in `data/raw/`, not taken from a paper.
Reproduce with `python -m src.preprocessing`.

**Files audited**

| File | Rows | Cols | Size |
|---|---|---|---|
| `UNSW_NB15_training-set.csv` | 82,332 | 45 | 15 MB |
| `UNSW_NB15_testing-set.csv` | 175,341 | 45 | 31 MB |
| `UNSW-NB15_1..4.csv` | ~2.54 M total | 49, no header | 562 MB |
| `NUSW-NB15_features.csv` | 49 rows | — | feature dictionary |
| `UNSW-NB15_LIST_EVENTS.csv` | — | — | attack subcategory counts |

We use **only the two partition files** (45 columns, with headers). The `UNSW-NB15_1..4.csv`
dumps are the 49-column raw Argus/Bro output with no header row, different column names,
`srcip`/`dstip`/`sport`/`dsport` identifiers, and blank cells for benign `attack_cat`.

---

## Finding 1 — The two partition files have their names swapped ⚠️

| File | Actual rows | Rows per Moustafa & Slay (2015) |
|---|---|---|
| `UNSW_NB15_training-set.csv` | **82,332** | 175,341 |
| `UNSW_NB15_testing-set.csv` | **175,341** | 82,332 |

The names are inverted relative to the published partition. This is a known defect in the
official distribution and a large fraction of published work using these files has the split
backwards without noticing.

**Decision: ignore the supplied partition entirely.** We concatenate both files (257,673 rows)
and build our own stratified 80/20 split. This sidesteps the swap and also removes the
distribution shift that the official partition bakes in.

**Verified safe to concatenate:** the two files are effectively disjoint — only **940 rows
(1.7%)** of the smaller file's unique rows also appear in the larger one. They are genuine
partitions, not overlapping samples.

---

## Finding 2 — Zero missing values, zero infinities

All 45 columns across all 257,673 rows: **0 NaN, 0 inf, 0 negative values.**

This is worth stating explicitly because it is unusual — the raw `UNSW-NB15_1..4.csv` dumps
*do* contain nulls and malformed `ct_ftp_cmd` strings. The curated partition files were
cleaned by the dataset authors.

**Decision:** no imputation needed. `clean()` still contains a justified median-imputation
path guarded by an assertion, so the pipeline degrades gracefully if the input ever changes,
and the audit claim is verified at runtime rather than assumed.

---

## Finding 3 — 36.8% of rows are exact duplicates, causing 42.6% test-set leakage ⚠️⚠️

This is the single most consequential finding.

| Measure | Value |
|---|---|
| Combined rows | 257,673 |
| Exact duplicate rows (ignoring `id`) | **94,928 (36.8%)** |
| — internal to `training-set.csv` | 26,387 (32.0% of that file) |
| — internal to `testing-set.csv` | 67,601 (38.6% of that file) |
| Rows after `drop_duplicates()` | **162,745** |

**The consequence, measured directly:**

| Split built from | Test rows with an *exact twin* in train |
|---|---|
| Duplicates kept | **21,958 / 51,535 = 42.6%** |
| Duplicates dropped | **0 / 32,549 = 0.0%** |

With duplicates left in, nearly half the test set is memorised rather than predicted. Every
accuracy, F1 and R² would be inflated, and the inflation would be invisible in the notebook.

**Decision: drop duplicates before splitting.** Rubric B1 requires duplicates be "checked and
treated"; here treating them is not cosmetic, it is what makes the entire evaluation valid.

### Side effect that must be reported: the class prior shifts substantially

| Class | Before dedup | After dedup | Change |
|---|---|---|---|
| Normal | 93,000 | 85,722 | −7.8% |
| Exploits | 44,525 | 27,434 | −38.4% |
| Fuzzers | 24,246 | 20,960 | −13.6% |
| Reconnaissance | 13,987 | 9,991 | −28.6% |
| **Generic** | **58,871** | **7,599** | **−87.1%** ⚠️ |
| DoS | 16,353 | 5,500 | −66.4% |
| Analysis | 2,677 | 2,032 | −24.1% |
| Backdoor | 2,329 | 1,880 | −19.3% |
| Shellcode | 1,511 | 1,456 | −3.6% |
| Worms | 174 | 171 | −1.7% |

`Generic` collapses by 87%. That is not an error — `Generic` is an attack against block ciphers
that emits near-identical flow records by construction, so it duplicates far more than any
other class. Post-dedup, `Generic` drops from the 2nd largest class to the 5th.

**Viva answer:** the duplicates are real repeated observations, but a random split places
identical rows on both sides of it, which is textbook train/test contamination. Deduplicating
costs realism in the class prior and buys a valid evaluation. We report the shift rather than
hiding it.

---

## Finding 4 — Five columns are algebraically tied to the regression target ⚠️

Target: `sloss` (source packets retransmitted or dropped).

| Relationship | Match rate | r with `sloss` |
|---|---|---|
| `sbytes` | — | **0.9967** |
| `spkts` (and `sloss ≤ spkts` by definition) | 13 violations / 162,745 | **0.9738** |
| `smean == sbytes / spkts` | **100.0% exact** | 0.2197 |
| `sload ≈ sbytes × 8 / dur` | r = 0.998 | 0.0212 |
| `rate ≈ (spkts + dpkts − 1) / dur` | **99.7% exact** | 0.0299 |

`LEAKY_FOR_SLOSS = ["sbytes", "spkts", "smean", "sload", "rate"]`, exposed via
`get_dataset("regression", drop_leaky=True)`.

### Two other exact redundancies found

- **`ct_ftp_cmd` is byte-for-byte identical to `is_ftp_login`** in 100% of rows (r = 0.9990).
  `ct_ftp_cmd` is dropped permanently — keeping both double-counts one signal and inflates its
  apparent importance in tree models.
- **`tcprtt == synack + ackdat`** in 100% of rows, exactly as the feature dictionary defines it.
  All three are kept (trees are unaffected by the collinearity and the components carry
  directional information), but this must be acknowledged when interpreting linear coefficients.
- `dmean == dbytes / dpkts` in 100% of rows — the destination-side mirror of `smean`. Harmless
  for `sloss`, but relevant if anyone switches the target to `dloss`.

---

## Finding 5 — `sloss` is so heavy-tailed that R² is meaningless on the raw scale ⚠️

| Statistic | Value |
|---|---|
| mean / median | 7.49 / 2.0 |
| std | 81.78 |
| 75th / 95th / 99th percentile | 5 / 19 / 53 |
| max | 5,319 |
| zeros | 28.3% |
| skew | 38.0 |

**Variance concentration in the 32,549-row test set:**

| Top-N rows by squared deviation | Share of total variance |
|---|---|
| 10 rows (0.03%) | **62.7%** |
| 100 rows (0.31%) | **98.7%** |
| 1,000 rows (3.07%) | 99.5% |

R² is a variance-explained metric, so **ten rows decide two-thirds of it.** Measured
consequence: every tree model scores R² ≈ 0.998 whether or not the leaky columns are removed.

| Framing | Linear Regression | Random Forest |
|---|---|---|
| raw `sloss`, all features | 0.6926 | 0.9985 |
| raw `sloss`, `drop_leaky=True` | 0.6818 | 0.9984 |
| **log1p(`sloss`)**, all features | 0.9633 | 0.9988 |
| **log1p(`sloss`)**, `drop_leaky=True` | **0.8591** | **0.9966** |

Rubric C2 requires ten models "ranked by R²". On the raw target that table is ten near-identical
0.99s and the ranking is numerical noise. On `log1p` the model families separate cleanly.

**Decision: `log_target=True` is the recommended primary framing** for the regression track.
Report R²/RMSE in log space and MAE back-transformed with `np.expm1()` so the error stays
interpretable in packets. Keep the raw-target run as a secondary table with this explanation.

### And the honest framing is *still* highly predictable — that is a real result, not a bug

Verified with a shuffled-target control: `RandomForestRegressor` on `drop_leaky=True` with
`y_train` shuffled scores **R² = −0.097**, i.e. the pipeline has no structural leak.
Restricted to the 99% of rows below the 99th percentile, the honest model still scores
R² = 0.9932, MAE = 0.233 packets.

The explanation is that Argus derives all 40-odd flow statistics from the same packet stream,
so they are mutually constraining. An ablation shows how deep this goes:

| Feature set | RF R² |
|---|---|
| everything | 0.9992 |
| − 5 leaky columns | 0.9981 |
| − also `sinpkt` | 0.9926 |
| − also `dbytes`, `dpkts`, `dmean` | 0.9707 |
| − also `dloss`, `dst_loss_ratio` | 0.9734 |
| − also `dur`, `dinpkt`, `dload`, `iat_ratio` | 0.6999 |

Only when duration is removed does the target become genuinely hard. Present this ablation —
it is a far stronger contribution than a single R² number.

---

## Finding 6 — Categorical columns need level folding, not naive one-hot

| Column | Levels | Detail |
|---|---|---|
| `proto` | **133** | top 10 cover **96.84%** of rows; 1 level has a single row |
| `service` | 13 | `'-'` is 62.2% of rows — the dataset's own "not much used service" marker, i.e. effectively unknown, but a legitimate category rather than missing data |
| `state` | 11 | `FIN` 111,833 · `INT` 29,483 · `CON` 19,198 · `REQ` 2,129 · `RST` 84 · then `ECO` 10, `ACC` 4, `CLO` 1, `PAR` 1, `URN` 1, `no` 1 |

One-hot encoding all 133 `proto` levels would add ~130 near-empty columns that dominate the
distance metric for KNN, SVR and SVC without adding signal.

`state` has levels occurring **once** in the entire dataset. A stratified split on `attack_cat`
gives no guarantee they land in train, so they arrive at `transform()` as unknown categories —
which sklearn silently encodes as all-zeros while emitting a `UserWarning` on every call.

**Decision:** fold rare levels into `"other"`, with the whitelist **fitted on the training split
only** — `proto` keeps its top 10, `service` and `state` keep every level with ≥ 20 training
rows. This eliminates the warning, keeps the feature space compact, and is itself a
fit-on-train-only transform.

Levels kept (learned from train): `proto` → tcp, udp, unas, ospf, arp, sctp, any, gre, sun-nd,
sep · `service` → all 12 non-rare · `state` → FIN, INT, CON, REQ, RST.

---

## Finding 7 — Structural zero-blocks are informative, not missing data

Several groups of columns share an identical zero rate, which reveals the data's structure:

| Zero rate | Columns | Meaning |
|---|---|---|
| **19.4%** | `dpkts`, `dbytes`, `dttl`, `dload`, `dinpkt`, `dmean` | flows where the destination **never replied** — unanswered probes, scans, backscatter |
| **~29.3%** | `stcpb`, `dtcpb`, `dwin`, `tcprtt`, `synack`, `ackdat` | **non-TCP** flows, which have no handshake or sequence numbers |
| 85–99% | `trans_depth`, `response_body_len`, `ct_flw_http_mthd`, `is_ftp_login`, `is_sm_ips_ports` | protocol-specific counters, zero unless the flow is HTTP/FTP |

These are genuine zeros with meaning, **not** missing values to impute. The 19.4% no-reply
population is a discontinuity rather than a continuum, which is why it is made explicit as the
engineered `no_dst_response` flag.

Also noted: `dur` is capped at exactly 59.999999 s and `rate` at exactly 1,000,000 — collection
limits of the capture harness, not outliers to remove.

---

## Finding 8 — `label` is a deterministic function of `attack_cat`

Crosstab is perfectly block-diagonal: `Normal` → `label` 0 (93,000 rows), all nine attack
families → `label` 1, with zero off-diagonal entries.

**Decision:** both `attack_cat` and `label` are removed from `X` for every task. Leaving `label`
in the classification feature set would hand any model the benign/attack boundary for free.

---

## Preprocessing decisions — final summary

| Step | Decision | Rubric |
|---|---|---|
| Load | Concatenate both partition files (257,673 × 45); ignore the swapped official split | A1 |
| Drop `id` | Ranges 1..175,341 overlap across files, so not unique and carries no signal | B1 |
| Drop `ct_ftp_cmd` | 100% identical to `is_ftp_login` | B1 |
| Missing values | None present (asserted at runtime); guarded median-impute path retained | B1 |
| Duplicates | **Drop 94,928 (36.8%)** — removes 42.6% test-set contamination | **B1** |
| Outliers | **Clip**, not delete, at train-set 1st/99th percentile; bounded/discrete columns exempt | B1 |
| Feature engineering | 6 domain features + 2 classification-only (see below) | **B3** |
| Split | Stratified 80/20 on `attack_cat`, `random_state=42`, identical rows for all three tracks | **B2** |
| Rare categorical levels | Folded to `"other"`, whitelist fit on train only | B2 |
| Encoding | `OneHotEncoder(handle_unknown='ignore', drop='first')` fit on train only | **B2** |
| Scaling | `StandardScaler` fit on train only — verified: train mean = 2.0e-16, test mean = 0.0213 | **B2** |
| Subsampling | Applied *after* fit/transform, so the feature space and test matrix are byte-identical to a full run | fair comparison |

**Outliers — why clip rather than drop:** the extreme values *are* the attack signal. A DDoS
flow with `sbytes` in the millions is the phenomenon being modelled, so deleting it would
remove the thing we are trying to detect. Clipping at the 1st/99th percentile bounds the
influence on scale-sensitive models (SVR, KNN, linear) while preserving the row and its label.
Bounds come from the training split only.

### Final shapes

| Task | X_train | X_test | Features |
|---|---|---|---|
| regression | (130196, 72) | (32549, 72) | 72 |
| regression, `drop_leaky=True` | (130196, 65) | (32549, 65) | 65 |
| classification | (130196, 73) | (32549, 73) | 73 |
| clustering | (130196, 22) | (32549, 22) | 22 |

Per-class split (all 10 classes present in both sides, `Worms` = 137 train / 34 test):

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
| Worms | 137 | 34 |

---

## Engineered features (rubric B3) — with the two rejected candidates

**Kept — valid in both regression framings:**

| Feature | Formula | Domain justification |
|---|---|---|
| `dst_loss_ratio` | `dloss / (dpkts + ε)` | Destination-side loss fraction. Normalises loss by volume so a 10-packet and a 10,000-packet flow are comparable. Destination side only — the source-side equivalent would divide the regression target by one of its own predictors. |
| `ttl_diff` | `sttl − dttl` | Crafted/spoofed packets carry TTLs inconsistent with the return path. A classic IDS heuristic; `sttl` has 13 distinct values and `dttl` 9, so the difference is a compact near-categorical signal. |
| `jitter_ratio` | `sjit / (djit + ε)` | One-sided jitter indicates congestion or queue exhaustion in a single direction — the direct mechanism behind QoS degradation, which is what the regression track predicts. |
| `iat_ratio` | `sinpkt / (dinpkt + ε)` | Automated tooling emits on a fixed cadence while victims reply irregularly. Separates machine-generated from interactive traffic. |
| `no_dst_response` | `(dpkts == 0)` | Makes the 19.4% no-reply population explicit. It is a discontinuity, not a continuum — linear models cannot express it otherwise, and trees would spend splits rediscovering it. |
| `conn_fanout` | `ct_srv_src / (ct_dst_ltm + ε)` | Reconnaissance touches many services on one host; a flood repeats one service. Separates the two shapes. |

**Kept but dropped under `drop_leaky=True`** (algebraic descendants of the leaky columns —
still valuable for classification and clustering):

| Feature | Formula |
|---|---|
| `pkt_dir_ratio` | `spkts / (dpkts + ε)` |
| `byte_dir_asymmetry` | `\|sbytes − dbytes\| / (sbytes + dbytes + ε)` |

**Rejected — both were in the original plan and are degenerate on this dataset:**

| Rejected candidate | Why |
|---|---|
| `bytes_per_pkt_src = sbytes / spkts` | **Already exists as `smean`**, matching in 100% of rows. Presenting it as engineered would be wrong and an examiner checking the feature dictionary would catch it. |
| `handshake_share = (synack + ackdat) / tcprtt` | **Identically 1.0 for every row**, since `tcprtt` is *defined* as `synack + ackdat` and matches in 100% of rows. A constant column with zero variance. |

Catching these two before writing the notebook is exactly why the audit ran first.

---

## Empirical evidence the engineered features earn their place

Random Forest importances in the honest regression framing (`drop_leaky=True`) put
`iat_ratio` **4th of 65 features** at 0.061, ahead of `dur`, `djit` and every categorical
dummy. `dst_loss_ratio` alone reaches R² = 0.5445 as a single-feature decision tree.

That is the direct evidence rubric B3 asks for — not just a justification paragraph.

---

## What is NOT done here, deliberately

- **No SMOTE / class weighting.** Resampling belongs inside the model pipeline so it is fitted
  on the training fold only. Doing it in `src.preprocessing` would leak into cross-validation.
  Person C owns this (`imblearn.pipeline.Pipeline`).
- **No feature selection.** All 10 regression algorithms must see the same feature set; Lasso's
  coefficient sparsity is itself a rubric deliverable.
- **No `y` clipping.** Features are clipped, the target never is — that would distort what the
  model is asked to predict.
