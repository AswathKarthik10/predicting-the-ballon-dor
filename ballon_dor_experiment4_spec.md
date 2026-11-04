# Experiment 4 Spec: A Hypothesis-Free, Data-Driven Pipeline (v4)

Nov 4, 2026 · companion to `ballon_dor_experiment3_spec.md` (v3)

## 0. Read this first

**Question:** if we drop every hand-built feature and let a standard data science pipeline choose features and models from the raw columns, how well does it predict the Ballon d'Or, which features does it pick, and how does it compare with the hypothesis-driven model (v3, E3-S) on the same editions?

**Design in one line:** the raw T1 snapshot columns of the 2022-2026 editions → automatic preprocessing → automatic feature selection → a model chosen from a fixed menu, with **every choice made by inner cross-validation inside each training fold**. Scored by leave-one-edition-out on 2022-2025, then a 2026 forecast.

**Rules**

- Repo `predicting-the-ballon-dor`, new branch `experiments-v4` from `experiments-v3`. Outputs in `results/v4/`.
- **No hypotheses:** no composites (F1-F12), no residualisation, no hand weights, no use of the v1-v3 feature definitions or coefficients. Column exclusions are allowed only for the reasons in section 2.2 (identifiers, targets, dates, provenance, leakage), each listed.
- **No leakage from selection:** feature selection, preprocessing choices and hyperparameters are fitted inside the training folds only. The held-out edition is never used to choose anything.
- **Fixed menu:** the search space in sections 3-5 is fixed now. Do not add methods, features or grid values after seeing results. Anything changed goes in `results/v4/deviations.md`, labelled exploratory.
- **2026:** the result is known. The v4 2026 forecast is EXPLORATORY and post-hoc. `forecast-v2` remains the forecast of record.
- Stop for review after the back-test (section 6), before the 2026 forecast (section 8).

## 1. Data

- **Same data as v3:** dataset commit pinned in `results/v3/environment.txt`. Record it again in `results/v4/environment.txt`.
- **Rows:** `player_snapshot_t1`, editions 2022-2026 (150 rows; 120 labelled). T1 only; the post-window block is out of scope.
- **Target:** the within-edition ranking from `bdo_points` (zero-point nominees tied at the bottom), exactly as in v1-v3. Never `bdo_rank` for ordering.
- **Not used:** `fact_award_events` difficulty weights (hand-built in v2.1), the odds table, the juror tables (except for the optional simulation in 8.2), T2 columns.

## 2. Candidate features

### 2.1 Start from every column

Take every column of `player_snapshot_t1` and apply only the mechanical rules below.

### 2.2 Excluded columns (and why)

| Reason | Columns |
|---|---|
| Identifiers and labels | `edition`, `snapshot`, `player_key`, `player_name`, `club_key`, `club_team_id`, `league_key` |
| Targets | `bdo_points`, `bdo_rank`, `vote_share` |
| Dates and build metadata | `snapshot_date`, `tm_value_date`, `ea_update_date`, `built_at`, `pipeline_version` |
| Provenance text | `honors_source`, `tourn_individual_award` (free text, null for the API rows) |
| Constant within an edition (cannot affect a within-edition ranking) | `eligibility_format`, `ballot_picks`, `jurors_voted`, and any column that is constant inside every edition |
| Zero variance across all rows | detected automatically (e.g. `jurors_same_nation`, `prev_bdo_rank_partial`) |

Everything else stays: performance, xG and non-xG rates, ratings, honours, awards, market value, EA rating, attention, news tone, juror-exposure counts (`jurors_same_confed`, `teammates_on_shortlist`, `nat_teammates_on_shortlist`), position, age, prior Ballon d'Or history. Write the final candidate list with counts to `results/v4/candidates.csv`.

### 2.3 Mechanical encoding (fixed, applied before the pipeline)

- **Booleans** → 0/1.
- **Categoricals** (`position_group`, `nationality`, `tourn_name`, `ucl_stage_reached`, `nt_tourn_stage_reached`): one-hot. Categories with fewer than 5 labelled rows are merged into `other`. `nationality` is one-hot only if a category has at least 5 labelled rows; otherwise dropped (record which).
- **Missing values:** a `<col>_missing` indicator for every column with any null, then the value is imputed by the training-fold median (inside the pipeline).
- **No ordinal maps** (stage columns stay categorical). No hand-built ratios or totals.

## 3. Preprocessing options (chosen by inner CV)

| Option | Values |
|---|---|
| Scaling | (a) within-edition z-score; (b) global z-score fitted on the training fold; (c) within-edition percentile rank |
| Skew transform | (a) none; (b) `log1p` on columns with training-fold skewness > 1 and no negative values |

Six combinations. The within-edition options use only the row's own edition (leak-free); the global option is fitted on the training fold.

## 4. Feature selection options (chosen by inner CV)

| ID | Method | Setting grid |
|---|---|---|
| FS0 | None (all candidates) | |
| FS1 | Univariate filter: absolute within-edition Spearman with points, averaged over training editions; keep the top k | k ∈ {5, 10, 15, 25} |
| FS2 | L1 (lasso) Plackett-Luce: features with non-zero weight | penalty on a 13-point log grid 0.001-10 |
| FS3 | Stability selection: lasso PL on 100 half-samples (players resampled within edition); keep features selected in at least π of runs | π ∈ {0.6, 0.75, 0.9} |
| FS4 | Tree importance: LightGBM ranker gain importance; keep the top k | k ∈ {5, 10, 15, 25} |
| FS5 | Correlation pruning then FS1: drop one of each pair with within-edition Spearman > 0.9 (keep the one more correlated with points), then the top k | k ∈ {5, 10, 15, 25} |

Each selector is fitted on the inner training editions only.

## 5. Models (chosen by inner CV)

| ID | Model | Settings |
|---|---|---|
| P-R | Plackett-Luce, ridge | λ on the v2 grid (0.01-100) |
| P-L | Plackett-Luce, lasso | λ on the 0.001-10 grid |
| P-EN | Plackett-Luce, elastic net | λ grid × mixing ∈ {0.25, 0.5, 0.75} |
| LGB | LightGBM LambdaMART | `num_leaves ∈ {4, 8}`, `n_estimators ∈ {100, 300}`, `learning_rate 0.05`, `min_data_in_leaf ∈ {5, 10}`, `lambda_l2 5`, seed 7 |
| XGB | XGBoost pairwise ranker | `max_depth ∈ {2, 3}`, `n_estimators ∈ {100, 300}`, `learning_rate 0.05`, `subsample 0.8`, `reg_lambda 5`, seed 7 |
| RF | Random forest regressor on within-edition points percentile | `max_depth ∈ {3, 5}`, `min_samples_leaf ∈ {3, 8}`, 500 trees, seed 7 |

Non-PL outputs are converted to PL utilities with the fitted temperature, as in v1 (temperature fitted on training editions only).

## 6. Validation (stop for review after this)

### 6.1 Fully nested leave-one-edition-out

- **Outer loop:** hold out each of 2022, 2023, 2024, 2025 in turn (4 folds).
- **Inner loop (per outer fold):** leave-one-edition-out over the 3 training editions. For every combination of preprocessing (3) × selection (FS0-FS5 with their grids) × model (with its grid), compute the mean inner held-out PL log-likelihood.
- **Choice:** the combination with the best mean inner log-likelihood is refitted on all 3 training editions and scored on the outer held-out edition.
- **Search budget:** if the full grid is too large to run, use random search with 300 configurations per outer fold, the same 300 (seed 7) in every fold. Record which was used.
- **One-standard-error rule** (fixed): among configurations within one standard error of the best inner score, choose the one with the fewest selected features. This counters overfitting with only three inner folds.

### 6.2 Metrics (outer held-out editions)

Held-out PL log-likelihood per voted player (primary), winner's predicted rank, top-1 hit, top-3 overlap, top-10 accuracy, NDCG@10, Spearman, pairwise accuracy. Per fold and mean; 5-pick (2022-2023) and 10-pick (2024-2025) means shown separately.

### 6.3 Comparisons on the same four folds

| Comparison | Source |
|---|---|
| v4 pipeline | this experiment |
| v3 E3-S (hypothesis-driven, v2.1 features, same training span) | `results/v3/` |
| v1 M1 (hypothesis-driven, v1 features) | `results/` on `juror-simulation-ballots` |
| Uniform utility (U0) | reference |

Paired within-edition bootstrap (1,000 draws) for v4 − v3 E3-S and v4 − v1: mean difference, 90% interval, folds better.

### 6.4 Overfitting check

Report, per outer fold, the inner-CV score of the chosen configuration and its outer score. A large gap (inner much better than outer) means selection overfitted; report the mean gap and compare it with the same gap for the fixed v3 model.

## 7. What the pipeline chose

- **Per fold:** the chosen preprocessing, selector, model, hyperparameters and the selected feature list.
- **Stability:** for each candidate feature, the share of outer folds in which it was selected; and from a separate stability-selection run on all four labelled editions (100 half-samples, lasso PL at the median chosen penalty), its selection frequency.
- **Importance:** for the final model (section 8.1): PL coefficients with Laplace and bootstrap 90% intervals, or, for a tree model, mean |SHAP| and permutation importance on the training editions.
- **Overlap with the hypothesis set:** map each selected raw column to the v3 block it would have fed (season output, big stage, trophies and team, awards, prior standing, fame and attention, market value, position) or `none` (a column v3 did not use, e.g. `teammates_on_shortlist`, `club_minutes`, `news_tone_in`). Report the share of selected features per block. Descriptive only.

## 8. 2026 (EXPLORATORY; after the review)

### 8.1 Final fit

Run the inner-CV selection of section 6.1 once on all four labelled editions (leave-one-edition-out over 2022-2025) to choose the configuration; refit it on all four; score the 2026 rows.

### 8.2 Probabilities

- Plain PL first-pick probabilities from the final model's utilities.
- If the final model is PL, or after temperature conversion for a tree model: the juror simulation exactly as v3 (2022-2025 jury bias, the 2026 jury, link re-estimated on 2022-2025 from out-of-fold v4 utilities, 10,000 runs, seed 7).

### 8.3 Comparison with the actual result

Add a row for v4 to the comparison table (`comparison_2026_all_versions.csv`): pick, Kane's predicted rank and P(win), top-3 and top-10 overlap, Spearman on the actual top 10 (Kane, Mbappé, Yamal, Rodri, Messi, Kvaratskhelia, Dembélé, Haaland, Olise, Ruiz). Also list the top 10 features driving the 2026 top 5.

## 9. How to read the result

- 120 labelled rows and about 100+ candidate columns: expect the pipeline to choose few features and the result to vary by fold. Report that variation; do not average it away.
- With four outer folds, differences below about 0.03 log-likelihood per voted player are within noise; claim a difference only if the paired 90% interval excludes 0.
- A worse v4 score is a valid result: it would show that domain structure helps at this sample size.

## 10. Tests (must pass before the real run)

- The outer held-out edition never appears in any inner fit (assert per fold: selector, scaler, imputer, temperature and model saw only training editions).
- Candidate list excludes every column in section 2.2 and contains no target-derived column.
- Within-edition scaling uses only the row's own edition.
- The one-standard-error rule picks the smallest feature set among tied configurations (unit test on a toy grid).
- No 2026 target is read before section 8 (existing guard).
- Determinism: two runs with seed 7 give identical choices and scores.

## 11. Outputs

```
results/v4/
  environment.txt
  deviations.md
  candidates.csv                 # every candidate column, type, missing share, why kept
  search_log_v4.csv              # every configuration × outer fold: inner score
  chosen_by_fold_v4.csv          # preprocessing, selector, model, hyperparameters, n features, inner and outer score
  selected_features_by_fold.csv  # feature × fold (0/1)
  stability_selection_v4.csv     # feature, selection frequency
  metrics_by_fold_v4.csv
  metrics_summary_v4.csv
  paired_v4.csv                  # v4 − v3 E3-S, v4 − v1
  importance_final_v4.csv        # coefficients or SHAP / permutation importance
  block_overlap_v4.csv           # selected features mapped to v3 blocks
  forecast_2026_v4_T1.csv        # EXPLORATORY
  comparison_2026_all_versions.csv  # v1, v2, v2.1, v3, v4
```

## 12. Order of work

1. Branch, candidate list and encoding (section 2), tests (section 10).
2. Nested back-test (sections 3-6) → stop for review.
3. What the pipeline chose (section 7).
4. 2026 final fit, forecast and comparison (section 8), labelled exploratory.

## 13. What not to do

- Do not reuse v1-v3 composites, residuals, weights or coefficients.
- Do not select features, preprocessing or hyperparameters on the outer held-out edition.
- Do not widen the search after seeing results.
- Do not drop a column by judgement; only by the rules in section 2.2.
- Do not present the v4 2026 forecast as a forecast of record.
