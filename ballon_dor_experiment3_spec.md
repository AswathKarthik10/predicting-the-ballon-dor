# Experiment 3 Spec: Does the Format Change Affect Prediction Quality?

Nov 3, 2026 · builds on `ballon_dor_experiments_spec.md` (v2) and `ballon_dor_awards_fix_spec.md` (v2.1)

## 0. Read this first

**Question:** the Ballon d'Or changed format in 2022: from a calendar-year window with about 175 juror nations (2016-2021) to an August-July season window with the FIFA top 100 (2022 onward). Does training on the old-format editions help or hurt predictions for the new format?

**Design in one line:** score the four season-format editions (2022-2025) under three training sets, with everything else held fixed:

| Run | Trains on (per held-out season edition) | What it answers |
|---|---|---|
| **E3-S** (season only) | The other 3 season editions | The user's experiment: v2.1 features, first-experiment span |
| **E3-A** (all formats) | The other 8 editions (5 calendar + 3 season) | Already run as v2.1; reused, not refitted |
| **E3-C** (calendar only) | The 5 calendar editions | Pure transfer: does the old format predict the new one at all? |

If E3-S beats E3-A, the old-format editions hurt; if E3-A beats E3-S, they help (more data outweighs the format change); if E3-C is close to both, the drivers of the vote did not change with the format.

**Rules**

- Repo `predicting-the-ballon-dor`, new branch `experiments-v3` from `experiments-v2.1`. Outputs in `results/v3/`.
- **Features are exactly v2.1:** F1-F12 with the difficulty-weighted F11, the expanded attention features (F9, F10), market value (F12), position flags. Same within-edition standardisation, same fold-internal residualisation, same λ grid, seeds and code paths. The only thing that changes is which editions are in the training set.
- **Pre-registration:** sections 1-4 are fixed before any run. Anything decided after seeing results goes in `results/v3/deviations.md`, labelled exploratory.
- **2026:** the result is known. Every 2026 output in this experiment is EXPLORATORY and post-hoc. `forecast-v2` remains the forecast of record.
- Stop for review after section 2 (the back-test comparison), before section 4 (2026).

## 1. Models and data

- **Main model:** M1 (Plackett-Luce, ridge). Also run X1 (XGBoost ranker) and B1 (ridge on vote share) to check that the conclusion does not depend on the model class. Fixed settings as in v2.
- **Data:** dataset commit pinned in `results/v2_1/environment.txt` (with `fact_award_events`). Record it again in `results/v3/environment.txt`.
- **T1 rows only.** M2 (post-window) is out of scope here.

### 1.1 Tuning with few training editions

- E3-S has only 3 training editions per fold, so the inner leave-one-edition-out for λ has 3 inner folds of 2 training editions. Use it as is, on the same 13-point grid.
- E3-C tunes λ by inner leave-one-edition-out over its 5 calendar editions.
- Report the λ chosen per fold for every run.

## 2. Back-test comparison (stop for review after this)

### 2.1 Main table

For each run (E3-S, E3-A, E3-C) and each model (M1, X1, B1), on the four held-out season editions 2022, 2023, 2024, 2025:

- Held-out PL log-likelihood per voted player (primary), winner's predicted rank, top-1 hit, top-3 overlap, top-10 accuracy, NDCG@10, Spearman.
- Per fold and mean. Note 2022-2023 are 5-pick ballots and 2024-2025 are 10-pick.

### 2.2 Paired differences

Within-edition bootstrap (1,000 draws, players resampled inside each held-out edition), paired across runs on the same draws:

- E3-S − E3-A (the main contrast)
- E3-C − E3-S
- E3-C − E3-A

Report the mean difference, 90% interval and the number of folds where each run is better.

### 2.3 Separating features from data (reference cells)

Put the result in this grid, all scored on 2022-2025:

| | Trained on season editions only | Trained on all editions |
|---|---|---|
| v1 features | v1 M1 (−2.3348, already reported) | not possible (no xG before 2021) |
| v2.1 features | **E3-S** | E3-A (v2.1) |

The row difference (v1 → v2.1 features, season only) is the feature effect; the column difference (season only → all editions) is the effect of adding the old format.

Also report the earlier v2 reference: M1-season4 with v2 features (−2.2773) against v2 nine-edition M1 (−2.2753).

### 2.4 Mirror check

Score the five calendar-format editions (2016-2021) under:

- **E3-Cself:** trained on the other 4 calendar editions.
- **E3-A:** trained on the other 8 editions (from v2.1).
- **E3-Scal:** trained on the 4 season editions only.

Same metrics and paired differences. This shows whether transfer is symmetric.

## 3. Did the drivers change with the format?

### 3.1 Coefficients by format

Fit M1 (fixed λ = the median λ chosen across E3-A folds, so penalties are comparable) on:

- all 4 season editions (2022-2025), and
- all 5 calendar editions (2016-2021).

For each feature report β with Laplace and bootstrap 90% intervals (bootstrap: 500 within-edition resamples), the difference season − calendar with a bootstrap 90% interval, and whether the interval excludes 0.

### 3.2 Pooled format-interaction test

Fit M1 on all nine editions with `feature × is_season` interactions for the blocks F9-F10 (fame and attention), F11 (awards), F4 (major titles) and F1 (attack), one block at a time. Report each interaction with its 90% interval and the change in held-out log-likelihood on 2022-2025 (leave-one-edition-out).

### 3.3 Confounds (state them; do not try to separate them)

The format change coincides with:

- the jury change (about 175 nations → FIFA top 100);
- the window change (calendar year → August-July, with T2 available from 2022);
- within the season era, the ballot change (5 picks in 2022-2023, 10 from 2024).

The experiment measures the combined effect of moving from the 2016-2021 setting to the 2022-2025 setting. It cannot attribute it to any one change. Report a 5-pick vs 10-pick split of the season folds as descriptive only.

## 4. 2026 (EXPLORATORY; after the review)

- Final fit of M1 on the 4 season editions (E3-S setting), λ by leave-one-edition-out over those 4.
- Juror simulation exactly as v2.1 (2022-2025 jury bias terms, the 2026 jury, 10,000 runs, seed 7). Re-estimate the link on the 4 season editions; report k and the shared SD.
- Output `results/v3/forecast_2026_v3_T1.csv`, headed EXPLORATORY, post-hoc.
- Compare with the actual 2026 result (`data/holdout/bdo_2026_result.csv`; if absent, use the published top 10: Kane, Mbappé, Yamal, Rodri, Messi, Kvaratskhelia, Dembélé, Haaland, Olise, Ruiz). Report a table of v1, v2 (locked), v2.1 and v3: pick, Kane's predicted rank and P(win), top-3 and top-10 overlap, Spearman on the actual top 10.

## 5. Power and how to read the result

- Only four held-out editions: differences in mean log-likelihood below about 0.03 per voted player are within noise (the v2 → v2.1 change was +0.008 with an interval spanning zero). Say so explicitly; do not claim an effect unless the paired 90% interval excludes 0.
- A winner-rank or top-1 difference rests on four winners; report it, do not lean on it.
- If E3-S and E3-A are indistinguishable, the conclusion is: "adding the old-format editions neither helps nor hurts at this sample size". That is a valid result.

## 6. Tests (must pass before the real run)

- The training set of each fold matches the table in section 0 (assert edition lists per fold for E3-S, E3-A, E3-C, and the mirror runs).
- E3-A reproduces the v2.1 per-fold numbers on 2022-2025 to 1e-6 (same code path, same seeds).
- Features for a given row are identical across runs (only the training set changes).
- No 2026 target is read before section 4 (existing guard).

## 7. Outputs

```
results/v3/
  environment.txt
  deviations.md
  metrics_by_fold_v3.csv          # run × model × fold × metric (season folds and mirror)
  metrics_summary_v3.csv
  paired_runs_v3.csv               # E3-S − E3-A, E3-C − E3-S, E3-C − E3-A (and mirror)
  feature_vs_data_grid.csv         # section 2.3
  tuning_by_fold_v3.csv
  coefficients_by_format.csv       # section 3.1
  format_interactions.csv          # section 3.2
  forecast_2026_v3_T1.csv          # EXPLORATORY
  comparison_2026_all_versions.csv # v1, v2, v2.1, v3 vs actual
```

## 8. Order of work

1. Branch, tests (section 6).
2. E3-S and E3-C back-tests; reuse E3-A; mirror runs (section 2) → stop for review.
3. Coefficients by format and interactions (section 3).
4. 2026 forecast and comparison (section 4), labelled exploratory.

## 9. What not to do

- Do not change features, weights, grids or seeds relative to v2.1.
- Do not refit E3-A; reuse the v2.1 outputs and verify they reproduce.
- Do not claim the format change caused a difference: the jury, window and ballot changes are confounded with it.
- Do not present the v3 2026 forecast as a forecast of record.
