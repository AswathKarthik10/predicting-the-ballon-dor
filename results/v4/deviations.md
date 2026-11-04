# Deviations and interpretation choices, Experiment 4

Everything below was decided before any held-out score was seen. Items marked EXPLORATORY change nothing in the fixed menu;
they fill gaps the spec leaves open.

## Search budget (spec 6.1)
- The full grid is 6 preprocessing x 29 selector settings x 81 model settings = **14,094 configurations**, too large for
  12 inner fits each with the stability-selection and tree models. The **random-search fallback was used: 300 configurations,
  seed 7, drawn uniformly without replacement from the full grid, the same 300 in every outer fold.**
- Uniform sampling follows the grid's composition, so the 300 are dominated by the model and selector families with the most
  settings (P-EN 132, P-R 45, P-L 43, LGB 34, RF 25, XGB 21; FS2 140, FS4 43, FS1 41, FS5 35, FS3 28, FS0 13). This is not
  rebalanced after the fact.
- Model grid sizes: P-R 13, P-L 13, P-EN 39, LGB 8, XGB 4, RF 4 (XGB and RF have two settings on two axes each).

## Settings the spec does not fix
- **FS3 lasso penalty:** the spec gives only pi for stability selection. The half-sample lasso penalty is fixed at **1.0**
  (a point of the 0.001-10 grid), chosen on compute and sparsity grounds (about 60-70 of 132 columns active on the full training
  set, so half-samples stay sparse enough for pi to discriminate), before any held-out scoring. Section 7's stability run on
  all four editions uses the median penalty chosen by P-L / FS2 folds when available, otherwise 1.0.
- **Lasso / elastic-net parameterisation:** minimise -sum of edition log-likelihoods + lam * (mix * |b|_1 + (1 - mix) * |b|_2^2),
  the same scaling as the v2 ridge (lam * |b|^2 on the summed log-likelihood). Lasso is mix = 1.
- **Solver:** FISTA with backtracking and adaptive restart, convergence judged on the objective (relative change < 1e-8, up to
  20,000 iterations). A fit that hits the cap is kept and flagged `converged = False` in `search_log_v4.csv`.
- **Missing indicators** are built for every column with any null across the 150 rows (features only, no targets). One-hot and
  indicator columns constant on the labelled rows are dropped and listed in `candidates.csv`.
- **Category merging:** levels with fewer than 5 labelled rows go to `other`. `nationality` keeps the 8 levels with >= 5 labelled
  rows plus `other`.
- **Within-edition percentile:** average rank scaled to [0, 1]; constant columns become 0. Within-edition z-score of a column
  constant in an edition becomes 0. `log1p` clips apply-time negatives at 0 (training columns with negatives are never logged).
- **Selectors:** FS1 averages the absolute within-edition Spearman over training editions. FS5 prunes greedily in order of
  FS1 score and drops a column when its edition-averaged absolute Spearman with a kept column exceeds 0.9. FS4 uses a fixed
  LightGBM ranker (4 leaves, 100 trees, rate 0.05, min leaf 5, lambda_l2 5, seed 7) and keeps the top k columns with non-zero gain.
- **Tree labels:** dense-rank relevance from `baselines.relevance_labels` (a pure points transform).
- **Empty selection:** a configuration whose selector keeps no column scores as the uniform utility. It is a legitimate
  member of the search and the 1-SE rule can pick it.
- **Temperature for tree models:** fitted on out-of-fold scores of a leave-one-training-edition-out loop. Column selection is
  held fixed inside that loop (re-running stability selection per sub-fold was too costly), so the temperature is calibrated on
  training editions only but not with a fully nested selection. Slightly optimistic sharpness is possible; the outer held-out
  edition is never involved.
- **Mean feature count** for the 1-SE rule is the mean number of selected columns over the three inner folds.

## Data
- The spec's 150 rows are the 2022-2026 rows of `data/v2/player_snapshot_t1.csv` (the 300-row file also holds 2016-2021). The
  root `player_snapshot_t1.csv` is the older v1 file and is not used for v4.

## Importance (spec 7)
- `shap` is not installed. Tree importance will use LightGBM `pred_contrib` / XGBoost `pred_contribs` mean |value| plus permutation
  importance instead.

## Comparators
- v3 E3-S (M1) and v1 M1 utilities are recomputed with the frozen code and verified against the saved held-out
  log-likelihoods (max abs gap < 1e-6) before use; players are aligned to the v4 row order by `player_key`.
- `overfitting_gap_v4.csv` (spec 6.4) is an extra output; for the fixed v3 model the inner score is the best mean inner
  leave-one-edition-out log-likelihood over the v2 penalty grid.
