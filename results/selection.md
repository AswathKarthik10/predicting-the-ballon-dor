# Model selection (spec rule 5.4)

# sha256_t1=249bb1ac9a623070c0573e5f2f40d1536f0122d86ad5f33a91ce170363e64f31
# sha256_t2=f9af1fd20747536eb06752cf202d0626c0ed55bd7f66507935ca83da27ad9cc0
# git_commit=36b162e

Primary metric: mean held-out Plackett-Luce log-likelihood per voted player over four leave-one-edition-out folds (higher is better). No 2026 output was used.

## Held-out log-likelihood per voted player

| model | mean | fold_2022 | fold_2023 | fold_2024 | fold_2025 |
|---|---|---|---|---|---|
| M1 | -2.335 | -2.501 | -2.392 | -2.155 | -2.291 |
| M1-G | -2.336 | -2.502 | -2.395 | -2.156 | -2.293 |
| X1 | -2.366 | -2.585 | -2.281 | -2.238 | -2.361 |
| M2 | -2.376 | -2.537 | -2.381 | -2.262 | -2.324 |
| B1 | -2.597 | -2.850 | -2.608 | -2.428 | -2.501 |
| B2 | -2.605 | -2.854 | -2.699 | -2.537 | -2.332 |
| B0 | -2.642 | -2.800 | -2.553 | -2.421 | -2.792 |
| U0 | -2.752 | -3.040 | -2.815 | -2.654 | -2.498 |

U0 is a uniform-utility reference (a model that knows nothing); any useful model must beat it.

## Selection outcome

**Published model: M1.** M2 is reported alongside.

- M2 minus M1 mean held-out log-likelihood = -0.0409 (margin 0.02); published model = M1.

## Secondary metrics (mean over folds)

| model | spearman | pairwise_acc | top1_hit | top3_overlap | winner_rank |
|---|---|---|---|---|---|
| U0 | nan | 0.500 | nan | nan | nan |
| B0 | 0.572 | 0.716 | 0.250 | 1.250 | 6.500 |
| B1 | 0.716 | 0.778 | 0.500 | 1.500 | 5.500 |
| B2 | 0.604 | 0.729 | 0.500 | 1.500 | 2.250 |
| X1 | 0.678 | 0.764 | 0.500 | 1.750 | 3.500 |
| M1 | 0.747 | 0.797 | 0.500 | 1.750 | 2.000 |
| M1-G | 0.748 | 0.798 | 0.500 | 1.750 | 2.000 |
| M2 | 0.724 | 0.784 | 0.500 | 1.750 | 2.000 |

## Back-test coefficients of M1 (provisional: per-fold fits, final fit and intervals come later)

Top 3 by absolute mean weight: **F3_big_stage** (+0.40, same sign in 4/4 folds), **F9_fame** (+0.34, same sign in 4/4 folds), **F4_major_titles** (+0.32, same sign in 4/4 folds)

| feature | mean_beta | sd_across_folds | same_sign_folds | abs_mean_beta | fold_2022 | fold_2023 | fold_2024 | fold_2025 |
|---|---|---|---|---|---|---|---|---|
| F3_big_stage | 0.395 | 0.028 | 4.000 | 0.395 | 0.396 | 0.432 | 0.364 | 0.388 |
| F9_fame | 0.341 | 0.067 | 4.000 | 0.341 | 0.434 | 0.293 | 0.346 | 0.290 |
| F4_major_titles | 0.319 | 0.035 | 4.000 | 0.319 | 0.292 | 0.329 | 0.291 | 0.364 |
| F5_other_honors | 0.312 | 0.062 | 4.000 | 0.312 | 0.297 | 0.273 | 0.403 | 0.273 |
| F6_team_strength | 0.264 | 0.103 | 4.000 | 0.264 | 0.227 | 0.358 | 0.135 | 0.336 |
| F7_prior_standing | 0.222 | 0.099 | 4.000 | 0.222 | 0.143 | 0.254 | 0.143 | 0.347 |
| is_defender | -0.140 | 0.062 | 4.000 | 0.140 | -0.125 | -0.110 | -0.095 | -0.232 |
| F10_season_attention | 0.130 | 0.063 | 4.000 | 0.130 | 0.161 | 0.051 | 0.111 | 0.198 |
| F2_form_rating | 0.113 | 0.079 | 3.000 | 0.113 | 0.151 | 0.167 | 0.137 | -0.004 |
| F1_attack | 0.109 | 0.045 | 4.000 | 0.109 | 0.065 | 0.097 | 0.105 | 0.171 |
| is_goalkeeper | 0.101 | 0.030 | 4.000 | 0.101 | 0.123 | 0.068 | 0.130 | 0.084 |
| F8_defending_winner | -0.082 | 0.055 | 3.000 | 0.082 | -0.115 | 0.000 | -0.102 | -0.109 |

## Selected penalty per fold (PL models)

| fold | M1 | M1-G | M2 |
|---|---|---|---|
| 2022 | 10.000 | 10.000 | 21.544 |
| 2023 | 10.000 | 10.000 | 10.000 |
| 2024 | 10.000 | 10.000 | 4.642 |
| 2025 | 10.000 | 10.000 | 10.000 |

## Diagnostics

| fold | diagnostic | value |
|---|---|---|
| 2022 | rf_raw_win_probability_sum | 3.803 |
| 2023 | rf_raw_win_probability_sum | 3.832 |
| 2024 | rf_raw_win_probability_sum | 2.669 |
| 2025 | rf_raw_win_probability_sum | 3.439 |

The B2 random forest produces win probabilities that, before renormalisation, sum to the values above (a proper distribution sums to 1).
