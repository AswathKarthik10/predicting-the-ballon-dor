# Deviations and clarifications (experiments v2)

Only items decided after Stage 0 appear here. Anything decided after seeing results is labelled EXPLORATORY.

## Stage 0 (2026-11-02, before any model was run)

| # | Item | Reason |
| --- | --- | --- |
| D0.1 | Branch `experiments-v2` was created with `git switch -c`, so the uncommitted v1 edits (`forecast_ballots.py`, notebook, untracked files) travel with the working tree and are not committed. | Avoids committing or stashing work the user did not ask to commit. |
| D0.2 | `lightgbm` and `statsmodels` installed into the project venv and pinned in `requirements.txt`. | Needed for L1 and O1 (spec section 4). |

## Stage 1-2 (2026-11-02, before any v2 feature or model was built)

| # | Item | Reason |
| --- | --- | --- |
| D1.1 | `StudyData` now carries `labelled_editions` and `forecast_edition`; `cv`, `inference` and `forecast` read them instead of the module constant `TRAIN_EDITIONS`. `check_snapshots` takes a `StudySpec`. v1 behaviour is unchanged (all 83 v1 tests pass). | Spec section 14 step 1: generalise the loaders. |
| D1.2 | The ten-edition study (T1: 10 editions, T2: 5) has no loader yet. `load_study` refuses specs whose T1 and T2 editions differ. | v2 features (npga, awards, market value) are built in Stage 3; the loader arrives with them. |
| D1.3 | Bridge run result: the plan expected a small gap from three revised columns (`career_bdo_podiums_before`, `club_trophies_n`, `won_domestic_cup`). The gap is exactly zero. Only 5 cells changed in 2022-2026 and none touches a 2022-2025 feature that v1 uses. They do change Messi's 2026 row (`won_domestic_cup` 1 to 0, so his F5 other-honours score drops). | Measured, see `bridge_revised_columns.csv` and `bridge_vs_v1.csv`. |
| D1.4 | The "same four folds with the v2 12-feature set" part of plan step 2 moves to Stage 3, since it needs the new features. | Dependency order. |

## Stage 3 (2026-11-02, decisions fixed before the back-test was run)

| # | Item | Reason |
| --- | --- | --- |
| D3.1 | Toni Kroos 2024 has `tm_value_eur_t1 = 0` (retired, Transfermarkt zeroes retired players), which makes `log` undefined. F12 uses his `tm_value_eur_window_start` (15M) instead. Rule: any `tm_value_eur_t1 <= 0` falls back to the window-start value. | Only one row affected. Fallback is dated before T1. |
| D3.2 | F11 (individual awards) is z-scored within edition like every other continuous feature. | Spec section 3 general rule. |
| D3.3 | B0, B1, B2, X1, L1, O1 and D1 use the same 14 columns as M1 (12 features + 2 position flags). B0 uses the v2 `other_honors` (no `league_poty`). | Spec section 4: all models see the same features. |
| D3.4 | Bootstrap intervals are reported for held-out log-likelihood, Spearman, pairwise accuracy, top-3 overlap, top-10 accuracy and NDCG@10. Not for winner rank and top-1 hit (undefined when the resample drops the winner), and not for U0 (constant utility). Per-fold values are in `metrics_by_fold.csv`. | Statistical validity. |
| D3.5 | Selection rule 5.4(2) is applied as: publish the simplest PL model (M1, then M1-G, then M1-F) whose mean log-likelihood is within 0.02 of the best PL model. | The spec says "if two differ by less than 0.02, use the simpler"; with three models this needs a fixed reading. |
| D3.6 | Rank bands for O1 use competition ranks (ties share the best rank): winner; ranks 2-3; ranks 4-10; rank 11 or lower with votes; no votes. | Spec lists the bands without a tie rule. |
| D3.7 | D1 and the O1 score are converted to Plackett-Luce utilities with the same fitted-temperature step as the other non-PL models. D1 predicts log(max(share, 1e-4)). | Same conversion as v1 section 4.2. |
| D3.8 | NDCG@10 uses points as gain (proportional to vote share); top-10 accuracy is the overlap of predicted and actual top 10 divided by 10. | Spec 5.2. |
| D3.9 | `M1-season4` (v2 features trained on the three other season editions per fold) answers "what do the extra editions buy". It is not a candidate for selection. | Spec 6.1. |
| D3.10 | The ten-edition study has no T2 (`frame_t2` is None). M2 will use the 2022-2026 study with `post_npga` in Stage 4. | T2 only covers 2022-2026. |

## Stage 4 (2026-11-02, decisions fixed before the Stage 4 runs; D4.8-D4.10 are findings)

| # | Item | Reason |
| --- | --- | --- |
| D4.1 | Variance shares follow the v1 method: leave-one-edition-out ridge (alpha 10, no intercept) on the within-edition z-scored points percentile, Shapley split over the eight spec-3.2 blocks, 300 bootstrap draws resampling players within edition. The F10 residualiser is fitted once per fold on the original training rows; bootstrap draws resample rows of the resulting matrices. Total R2 is not comparable to v1's 0.63 (different feature set and blocks). | Spec 6.2 "v1 method". |
| D4.2 | Ablations read M1's per-fold values from Stage 3 `metrics_by_fold.csv` (same code path, same seed) instead of refitting M1. The 6-of-9 rule is applied as written; it counts folds, not size, so blocks with tiny effects can pass it (Big stage -0.009, Position -0.001). | Avoids a redundant 9-fold refit; the rule is the spec's. |
| D4.3 | Football-only model for the fame test: blocks season output, big stage, trophies and team, individual awards, prior standing and position (F1-F8, F11, position flags). Fame is residualised on exactly those columns by OLS fitted on the 8 training editions. The permutation test shuffles the high/low-fame label within edition (10,000 shuffles); the one-sided p tests the pre-registered direction (low-fame players finish worse); two-sided is also reported. | Spec 6.3. |
| D4.4 | Tournament check: "played the main tournament" means `tourn_name` is World Cup, EURO or Copa America in editions 2016, 2018, 2019, 2021, 2023, 2024 (133 players). "Tournament terms" is read as the `tourn_ko_goal_contribs x is_world_cup` interaction. Models compared: M1 without F11; M1; M1 plus the interaction. Residual = actual competition rank minus predicted rank (negative = finished better than predicted). | Spec 6.5 does not define the terms exactly. |
| D4.5 | Golden-ball term: `golden_ball_score` = 1 for the Golden Ball winner, 0.5 for second and third; estimated with F11 also in the model, on 2016-2025 (two World Cups: 2018, 2023) and on 2022-2025 (one: 2023). | Spec 6.5 step 4 allows "award_wc_golden_ball (or F11)". |
| D4.6 | Ballot model: `era2022` = 1 from the 2022 edition. 2016 uses the 3-pick scheme (5-3-1). Bootstrap draws: 500 for the pooled, era and interaction fits, 200 for per-edition and exploratory fits. Top tier = the official top 3 by points (ties can add a fourth). | Spec 7.1, 7.2. |
| D4.7 | Jury-local attention: a juror-country cell is covered when the attention share exists for at least 15 of 30 nominees. Inside a covered cell a nominee below the view threshold has no share and is set to 0; the share is then z-scored within the cell. Fits use covered juror-editions only. | Spec 7.3 coverage rule; missing-share handling is not specified. |
| D4.8 | FINDING: the spec says `post_npga_p90_opp_adj` exists for 2025. It is non-null for 1 of 30 players in 2025 (and 26-29 of 30 in the other editions), as in v1. With shrinkage to the edition mean P1 is constant in 2025, so the 2025 fold carries no information on P1. | Data check (`test_post_window_study_has_constant_p1_in_2025`). |
| D4.9 | FINDING: 2022-2025 ballot fit reproduces v1 exactly (home 2.3110, 90% interval 1.88-2.80). | Spec section 13. |
| D4.10 | FINDING: the fame effect replicates in direction but is smaller than v1's headline (v1: 4.9 vs 10.8, a 5.9-place gap; v2: 6.4 vs 9.6, a 3.2-place gap). The split variable also differs (v1 used an attention gap; v2 uses fame residualised on football). | Reported as measured. |

## Stage 5 (2026-11-02)

| # | Item | Reason |
| --- | --- | --- |
| D5.1 | Odds: a capture is one (edition, date); rows from several bookmakers on the same date are merged by taking each player's best (highest) implied probability. Non-nominee prices are dropped and the rest renormalised over the quoted nominees. Seven captures qualify under the spec rule (before T1 and within 60 days, or between T1 and the deadline): 2016-10-24, 2018-09-25, 2019-10-21, 2021-10-28, 2024-07-16, 2024-08-28, 2025-08-06. The 2019-06-07 and the 2022 and 2023 captures are more than 60 days before T1; after-deadline captures and 2026 are excluded. | Spec section 8. |
| D5.2 | Robustness runs re-run the headline model M1 only (not every comparison model). Deltas are against base M1 on the same folds. In R1-R4 the changed run trains on fewer editions than the base does, so the delta mixes training volume with the change under test. | Spec section 10 repeats "5", the back-test, for the headline model. |
| D5.3 | R5 (no ratings) and R8 (no fame and attention) also remove the dropped features from the F10 residualiser. R9 (format fixed effect) is a no-op in Plackett-Luce because a within-edition constant cancels in the likelihood; it was run as a numerical check and matches M1 to four decimals. | Keeps the runs clean; R9 as written cannot change a PL fit. |
| D5.4 | Each robustness run also reports 2026 utilities (features only, no result) as top-5 and overlap with the base model. The juror simulation is not repeated for robustness runs. These outputs were not used for any decision. | Spec section 10 "and the 2026 forecast". |
| D5.5 | FINDING: the 2026 top pick is specification-sensitive. Across the ten runs the same top player as the base model appears in 3, and the top five is the same set in 4 runs; Spearman correlation of the 2026 utilities with the base is 0.95-1.00. | `robustness.csv`. |

## Stage 6 and lock-in (2026-11-02)

| # | Item | Reason |
| --- | --- | --- |
| D6.1 | Published model: M1 with the v2 features, chosen by the pre-registered rule 5.4 from the 2016-2025 back-test (`selection.md`). Final fit on all nine labelled editions, ridge penalty 10 chosen by leave-one-edition-out. Intervals: Laplace and a 500-draw within-edition bootstrap (`coefficients_M1.csv`). | Spec 9.1. |
| D6.2 | Juror simulation: 100 juror nations for 2026 (`fact_juror_nations`, assumed as in the spec), 10-pick scheme, 10,000 runs, seed 7. Bias terms are the 2022-2025 era estimates (home 2.311, confed 0.366, league 0.478) with their Laplace covariance. | Spec 9.2. |
| D6.3 | Link re-estimated on nine editions: the ballot-model player strengths of each edition come from the fit of its own jury era (2016-2021 and 2022-2025 fitted separately), against out-of-fold M1 utilities. Result k = 1.347, shared SD = 1.631 (v1 four-edition: 1.588, 1.664). Using one pooled nine-edition ballot fit instead gives k = 1.346, SD = 1.633, and a very similar forecast (`forecast_2026_v2_sensitivity.csv`). | The spec does not say which ballot fit supplies the strengths. |
| D6.4 | Sensitivity forecasts: bias terms from all nine editions, and 5 of 100 jurors absent. The headline forecast is the 2022-2025-bias one. | Mirrors v1's sensitivity table. |
| D6.5 | The Messi 2026 row is revised in the expanded dataset (`won_domestic_cup` 1 to 0, `club_trophies_n` 2 to 1, `career_bdo_podiums_before` 11 to 14). The v2 forecast uses the revised values, so part of the change from v1 comes from the data, not the model. | Dataset revision, see D1.3. |
| D6.6 | LOCK-IN. Per spec 9.2 the 2026 result is recorded as publicly known when the tag `forecast-v2` was made. I have not seen it and no file containing it is in the repo (guard: `forecast_v2.refuse_if_result_present`, tested). Model selection used 2016-2025 only; `selection.md` was written before any 2026 forecast was computed. Robustness runs printed 2026 utilities before this forecast (D5.4); they changed no decision and the base specification was fixed beforehand. | Spec 9.2 and section 0. |
| D6.7 | FINDING: the v2 forecast differs a lot from the published v1 forecast (v1: Dembele 36%, Kane 23%, Yamal 20%; v2: Messi 31%, Mbappe 29%, Yamal 16%, Dembele 11%). The main drivers are the new individual-awards feature (largest coefficient) and the revised Messi row. The robustness runs show the order of the top four is specification-sensitive (D5.5). | Reported as measured. |
