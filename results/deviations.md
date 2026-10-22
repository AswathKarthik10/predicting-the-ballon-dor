# Deviations and implementation clarifications

Pipeline commit and input hashes are printed in the notebook header (`ballon_dor_2026.ipynb`, section A).

## Part 1. Implementation clarifications made BEFORE any model was run (2026-10-21)

These resolve points the spec leaves open. They were fixed while writing `src/model/features.py`, after exploratory
data analysis (sections A-E of the notebook) but before any model was fitted or scored. They are not post-result changes.

| # | Clarification | Reason |
| --- | --- | --- |
| C1 | Within-edition standardisation uses the sample standard deviation (ddof = 1). | Spec says "divide by the edition standard deviation" without naming ddof. |
| C2 | If an edition has zero standard deviation for a column, its z-score is set to 0. | Occurs for 2025 tournament features and the shrunk 2025 `post_npxg_xa_p90_opp_adj` (29 of 30 null, so all rows take the one non-null value). Avoids NaN. |
| C3 | Null `tourn_ko_goal_contribs` and `final_goal_contribs` (did not play) are set to 0 before z-scoring; null `scored_winner_in_final` is False. | Null means the competition was not played. |
| C4 | `ucl_stage_reached == "playoff"` (3 rows: Haaland 2025, Gyokeres 2025, Martinez 2026) is mapped to ordinal 2, the same as R32. | The spec's ordinal list omits `playoff`. It is the UCL knockout play-off round. Confirmed by the user. |
| C5 | Every continuous composite (F1-F7, F9, F10, P1-P3) is re-standardised within edition after it is built. F8 and the position flags are not. | Spec section 3 general rule: "Standardize every continuous feature within edition". Confirmed by the user. |
| C6 | Attention components are z-scored within edition individually, then averaged, then (for F10 and P2) residualised, then re-z-scored. | The two components (Wikipedia and news) have different scales; z-scoring first keeps them equally weighted. |
| C7 | F1 goalkeepers: the edition mean and SD are computed from non-goalkeeper rows only; goalkeepers are then set to 0 (the edition mean). | Spec says to set goalkeepers to the edition mean without saying how that mean is defined. |
| C8 | `days_in` (length of the eligibility window) is not used. P2 compares the log of the post-window daily rate with the log of the in-window level; the in-window level differs from its daily rate by a constant within an edition, which the within-edition z-score removes. | Window start dates are not in the data or the spec. |
| C9 | P1 shrinks toward the edition mean of the non-null `post_npxg_xa_p90_opp_adj` values; rows with a null value take weight 0 (so they equal that mean). | Spec: "null value -> w = 0" with "edition mean". |
| C10 | `club_rating_window_end` null rows are imputed with the 5th percentile of that edition's own club ratings (only Messi and Quinones, 2026), flag `club_rating_imputed = 1`. `big_game_rating` null rows use the player's `club_avg_rating`. | Per spec section 2. |
| C11 | Raw (unstandardised) `major_titles`, `other_honors` and `prior_standing` are kept as columns for the naive baseline B0. | Spec: B0 uses raw values. |
| C12 | Residualisation (F10, P2) is OLS with an intercept, fitted on the training-fold rows only and applied to the held-out edition; the residual is then re-z-scored within its own edition. | Spec section 3.1 and the leakage rule. |

## Part 2. Clarifications for the modelling stage (recorded before the first CV run; applied in `src/model/cv.py` and `baselines.py`)

| # | Planned clarification | Reason |
| --- | --- | --- |
| M1 | For fitted comparison models (B1, B2, X1) the temperature T is fitted on out-of-sample scores obtained by an inner leave-one-edition-out loop over the training editions. | In-sample scores of a random forest or boosted trees are overconfident and would distort T. Still uses training editions only. |
| M2 | B1 predicted vote shares are clipped at 1e-4 before the log. | A ridge prediction can be zero or negative, and the spec converts with `log(predicted vote share)`. |

## Part 2b. Stage 3 clarifications (recorded 2026-10-21, before any Stage 3 code was run)

The selected model (M1, rule 5.4) is already fixed from the back-test. These clarifications concern the final fit, the juror simulation and the research tests; none of them looks at 2026 output.

| # | Clarification | Reason |
| --- | --- | --- |
| S1 | Jurors are built from dataset columns (no `fact_juror_nations` or `fact_juror_ballots` exist). Each nominee nationality is one home juror in its confederation; the rest of the confederation's jurors (from `jurors_same_confed`) have no nominee compatriot; the shortfall to 100 jurors is "unassigned" (2026: UEFA 41, CAF 20, CONMEBOL 10, CONCACAF 11 = 82, so 18 unassigned, consistent with AFC about 18-19 in 2022-23). Each league country has exactly one juror (`jurors_league_country` = 1). | Spec section 6.3 fallback. |
| S2 | League id to country: 47 England, 53 France, 54 Germany, 55 Italy, 57 Netherlands, 61 Portugal, 87 Spain, 130 USA, 536 Saudi Arabia. | `league_key` is a numeric id; the mapping is inferred from the clubs in each league. |
| S3 | Laplace and bootstrap intervals use the final lambda and the final residualisation fixed; they are not re-tuned or refitted per draw. In the within-edition bootstrap, a duplicated player with the same points is treated as a tied group. | Keeps the intervals about coefficient uncertainty at the selected specification. |
| S4 | `tau_juror` is calibrated on 2024 and 2025 using out-of-fold M1 utilities (each edition predicted by a model that did not see it), tau_shared = 1 and placeholder biases. Loss: sum of squared relative errors of winner vote share, number of players with points and the Herfindahl index. | Spec says "own fitted utilities"; out-of-fold avoids using in-sample fit. |
| S5 | Exploratory extra columns (`post_transfer`, `post_injury`, `teammates_on_shortlist`, `nat_teammates_on_shortlist`) are added as non-primary columns; they never enter M1, M2 or model selection. | Needed for the exploratory tests in spec 7.4. |
| S6 | Robustness run 2 (2024-25 only) has two folds, not three (the spec's "three folds" is impossible with two editions), and uses the lambda from the main final fit because inner tuning needs two or more training editions. Run 5 (deadline +/-7 days) stays pending: it needs a data rebuild. | Arithmetic and spec section 7.5. |
| S7 | `top_tier` (7.2) uses M1 out-of-fold utilities for 2022-25 and the final M1 utilities for 2026. | Avoids in-sample utilities defining the groups. |
| S8 | Bias parameters b_home = 0.8, b_confed = 0.2, b_league = 0.2 are placeholders, not estimates, and every simulation output says so. Any claim about juror bias needs the individual ballots. | Spec section 6.3. |

## Part 3. Changes made after seeing results

| # | Change | Effect on selection |
| --- | --- | --- |
| D1 | After the first cross-validation run (code commit `fbb3ff2`), the rank-based secondary metrics (top-1 hit, top-3 overlap, winner rank) of the uniform reference model U0 were set to NaN. With constant utilities these metrics only reflected tie-breaking by index (`winner_rank = 1`). | None. U0 is a reference, not a candidate; held-out log-likelihoods and all other models are unchanged. Labelled exploratory/reporting only. |
