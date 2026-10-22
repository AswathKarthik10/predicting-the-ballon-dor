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

## Part 4. Juror simulation from ballots (2026-10-22)

Decisions B1-B3 and the design choices were fixed in `juror_simulation_ballots_spec.md` and the implementation plan before any new estimate or 2026 output was produced. B4-B12 are implementation choices made while coding; none was changed after seeing the 2026 forecast.

| # | Decision | Reason |
| --- | --- | --- |
| B1 | The ballots and juror nation lists existed in the dataset repo (`fact_bdo_country_votes`, `fact_juror_nations`) but were not used in the first run; S1 and S8 described them as missing. This rerun implements spec 6.3 as written. The original forecast (`results/forecast_2026_T1.csv`, `_T2.csv`) is kept byte-identical and remains the timestamped pre-registered forecast. | Spec 6.3 already said to estimate the biases from ballots if they exist. This corrects an implementation gap, not the plan. |
| B2 | Noise calibration changes from matching vote concentration (S4, `calibrate_tau_juror`) to estimating it from individual ballots: per-juror noise is fixed at Gumbel(0, 1) by the ballot model, and the shared race-level shock is Normal(0, `sigma_shared`) with `sigma_shared` the residual SD of the alpha-on-u regression (section 4). Decided before seeing the new 2026 output. | Everything about juror agreement is learned from real ballots instead of tuned to a summary statistic. |
| B3 | MLS (`league_key` 130) is missing from `dim_league_context` and was mapped by hand to `c_usa`. Two rows of `dim_league_context` with a null `league_key` (`c_blr`, `c_cod`) are dropped; no snapshot uses them. | Needed to build the `league` indicator; all nine snapshot league keys then map. |
| B4 | The data was copied from `/Users/aswath/Documents/balon dor prediction/data/processed/` (the same tables are in `ballon_dor.duckdb`; row counts match the spec: 11,520 / 500 / 211 / 91 / 150 / 295 / 236). The dataset-repo release tag and commit (`data-v1`, `371021c`) could not be verified from here, so every ballot output records the SHA-256 of all seven parquet files instead. | The dataset repo named in the spec was not on disk. |
| B5 | The ballot model's identification uses the small ridge on alpha (lambda = 0.01, no penalty on beta), not a mean-zero constraint. | The ridge pins each edition's alpha mean and keeps the Hessian invertible, so Laplace standard errors need no constrained inverse. |
| B6 | Laplace intervals use the inverse penalised Hessian of the full alpha+beta problem restricted to beta. The bootstrap resamples whole ballots within edition (500 refits, seed 7, warm-started from the main fit), for the main fit and for every robustness fit. | Spec section 3; bootstrap reported next to Laplace. |
| B7 | The `low_evidence` flag follows the spec literally (fewer than 20 flagged player-juror pairs). No fit trips it, but `league` rests on only 5-7 ballots per edition (23 in the main fit) and `home` on 13-18 per edition, so those by-edition estimates are noisy; read the intervals, not the flag. | Honest reading of the evidence counts. |
| B8 | The link regression (section 4) is OLS of alpha on out-of-fold utility with per-edition intercepts and one common slope, via within-edition centring. `sigma_shared` is the residual SD with n - (editions + 1) degrees of freedom, and the reported R-squared is within-edition. The residual includes the estimation error in alpha, so `sigma_shared` slightly overstates the true race-level shock. | Direct implementation of the spec; the overstatement is a known conservative bias. |
| B9 | `forecast.oof_utilities` is left unchanged (M1 only); `forecast.oof_utilities_for(study, model, edition)` is added for M2. | The spec needs M2 out-of-fold utilities for the T2 link. |
| B10 | Validation (section 6) uses the fixed out-of-fold point strengths for both simulations (as `calibrate_tau_juror` did), draws beta from the Laplace distribution of the leave-one-edition-out fit, and uses 10,000 simulations. The leave-one-out link uses out-of-fold strengths of the other editions, which come from models that saw the held-out edition's results; the held-out edition's own ballots and strengths never enter its parameters. Past editions use only the nations that voted (93/92/99/100), not all 100 listed. | A fully nested design would need a different, weaker strength model per fold; the spec asks for the existing out-of-fold utilities. |
| B11 | The placeholder column in the validation is the original simulation run under each edition's true points scheme (5-pick for 2022-23), not its original 10-pick-only behaviour; `tau_juror` is the 2024-25 calibrated value (0.75), so it is in-sample for those two editions. `simulate.simulate` keeps its 10-pick behaviour and produces bit-identical output (checked). | A 10-pick placeholder cannot be compared with 5-pick actual concentration. The in-sample advantage favours the placeholder. |
| B12 | `real_jury(root, edition)` takes the repo root because the data is read from `data/jury/`. For 2026, `voted` is unknown, so all 100 nations vote. `build_jurors` is unchanged. The 95-juror sensitivity drops 5 random jurors per simulation. | Spec section 5. |
