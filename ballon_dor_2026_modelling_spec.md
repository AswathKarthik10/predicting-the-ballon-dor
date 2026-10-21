# Ballon d'Or 2026 Modelling Spec

Oct 20, 2026 · companion to `ballon_dor_2026_dataset_spec.md`

## 0. Read this first

You are building the modelling stage of a study that forecasts the 2026 men's Ballon d'Or ranking (all 30 places) and tests whether information from **after** the eligibility window influenced the vote.

- **Inputs:** `player_snapshot_t1.csv` and `player_snapshot_t2.csv` (pipeline commit `cd6d898`). The data is final and has passed QA. Do not modify it; derive everything in code.
- **Hard deadline:** the 2026 forecast must be committed to git and posted publicly **before 2026-10-25 23:59 UTC**. The ceremony is 2026-10-26.
- **Pre-registration:** sections 4, 5 and 7 are fixed before any model is run. Anything decided after seeing results must be logged in `results/deviations.md` and labelled exploratory.
- **The single most important rule:** the published 2026 forecast comes from the model with the best **held-out back-test** score (section 5). Never choose, tune or adjust a model by looking at its 2026 output.

## 1. Research design in one paragraph

Performance stops counting on the last day of the eligibility window, but jurors vote weeks later. Each player-season therefore has two rows. **T1** holds everything known at window close. **T2** holds the identical in-window columns plus a post-window block: new-season form, attention and events up to 3 days before the vote deadline. Model **M1** is fit on T1 rows; model **M2** on T2 rows. If M2 predicts held-out editions better than M1, and its post-window weights are consistently positive, voters used information they should not have.

## 2. The data

### Shape

| File | Rows | Columns | Grain |
| --- | --- | --- | --- |
| `player_snapshot_t1.csv` | 150 | 91 | edition × player, snapshot T1 |
| `player_snapshot_t2.csv` | 150 | 111 | edition × player, snapshot T2 (T1 columns + 20 post-window columns) |

Five editions × 30 nominees. Key: (`edition`, `player_key`). In-window columns are verified identical between T1 and T2.

- **Training / back-test editions:** 2022, 2023, 2024, 2025 (120 player-seasons with known results).
- **Forecast edition:** 2026 (30 player-seasons; `bdo_points`, `bdo_rank`, `vote_share` are null and must stay unused).

### Edition facts the code must encode

| Edition | Window end (T1) | T2 | Post-window days | Ballot | Points per ballot | Jurors | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | 2022-07-31 | 2022-09-16 | 47 | Top 5: 6-4-3-2-1 | 16 | 93 | 11 zero-point nominees |
| 2023 | 2023-07-31 | 2023-09-29 | 60 | Top 5: 6-4-3-2-1 | 16 | 92 | 5 zero-point nominees |
| 2024 | 2024-07-31 | 2024-09-27 | 58 | Top 10: 15-12-10-8-7-5-4-3-2-1 | 67 | 99 | 2 zero-point nominees |
| 2025 | 2025-07-13 | 2025-08-22 | 40 | Top 10: 15-12-10-8-7-5-4-3-2-1 | 67 | 100 | Post-window period nearly empty (median 1 match) |
| 2026 | 2026-07-19 | 2026-09-25 | 68 | Top 10: 15-12-10-8-7-5-4-3-2-1 | 67 | 100 (assume) | Forecast target |

The point totals in the data confirm these schemes exactly (1,488 = 93 × 16; 1,472 = 92 × 16; 6,633 = 99 × 67; 6,700 = 100 × 67). Use **15-12-10-8-7-5-4-3-2-1** for 2026, not 15-12-10-8-7-6-5-4-2-1.

### Known data quirks and how to handle them

| Quirk | Rows affected | Handling |
| --- | --- | --- |
| Zero-point nominees | 18 across 2022–2024 | Treat as tied at the bottom (section 4.3). Never impose an order among them |
| Tied point totals among voted players | A few in 2022–2024 | Treat as tied (section 4.3) |
| `intl_npxg_xa_p90*` null | 18 rows, all with < 270 international minutes | Use club output only for these players (section 3.1) |
| `intl_xg_missing = true` | Players whose international xG was imputed | Keep the imputed value; carry the flag into robustness checks |
| `club_rating_window_end` null | Messi and Quiñones 2026 (MLS, Saudi Pro League) | Impute the 5th percentile of all club ratings in that edition; set `club_rating_imputed = 1` |
| `big_game_rating` null | 3 rows | Impute with the player's `club_avg_rating` |
| `ucl_stage_reached`, `nt_tourn_stage_reached` null | Did not play in the competition | Ordinal 0 |
| Goalkeepers | 0–3 per edition (none in 2026) | Exclude from the attacking composite (set to edition mean); add a position indicator |
| Negative `club_npxg_xa_p90_opp_adj` | 10 rows, all GK or defenders | Expected; comes from rescaling near-zero values. Leave as is |
| 2025 post-window block | 29 of 30 have null `post_npxg_xa_p90` | Shrinkage rule in section 3.2 handles it; 2025 contributes little to the post-window form test, which is expected |
| No YouTube columns | All rows | Not available. Do not add placeholders |

## 3. Feature engineering

**General rules**

- Standardize every continuous feature **within edition** (subtract the edition mean, divide by the edition standard deviation). Voting is relative.
- Log-transform heavy-tailed counts (Wikipedia views, news volume) before standardizing: `log1p(x)`.
- Any statistic learned from data (residualization coefficients, imputation values, standardization) must be computed **inside each training fold only** and then applied to the held-out edition. The exception is within-edition standardization, which uses only that edition's own rows and is therefore leak-free.
- Keep the primary feature set to **10 features**. More will overfit 120 rows.

### 3.1 Primary feature set (used by M1 and every comparison model)

| # | Feature | Built from | Notes |
| --- | --- | --- | --- |
| F1 | `attack` | Minutes-weighted mean of `club_npxg_xa_p90_opp_adj` (weight `club_minutes`) and `intl_npxg_xa_p90_opp_adj` (weight `intl_minutes`) | Club only when the international value is null. Goalkeepers set to edition mean |
| F2 | `form_rating` | Mean of z(`club_avg_rating`) and z(`big_game_rating`) | Dropped in the rating-free robustness run |
| F3 | `big_stage` | Sum of z(`ucl_ko_goal_contribs`), z(`tourn_ko_goal_contribs`), z(`final_goal_contribs`), plus 1 for `scored_winner_in_final`, plus 1 if `tourn_individual_award` is non-null | Re-standardize the sum |
| F4 | `major_titles` | `won_ucl` + `won_world_cup` + `won_continental` | Count 0–3; standardize |
| F5 | `other_honors` | `won_league` + `won_domestic_cup` + `won_other_uefa` + `league_poty` + `ucl_back_to_back` | Standardize |
| F6 | `team_strength` | Mean of z(`club_rating_window_end`), z(ordinal `ucl_stage_reached`), z(ordinal `nt_tourn_stage_reached`) | Ordinals: none 0, group/league_phase 1, R32 2, R16 3, QF 4, SF 5, final 6, winner 7 |
| F7 | `prior_standing` | `31 − prev_bdo_rank` (0 if null) | Standardize |
| F8 | `defending_winner` | `is_defending_winner` | Binary, not standardized. Separates voter fatigue from momentum |
| F9 | `fame` | z(`log1p(wiki_views_pre_juror_w)`) | Standing fame before the season, a celebrity control |
| F10 | `season_attention` | Mean of [log1p(`wiki_views_in_juror_w`) − log1p(`wiki_views_pre_juror_w`)] and [log1p(`news_vol_in_juror_w`) − log1p(`news_vol_pre_juror_w`)], then **residualized** on F1–F7 | Residualization: OLS on the training fold's rows, applied to the held-out edition. Keeps attention from double-counting performance |

**Position indicators** (not counted in the 10, always included): `is_defender` (position_group = DEF), `is_goalkeeper`.

`consecutive_podiums` is reported descriptively only (it is 1 for both defending winners in the data, so it adds nothing beyond F8).

### 3.2 Post-window block (M2 only; T2 rows)

Compute `days_post` = `snapshot_date` − window end, and `days_in` = the eligibility window length in days.

| # | Feature | Built from | Notes |
| --- | --- | --- | --- |
| P1 | `post_form` | `post_npxg_xa_p90_opp_adj` shrunk toward the edition mean: `w × value + (1 − w) × mean`, with `w = post_minutes / (post_minutes + 450)`; null value → `w = 0` | Shrinkage stops a single 90-minute match dominating |
| P2 | `post_attention` | Mean of [log(`wiki_views_post_juror_w` / days_post) − log(`wiki_views_in_juror_w` / days_in)] and the same for `news_vol_*_juror_w`, residualized on P1 and F1–F7 | Change in daily attention rate, not level |
| P3 | `final_spike` | log(`wiki_views_final14_juror_w` / 14) − log(`wiki_views_post_juror_w` / days_post) | Attention in the run-up to the vote relative to the rest of the post-window period |

Event flags `post_transfer` and `post_injury` are used only in the exploratory analysis (section 7.4).

### 3.3 Interactions for the research tests (section 7)

- `first_time` = 1 if `career_bdo_podiums_before == 0`; interaction `season_attention × first_time` (pre-registered hypothesis: attention overrates first-time contenders).
- `days_scaled` = days_post / 68; interactions `P1 × days_scaled`, `P2 × days_scaled` (dose-response).
- `trophy_gate` = 1 if `major_titles == 0` (model 7).

## 4. Models to run

All models use the same features (section 3.1, plus 3.2 for M2), the same folds and the same metrics.

| ID | Model | Represents | Specification |
| --- | --- | --- | --- |
| B0 | Naive baseline | "Do features add anything?" | Score = 2 × `major_titles` + `other_honors` + 0.5 × `prior_standing` (raw, not standardized). No fitting except the temperature (section 5.3) |
| B1 | Ridge regression on vote share | WashU (Olmeta & Kras 2026), prior published work | Target `vote_share`; `sklearn.linear_model.Ridge`; prediction converted by within-edition softmax with temperature |
| B2 | Random forest binary classifier | AVISIA, the most common industry approach | Target 1 = winner; `RandomForestClassifier(n_estimators=500, max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=7)`; probabilities renormalized to sum to 1 within edition |
| **M1** | **Plackett-Luce, ridge penalty, T1** | **Main model** | Section 4.1 |
| **M2** | **Plackett-Luce, ridge penalty, T2** | **Research test** | M1 + P1–P3 |
| X1 | XGBoost ranking (LambdaMART) | Check for missed non-linearities | `XGBRanker(objective="rank:pairwise", max_depth=2, n_estimators=100, learning_rate=0.05, subsample=0.8, reg_lambda=5, random_state=7)`; editions as query groups; relevance label = `vote_share` rank, zero-point players all 0 |
| M1-G | Plackett-Luce with trophy gate | "No major title, no win" | M1 + a `trophy_gate` term with its own weight (expected negative) |

Optional, if time allows: **C1** conditional logit on the winner only (the first stage of PL alone), and **M1-B** Bayesian Plackett-Luce in PyMC with Normal(0, 1) priors, as a check on the Laplace intervals in section 6.2.

### 4.1 Plackett-Luce: exact likelihood

For one edition, let each player have utility `u_i = β · x_i`. Order voted players by points. Let `R_k` be the set of players not yet placed before position `k` (all remaining voted players **plus all zero-point players**).

```
log L_edition = Σ_k [ u_(k) − log Σ_{j ∈ R_k} exp(u_j) ]   (sum over voted players only)
```

The sum stops after the last voted player, so zero-point players contribute only as "not chosen yet" and are never ordered among themselves.

**Ties among voted players** (equal points): use the Breslow approximation. For a tied group G occupying |G| consecutive places, add `Σ_{i ∈ G} u_i − |G| × log Σ_{j ∈ R} exp(u_j)`, where R is the remaining set before the group, then remove the whole group.

**Objective:** maximize `Σ_editions log L_edition − λ × ||β||²` (no penalty on the intercept; PL has none). Each edition counts equally. Optimize with `scipy.optimize.minimize(method="L-BFGS-B")` using the analytic gradient.

### 4.2 Converting every model's output to a common score

To compare log-likelihoods, every model's output becomes a PL utility, `u_i = s_i / T`:

| Model | Raw score `s_i` |
| --- | --- |
| B0 | Baseline score |
| B1 | log(predicted vote share) |
| B2 | log(renormalized win probability + 1e-6) |
| X1 | Ranker output margin |
| M1, M2, M1-G | `β · x_i` (T fixed at 1) |

`T` is fit by maximum PL likelihood on the **training editions only** of each fold, then applied to the held-out edition.

## 5. Validation and model selection

### 5.1 Folds

Leave one edition out over 2022–2025: four folds. Each fold trains on three editions and predicts the fourth.

### 5.2 Tuning (nested, inside each training fold)

Tune each model's penalty or hyperparameter by inner leave-one-edition-out on the three training editions, maximizing mean inner held-out PL log-likelihood.

- PL and ridge: λ (alpha) on a log grid of 13 values from 0.01 to 100.
- XGBoost and random forest: use the fixed settings in section 4. Do not grid-search them; with this little data a search will overfit the inner folds.

### 5.3 Metrics (computed on each held-out edition)

| Metric | Definition | Role |
| --- | --- | --- |
| **Held-out log-likelihood** | PL log L of the actual ranking (section 4.1), divided by the number of voted players | **Primary. Selects the model** |
| Spearman ρ | Rank correlation between predicted utility and actual points, all 30 players | Secondary |
| Pairwise accuracy | Share of player pairs with different points that the model orders correctly | Secondary |
| Top-1 hit | Predicted winner = actual winner | Secondary |
| Top-3 overlap | Size of overlap between predicted and actual top 3, out of 3 | Secondary |
| Winner's predicted rank | Where the actual winner sits in the model's ordering | Secondary |

Report every metric per fold and averaged. Because the 5-pick and 10-pick editions differ in difficulty, always show per-fold numbers, not just the average.

### 5.4 Selection rule (fixed now)

1. Rank models by mean held-out log-likelihood over the four folds.
2. Between **M1** and **M2**, the published forecast uses whichever has the higher mean held-out log-likelihood. If they differ by less than 0.02 per voted player, publish **M1** (the simpler model) and report M2 alongside it.
3. If a comparison model (B1, B2, X1) beats both M1 and M2 on held-out log-likelihood, still publish the better of M1/M2 as the main forecast, report the comparison model's forecast alongside it, and log the result in `results/deviations.md`. The juror simulation needs PL utilities.
4. If X1 beats M1, inspect which features drive its splits and test the corresponding interaction in PL as exploratory analysis.

## 6. Final fit and 2026 forecast

### 6.1 Final fit

Refit the selected PL model(s) on all four back-test editions, with λ chosen by leave-one-edition-out over those four. Produce 2026 utilities from the T1 rows (M1) and the T2 rows (M2). Produce both forecasts regardless of which is published.

### 6.2 Parameter uncertainty

Use the Laplace approximation: draw coefficient vectors from Normal(β̂, H⁻¹), where H is the Hessian of the penalized negative log-likelihood at β̂. As a check, run a within-edition bootstrap (resample editions' player sets with replacement inside each edition, refit, 500 times) and report both sets of coefficient intervals.

### 6.3 Juror ballot simulation

Run 10,000 simulations per snapshot.

**Jurors.** If `fact_juror_nations` for 2026 exists, use it. Otherwise build 100 jurors from the dataset:
- Map each nominee's `nationality` to a confederation (UEFA, CONMEBOL, CAF, AFC, CONCACAF, OFC).
- `jurors_same_confed` gives the number of juror nations in each player's confederation. Read one count per confederation and check that the counts sum to about 100. Fill any shortfall with OFC or unassigned jurors.
- Every nominee nationality with `jurors_same_nation == 1` is one juror in its confederation; the remaining jurors in that confederation have no nominee compatriot.
- League-country jurors: a juror whose nation equals the country of a player's league.

**Each simulation:**
1. Draw β from the posterior (section 6.2) and compute strengths `s_i`.
2. Draw one shared shock per player, `τ_shared × Gumbel(0, 1)`. This is the race-level uncertainty and produces the "consensus top tier".
3. For each juror j and player i: `v_ij = s_i + shock_i + b_home·[same nation] + b_confed·[same confederation] + b_league·[juror nation = league country] + τ_juror × Gumbel(0, 1)`.
4. Each juror ranks their top 10 by `v_ij` and awards 15-12-10-8-7-5-4-3-2-1.
5. Sum points across jurors; rank players (ties broken by first-place votes, then second-place votes).

**Bias parameters.**
- If `fact_juror_ballots` exists: estimate `b_home`, `b_confed`, `b_league` with a rank-ordered logit on individual ballots (2023–2025), with one fixed effect per player-edition plus the three indicators. Use those estimates.
- If not: use placeholders `b_home = 0.8`, `b_confed = 0.2`, `b_league = 0.2` (in utility units), label them as placeholders in every output, and run the sensitivity grid `b_home ∈ {0, 0.8, 1.6}` × `b_confed ∈ {0, 0.2, 0.4}`.

**Noise calibration.**
- `τ_shared = 1` by default (consistent with the PL model).
- Calibrate `τ_juror` by simulating the 2024 and 2025 editions with their own fitted utilities and choosing the value that best matches the observed concentration: the winner's vote share, the number of players receiving any points, and the Herfindahl index of vote shares. Search `τ_juror ∈ {0.5, 0.75, 1, 1.5, 2, 3}`.
- Report results at `τ_shared ∈ {0.5, 1, 1.5}` as sensitivity.

**Outputs per player per snapshot:** P(win), P(top 3), P(top 10), expected rank, 80% rank interval, expected points, expected vote share.

## 7. Research tests (reported, not used for selection)

### 7.1 Main test

- Held-out log-likelihood of M2 minus M1, per fold and averaged. Positive in most folds = post-window information improves prediction.
- Coefficients of P1–P3 in the final M2 fit with 90% intervals; report the sign in each fold's fit.
- The 2026 forecast shift: the change in P(win) and expected rank between T1 and T2 for each player.

### 7.2 Where on the ballot it matters

Refit M2 with the post-window block interacted with `top_tier` = 1 if the player is in the top 3 of M1's predicted utility in that edition. Report the post-window effect separately for the predicted top 3 and for the rest.

### 7.3 Dose-response

Refit M2 with the `P1 × days_scaled` and `P2 × days_scaled` interactions. A positive interaction means the effect grows with exposure. With four editions this is descriptive only; say so.

### 7.4 Pre-registered and exploratory hypotheses

- **Pre-registered:** `season_attention × first_time` coefficient (expected positive).
- **Exploratory, label as such:** effects of `post_transfer`, `post_injury`, `teammates_on_shortlist` (vote splitting), `nat_teammates_on_shortlist`.

### 7.5 Robustness runs (repeat section 5 and the 2026 forecast)

1. Without F2 (`form_rating`): xG-based features only.
2. Excluding the 2022 and 2023 editions (5-pick ballots): three folds, 2024–2025 train/test only, treat as descriptive.
3. Without F9 and F10 (no attention at all).
4. With players flagged `intl_xg_missing` or `club_rating_imputed` removed from training.
5. Deadline ±7 days: **requires a data rebuild**, not done in this stage. Record as pending.

## 8. Outputs

```
results/
  metrics_by_fold.csv          # model × fold × metric
  metrics_summary.csv          # model × metric, mean and per-fold
  selection.md                 # which model was selected and why, per rule 5.4
  coefficients_M1.csv          # β, SE, 90% interval (Laplace and bootstrap)
  coefficients_M2.csv
  forecast_2026_T1.csv         # player × output columns from 6.3
  forecast_2026_T2.csv
  forecast_2026_comparison.csv # every model's 2026 predicted rank, for transparency
  research_tests.md            # 7.1–7.4 with tables
  robustness.csv
  sensitivity_juror.csv        # bias and noise grids
  deviations.md                # anything changed after seeing results
figures/
  heldout_ll_by_model.png      # per-fold dots + mean per model
  forecast_2026_win_prob.png   # T1 vs T2 win probability, top 10
  rank_intervals_2026.png      # expected rank with 80% interval, all 30
  coefficients.png             # M1 and M2 coefficients with intervals
```

## 9. Code structure

```
src/model/
  features.py      # section 3; pure functions; fold-aware fit/transform
  pl.py            # Plackett-Luce likelihood, gradient, ties, ridge, Laplace
  baselines.py     # B0, B1, B2, X1
  cv.py            # folds, nested tuning, temperature fitting, metrics
  simulate.py      # section 6.3
  report.py        # tables and figures
tests/
  test_pl.py       # gradient check; recovers known β on synthetic data;
                   # zero-point players never ordered; ties handled
  test_features.py # no target columns used; no fold leakage; 2026 targets untouched
```

**Tests that must pass before any real run:**
- PL gradient matches finite differences to 1e-5.
- On synthetic rankings drawn from a known β, PL recovers β within sampling error.
- No feature function reads `bdo_points`, `bdo_rank` or `vote_share`.
- Fitted statistics (residualization, imputation) use training-fold rows only.

## 10. Reproducibility and guardrails

- Fix random seeds (7 everywhere). Log library versions to `results/environment.txt`.
- Record the SHA-256 of both input CSVs and the git commit in every output file's header.
- Never read 2026 target columns. Assert they are entirely null at load time.
- Never rerun selection after looking at the 2026 forecast. If something must change, log it in `deviations.md`, label it exploratory, and keep the original forecast published.
- Commit and push before posting. The commit timestamp is the pre-registration evidence.

## 11. Timeline

| Date | Work |
| --- | --- |
| Oct 20–21 | `features.py`, `pl.py` with tests passing; baselines |
| Oct 22 | Cross-validation for all models; selection per rule 5.4 |
| Oct 23 | Final fit, Laplace and bootstrap intervals, juror simulation and calibration |
| Oct 24 | Research tests, robustness runs, figures |
| Oct 25 | Freeze: commit, tag `forecast-2026`, post publicly |
| Oct 26 | Ceremony; fill 2026 targets; score the forecast |

## 12. Interpretation notes for the write-up

- Four editions is little evidence. Report the post-window test as "consistent sign across editions" plus effect size with intervals, not as a significance claim.
- 2025 contributes almost nothing to the post-window **form** test (a 40-day gap with almost no matches). It still contributes to the attention test.
- If B2 (the AVISIA-style random forest) produces win probabilities summing well above 100% before renormalization, report that number. It illustrates why a ranking model is used.
- Juror bias parameters are placeholders unless estimated from `fact_juror_ballots`. Any claim about bias needs those ballots.
