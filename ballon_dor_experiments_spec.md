# Ballon d'Or Experiments Spec: Ten-Edition Dataset

Nov 2, 2026 · companion to `ballon_dor_2026_modelling_spec.md` (v1) and `ballon_dor_expansion_spec.md` (dataset v2)

## 0. Read this first

You are running the experiments for the paper **"What predicts a Ballon d'Or?"** on the expanded dataset: nine labelled editions (2016-2019, 2021-2025, 270 player-editions) plus the 2026 forecast edition. The paper is a **5-page short paper** (ACM Web Science 2027, deadline 15 Dec 2026), so this spec separates the experiments that go in the paper (P) from those that go in the supplement (S).

- **Repo:** `predicting-the-ballon-dor`. Work on a new branch `experiments-v2` from `juror-simulation-ballots`. Reuse `src/model/` (PL likelihood, CV, baselines, juror panel, simulation); do not rewrite what exists.
- **Inputs:** the dataset repo `ballon-dor-dataset`, branch `expansion-2016`, commit `2a4437e` (proposed tag `data-v2`). Copy the exports (`player_snapshot_t1.csv`, `player_snapshot_t2.csv`) and the tables listed in section 2 into `data/v2/`, record the commit hash in `results/v2/environment.txt`, and never modify them.
- **2026 is held out.** Its targets are null in the dataset by design. No experiment may read the 2026 result until step 9.3, and every model choice is made on 2016-2025 only.
- **Pre-registration:** sections 3 to 8 are fixed before any model is run. Anything decided after seeing results goes in `results/v2/deviations.md`, labelled exploratory.
- **Selection rule:** the 2026 forecast comes from the model with the best held-out back-test score (section 5.4). Never choose, tune or adjust a model by looking at 2026 output.
- **Stop for review** after sections 4-5 (back-test table), after 6-8 (research tests) and before 9.3 (opening the 2026 result).

## 1. What changed since v1

| | v1 (Oct 2026) | v2 (this spec) |
|---|---|---|
| Labelled editions | 4 (2022-2025) | 9 (2016-2019, 2021-2025) |
| Rows for fitting | 120 | 270 |
| Formats | Season only | Season (2022+) and calendar year (2016-2021, T1 only) |
| Ballots | 5 and 10 picks | 3 (2016), 5 (2017-2023), 10 (2024+) |
| Per-country ballots | 384 | 1,259 (875 new) |
| Attacking output | npxG + xA (2021+ only) | Non-penalty goals + assists, every edition |
| New features | | Tournament awards, league and UEFA player awards, market value, EA rating |
| New tables | | Juror-by-nominee attention, betting odds |
| World Cup editions labelled | 1 (2023) | 2 (2018, 2023) plus 2 Euros and 2 Copas |

## 2. Data

| File | Rows | Use |
|---|---|---|
| `player_snapshot_t1` | 300 (10 editions × 30) | All models; M1 |
| `player_snapshot_t2` | 150 (2022-2026) | M2 only (post-window test) |
| `fact_bdo_country_votes` | 37,770 | Ballot-level jury model (section 7) |
| `fact_juror_nations` | per edition | Juror panel; 2026 simulation |
| `fact_juror_player_attention` | 41,250 | Jury-local attention test (section 7.3) |
| `fact_bdo_odds` | 104 prices, 18 captures | Odds baseline only (section 8). **Never a feature** |
| `dim_player`, `dim_country` | | Nationality, confederation |

### Edition facts the code must encode (read from the data, assert them)

| Edition | Format | T1 | Ballot | Points/ballot | Jurors | Zero-point nominees |
|---|---|---|---|---|---|---|
| 2016 | calendar | 2016-11-11 | 3: 5-3-1 | 9 | 173 | 11 |
| 2017 | calendar | 2017-11-06 | 5: 6-4-3-2-1 | 16 | 176 | 2 |
| 2018 | calendar | 2018-11-02 | 5 | 16 | 180 | 2 |
| 2019 | calendar | 2019-11-01 | 5 | 16 | 176 | 3 |
| 2021 | calendar | 2021-10-29 | 5 | 16 | 170 | 2 |
| 2022 | season | 2022-07-31 | 5 | 16 | 93 | 11 |
| 2023 | season | 2023-07-31 | 5 | 16 | 92 | 5 |
| 2024 | season | 2024-07-31 | 10: 15-12-10-8-7-5-4-3-2-1 | 67 | 99 | 2 |
| 2025 | season | 2025-07-13 | 10 | 67 | 100 | 0 |
| 2026 | season | 2026-07-19 | 10 | 67 | 100 (assumed) | forecast |

Check: `points.sum() == jurors × points_per_ballot` per labelled edition; fail otherwise.

### Known data properties and how to handle them

| Property | Handling |
|---|---|
| No xG before 2021 (`club_npxg*`, `intl_npxg*` null 2016-2019) | Main features use `*_npga_p90_opp_adj` (every edition). xG only in robustness R4 (2021+) |
| `intl_npga_p90*` null below 270 international minutes (8-20% of rows) | Club value only for those players, as in v1 |
| 2021 and 2022 windows overlap (Aug-Oct 2021) | Expected (format change). Players nominated in both editions share some matches; LOEO handles it because each edition is held out whole |
| 2016 is a 3-pick ballot with 11 zero-point nominees | PL handles it (zero-point players are never ordered). Robustness R2 drops 2016 |
| `tourn_individual_award` null for 86 API-sourced rows | Do not use. Use the structured `award_*` columns |
| `league_poty` (Transfermarkt's own award) vs `league_player_of_season` (league-run) | Main: `league_player_of_season`. `league_poty` only in robustness |
| EA FC 25 is a June 2025 scrape (in-season ratings); every other game is a launch snapshot | EA enters only as a robustness feature (R6) |
| Market value can be months old at T1 (`tm_value_date`) | Use as is; it is dated before T1 |
| `club_rating_window_end` null (2 rows of 2026, MLS/Saudi) | 5th percentile of the edition, `club_rating_imputed = 1` (v1 rule) |
| Zero-variance columns (`jurors_same_nation`, `jurors_league_country`, `prev_bdo_rank_partial`) | Drop |
| `wiki_views_pre` for 2016 covers 184 of 365 days (scaled) | Use as is; R2 drops 2016 |
| Goalkeepers | Attacking feature set to the edition mean; `is_goalkeeper` indicator |

## 3. Features

General rules carry over from v1: standardise within edition; `log1p` heavy-tailed counts; anything learned from data (residualisation, imputation) is fitted inside the training fold only.

### 3.1 Main feature set (M1 and every comparison model)

Twelve features. With 270 rows this is still small enough; the v1 rule was 10 for 120 rows.

| # | Feature | Built from | Change from v1 |
|---|---|---|---|
| F1 | `attack` | Minutes-weighted mean of `club_npga_p90_opp_adj` and `intl_npga_p90_opp_adj` | npga replaces npxG + xA |
| F2 | `form_rating` | Mean of z(`club_avg_rating`), z(`big_game_rating`) | Same |
| F3 | `big_stage` | z(`ucl_ko_goal_contribs`) + z(`tourn_ko_goal_contribs`) + z(`final_goal_contribs`) + `scored_winner_in_final`, re-standardised | Award term moved to F11 |
| F4 | `major_titles` | `won_ucl` + `won_world_cup` + `won_continental` | Same |
| F5 | `other_honors` | `won_league` + `won_domestic_cup` + `won_other_uefa` + `ucl_back_to_back` | `league_poty` removed (moved to F11) |
| F6 | `team_strength` | Mean of z(`club_rating_window_end`), z(ordinal `ucl_stage_reached`), z(ordinal `nt_tourn_stage_reached`) | Same |
| F7 | `prior_standing` | `31 − prev_bdo_rank` (0 if null) | History now from 2005 |
| F8 | `defending_winner` | `is_defending_winner` | Same |
| F9 | `fame` | z(log1p(`wiki_views_pre_juror_w`)) | Same |
| F10 | `season_attention` | As v1 (wiki and news in-window over pre-window, juror-weighted), residualised on F1-F7 and F11 in the training fold | Residualised on awards too |
| F11 | `individual_awards` | `major_individual_awards_n` + `league_player_of_season` + `uefa_player_award` + 0.5 × [`award_wc_golden_ball` in {2, 3}] (gold already counted) | **New** |
| F12 | `market_value` | z(log(`tm_value_eur_t1`)) | **New**; a "standing quality" control |

Position indicators (always in, not counted): `is_defender`, `is_goalkeeper`.

### 3.2 Feature blocks (for shares and ablations)

| Block | Features |
|---|---|
| Season output | F1, F2 |
| Big stage | F3 |
| Trophies and team | F4, F5, F6 |
| Individual awards | F11 |
| Prior standing | F7, F8 |
| Fame and attention | F9, F10 |
| Market value | F12 |
| Position | indicators |

### 3.3 Post-window block (M2, T2 rows, 2022-2025 only)

As v1 section 3.2, with `post_npga_p90_opp_adj` replacing `post_npxg_xa_p90_opp_adj` in P1 (so it exists for 2025 too).

## 4. Models

Same common-score conversion and temperature fitting as v1 (section 4.2). All models see the same features and folds.

| ID | Model | Spec |
|---|---|---|
| B0 | Naive trophies score | v1 B0 |
| B1 | Ridge on vote share, softmax | v1 B1 |
| B2 | Random forest, winner classifier | v1 B2 |
| **M1** | **Plackett-Luce, ridge, T1** | v1 4.1, features 3.1 |
| M1-G | M1 + trophy gate | v1 |
| M1-F | M1 + format interaction: `is_calendar` × {F9, F10, F11} | **New**; tests whether fame and awards weigh differently under the old format |
| X1 | XGBoost LambdaMART | v1 fixed settings |
| L1 | LightGBM LambdaMART | `num_leaves=4, n_estimators=150, learning_rate=0.05, min_data_in_leaf=10, lambda_l2=5, seed=7`; editions as groups |
| O1 | Ordinal logit on rank band | Bands: 1, 2-3, 4-10, 11+ voted, zero points; `statsmodels` OrderedModel; score = expected band (negated) |
| D1 | Decision tree on vote share | `max_depth=3, min_samples_leaf=8`; for an interpretable figure, not for selection |
| M2 | M1 + post-window block, T2 | 2022-2025 only (section 6.4) |

## 5. Validation and selection

### 5.1 Folds

Leave one edition out over the nine labelled editions. Nested inner LOEO on the eight training editions for λ (PL and ridge: 13-point log grid 0.01-100). Tree models use fixed settings.

### 5.2 Metrics (per held-out edition, then averaged)

| Metric | Role |
|---|---|
| Held-out PL log-likelihood per voted player | **Primary; selects the model** |
| Winner's predicted rank; top-1 hit | Headline |
| Top-3 overlap (/3); top-10 accuracy (overlap of predicted and actual top 10, /10) | Secondary |
| NDCG@10 (gain = vote share) | Secondary |
| Spearman ρ (all 30), pairwise accuracy | Secondary |

Always report per-fold values: 3-, 5- and 10-pick editions differ in difficulty. Report also the mean over the four v1 editions (2022-2025) so v1 and v2 can be compared directly.

### 5.3 Uncertainty

Within-edition bootstrap of the held-out metrics (resample players inside the held-out edition, 1,000 draws) for 90% intervals; paired bootstrap for model differences.

### 5.4 Selection rule (fixed now)

1. Rank by mean held-out log-likelihood over nine folds.
2. The published 2026 forecast uses the best PL model (M1, M1-G or M1-F); if two differ by less than 0.02 per voted player, use the simpler.
3. If a non-PL model wins, still publish the best PL model (the juror simulation needs PL utilities), report the winner beside it and log it.

## 6. Research experiments: what predicts the vote

### 6.1 (P) Model comparison table

All models in section 4 × the metrics in 5.2, nine-fold means with 90% intervals. Also show v1's M1 retrained on the four season editions with the v2 features, scored on the same four folds, to show what the extra editions buy.

### 6.2 (P) Variance shares and block ablations

- **Shares:** v1 method (out-of-sample explained variance by block, section 3.2 blocks), now over nine editions. Report with intervals.
- **Ablations:** refit M1 dropping one block at a time; report the change in held-out log-likelihood and winner rank. A block matters if dropping it worsens log-likelihood in at least 6 of 9 folds.
- **SHAP (S):** TreeSHAP on L1; mean |SHAP| per feature and a beeswarm plot.

### 6.3 (P) The fame finding, replicated

v1 result: among players whose football alone pointed to a top-10 finish, those with at least expected attention finished about 4.9th on average and those with less about 10.8th.

1. Fit a football-only PL (blocks: season output, big stage, trophies and team, awards, prior standing, position) in LOEO; take each held-out player's predicted rank.
2. Keep players with predicted rank ≤ 10. Split by `fame` (F9) residualised on football features in the training fold: ≥ 0 vs < 0.
3. Report mean actual rank, share finishing top 10 and a permutation p-value (shuffle the split within edition, 10,000 times), for all nine editions, for 2016-2021 and for 2022-2025 separately.
4. Pre-registered hypothesis (from v1): `season_attention × first_time` positive. Report the coefficient with its 90% interval.

### 6.4 (P) Does post-window information matter? (M2 vs M1)

v1 design unchanged, on 2022-2025 (four folds, T2): held-out log-likelihood difference per fold, P1-P3 coefficients and their sign per fold. The calendar editions have no post-window period, so this test cannot grow; say so.

### 6.5 (P) World Cup and tournament underrating check

The 2026 misses (Rodri 12th predicted → 4th actual; Haaland, Mbappé underrated) were attributed to tournament performance being underweighted with one World Cup in training.

1. Identify tournament editions: 2016 (Euro), 2018 (World Cup), 2019 (Copa), 2021 (Euro, Copa), 2023 (World Cup 2022), 2024 (Euro, Copa).
2. For players who played the main tournament, compare held-out residuals (actual minus predicted rank) between M1 **without** F11 and tournament terms and M1 **with** them.
3. Fit M1 with an interaction `tourn_ko_goal_contribs × is_world_cup` and report it.
4. Report the coefficient on `award_wc_golden_ball` (or F11) estimated with two World Cups (2018, 2023) vs one (2023 only, the v1 situation).

## 7. Jury experiments (ballot level)

Use the v1 ballot-level Plackett-Luce (`src/model/ballots.py`): one strength per player-edition (fixed effect) plus juror-specific terms. A juror's ranks only the top `ballot_picks` players; unpicked nominees enter as "not chosen".

### 7.1 (P) Home bias: persistence across juries

- Terms: `home` (juror nation = player nationality), `confed` (same confederation), `league` (juror nation = player's league country).
- Fit on 2016-2025 pooled, then with era interactions: `home × era2022` where era2022 = 1 for 2022-2025 (FIFA top-100 jury) and 0 for 2016-2021 (all-country jury).
- Report coefficients with 90% intervals; the descriptive rate (compatriots ranking their own player 1st vs other jurors), per era; and the translation into places for a player a neutral juror ranks 5th (v1: about 2.9 places).
- Question answered: did the 2022 reform (jury cut from about 175 to 100 nations) change home bias? Frame as persistence, citing the *Perspectives on Politics* (2020) and *Social Science Quarterly* papers.

### 7.2 (S) Bias by confederation and by player tier

Interact `home` with the player's confederation and with `top_tier` (top 3 by strength). Exploratory.

### 7.3 (P or S) Jury-local attention

Does attention inside a juror's own country predict that juror's ranking beyond the player's global strength?

- Add `local_attention` = z(`wiki_country_share_in`) per juror-nominee to the ballot model, with the player-edition fixed effects (which absorb global attention).
- Main version: the 3,500-view threshold, cells where the share exists (at least 15 of 30 nominees present in that country). Sensitivity: the 1,000-view version (`*_r1000`, 2018 onward); news share (`news_country_share_in`, 24 countries).
- **Missingness:** only 5-34% of ballot points sit in usable cells, and those are large countries. Report coverage per edition, fit only on covered juror-editions, and compare the covered jurors' other coefficients (home, confed) with the full fit to show whether they differ.
- Goes in the paper only if the coefficient is clear; otherwise the supplement, with the coverage table.

## 8. (S) Betting-odds baseline

- Use captures with `capture_phase = before_t1` within 60 days of T1, or `t1_to_deadline` (2016, 2018, 2019, 2021, 2024, 2025 have at least one). Exclude `after_deadline`.
- For each usable capture: does the market favourite equal the actual winner? Compare with M1's held-out pick for the same edition. Compute PL log-likelihood of the actual order **restricted to the quoted players** for the market (`implied_prob_norm` as utilities) and for M1 (its utilities renormalised over the same players).
- Report per capture; no pooled claim (n is tiny). Never feed odds into any model.

## 9. 2026 forecast and evaluation

### 9.1 Final fit

Refit the selected PL model on all nine editions (λ by LOEO). Produce 2026 utilities from T1 and, if M2 was selected, T2. Laplace intervals plus a within-edition bootstrap (500), as v1.

### 9.2 Juror simulation

v1 ballot simulation on the real 2026 jury (`fact_juror_nations`, 100 nations, 15-12-10-8-7-5-4-3-2-1), with bias terms from section 7.1 (2022-2025 era estimates). 10,000 runs, the v1 link (`k`, shared race-level SD) re-estimated on nine editions. Outputs as v1: P(win), P(top 3), P(top 10), expected rank, 80% interval, expected points.

**Lock-in:** commit the v2 forecast and tag `forecast-v2` **before** any file containing the 2026 result enters the repo. Record in `deviations.md` that the result was publicly known when the tag was made, and that model selection used 2016-2025 only.

### 9.3 Compare with the actual 2026 result (after the tag)

The user adds `data/holdout/bdo_2026_result.csv` (rank 1-30, points if published). Then report:

- v2 vs v1 (published 22 Oct) vs market (22 Oct): winner's predicted rank, P(win) of the actual winner, top-10 overlap, Spearman.
- The misses: Rodri, Haaland, Mbappé (underrated in v1), Hakimi and Vitinha (overrated). Did the extra tournament editions and awards fix them?
- If points are not published, use rank-based metrics only.

## 10. (S) Robustness runs (repeat 5 and the 2026 forecast)

| ID | Run |
|---|---|
| R1 | Season editions only (2022-2025): v1 design with v2 features |
| R2 | Drop 2016 (3-pick ballot, half-covered baseline) |
| R3 | Calendar editions only (2016-2021), five folds |
| R4 | 2021-2025 with xG (`npxg_xa` instead of `npga` in F1) |
| R5 | Without ratings (F2), the rating-free run |
| R6 | Add EA rating (z(`ea_overall_t1`)) |
| R7 | Without F12 (market value), and with `league_poty` instead of `league_player_of_season` |
| R8 | Without fame and attention (F9, F10) |
| R9 | Edition-format fixed effect only (no M1-F interactions) |

## 11. The paper (5 pages)

| Element | Content | From |
|---|---|---|
| Table 1 | Model comparison: B0, B1, X1, L1, M1, M1-G, M1-F; log-likelihood, winner rank, top-3, top-10, NDCG@10 (nine-fold means) | 6.1 |
| Figure 1 | Variance shares by block with intervals, plus the ablation deltas | 6.2 |
| Figure 2 | Fame effect: actual rank for football-predicted top-10 players, above vs below expected fame, by era | 6.3 |
| Table 2 | Jury: home, confederation, league coefficients by era; places gained at rank 5 | 7.1 |
| Text | M2 vs M1 result (one paragraph); World Cup check (one paragraph); 2026 forecast vs result (one paragraph, with the lock-in tag) | 6.4, 6.5, 9 |

Everything else goes in a supplement on GitHub.

## 12. Outputs

```
results/v2/
  environment.txt              # dataset commit, package versions, seeds
  deviations.md
  metrics_by_fold.csv          # model × fold × metric
  metrics_summary.csv          # with 90% intervals; also the 2022-2025 subset
  selection.md
  coefficients_M1.csv          # Laplace and bootstrap intervals
  variance_shares.csv, ablations.csv, shap_L1.csv
  fame_test.csv                # 6.3, by era
  post_window_test.csv         # 6.4
  tournament_check.csv         # 6.5
  jury_bias_by_era.csv         # 7.1
  jury_local_attention.csv     # 7.3, with coverage per edition
  odds_baseline.csv            # 8
  forecast_2026_v2_T1.csv      # 9.2, tagged forecast-v2
  forecast_2026_comparison_v2.csv   # 9.3, after the tag
  robustness.csv               # 10
figures/v2/                    # fig1_shares.pdf, fig2_fame.pdf, plus supplement figures
```

## 13. Tests (must pass before the real run)

- PL likelihood: hand-checked on a 4-player edition with a tie and a zero-point player; 3-pick ballots handled.
- Within-edition standardisation never uses another edition; residualisation fitted on training folds only (assert by refitting with the held-out edition perturbed: training outputs must not change).
- No 2026 target is read anywhere before 9.3 (assert `bdo_points` for 2026 is all null in every loaded frame; grep for the holdout path).
- Odds never appear in a feature matrix.
- Ballot model reproduces v1's 2022-2025 home coefficient (2.31, 90% CI 1.88-2.80) when fitted on 2022-2025 only.
- Point-sum assertions per edition (section 2).

## 14. Order of work

1. Branch, copy data, tests (13).
2. Features (3), models (4), LOEO (5) → stop: Table 1 and selection.
3. Research tests 6.2-6.5 and jury 7.1-7.3 → stop: review.
4. Odds (8), robustness (10).
5. Final fit, simulation, tag `forecast-v2` (9.1-9.2) → stop.
6. Add the 2026 result, comparison (9.3), figures and tables for the paper (11).

## 15. What not to do

- Do not tune anything on 2026 or on the full nine editions outside the nested folds.
- Do not use odds, `tourn_individual_award`, or 2026 targets as inputs.
- Do not grid-search tree models.
- Do not pool the M2 test with calendar editions.
- Do not claim the jury-local result generalises to small countries.
- Do not edit the dataset; derive everything in code.
