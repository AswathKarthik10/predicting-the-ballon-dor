# Ballon d'Or 2026: feature-level EDA findings

Source: `ballon_dor_2026.ipynb`, sections B–D. Inputs: `player_snapshot_t1.csv` and `player_snapshot_t2.csv` (git `4b192ae`). Date: 2026-10-21.

**Status: descriptive and exploratory.** The 10-feature set in the modelling spec (section 3.1) is pre-registered. Nothing here was used to drop, add or re-weight a feature, and no model has been fitted. All outcome-related numbers use the 120 labelled player-seasons (editions 2022–25); 2026 targets are untouched. Correlations are computed **within edition** (voting is relative), then summarised across the four editions.

---

## 1. Data check against the spec (section 2)

Every claim I tested in spec section 2 matches the data:

| Check | Result |
| --- | --- |
| Shape | T1 150 × 91, T2 150 × 111 (20 extra post-window columns) |
| Post-window days | 47 / 60 / 58 / 40 / 68 for 2022–26, as specified |
| Point totals | 1,488 / 1,472 / 6,633 / 6,700 for 2022–25, as specified |
| Zero-point nominees | 11 / 5 / 2 / 0 for 2022–25 (18 in 2022–24) |
| Null international xG | 18 rows, all with fewer than 270 international minutes |
| Null `club_rating_window_end` | 2 rows, Messi and Quiñones (2026) |
| Null `big_game_rating` | 3 rows, Mané, Quiñones, Messi (2026) |
| Negative `club_npxg_xa_p90_opp_adj` | 10 rows, 7 GK and 3 DEF |
| Goalkeepers | 2 / 3 / 1 / 1 / 0 for 2022–26 |

Findings the spec does not state:

- **2025 had no international tournament.** All `tourn_*` columns are null or constant, so `tourn_ko_goal_contribs` and the tournament-stage ordinal have zero variance that year. 2022 has only 4 players in a tournament. The 2024 edition mixes EURO, Copa América and AFCON.
- **`bdo_rank` cannot define the ordering.** It is not unique in 2022–24, because zero-point nominees share ranks. The Plackett-Luce ordering must be built from `bdo_points`.
- **Tied voted players:** 7 / 7 / 4 / 4 rows in 2022–25 (cases for the Breslow approximation).
- **Winner's share of points:** 37% / 31% / 18% / 21% (2022–25). The 5-pick editions are more lopsided.
- **Unlisted ordinal value:** `ucl_stage_reached` contains `playoff` (3 rows: Haaland 2025, Gyökeres 2025, Martínez 2026), which the spec's ordinal map omits. I treated it as the R32-equivalent round (ordinal 2) for the EDA. This needs an explicit decision and a `deviations.md` entry.
- **`defending_winner` has one labelled row** (2023). The other defending winner sits in the 2026 pool, so feature F8 can barely be estimated from the back-test data.
- **2025 post-window form is almost entirely null.** `post_npxg_xa_p90` is null whenever post-window minutes are below about 270. Only 1 of 30 players in 2025 reaches that.

Figures: `figures/eda/b_missingness_by_edition.png`, `b_points_by_edition.png`.

## 2. Which raw features move the outcome

Method: Spearman correlation between each candidate input and points, within edition, for 36 candidates (7 groups). Pooled p-values use within-edition percentile ranks, with Benjamini–Hochberg correction across the 36 tests. The grey dots in the figure show each edition separately.

| Rank | Feature | Group | Mean ρ (4 editions) | Same sign in |
| --- | --- | --- | --- | --- |
| 1 | `log_wiki_in` | attention | 0.57 | 4 of 4 |
| 2 | `log_news_in` | attention | 0.57 | 4 of 4 |
| 3 | `other_honors` | honors | 0.45 | 4 of 4 |
| 4 | `log_wiki_pre` | attention | 0.44 | 4 of 4 |
| 5 | `ucl_goal_contribs` | big stage | 0.43 | 4 of 4 |
| 6 | `log_news_pre` | attention | 0.42 | 4 of 4 |
| 7 | `major_titles` | honors | 0.42 | 4 of 4 |
| 8 | `final_goal_contribs` | big stage | 0.41 | 4 of 4 |
| 9 | `ucl_ko_goal_contribs` | big stage | 0.41 | 4 of 4 |
| 10 | `club_avg_rating` | performance | 0.40 | 4 of 4 |
| … | `club_attack_p90` | performance | 0.32 | 4 of 4 |
| … | `is_defender` | position | −0.29 | 4 of 4 |
| … | `defending_winner`, `age`, `club_minutes`, `wiki_change`, `news_change` | various | about 0 or inconsistent | n/a |

Reading:

- Attention is the strongest group, followed by honors and big-stage output. The raw attack rate is only moderate, much weaker than trophies and fame.
- 2025 is the weakest edition for attention and prior standing (pre-window attention ρ between 0.05 and 0.19; `prior_standing` is −0.06 in 2025).
- Most of the top features survive the multiple-testing correction. `tourn_ko_goal_contribs` sits exactly at p_BH = 0.05, and it has no 2025 value.
- Defenders are consistently penalised (−0.36, −0.36, −0.36, −0.07 across editions).

Category splits (Kruskal–Wallis on within-edition points percentile):

| Split | p-value | Note |
| --- | --- | --- |
| Major titles (0 / 1 / 2) | 0.00005 | group sizes 82 / 36 / 2 |
| Position | 0.005 | |
| UCL stage reached | 0.012 | |
| First-time contender | 0.34 | no difference |
| Tournament stage reached | 0.27 | 66 of 120 did not play a tournament |
| Defending winner | not testable | 1 labelled row |

Figures: `c_spearman_vs_points.png`, `c_top_features_scatter.png`, `c_points_by_group.png`, `c_feature_distributions.png`.

## 3. Overlap between features

- **Near-duplicates:** `club_attack_p90` and `club_goals` (ρ 0.91); `club_rating` and `teammates_on_shortlist` (0.88); `log_news_pre` and `log_news_in` (0.84); `career_podiums_before` and `first_time` (−0.83); `log_wiki_pre` and `log_news_pre` (0.81).
- **Variance inflation:** the attention columns are the worst (`log_news_pre` 341, `log_news_in` 244, `news_change` 65, `log_wiki_pre` 62). This is expected for pre- and in-window counts of the same quantity, and supports the spec's choice to residualise attention. These are diagnostics only; nothing is dropped on this basis.
- **PCA** on 34 within-edition standardised features: 11 components reach 80% of variance.
  - PC1 (22%): a general fame-and-output axis (Wikipedia views, attack rate, prior standing). Spearman with points 0.46.
  - PC2 (15%): team and trophy context (teammates on the shortlist, UCL stage, titles, club rating). 0.38.
  - PC3 (11%): the *change* in attention. 0.03, so no relationship with points.
- **Rough multivariate look** (one input per group, in-sample, descriptive only): R² = 0.60. The largest standardised terms are `log_wiki_pre` (0.45), `major_titles` (0.28), `wiki_change` (0.32), `other_honors` (0.23), `club_rating` (0.19) and `prior_standing` (0.21). The attack rate adds little once fame and honors are in the model. Cross-validation in the modelling stage is the real test.

Figures: `c_feature_correlation.png`, `c_vif.png`, `c_pca.png`.

## 4. Recency bias: does late "hype" earn extra points?

**Question.** Does a player who is talked about more in the weeks before the vote get more points than their season and baseline fame justify?

**Data and method.**

- Juror-weighted Wikipedia views and news volume in four slices: `pre` (before the window), `in` (during the window), `post` (window end to 3 days before the vote deadline, T2 only) and `final14` (the last 14 days of that period, Wikipedia only).
- Log-transformed, then standardised within edition. Period length only shifts a log value by a constant within an edition, so the missing window start dates do not affect these tests.
- `final_spike` = log daily rate in the last 14 days minus log daily rate over the whole post-window period (spec P3).
- Because performance drives attention, the main test is a **partial** Spearman correlation with points after controlling for attack rate, big-game rating, UCL knockout output, major titles, other honors, club rating and prior standing (control set A). Control set B also controls for baseline fame (`wiki_pre`) and in-window attention (`wiki_in`). 90% bootstrap intervals, 2,000 draws, n = 120.

### 4.1 Raw correlation with points, by period (mean of 4 editions)

| Attention measure | Mean ρ | 2022 | 2023 | 2024 | 2025 |
| --- | --- | --- | --- | --- | --- |
| `wiki_pre` | 0.44 | 0.48 | 0.54 | 0.55 | 0.19 |
| `wiki_in` | **0.57** | 0.60 | 0.56 | 0.74 | 0.41 |
| `wiki_post` | 0.48 | 0.49 | 0.43 | 0.60 | 0.39 |
| `wiki_final14` | 0.52 | 0.55 | 0.47 | 0.70 | 0.36 |
| `news_pre` | 0.42 | 0.43 | 0.59 | 0.62 | 0.05 |
| `news_in` | **0.57** | 0.58 | 0.70 | 0.71 | 0.30 |
| `news_post` | 0.42 | 0.28 | 0.41 | 0.68 | 0.29 |
| `final_spike` | 0.09 | 0.17 | −0.06 | 0.36 | −0.12 |

Attention in every period correlates positively with points, but the strongest link is with attention *during the window*, not with attention in the final weeks. The late-acceleration measure itself (`final_spike`) is not related to points (pooled ρ = 0.09, p = 0.32).

### 4.2 Partial correlations: the recency test

| Attention measure | Raw ρ | Partial ρ, controls A (90% CI) | Partial ρ, controls B (90% CI) | Positive editions (A / B) |
| --- | --- | --- | --- | --- |
| `wiki_post` | 0.48 | 0.42 (0.27, 0.54) | 0.13 (−0.03, 0.27) | 4 / 2 |
| `wiki_final14` | 0.50 | 0.41 (0.27, 0.53) | 0.13 (−0.02, 0.28) | 4 / 4 |
| `final_spike` | 0.09 | 0.03 (−0.12, 0.18) | 0.03 (−0.13, 0.19) | 4 / 4 |
| `news_post` | 0.44 | 0.33 (0.16, 0.48) | 0.16 (−0.01, 0.32) | 4 / 3 |

Incremental explanatory power (pooled, in-sample, nested blocks):

| Step | R² | ΔR² | p-value |
| --- | --- | --- | --- |
| Controls A (7 performance, honors, team and standing inputs) | 0.525 | n/a | n/a |
| + pre-window attention | 0.575 | +0.050 | 0.002 |
| + in-window attention | 0.645 | +0.070 | < 0.001 |
| + post-window attention | 0.653 | **+0.008** | 0.31 |
| + final-14-day spike | 0.654 | **+0.001** | 0.51 |

### 4.3 Did the top finishers have a late spike?

- Final-14-day spike for the actual top 3 versus the rest, pooled: Mann–Whitney p = 0.997. The top 3 had a higher mean spike than the rest only in 2024 (1.68 vs 1.42).
- Late risers did not reliably finish high. Rodri (2024) had one of the biggest spikes and won. But Rüdiger was a top-3 riser in 2022 and 2024 and finished 20th and 22nd, Musiala (2023) finished 26th, Olise (2025) 30th, and Leão (2022) 13th.
- The bottom 3 in spike each year (attention fading into the vote) mostly finished 13th–26th (Cole Palmer 2025 was 8th), but most of the top-3 risers also finished outside the top 10. Neither tail stands out from the field.

### 4.4 Interpretation

1. **No evidence of an extra recency effect beyond what fame and the season already explain.** Late attention correlates with points (0.4–0.5 raw), but once the player's baseline fame and in-window attention are controlled, the partial correlations drop to about 0.13–0.16, with intervals that include zero. Adding post-window attention or the final spike to a model that already has pre- and in-window attention raises R² by under 1 percentage point (p ≥ 0.31).
2. **The "accelerating into the vote" measure carries no signal on its own** (raw ρ 0.09, partial ρ 0.03).
3. **This cannot rule out a small effect.** The upper ends of the control-B intervals are 0.27–0.32, and with 4 editions and 120 rows the test has limited power.
4. **A positive result would not prove influence either.** Attention is partly a *consequence* of being a favourite: media cover likely winners. The attention-based tests are suggestive at most. Any claim about juror behaviour needs the individual ballots (`fact_juror_ballots`).
5. This matches the spec's own caution (section 12: report consistency of sign and effect size, not significance claims). The result is **exploratory**; the pre-registered tests (P1–P3 in M2, and `season_attention × first_time`) are unchanged.

Figures: `d_attention_persistence.png`, `d_attention_vs_points.png`, `d_partial_recency.png`, `d_final_spike.png`.

### 4.5 2026 forecast pool (features only, no outcome)

- Largest late acceleration (`final_spike`): Willian Pacho (2.24), Harry Kane (2.11), Michael Olise (2.08), Sadio Mané (1.96), Bruno Fernandes (1.90).
- Biggest post-window attention gain over their in-window level (within-edition z difference): Rodri (1.71), Ferran Torres (1.61), Pau Cubarsí (1.29), Marc Cucurella (0.85), Fabián Ruiz (0.81).

These are descriptive. They are not a forecast, and given section 4.4 they should not be read as predicting a higher finish.

## 5. How much do performance and attention explain, and does anyone miss out?

Source: notebook section E. Exploratory, not model selection.

### 5.1 Variance decomposition

Inputs are grouped into five blocks: **Performance** (attack rate, ratings, big-stage output; 9 features), **Trophies and team success** (5), **Standing** (prior ballot rank, past podiums; 2), **Attention** (Wikipedia and news, pre-window and in-window; 4) and **Position** (1). For every subset of blocks I fitted a ridge regression (fixed alpha 10, no tuning) of the within-edition points percentile and scored it by **leave-one-edition-out R²** (out-of-sample). Each block's Shapley value (its average marginal gain over all block orderings) splits the total R² fairly when blocks overlap. 300 bootstrap draws (resampling within edition) give the 90% intervals.

| Block | R² alone | Share of explained variance | 90% interval | Unique R² (loss if dropped) |
| --- | --- | --- | --- | --- |
| Trophies and team | 0.28 | **39%** | 24–67% | 0.20 |
| Attention | 0.30 | **33%** | 18–55% | 0.16 |
| Performance | 0.30 | **18%** | −4–30% | −0.02 |
| Standing | 0.05 | 6% | −15–20% | 0.02 |
| Position | 0.07 | 4% | −2–12% | −0.00 |
| **All blocks** | | out-of-sample R² **0.63** (bootstrap 90%: 0.43–0.69; in-sample 0.73) | | |

- About **two-thirds of the explainable variation sits in trophies/team success plus attention**, roughly equal in size and each larger than the pure performance block. Performance alone explains as much as attention alone (R² 0.30 each), but it shares a lot with the other blocks, and once trophies and attention are in the model it adds nothing out of sample (unique R² −0.02).
- Removing attention costs a lot: out-of-sample R² falls from 0.63 to 0.47. Attention is not just a proxy for performance.
- The ordering is stable across variants (points percentile or log points as the target; in-sample or out-of-sample): trophies/team 30–39%, attention 29–35%, performance 18–27%, standing 5–11%, position 3–4%.
- The performance share has a wide interval that includes zero, so with 120 rows I would not claim performance matters less than attention in absolute terms. What the data does support is that **the stat-based performance block is not the dominant driver**.
- Shares are attribution of overlapping information, not causal effects.

### 5.2 Do better performers with less attention miss out?

I built an out-of-sample **expected finish** from on-field information only (performance + trophies/team + position, no attention), and an **attention gap** for each player (their attention minus what that on-field profile predicts, also fitted on the other editions).

| Expectation model | Spearman(attention gap, finish residual) | 90% CI | Per edition (2022 / 23 / 24 / 25) |
| --- | --- | --- | --- |
| On-field (performance + trophies/team + position) | **0.54** | 0.42–0.64 | 0.58 / 0.69 / 0.67 / 0.37 |
| Performance only | 0.45 | 0.32–0.56 | 0.41 / 0.57 / 0.59 / 0.20 |

Players with less attention than their on-field profile predicts finish below their expected placing, in all four editions.

Of the 40 player-seasons the on-field model expects in the top 10 (4 editions × 10):

| | n | Mean expected rank | Mean actual finish | Mean rank error | Finished top 10 |
| --- | --- | --- | --- | --- | --- |
| Attention at or above expectation | 16 | 5.4 | 4.9 | −0.5 | 94% |
| Attention below expectation | 24 | 5.5 | **10.8** | **+5.2** | 54% |

The same expected strength, but a 5-place gap in finishing position (Mann–Whitney one-sided p = 0.001).

Specific cases (expected top 8, finished at least 6 places worse): Olise 2025 (expected 8th, finished 30th, attention rank 22nd), Fabián Ruiz 2025 (5th → 24th, attention 26th), Doué 2025 (2nd → 14th, attention 28th), Nkunku 2022 (8th → 20th, attention 30th), Lookman 2024 (8th → 14th), Gündoğan 2023 (3rd → 14th) and Kvaratskhelia 2025 (3rd → 12th). **All 7 had less attention than their profile predicts.** The reverse also holds: of 8 players expected 12th or worse who finished top 8 (Yamal 2025: 14th → 2nd; Carvajal, Salah, Haaland, Courtois, Osimhen, Hakimi, Palmer), 6 had above-expected attention.

### 5.3 How to read this (important)

- **Consistent with attention mattering, not proof of bias.** The attention gap is not independent of quality. If a player is better than my features capture (for example, creative output or influence the stats miss), that also raises both his attention and his finish. The 0.54 correlation therefore mixes "voters reward visibility" with "visibility tracks quality my features miss". The data cannot separate the two. Only the ballot data and an experiment-style design could.
- **The in-window attention measure overlaps with the season itself.** It is also the measure with the strongest link to points. That is why section 4 (post-window attention) is the more targeted test for influence from *after* the eligibility window, and found no extra effect.
- **Small samples at the extremes.** The top-10 comparison has 40 player-seasons and the named cases are individual anecdotes. They illustrate the pattern; the correlation and the 5-place gap carry the weight.
- **2025 is again the weakest edition** (0.37), consistent with the other 2025 findings.
- Answering the question directly: **yes, the data shows players with strong on-field profiles and below-expected attention finishing well below where they would be expected (in all four editions), and high-attention players overperforming.** Whether that is a voting bias or unmeasured quality is not identifiable from these data alone.

Figures: `e_variance_decomposition.png`, `e_missing_out.png`.

## 6. What this means for the preparation step

- **Constant columns.** Edition-level z-scoring needs a rule for zero variance (z = 0): 2025 tournament features and 2025 `post_npxg_xa_p90` shrunk values are both constant.
- **Decisions needed, logged in `results/deviations.md`:** the `playoff` ordinal; z-score ddof (I used ddof = 1); and the zero-for-non-participants rule for tournament counts.
- **F8 (`defending_winner`)** will be close to unidentifiable in training; expect the ridge penalty to shrink it to about zero.
- **F10 and P2** (the residualised attention features) must be fit per training fold, as the spec requires. The partial-correlation results above are *not* a substitute and must not feed the CV.
- **Collinearity among attention inputs** supports residualising attention on F1–F7, as the spec already does.

## 7. Limitations

- Four labelled editions and 120 rows. Many significance tests are exploratory; effect sizes and sign consistency matter more than p-values.
- 2025 behaves differently (no tournament, no post-window form, weaker attention link), so pooled numbers lean on 2022–24.
- Raw EDA features are simplified (nulls filled with 0 for non-participation, mean-imputed in PCA). The final engineered features follow spec section 3 exactly.
- Attention measures are juror-weighted views and news volume, not a direct measure of what individual jurors read or saw.
