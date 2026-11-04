# Individual Awards Fix Spec (v2.1)

Nov 3, 2026 · amends `ballon_dor_experiments_spec.md` section 3.1 (feature F11)

## 0. Read this first

The v2 experiments replaced v1's crude award flag with F11 = an unweighted count of individual awards. On the 2026 forecast this failed:

- **Second and third places outscore wins.** The dataset counts any Golden Ball or Golden Boot place as 1, and the v2 formula added +0.5 for a 2nd or 3rd-place Golden Ball on top. Messi's silver Golden Ball scores 1.5; Rodri's gold scores 1.
- **Every award weighs the same.** MLS MVP = World Cup Golden Ball = World Cup Best Young Player = 1.
- **One achievement stacks.** Mbappé's World Cup counts twice (Golden Ball 3rd plus Golden Boot 1st).
- **Skew.** z-scoring a count of 0 to 3.5 within edition makes two players about 3 standard deviations above the rest, and F11 has the largest coefficient (0.60).

Result: Messi and Mbappé share the top score and are forecast 1st and 2nd. Rodri (Golden Ball winner) scores the same as a single league award. Kane (the actual winner) scores nothing.

This spec replaces F11 with a **weighted sum against the difficulty of the competition**, fixed here before any rerun.

**Rules for this work**

- **The v2 forecast stays locked.** Do not edit `results/v2/`. First push the tag (`git push origin forecast-v2`), then run the 2026 comparison for v2 as planned (section 4).
- **Everything in this spec is a correction made after the 2026 result was seen.** Log it in `results/v2_1/deviations.md`, label every v2.1 2026 output EXPLORATORY, and never present v2.1 as the forecast.
- **The weights in section 2 are fixed now.** Do not change any weight, cap or floor after running v2.1. If a rule turns out to be unworkable, stop and ask; log any change.
- **Two repos:** Part A in `ballon-dor-dataset` (branch `expansion-2016`), Part B in `predicting-the-ballon-dor` (new branch `experiments-v2.1` from `experiments-v2`).
- Stop for review after Part A, after the v2.1 back-test (B3) and before the v2.1 2026 forecast (B5).

## 1. The new feature

For each player-edition:

```
award_score = Σ over award events e of  type_w(e) × place_w(e) × difficulty(e)
```

- **Award event:** one award (type and place) in one competition-season, counted only if `announced_date` is inside the edition window and not after the snapshot date (the existing dataset rule).
- **One event per competition-season:** if a player has several awards from the same competition-season (for example Golden Ball 3rd and Golden Boot 1st at World Cup 2026), keep only the highest-scoring one.
- **F11 (v2.1)** = z(log1p(`award_score`)) within edition, then clipped to [−3, 3].

## 2. Weights (fixed)

### 2.1 Award type

| Type | Awards | `type_w` |
|---|---|---|
| Best player | World Cup Golden Ball; continental tournament best player; Champions League Player of the Season; UEFA Men's Player of the Year; league player of the season (league-run, `league_player_of_season` labels); Europa League and Conference League Player of the Season | 1.0 |
| Top scorer | World Cup Golden Boot; continental top scorer; Champions League top scorer; **league top scorer** (new, Part A2) | 0.6 |
| Other | World Cup Golden Glove; World Cup Best Young Player; any young-player award | 0.3 |

### 2.2 Place

| Place | `place_w` |
|---|---|
| 1st | 1.0 |
| 2nd | 0.5 |
| 3rd | 0.25 |

League and UEFA awards have only a winner (1.0).

### 2.3 Difficulty

All difficulties lie in [0.1, 1.0]; 1.0 is the strongest competition of its kind.

| Competition | `difficulty` | Source |
|---|---|---|
| Domestic league | `league_strength` (A1), scaled as below | Club ratings |
| International tournament | `tournament_strength` (A3), scaled as below | National Elo |
| Champions League; UEFA Men's Player of the Year | 1.0 | Fixed: the strongest club competition |
| Europa League | 0.5 | Fixed |
| Conference League | 0.3 | Fixed |

**Scaling.** For leagues: `d = 0.1 + 0.9 × (s − 1500) / (s_max − 1500)`, where `s` is the league's strength in that season and `s_max` the strongest league in the same season, clipped to [0.1, 1.0]. For tournaments: the same formula with `s_max` the strongest tournament in the whole 2016-2026 table (so a World Cup and a Euro of the same summer are comparable). 1500 is the dataset's floor rating for unrated clubs (`unrated_club_floor_rating`) and roughly a weak national team's Elo.

### 2.4 What this gives on 2026 (sanity targets, checked by tests)

- Rodri's Golden Ball (1st) > Messi's Golden Ball (2nd).
- Mbappé's World Cup counts once (his best World Cup award).
- MLS MVP < Bundesliga or Premier League Player of the Season.
- Kane receives a top-scorer event for the 2025/26 Bundesliga if he was its top scorer (A2), on the same rule as any other league.

These are properties of the rule, not targets for the forecast. Do not tune anything towards them.

## Part A: dataset (`ballon-dor-dataset`, branch `expansion-2016`)

A new build part, **Part 27**. No existing column changes; rule 2.1 of the expansion spec (released values do not move) still holds.

### A1. League strength

`league_strength(league_key, season_year)` = mean rating of the league's **five** strongest clubs that season, filling places with no rated club at 1500.

- Build it from `fact_club_rating` and `dim_club.league_key`.
- Why: the existing `league_mean_rating` averages only the rated clubs, so leagues with one to four rated clubs look too strong (2026: Portugal 1786 with 4 clubs above Germany 1763 with 10; Norway 1748 with 1).
- New table `fact_league_strength`: `league_key, season_year, top5_mean_rating, n_rated_in_top5`.
- The season used for an award is the season the award is for (2025/26 → `season_year` 2026).
- MLS and the Saudi league get their real values: few or no rated clubs, so close to 1500.

### A2. League top scorers

New hand-built reference `data/reference/league_top_scorers.csv`: `league_key, season, player_name, wiki_title, goals, announced_date, source_url`.

- Cover the leagues any nominee played in during any window, seasons 2014/15 to 2025/26.
- Source: the Wikipedia season page of each league (top scorers table), one URL per row. Joint top scorers: one row each.
- `announced_date` = the last day of the league season (from `competition_calendar.csv` or the season page).
- Resolve `wiki_title` to `player_key`; non-nominees keep a null key.
- Rationale: the tournament part of F11 already counts top scorers, so leagues are treated the same way. This was decided after seeing that Kane scored nothing in v2; log it as such (section 5).

### A3. Tournament strength

`tournament_strength(competition, season)` = mean Elo of the **eight** strongest entrants on the day before the tournament starts.

- Entrants: from the international results dataset (`martj42`, already used for Elo), all teams with a match in that tournament.
- Elo: the existing national Elo code, evaluated on the day before the first match.
- Cover every tournament in `tournament_awards.csv` (World Cup, Euro, Copa América, Africa Cup of Nations, Asian Cup, Gold Cup, Confederations Cup, 2016-2026).
- New table `fact_tournament_strength`: `competition, season, start_date, top8_mean_elo, n_entrants`.

### A4. Award events table

New table `fact_award_events`, one row per player and award, for nominees only:

`player_key, edition, competition, season, award_type (best_player | top_scorer | other), place, difficulty, announced_date, source`

- Built from `fact_tournament_awards`, `fact_individual_awards` and A2, with difficulty from A1 and A3 and the fixed values in 2.3.
- Assigned to editions by the existing window and snapshot-date rule.
- Store the components; the score itself is computed in Part B.

### A5. Checks (the build fails if any fails)

- Every award event has a difficulty in [0.1, 1.0].
- No event is dated after its edition's snapshot date.
- Every tournament in `tournament_awards.csv` has a strength row; every league with an award has a strength row for that season.
- Spot facts: 2026 Bundesliga and Premier League strength > MLS; World Cup 2022 and 2026 strength ≥ Gold Cup of the same years.
- Released snapshot values unchanged (the existing regression check).

## Part B: modelling (`predicting-the-ballon-dor`, branch `experiments-v2.1`)

### B1. Feature

Compute `award_score` and F11 (v2.1) as in sections 1-2, from `fact_award_events`. All other features, folds, penalties grids, seeds and models are exactly as v2.

### B2. Tests (must pass before any run)

- Rodri 2026 `award_score` > Messi 2026 `award_score` on World Cup awards alone.
- Mbappé 2026: one World Cup event, not two.
- A Premier League Player of the Season in 2026 scores more than an MLS MVP in 2026.
- Place weights: a 2nd place scores exactly half the 1st place at the same competition.
- The dedupe keeps the highest-scoring award per competition-season.
- No award event enters an edition after its snapshot date.
- No 2026 target is read before B6 (the existing guard).

### B3. Back-test (stop for review)

Rerun the v2 back-test (spec sections 5 and 6.1-6.2) with F11 (v2.1): all models, nine folds, the same metrics. Report:

- Table 1 for v2.1 next to v2 (held-out log-likelihood, winner rank, top-3, top-10, NDCG@10), per fold and mean.
- The paired difference v2.1 − v2 for M1 with 90% intervals.
- The ablation of the awards block and the variance shares.
- M1 coefficients (F11 and F9 in particular).

### B4. Sensitivity runs (back-test only)

| ID | Change |
|---|---|
| S1 | Without league top scorers (A2) |
| S2 | Uniform difficulty (all 1.0): isolates the type and place weights |
| S3 | v2.1 plus a World Cup indicator: 1 if the player's best award event is from a World Cup |
| S4 | No log1p and no clipping on `award_score` |

### B5. 2026 (stop before running; EXPLORATORY)

Only after the review in B3:

- Final fit of M1 (v2.1) on the nine editions and the juror simulation, exactly as v2 section 9.1-9.2.
- Output to `results/v2_1/forecast_2026_v2_1_T1.csv`, every file headed EXPLORATORY, post-hoc.
- Also report the 2026 top 5 for S1-S4.

### B6. Comparison with the actual 2026 result

Add `data/holdout/bdo_2026_result.csv` (rank 1-30; points if published) only after the v2 comparison (section 4) is committed. Report v1, v2 (locked) and v2.1 (exploratory) side by side: winner's predicted rank, P(win) of the actual winner, top-3 and top-10 overlap, Spearman, and the named misses (Kane, Rodri, Haaland, Mbappé; Messi, Hakimi, Vitinha).

## 3. Outputs

```
results/v2_1/
  deviations.md
  award_events_check.csv        # per player-edition: events, difficulty, score
  metrics_summary_v2_1.csv
  metrics_by_fold_v2_1.csv
  paired_v2_1_vs_v2.csv
  ablations_v2_1.csv
  variance_shares_v2_1.csv
  coefficients_M1_v2_1.csv
  sensitivity_awards.csv        # S1-S4
  forecast_2026_v2_1_T1.csv     # EXPLORATORY
  comparison_2026.csv           # v1, v2, v2.1 vs actual (after B6)
```

## 4. Order of work

1. `git push origin forecast-v2` (the v2 lock-in tag is not on GitHub).
2. Add the 2026 result and run the v2 comparison (experiments spec 9.3); commit.
3. Part A1-A5 in the dataset repo → stop for review.
4. Part B1-B2 → B3 back-test → stop for review.
5. B4 sensitivity.
6. B5 forecast (exploratory) → B6 comparison.

## 5. Deviations to log (at least)

- F11 redefined after the 2026 result was known; the reason (silver over gold, equal weights, stacking, skew) and that the weights were fixed in this spec before the rerun.
- League top scorers added after seeing Kane score zero in v2; justified by symmetry with tournament top scorers; S1 reports the result without them.
- The spec's own error: the v2 formula's "+0.5 for a 2nd or 3rd-place Golden Ball (gold already counted)" assumed the count held winners only.

## 6. What not to do

- Do not change any weight in section 2 after seeing v2.1 results.
- Do not present v2.1 as the forecast or overwrite any v2 file.
- Do not add award types not listed here without logging them.
- Do not use the 2026 result in any fit, tuning or rule.
