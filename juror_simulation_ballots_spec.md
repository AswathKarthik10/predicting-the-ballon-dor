# Juror simulation from real ballots: spec

Oct 22, 2026 · for Claude Code, in the `predicting-the-ballon-dor` repo

## 0. Read this first

The current juror simulation (`src/model/jurors.py`, `src/model/simulate.py`, `src/model/forecast.py`) rebuilds the jury from summary columns and uses **placeholder** bias values (`b_home=0.8, b_confed=0.2, b_league=0.2`). Its docstrings and `results/deviations.md` (S1, S8) say no ballots existed. They did: the dataset repo has every country's ballot for 2022–2025 and the real 100 juror nations for every edition, 2026 included.

This task replaces the placeholders with values **estimated from the real ballots** and reruns the 2026 simulation on the **real 2026 jury**. Modelling spec section 6.3 already specified this ("if ballots exist, estimate the biases from them"), so this corrects an implementation gap rather than changing the plan.

**Hard rules**

- **Do not refit or reselect M1 or M2.** Use the existing final fits and Laplace draws exactly as they are (`src/model/inference.py`). Selection (M1 published) stands.
- **Do not overwrite any existing result file.** Every new output gets a `_ballots` suffix. The original forecast stays on record.
- **Never read 2026 targets** (they are null; assert it).
- Log every decision made while implementing in `results/deviations.md`, Part 4 (section 9).
- Seeds fixed at 7. Record input file SHA-256 hashes in every output header, as the existing outputs do.

## 1. Data to copy into this repo

Copy these seven files from the dataset repo (`ballon-dor-dataset`, release `data-v1`, commit `371021c`) into a new folder `data/jury/` in this repo:

| File (from `ballon-dor-dataset/data/processed/`) | Rows | What it gives |
| --- | --- | --- |
| `fact_bdo_country_votes.parquet` | 11,520 | Points each voting country gave each nominee, 2022–2025 (zero for unpicked nominees). One ballot = one `(edition, country_key)` |
| `fact_juror_nations.parquet` | 500 | The 100 juror nations per edition, 2022–2026, with `confederation` and `voted` (null for 2026) |
| `dim_country.parquet` | 211 | `country_key` → `confederation`, `name` |
| `dim_player.parquet` | 91 | `wiki_title` → `player_key`, plus `nationality_country_key` |
| `fact_bdo_results.parquet` | 150 | Official points per nominee, used to verify the ballots sum correctly |
| `dim_league_context.parquet` | 295 | `league_key` → `country_key` (the league's country) |
| `dim_club.parquet` | 236 | Backup for club → country if needed |

From the repo root:

```
mkdir -p data/jury
cp ../ballon-dor-dataset/data/processed/{fact_bdo_country_votes,fact_juror_nations,dim_country,dim_player,fact_bdo_results,dim_league_context,dim_club}.parquet data/jury/
```

(Adjust `../ballon-dor-dataset` to wherever the dataset repo is cloned.)

**Dependency:** reading Parquet needs `pyarrow`. Add `pyarrow` to `requirements.txt` (pin the installed version) and install it.

**Data facts to rely on (verified):**

- Ballots: 93 (2022), 92 (2023), 99 (2024), 100 (2025) countries. Points schemes: 2022–23 top 5 `[6,4,3,2,1]`; 2024–25 top 10 `[15,12,10,8,7,5,4,3,2,1]`. Every ballot uses each scheme value exactly once.
- `fact_bdo_country_votes.wiki_title` matches `dim_player.wiki_title` for every row.
- 2026 jury: 100 nations; UEFA 41, CAF 20, AFC 17, CONCACAF 11, CONMEBOL 10, OFC 1. `voted` is null for 2026 (not yet known).
- `league_key` 130 (MLS) is missing from `dim_league_context`: map it to `c_usa`. All other snapshot league keys map. Snapshot `league_key` is an integer; `dim_league_context.league_key` is a string; cast before joining.
- Player nationality: use `dim_player.nationality_country_key`. Confederation of a player = `dim_country.confederation` of that key.

## 2. Build the ballot table

New module `src/model/ballots.py`.

`load_ballots(root) -> pd.DataFrame` with one row per `(edition, juror_country_key, player_key)` and columns:

| Column | Definition |
| --- | --- |
| `edition`, `juror_country_key`, `player_key` | Keys |
| `points` | Points given (0 if unpicked) |
| `pick_rank` | 1 for the highest points on that ballot, 2 for the next, … ; null if 0 points |
| `home` | 1 if the player's nationality equals the juror country |
| `confed` | 1 if the player's confederation equals the juror's confederation, **and** `home == 0` |
| `league` | 1 if the player's league country (from his snapshot `league_key`) equals the juror country |

`home` and `confed` are kept mutually exclusive so each coefficient means one thing: `confed` is "same continent, different country".

**Checks (raise on failure):**

1. Summing `points` per `(edition, player_key)` reproduces `fact_bdo_results.points` exactly for 2022–2025.
2. Every ballot uses each scheme value exactly once and has 5 (2022–23) or 10 (2024–25) picks.
3. Ballot counts per edition are 93, 92, 99, 100.
4. Every voting country appears in `fact_juror_nations` for that edition with `voted == True`.

## 3. Estimate juror bias from the ballots

Model: a rank-ordered logit (Plackett-Luce) on **individual ballots**, top-k partial rankings (unpicked nominees tied at the bottom, never ordered among themselves; same treatment as `src/model/pl.py`).

For juror j in edition e and player i:

```
v_ij = α_(i,e) + β_home · home_ij + β_confed · confed_ij + β_league · league_ij + ε_ij,   ε ~ Gumbel(0, 1)
```

- `α_(i,e)` is a free strength per player-edition (about 120 parameters). It soaks up everything about how good the player's season was, so the β's measure only how juror nationality shifts the ballot.
- Identification: fix the mean of α within each edition to 0, or add a very small ridge penalty (λ = 0.01) on α only. **No penalty on the β's.**
- Fit by maximum likelihood with `scipy.optimize.minimize(method="L-BFGS-B")` and an analytic gradient. Reuse the likelihood code in `src/model/pl.py` where possible.
- Uncertainty for the β's: Laplace (inverse Hessian), plus a **ballot bootstrap**: resample whole ballots with replacement within each edition, refit, 500 times. Report both.

**Report** in `results/juror_bias_estimates_ballots.csv`: each β with standard error, 90% Laplace interval, 90% bootstrap interval, and the number of ballots and player-juror pairs where the indicator is 1 (so readers can see how much evidence sits behind each β).

**Robustness fits** (same file, a `spec` column):

- `main`: all four editions.
- `10pick_only`: 2024–2025 only.
- `by_edition`: one fit per edition (β's only; flag any edition where an indicator has fewer than 20 non-zero pairs).

**Plain-language translation** (for the article and paper), in `results/juror_bias_summary_ballots.md`:

- For an average nominee ranked 5th by a neutral juror, how many places higher does a compatriot juror rank him? Compute by simulation from the fitted model: hold α fixed, draw ballots with and without `home = 1`, compare the average rank.
- Same for `confed` and `league`.
- Descriptive, no model: across all ballots, the share of compatriot jurors who ranked their own nominee 1st, versus that player's share of 1st-place votes from all other jurors.

## 4. Link the published model's strengths to juror utilities

The published M1 and M2 strengths `u_i` are on the scale of the **aggregate** ranking. Ballot-level utilities α are on the scale of **one juror's** ballot. Map one to the other using the back-test editions:

1. For 2022–2025, take the **out-of-fold** M1 utilities (`forecast.oof_utilities`, which already exists) for each player-edition, and the fitted `α_(i,e)` from section 3.
2. Within each edition, regress α on u: `α_(i,e) = c_e + k · u_(i,e) + r_(i,e)` (one common slope k, per-edition intercepts c_e).
3. `k` is the scale factor from model strength to juror utility. The residual standard deviation `σ_shared` is the race-level uncertainty the model does not explain, shared by all jurors in an edition.
4. Repeat with M2 out-of-fold utilities on T2 rows to get `k_M2` and `σ_shared_M2`.

Save `k`, `σ_shared`, and the regression R² per snapshot to `results/juror_link_ballots.csv`.

Why this design: juror noise is fixed at Gumbel(0, 1) by the ballot model, so everything about how strongly jurors agree is learned from real ballots rather than tuned to match vote concentration (which is what `calibrate_tau_juror` did).

## 5. Simulation on the real 2026 jury

Changes to the simulation code:

- **Jurors:** new function `real_jury(edition)` in `src/model/jurors.py` returning the 100 nations from `fact_juror_nations` with their confederations. Build `nation_match`, `confed_match` (excluding home, as in section 2) and `league_match` from them. Keep `build_jurors` (the old reconstruction) for comparison, unchanged.
- **Ballot scheme per edition:** `simulate.py` hard-codes 10 picks. Make the points scheme a parameter (`[6,4,3,2,1]` for 2022–23, the 10-pick scheme for 2024–26) without changing existing default behaviour.
- **Noise:** shared shock per player is `Normal(0, σ_shared)` (from section 4); per-juror noise is `Gumbel(0, 1)`. Utilities are `k · u_i` with u drawn from the existing Laplace draws.

For each simulation:

```
v_ij = k · u_i(draw) + shared_i + β_home·home_ij + β_confed·confed_ij + β_league·league_ij + Gumbel(0,1)
```

Draw β's from their Laplace distribution too, so bias uncertainty carries into the forecast.

- 10,000 simulations per snapshot (T1 with M1 strengths, T2 with M2 strengths).
- Jurors: all 100 vote. Sensitivity: 95 jurors, dropping 5 at random per simulation (past participation was 92–100).
- Ties broken as in the existing `rank_players`.

**Outputs:** `results/forecast_2026_T1_ballots.csv`, `results/forecast_2026_T2_ballots.csv`, same columns as the existing forecast files.

## 6. Validate on past editions before trusting 2026

For each back-test edition (2022–2025), simulate with **out-of-fold** strengths, that edition's real jury and points scheme, and bias and link parameters estimated **without that edition** (leave one edition out for sections 3 and 4).

Report in `results/juror_validation_ballots.md`, for the new simulation and for the original placeholder simulation side by side:

| Metric | Definition |
| --- | --- |
| P(actual winner wins) | Per edition |
| Log score | Sum over editions of log P(actual winner) |
| Actual winner's expected rank | Per edition |
| Concentration | Simulated vs actual winner vote share, number of players with points, Herfindahl |
| Home-vote check | Simulated vs actual share of compatriot jurors ranking their own nominee 1st |

If the new simulation is worse than the placeholder one on log score in 3 or more of 4 editions, say so plainly in the report and keep the placeholder forecast as the main one; still publish the ballot-based one as secondary.

## 7. Comparison with the published forecast

`results/forecast_2026_comparison_ballots.csv`: for every player, P(win), P(top 3) and expected rank under (a) the original placeholder simulation and (b) the new ballot-based simulation, for T1 and T2, plus the change.

Figure `figures/forecast_2026_win_prob_ballots.png`: top 10 by P(win), original vs ballot-based, T1 and T2. Figure `figures/juror_bias_ballots.png`: the three β's with 90% intervals, main fit plus the `10pick_only` fit.

## 8. Tests (must pass before the real run)

`tests/test_ballots.py`:

- Ballot points reproduce official totals for every back-test edition.
- The ballot PL gradient matches finite differences to 1e-5.
- On synthetic ballots drawn with known α and β (100 jurors, 30 players, 4 editions), the fit recovers β within 2 standard errors.
- `home` and `confed` are never both 1.
- `real_jury(2026)` returns exactly 100 nations with the confederation counts in section 1.
- The 5-pick scheme simulation awards exactly 16 points per juror; the 10-pick, 67.
- No function reads `bdo_points`, `bdo_rank` or `vote_share` for 2026.

## 9. Deviations log

Add **Part 4. Juror simulation from ballots (2026-10-22)** to `results/deviations.md`:

- B1: The ballots and juror nation lists existed in the dataset repo (`fact_bdo_country_votes`, `fact_juror_nations`) but were not used in the first run; S1 and S8 described them as missing. This rerun implements spec 6.3 as written. The original forecast is kept unchanged and remains the timestamped pre-registered forecast.
- B2: Noise calibration changes from matching vote concentration (S4) to estimating it from individual ballots (section 4). State this was decided before seeing the new 2026 output.
- B3: MLS league key 130 mapped to `c_usa` by hand.
- Any other implementation choice made while coding.

Also correct the docstring at the top of `src/model/jurors.py`, which says no ballots exist.

## 10. Order of work

1. Copy data, add `pyarrow`, write `ballots.py` and its checks (section 2). Run the checks.
2. Tests for the ballot model on synthetic data (section 8), then fit on real ballots (section 3).
3. Link (section 4).
4. Validation on past editions (section 6). **Read the result before running 2026.**
5. 2026 simulation (section 5) and comparison (section 7).
6. Deviations log and a short summary in `results/juror_summary_ballots.md`: the bias estimates in plain words, how the 2026 forecast changed, and whether the ballot-based simulation beat the placeholder one on past editions.

## 11. What not to do

- Do not change features, refit M1/M2, or rerun model selection.
- Do not tune `k`, `σ_shared` or the β's to make the 2026 output look different; they come only from sections 3 and 4.
- Do not delete or overwrite the original forecast files or figures.
- Do not claim causal juror "bias" in any summary text: the β's measure how ballots differ by juror nationality after controlling for each player's overall standing that year.
