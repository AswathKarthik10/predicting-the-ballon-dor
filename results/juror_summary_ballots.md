# Juror simulation from real ballots: summary

Written 2026-10-22. All figures come from the files named in each section. Seed 7 throughout; input hashes are in each file's header. The original placeholder forecast (`results/forecast_2026_T1.csv`, `_T2.csv`) is unchanged and remains the pre-registered forecast.

## What the ballots say about juror nationality

Source: `results/juror_bias_estimates_ballots.csv`, `results/juror_bias_summary_ballots.md`. Estimated from 384 ballots (2022-2025) with a Plackett-Luce model that gives every player-edition its own strength, so the coefficients only measure how a juror's nationality shifts the ballot.

| Effect | Coefficient (90% bootstrap interval) | Places gained for a nominee a neutral juror ranks 5th | Placeholder used before |
| --- | --- | --- | --- |
| Same country as the nominee | 2.31 (1.88 to 2.80) | about 2.9 (2.6 to 3.2) | 0.8 |
| Same confederation, different country | 0.37 (0.24 to 0.49) | about 0.7 (0.4 to 0.9) | 0.2 |
| Nominee plays in the juror's country | 0.48 (0.03 to 0.95) | about 0.8 (0.0 to 1.6) | 0.2 |

- Descriptively, with no model: compatriot jurors ranked their own nominee 1st in 11.7% of cases (120 juror-nominee pairs), against 3.2% for the same nominees among all other jurors.
- The home effect is about three times the placeholder, and clearly positive in every edition (1.4 to 3.6 by edition). In the 2024-25 ten-pick editions alone it is smaller, 1.75 (1.28 to 2.45).
- The league effect rests on only 23 ballots (5-7 per edition) and its interval nearly touches zero. Treat it as weak evidence of a small positive effect.
- These coefficients describe how ballots differ by juror nationality after controlling for each player's overall standing that year. They are not a causal estimate of juror bias.

## How strongly model strength maps to juror utility

Source: `results/juror_link_ballots.csv`. One unit of the published model's strength is worth k = 1.59 (T1, M1) or 1.54 (T2, M2) units of single-juror utility. The model explains 57% (T1) and 53% (T2) of the within-edition variation in juror-level strengths; the rest, with SD 1.66 (T1) and 1.75 (T2), is race-level uncertainty shared by all jurors in an edition.

## Did the ballot-based simulation beat the placeholder on past editions?

Source: `results/juror_validation_ballots.md`. Leave-one-edition-out, out-of-fold strengths, each edition's real jury and points scheme.

- Yes on the log score of the actual winner: T1 -6.28 against -7.64, T2 -7.22 against -8.27 (higher is better). It was better in 3 of 4 editions and slightly worse in 2024, where both versions gave the winner under 6%. The rule to fall back to the placeholder (worse in 3 or more of 4) was not triggered.
- It reproduces the home-vote share (simulated 9-11% against actual 7-20%); the placeholder gave about 6%.
- Concentration is mixed: both versions are close on the Herfindahl index and winner share, and the ballot version has fewer players receiving points than actually did in 2023 (18.7 simulated against 25 actual).
- Four editions is a small sample. The placeholder's noise level was tuned on 2024-25, which favours it on those two editions.

## How the 2026 forecast changed

Source: `results/forecast_2026_T1_ballots.csv`, `_T2_ballots.csv`, `forecast_2026_comparison_ballots.csv`, `figures/forecast_2026_win_prob_ballots.png`. 10,000 simulations on the real 2026 jury (100 nations).

| Nominee | P(win) T1 original | P(win) T1 ballots | P(win) T2 original | P(win) T2 ballots |
| --- | --- | --- | --- | --- |
| Ousmane Dembélé | 23.2% | 35.8% | 28.4% | 40.0% |
| Harry Kane | 16.2% | 22.8% | 13.5% | 18.0% |
| Lamine Yamal | 14.8% | 20.4% | 14.0% | 18.6% |

- The top three are the same under both versions and both snapshots. The ballot-based forecast is more concentrated on them (combined P(win) 54% to 79% in T1, 56% to 77% in T2) and gives less to the tail (for example Messi 4.2% to 3.0% in T1).
- Dropping 5 random jurors (95 voting) barely changes anything (`forecast_2026_sensitivity_95jurors_ballots.csv`; Dembélé 35.8% to 35.9% in T1).
- The ballot-based forecast has a better past record than the placeholder, but it is still a forecast from four editions of ballots; uncertainty in the model strengths, the link and the bias coefficients is carried into the simulation, and none of it is tuned to move the 2026 result.
