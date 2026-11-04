# Deviations, Experiment 3

Nothing decided after seeing results yet. Entries below are exploratory.

- E3-A is recomputed through the same code path only to supply held-out utilities for the paired bootstrap; its per-fold metrics are asserted equal to results/v2_1 (max abs gap < 1e-6) and are not treated as a new fit.
- metrics_summary_v3.csv initially averaged E3-A over all nine folds. Fixed: the summary now has a `subset` column (season4, calendar5) and each group is averaged on its own. Regenerated from metrics_by_fold_v3.csv; no result changed.
- Section 4 (exploratory, post-hoc): the published 2026 top ten is a constant in src/model/compare_2026_v3.py because data/holdout/ does not exist and creating it would trip forecast_v2.refuse_if_result_present. The forecast code (run_v3_forecast.py) never reads it.
- Section 4 link: ballot_params hardcodes the nine-edition link, so run_v3_forecast.py builds BallotParams directly with link.fit_link on 2022-2025 and out-of-fold strengths from 3-edition fits.
