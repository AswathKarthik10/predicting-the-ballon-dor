# Validation of the ballot-based juror simulation on 2022-2025

<!-- sha256_t1=249bb1ac9a623070c0573e5f2f40d1536f0122d86ad5f33a91ce170363e64f31 -->
<!-- sha256_fact_bdo_country_votes=2cdfaa58a5ba83505eedfa9366add2dcf8d8cd2bb03b5ecd555f3c73bbef5920 -->
<!-- sha256_fact_juror_nations=3cdd55cd1e1c86ffd0e4395ae20c94c3e2236fac11abcbf70ce5bbfb258070bb -->
<!-- sha256_dim_country=ed8fa9634bd088cfef5783212f2e2bb8db6050e8e949e4f40211ec2064f3083f -->
<!-- sha256_dim_player=f5fe0f6a9d0fe7c02a6dcd34d8e7ae93d2d4a9b9c0f31eef1b766cc71a9b0d90 -->
<!-- sha256_fact_bdo_results=f6ecfe365fd0a2b9f51d7023077938a83417735ae1c2419a9709d6a491ad86a2 -->
<!-- sha256_dim_league_context=abc18fc4e497399d62090fbe7b6e1b725d7ebbcaff8d1cf49db283e75b81a239 -->
<!-- sha256_dim_club=ad7268f198eb21a7718488f242587d01c0c670add3afb3ddaa29a9648c5d503a -->
<!-- git_commit=9fb34fc -->
<!-- seed=7 -->
<!-- n_simulations=10000 seed=7 leave_one_edition_out=true -->

Each edition is simulated with out-of-fold model strengths, that edition's real jury and points scheme, and bias, link and noise parameters estimated from the other three editions' ballots. The placeholder column is the original simulation (reconstructed panel, b = 0.8/0.2/0.2, tau_juror from the 2024-25 concentration calibration) run under the same points scheme. With four editions each metric is a handful of observations, so differences are suggestive, not conclusive.

## T1 (M1)

Sum of log P(actual winner): ballots -6.283, placeholder -7.637. Accepted: the ballot simulation is worse on log score in 1 of 4 editions (rule: reject at 3 or more).

### Winner probability and expected rank

| edition | p_actual_winner [ballots] | p_actual_winner [placeholder] | log_p_actual_winner [ballots] | log_p_actual_winner [placeholder] | winner_expected_rank [ballots] | winner_expected_rank [placeholder] |
|---|---|---|---|---|---|---|
| 2022 | 0.591 | 0.369 | -0.526 | -0.997 | 1.931 | 2.486 |
| 2023 | 0.210 | 0.144 | -1.562 | -1.939 | 4.180 | 5.107 |
| 2024 | 0.052 | 0.056 | -2.964 | -2.884 | 8.684 | 8.762 |
| 2025 | 0.292 | 0.162 | -1.230 | -1.818 | 4.112 | 4.996 |

### Concentration (simulated mean vs actual)

| edition | winner_share_sim [ballots] | winner_share_sim [placeholder] | winner_share [actual] |
|---|---|---|---|
| 2022 | 0.290 | 0.298 | 0.369 |
| 2023 | 0.282 | 0.290 | 0.314 |
| 2024 | 0.188 | 0.190 | 0.176 |
| 2025 | 0.182 | 0.188 | 0.206 |

| edition | n_with_points_sim [ballots] | n_with_points_sim [placeholder] | n_with_points [actual] |
|---|---|---|---|
| 2022 | 23.105 | 22.413 | 19.000 |
| 2023 | 18.674 | 21.094 | 25.000 |
| 2024 | 27.302 | 28.292 | 28.000 |
| 2025 | 26.750 | 28.071 | 30.000 |

| edition | herfindahl_sim [ballots] | herfindahl_sim [placeholder] | herfindahl [actual] |
|---|---|---|---|
| 2022 | 0.166 | 0.174 | 0.194 |
| 2023 | 0.170 | 0.167 | 0.199 |
| 2024 | 0.103 | 0.104 | 0.106 |
| 2025 | 0.101 | 0.101 | 0.109 |

### Home-vote check: share of compatriot jurors ranking their own nominee 1st

| edition | home_first_share_sim [ballots] | home_first_share_sim [placeholder] | home_first_share [actual] |
|---|---|---|---|
| 2022 | 0.092 | 0.062 | 0.100 |
| 2023 | 0.092 | 0.064 | 0.100 |
| 2024 | 0.114 | 0.063 | 0.067 |
| 2025 | 0.106 | 0.063 | 0.200 |

## T2 (M2)

Sum of log P(actual winner): ballots -7.215, placeholder -8.267. Accepted: the ballot simulation is worse on log score in 1 of 4 editions (rule: reject at 3 or more).

### Winner probability and expected rank

| edition | p_actual_winner [ballots] | p_actual_winner [placeholder] | log_p_actual_winner [ballots] | log_p_actual_winner [placeholder] | winner_expected_rank [ballots] | winner_expected_rank [placeholder] |
|---|---|---|---|---|---|---|
| 2022 | 0.424 | 0.270 | -0.858 | -1.308 | 2.989 | 3.317 |
| 2023 | 0.244 | 0.169 | -1.411 | -1.781 | 3.984 | 4.581 |
| 2024 | 0.033 | 0.042 | -3.411 | -3.165 | 9.160 | 9.183 |
| 2025 | 0.216 | 0.134 | -1.534 | -2.013 | 5.569 | 5.910 |

### Concentration (simulated mean vs actual)

| edition | winner_share_sim [ballots] | winner_share_sim [placeholder] | winner_share [actual] |
|---|---|---|---|
| 2022 | 0.266 | 0.290 | 0.369 |
| 2023 | 0.286 | 0.291 | 0.314 |
| 2024 | 0.195 | 0.194 | 0.176 |
| 2025 | 0.180 | 0.187 | 0.206 |

| edition | n_with_points_sim [ballots] | n_with_points_sim [placeholder] | n_with_points [actual] |
|---|---|---|---|
| 2022 | 24.645 | 23.728 | 19.000 |
| 2023 | 18.681 | 20.956 | 25.000 |
| 2024 | 26.451 | 27.705 | 28.000 |
| 2025 | 26.918 | 28.255 | 30.000 |

| edition | herfindahl_sim [ballots] | herfindahl_sim [placeholder] | herfindahl [actual] |
|---|---|---|---|
| 2022 | 0.145 | 0.165 | 0.194 |
| 2023 | 0.173 | 0.168 | 0.199 |
| 2024 | 0.107 | 0.106 | 0.106 |
| 2025 | 0.100 | 0.100 | 0.109 |

### Home-vote check: share of compatriot jurors ranking their own nominee 1st

| edition | home_first_share_sim [ballots] | home_first_share_sim [placeholder] | home_first_share [actual] |
|---|---|---|---|
| 2022 | 0.106 | 0.064 | 0.100 |
| 2023 | 0.089 | 0.063 | 0.100 |
| 2024 | 0.102 | 0.061 | 0.067 |
| 2025 | 0.110 | 0.063 | 0.200 |

## Caveats

- The placeholder tau_juror was calibrated on the 2024-25 concentration statistics, so the placeholder is in-sample for those editions.
- Out-of-fold strengths for the other editions used to fit the link were produced by models that saw the held-out edition's results; the held-out edition's own ballots and strengths never enter its parameters.
