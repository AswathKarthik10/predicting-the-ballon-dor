# Juror nationality effects estimated from real ballots (2022-2025)

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
<!-- ridge_alpha=0.01 n_bootstrap=500 spec=ballot Plackett-Luce, alpha per player-edition -->

These coefficients describe how ballots differ by juror nationality *after* controlling for each nominee's overall standing that year. They are not a causal estimate of juror bias.

## Coefficients (log-odds scale, main fit)

| indicator | beta | se_laplace | boot_lo90 | boot_hi90 | n_ballots | n_pairs |
|---|---|---|---|---|---|---|
| home | 2.311 | 0.172 | 1.880 | 2.804 | 66 | 120 |
| confed | 0.366 | 0.064 | 0.244 | 0.488 | 313 | 3897 |
| league | 0.478 | 0.230 | 0.028 | 0.946 | 23 | 120 |

## What they mean in places (a nominee a neutral juror would rank 5th)

Places gained on one juror's ballot when the juror is a compatriot / same-continent / same league country. Simulated from the fitted model with each nominee's strength held fixed; interval is the 90% bootstrap range.

| indicator | beta | places_gained | lo90 | hi90 |
|---|---|---|---|---|
| home | 2.311 | 2.855 | 2.562 | 3.161 |
| confed | 0.366 | 0.661 | 0.431 | 0.875 |
| league | 0.478 | 0.842 | -0.008 | 1.585 |

## Descriptive check, no model

Share of flagged jurors who ranked the nominee 1st, against the share of all other jurors who did, pooled over nominees with at least one flagged juror (weighted by number of flagged jurors).

| indicator | n_nominees | n_flagged_jurors | share_flagged_rank_first | share_others_rank_first |
|---|---|---|---|---|
| home | 120 | 120 | 0.117 | 0.032 |
| confed | 120 | 3897 | 0.033 | 0.031 |
| league | 120 | 120 | 0.042 | 0.033 |
