"""Stage 4, jury experiments on the 2016-2025 ballots (experiments spec sections 7.1-7.3). No 2026 data is used."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ballots as bl
from src.model import ballots_v2 as b2
from src.model import jury_v2 as jv
from src.model.data import sha256_of

OUT_DIR = Path("results") / "v2"
N_BOOT_MAIN = 500
N_BOOT_EXPLORATORY = 200
HOME_ERA_COLUMN = b2.HOME_ERA_COLUMN
V1_ESTIMATES = Path("results") / "juror_bias_estimates_ballots.csv"
V1_HOME_TOLERANCE = 1e-6
CONFED_HOME_COLUMNS: tuple[str, ...] = ("home_uefa", "home_conmebol", "home_other", "confed", "league")
TIER_HOME_COLUMNS: tuple[str, ...] = ("home_top_tier", "home_not_top_tier", "confed", "league")


def log(message: str, started: float) -> None:
    print(f"[{time.time() - started:6.0f}s] {message}", flush=True)


def check_v1_reproduction(root: Path, season_table: pd.DataFrame) -> pd.DataFrame:
    """Section 13: fitted on 2022-2025 only, the ballot model must reproduce v1's coefficients (home 2.31, 90% CI 1.88-2.80)."""
    v1 = pd.read_csv(root / V1_ESTIMATES, comment="#")
    v1 = v1[v1.spec == "main"].set_index("indicator")
    new = season_table.set_index("indicator")
    out = pd.DataFrame({"v1_beta": v1.beta, "v2_beta": new.beta, "v1_boot_lo90": v1.boot_lo90, "v2_boot_lo90": new.boot_lo90,
                        "v1_boot_hi90": v1.boot_hi90, "v2_boot_hi90": new.boot_hi90}).loc[list(b2.BASE_INDICATORS)]
    if not np.allclose(out.v1_beta, out.v2_beta, atol=V1_HOME_TOLERANCE):
        raise ValueError(f"2022-2025 fit does not reproduce v1:\n{out}")
    return out.reset_index()


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    ballots = b2.load_ballots_v2(root)
    attention = b2.read_tables_v2(root).attention
    all_eds = b2.BALLOT_EDITIONS_V2
    base = b2.BASE_INDICATORS

    pooled, pooled_fit, pooled_boot = jv.fit_spec(ballots, "pooled", all_eds, base, N_BOOT_MAIN)
    calendar, calendar_fit, calendar_boot = jv.fit_spec(ballots, "by era", jv.CALENDAR_ERA, base, N_BOOT_MAIN)
    season, season_fit, season_boot = jv.fit_spec(ballots, "by era", jv.SEASON_ERA, base, N_BOOT_MAIN)
    reproduction = check_v1_reproduction(root, season)
    reproduction.to_csv(out / "jury_v1_reproduction.csv", index=False)
    log(f"pooled and era fits done; v1 reproduced (home {season.beta.iloc[0]:.4f})", started)

    interaction_indicators = (*base, HOME_ERA_COLUMN)
    interaction, interaction_fit, interaction_boot = jv.fit_spec(ballots, "era interaction", all_eds, interaction_indicators, N_BOOT_MAIN)
    after = jv.home_after_reform(interaction_fit, interaction_boot, interaction_indicators)
    by_edition = pd.concat([jv.fit_spec(ballots, "by edition", (e,), base, N_BOOT_EXPLORATORY)[0] for e in all_eds], ignore_index=True)
    pd.concat([pooled, calendar, season, interaction, after, by_edition], ignore_index=True).to_csv(out / "jury_bias_by_era.csv", index=False)
    log("era interaction and per-edition fits done", started)

    places = pd.concat([jv.places_gained(pooled_fit, pooled_boot, "pooled 2016-2025"), jv.places_gained(calendar_fit, calendar_boot, "2016-2021"),
                        jv.places_gained(season_fit, season_boot, "2022-2025")], ignore_index=True)
    places.to_csv(out / "jury_places_gained.csv", index=False)
    jv.descriptive_by_era(ballots).to_csv(out / "jury_descriptive_by_era.csv", index=False)
    log("places gained and descriptive rates done", started)

    confed_fit = jv.fit_spec(ballots, "home by player confederation (exploratory)", all_eds, CONFED_HOME_COLUMNS, N_BOOT_EXPLORATORY)[0]
    tier_fit = jv.fit_spec(ballots, "home by player tier (exploratory)", all_eds, TIER_HOME_COLUMNS, N_BOOT_EXPLORATORY)[0]
    pd.concat([confed_fit, tier_fit], ignore_index=True).to_csv(out / "jury_bias_exploratory.csv", index=False)
    log("exploratory splits done", started)

    attention_results, coverage = jv.local_attention_fits(ballots, attention, N_BOOT_EXPLORATORY)
    full_reference = pooled.assign(spec="full fit, all jurors (reference)")
    pd.concat([attention_results, full_reference], ignore_index=True).to_csv(out / "jury_local_attention.csv", index=False)
    coverage.to_csv(out / "jury_local_attention_coverage.csv", index=False)
    log("local attention done", started)
    (out / "jury_inputs.txt").write_text("\n".join(f"sha256 {n}={sha256_of(root / 'data' / 'v2' / n)}" for n in
                                                   ("fact_bdo_country_votes.parquet", "fact_juror_nations.parquet", "fact_juror_player_attention.parquet")) + "\n")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
