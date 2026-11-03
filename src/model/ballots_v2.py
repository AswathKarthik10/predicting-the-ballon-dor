"""Ten-edition ballot table and jury experiments (experiments spec section 7).

Builds on the v1 ballot-level Plackett-Luce in ballots.py. Ballots exist for 2016-2025; 2026 has no ballots."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.model import ballots as bl
from src.model.data import EDITION_FACTS, T1_FILE, V2_DIR
from src.model.jurors import (COUNTRY_FILE, LEAGUE_FILE, NATIONS_FILE, PLAYER_FILE, RESULTS_FILE, VOTES_FILE,
                              load_juror_inputs_from, league_country_map)

ATTENTION_FILE = "fact_juror_player_attention.parquet"
BALLOT_EDITIONS_V2: tuple[int, ...] = (2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025)
FIRST_FIFA_TOP100_EDITION = 2022  # the jury was cut from about 175 nations to the FIFA top 100
SCHEME_3: tuple[int, ...] = (5, 3, 1)
TOP_TIER_PLACES = 3
BASE_INDICATORS: tuple[str, ...] = bl.INDICATORS
MIN_COVERED_NOMINEES = 15  # a juror-country cell is "covered" when its attention share exists for at least this many nominees
HOME_ERA_COLUMN = "home_x_era2022"
LOCAL_ATTENTION_COLUMN = "local_attention"
CONFED_GROUPS: tuple[str, ...] = ("UEFA", "CONMEBOL")
OTHER_GROUP = "other"


def scheme_for(edition: int) -> tuple[int, ...]:
    picks = EDITION_FACTS[edition].ballot_picks
    return {3: SCHEME_3, 5: bl.SCHEME_5, 10: bl.SCHEME_10}[picks]


@dataclass(frozen=True)
class JuryTablesV2:
    tables: bl.JuryTables
    attention: pd.DataFrame


def read_tables_v2(root: Path) -> JuryTablesV2:
    directory = root / V2_DIR
    read = lambda name: pd.read_parquet(directory / name)  # noqa: E731
    tables = bl.JuryTables(votes=read(VOTES_FILE), nations=read(NATIONS_FILE), countries=read(COUNTRY_FILE),
                           players=read(PLAYER_FILE), results=read(RESULTS_FILE), leagues=read(LEAGUE_FILE))
    return JuryTablesV2(tables=tables, attention=read(ATTENTION_FILE))


def validate_ballots_v2(ballots: pd.DataFrame, tables: bl.JuryTables) -> None:
    """Spec section 2 checks: points per player match the official result, every ballot uses its scheme exactly once,
    the ballot count equals the juror count, and every voter is a listed juror."""
    summed = ballots.groupby(["edition", "player_key"]).points.sum()
    official = tables.results[tables.results.edition.isin(BALLOT_EDITIONS_V2)].set_index(["edition", "player_key"]).points
    joined = pd.concat([summed.rename("ballots"), official.rename("official")], axis=1)
    bad = joined[joined.ballots.fillna(-1) != joined.official.fillna(-1)]
    if len(bad):
        raise ValueError(f"ballot totals differ from fact_bdo_results for {len(bad)} player-editions:\n{bad.head(10)}")
    for (edition, juror), group in ballots.groupby(["edition", "juror_country_key"]):
        picked = sorted(group.points[group.points > 0], reverse=True)
        if picked != sorted(scheme_for(int(edition)), reverse=True):
            raise ValueError(f"ballot {edition}/{juror} does not use the {len(scheme_for(int(edition)))}-pick scheme exactly once: {picked}")
    counts = ballots.groupby("edition").juror_country_key.nunique().to_dict()
    expected = {ed: EDITION_FACTS[ed].jurors for ed in BALLOT_EDITIONS_V2}
    if counts != expected:
        raise ValueError(f"ballot counts per edition {counts} != expected {expected}")
    bl.check_voters_are_jurors(ballots, tables.nations)
    bl.check_indicators_exclusive(ballots)


def add_era_and_tier(ballots: pd.DataFrame, results: pd.DataFrame, countries: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Era flag, home x era interaction, home split by the player's confederation group, and the top-tier flag."""
    out = ballots.copy()
    out["era2022"] = (out.edition >= FIRST_FIFA_TOP100_EDITION).astype(int)
    out[HOME_ERA_COLUMN] = out.home * out.era2022
    nationality = players.set_index("player_key").nationality_country_key
    confed = countries.set_index("country_key").confederation
    player_confed = out.player_key.map(nationality).map(confed)
    group = player_confed.where(player_confed.isin(CONFED_GROUPS), OTHER_GROUP)
    for name in (*CONFED_GROUPS, OTHER_GROUP):
        out[f"home_{name.lower()}"] = out.home * (group == name).astype(int)
    results = results[results.edition.isin(BALLOT_EDITIONS_V2)].copy()
    results["official_rank"] = results.groupby("edition").points.rank(ascending=False, method="min")
    top = results[results.official_rank <= TOP_TIER_PLACES][["edition", "player_key"]].assign(top_tier=1)
    out = out.merge(top, on=["edition", "player_key"], how="left")
    out["top_tier"] = out.top_tier.fillna(0).astype(int)
    out["home_top_tier"] = out.home * out.top_tier
    out["home_not_top_tier"] = out.home * (1 - out.top_tier)
    return out


def load_ballots_v2(root: Path) -> pd.DataFrame:
    jury = read_tables_v2(root)
    inputs = load_juror_inputs_from(root / V2_DIR / T1_FILE)
    ballots = bl.build_ballot_table(jury.tables, inputs)
    validate_ballots_v2(ballots, jury.tables)
    return add_era_and_tier(ballots, jury.tables.results, jury.tables.countries, jury.tables.players)


def local_attention_table(attention: pd.DataFrame, share_column: str) -> pd.DataFrame:
    """Per (edition, juror country, player): z-score of the attention share within the juror-country cell, for covered
    cells only. A nominee below the view threshold has no share; inside a covered cell his share is set to 0."""
    frame = attention[["edition", "country_key", "player_key", share_column]].copy()
    present = frame.groupby(["edition", "country_key"])[share_column].transform("count")
    covered = frame[present >= MIN_COVERED_NOMINEES].copy()
    covered["share"] = covered[share_column].fillna(0.0)
    cell = covered.groupby(["edition", "country_key"]).share
    sd = cell.transform("std", ddof=1)
    covered[LOCAL_ATTENTION_COLUMN] = ((covered.share - cell.transform("mean")) / sd).where(sd > 0, 0.0)
    return covered.rename(columns={"country_key": "juror_country_key"})[["edition", "juror_country_key", "player_key", LOCAL_ATTENTION_COLUMN]]


def attach_local_attention(ballots: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """Ballot rows of covered juror-editions only, with the local attention column."""
    return ballots.merge(local, on=["edition", "juror_country_key", "player_key"], how="inner", validate="1:1")


def coverage_table(ballots: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    """Per edition: covered jurors, share of jurors, and share of ballot points cast in covered cells."""
    covered_keys = local[["edition", "juror_country_key"]].drop_duplicates().assign(covered=1)
    marked = ballots.merge(covered_keys, on=["edition", "juror_country_key"], how="left")
    marked["covered"] = marked.covered.fillna(0)
    jurors = marked.drop_duplicates(["edition", "juror_country_key"]).groupby("edition").agg(jurors=("covered", "size"), covered_jurors=("covered", "sum"))
    points = marked.groupby("edition").apply(lambda g: pd.Series({"share_of_points": g.loc[g.covered == 1, "points"].sum() / g.points.sum()}),
                                            include_groups=False)
    out = jurors.join(points)
    out["share_of_jurors"] = out.covered_jurors / out.jurors
    return out.reset_index()
