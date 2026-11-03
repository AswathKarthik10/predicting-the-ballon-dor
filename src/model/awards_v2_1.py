"""F11 (v2.1): a weighted award score (awards fix spec v2.1, sections 1-2). Weights are fixed by the spec and are not tuned.

award_score = sum over award events of type_w x place_w x difficulty, one event per competition-season (the best kept);
F11 = clip(z(log1p(award_score)) within edition, -3, 3). Reads only the award events table, never an outcome column."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TYPE_WEIGHT: dict[str, float] = {"best_player": 1.0, "top_scorer": 0.6, "other": 0.3}
PLACE_WEIGHT: dict[int, float] = {1: 1.0, 2: 0.5, 3: 0.25}
F11_CLIP = 3.0
WORLD_CUP = "world_cup"
LEAGUE_PREFIX = "league_"
LEAGUE_SCORER_SOURCE = "league_top_scorers.csv"
SCORE_COLUMN = "award_score_raw"
BEST_IS_WC_COLUMN = "award_best_is_wc"
EVENTS_FILE = "fact_award_events.parquet"
KEYS: list[str] = ["player_key", "edition"]
SEASON_END_YEAR_DIGITS = 2
SEASON_END_YEAR_BASE = 2000


@dataclass(frozen=True)
class AwardVariant:
    """The main v2.1 rule and the sensitivity switches S1-S4 (spec B4)."""
    name: str
    league_top_scorers: bool
    uniform_difficulty: bool
    log_and_clip: bool


MAIN_AWARDS = AwardVariant("v2.1", True, False, True)
S1_NO_LEAGUE_SCORERS = AwardVariant("S1", False, False, True)
S2_UNIFORM_DIFFICULTY = AwardVariant("S2", True, True, True)
S4_RAW_SCORE = AwardVariant("S4", True, False, False)


def season_end_year(label: str) -> int:
    """'25/26', '2025/26' -> 2026; '2026' -> 2026."""
    if "/" in label:
        return SEASON_END_YEAR_BASE + int(label.split("/")[1][-SEASON_END_YEAR_DIGITS:])
    return int(label)


def event_scores(events: pd.DataFrame, variant: AwardVariant) -> pd.DataFrame:
    """One row per award event with its score and a competition-season key."""
    out = events.copy()
    if not variant.league_top_scorers:
        out = out[out["source"] != LEAGUE_SCORER_SOURCE]
    unknown = set(out["award_type"]) - set(TYPE_WEIGHT)
    if unknown or not set(out["place"]) <= set(PLACE_WEIGHT):
        raise ValueError(f"award events with an unknown type {sorted(unknown)} or place {sorted(set(out['place']))}")
    difficulty = 1.0 if variant.uniform_difficulty else out["difficulty"]
    out["score"] = out["award_type"].map(TYPE_WEIGHT) * out["place"].map(PLACE_WEIGHT) * difficulty
    out["competition_season"] = out["competition"] + "|" + out["season"].map(season_end_year).astype(str)
    return out


def best_per_competition_season(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep the highest-scoring event of each player's competition-season (ties: first by award type name)."""
    ordered = scored.sort_values(KEYS + ["competition_season", "score", "award_type"], ascending=[True, True, True, False, True])
    return ordered.drop_duplicates(KEYS + ["competition_season"], keep="first")


def award_scores(events: pd.DataFrame, variant: AwardVariant) -> pd.DataFrame:
    """Per player-edition: award_score_raw and whether the best single event comes from a World Cup."""
    kept = best_per_competition_season(event_scores(events, variant))
    total = kept.groupby(KEYS).score.sum().rename(SCORE_COLUMN)
    top = kept.sort_values(KEYS + ["score", "competition"], ascending=[True, True, False, True]).drop_duplicates(KEYS)
    best_wc = (top.set_index(KEYS).competition == WORLD_CUP).astype(float).rename(BEST_IS_WC_COLUMN)
    return pd.concat([total, best_wc], axis=1).reset_index()


def attach_award_columns(t1: pd.DataFrame, events: pd.DataFrame, variant: AwardVariant) -> pd.DataFrame:
    """T1 rows with the raw award score and the World Cup indicator (0 for players without events)."""
    scores = award_scores(events, variant)
    merged = t1.merge(scores, on=KEYS, how="left", validate="1:1")
    merged[SCORE_COLUMN] = merged[SCORE_COLUMN].fillna(0.0)
    merged[BEST_IS_WC_COLUMN] = merged[BEST_IS_WC_COLUMN].fillna(0.0)
    return merged


def f11_values(score: pd.Series, editions: pd.Series, variant: AwardVariant, zscore) -> pd.Series:  # noqa: ANN001
    """F11 from the raw score: log1p, z within edition, clipped to [-3, 3]; S4 uses the plain z-score of the raw score."""
    if not variant.log_and_clip:
        return zscore(score, editions, "award_score_raw")
    z = zscore(pd.Series(np.log1p(score), index=score.index), editions, "award_score_log")
    return z.clip(-F11_CLIP, F11_CLIP)
