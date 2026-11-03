from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.awards_v2_1 import (BEST_IS_WC_COLUMN, F11_CLIP, MAIN_AWARDS, S1_NO_LEAGUE_SCORERS, S2_UNIFORM_DIFFICULTY, SCORE_COLUMN,
                                   award_scores, best_per_competition_season, event_scores, f11_values)
from src.model.data import V2_SPEC, StudyData, load_study_v2, load_study_v2_1
from src.model.features import zscore_within_edition
from src.model.features_v2 import MAIN_OPTIONS, V2_FEATURES

ROOT = Path(__file__).resolve().parents[1]
EVENTS = pd.read_parquet(ROOT / "data" / "v2_1" / "fact_award_events.parquet")
PLAYERS = pd.read_parquet(ROOT / "data" / "v2" / "dim_player.parquet").set_index("player_key").wiki_title
SNAPSHOT_DATES = pd.read_csv(ROOT / "data" / "v2" / "player_snapshot_t1.csv", usecols=["edition", "snapshot_date"]).drop_duplicates()
MAIN_V2_1_OPTIONS = MAIN_OPTIONS.__class__(MAIN_OPTIONS.club_attack_column, MAIN_OPTIONS.intl_attack_column, MAIN_OPTIONS.league_award_column, MAIN_AWARDS)
TOLERANCE = 1e-12


def score_of(name: str, edition: int, variant=MAIN_AWARDS) -> float:
    keys = PLAYERS[PLAYERS == name].index
    scores = award_scores(EVENTS, variant)
    hit = scores[(scores.player_key.isin(keys)) & (scores.edition == edition)]
    return float(hit[SCORE_COLUMN].iloc[0]) if len(hit) else 0.0


def event(competition: str, season: str, award_type: str, place: int, difficulty: float, player: str = "a") -> dict:
    return {"player_key": player, "edition": 2026, "competition": competition, "season": season, "award_type": award_type,
            "place": place, "difficulty": difficulty, "announced_date": "2026-07-01", "source": "fact_tournament_awards"}


def test_rodri_gold_ball_outscores_messi_silver_on_world_cup_awards_alone() -> None:
    world_cup = EVENTS[(EVENTS.competition == "world_cup") & (EVENTS.edition == 2026)]
    assert score_of("Rodri", 2026) >= 1.0
    rodri = award_scores(world_cup, MAIN_AWARDS)
    keys = {n: PLAYERS[PLAYERS == n].index[0] for n in ("Rodri", "Lionel Messi")}
    value = rodri.set_index("player_key")[SCORE_COLUMN]
    assert value[keys["Rodri"]] > value[keys["Lionel Messi"]]


def test_mbappe_world_cup_counts_once() -> None:
    key = PLAYERS[PLAYERS == "Kylian Mbappé"].index[0]
    mine = EVENTS[(EVENTS.player_key == key) & (EVENTS.edition == 2026) & (EVENTS.competition == "world_cup")]
    assert len(mine) == 2  # golden ball 3rd and golden boot 1st are both in the table
    kept = best_per_competition_season(event_scores(mine, MAIN_AWARDS))
    assert len(kept) == 1 and kept.score.iloc[0] == pytest.approx(0.6)  # golden boot 1st (0.6) beats golden ball 3rd (0.25)


def test_premier_league_player_of_the_season_outscores_an_mls_mvp() -> None:
    mls = EVENTS[(EVENTS.competition == "league_130") & (EVENTS.edition == 2026) & (EVENTS.award_type == "best_player")]
    pl_events = EVENTS[(EVENTS.competition == "league_47") & (EVENTS.edition == 2026) & (EVENTS.award_type == "best_player")]
    assert event_scores(pl_events, MAIN_AWARDS).score.min() > event_scores(mls, MAIN_AWARDS).score.max()


def test_second_place_scores_half_of_first_place_at_the_same_competition() -> None:
    events = pd.DataFrame([event("world_cup", "2026", "best_player", 1, 1.0, "a"), event("world_cup", "2026", "best_player", 2, 1.0, "b")])
    s = event_scores(events, MAIN_AWARDS).set_index("player_key").score
    assert s["b"] == pytest.approx(0.5 * s["a"])


def test_dedupe_keeps_the_highest_scoring_award_of_a_competition_season() -> None:
    events = pd.DataFrame([event("league_55", "25/26", "best_player", 1, 0.8), event("league_55", "2025/26", "top_scorer", 1, 0.8),
                           event("champions_league", "2025/26", "top_scorer", 1, 1.0)])
    kept = best_per_competition_season(event_scores(events, MAIN_AWARDS))
    assert len(kept) == 2
    assert kept[kept.competition == "league_55"].score.iloc[0] == pytest.approx(0.8)  # season labels 25/26 and 2025/26 are one season


def test_no_event_enters_an_edition_after_its_snapshot_date() -> None:
    dates = SNAPSHOT_DATES.set_index("edition").snapshot_date
    late = EVENTS[pd.to_datetime(EVENTS.announced_date) > EVENTS.edition.map(pd.to_datetime(dates))]
    assert late.empty


def test_uniform_difficulty_and_dropped_league_scorers_change_only_what_they_claim() -> None:
    base = event_scores(EVENTS, MAIN_AWARDS)
    uniform = event_scores(EVENTS, S2_UNIFORM_DIFFICULTY)
    assert np.allclose(uniform.score, base.award_type.map({"best_player": 1.0, "top_scorer": 0.6, "other": 0.3}) * base.place.map({1: 1.0, 2: 0.5, 3: 0.25}))
    no_scorers = event_scores(EVENTS, S1_NO_LEAGUE_SCORERS)
    assert len(no_scorers) < len(base) and (no_scorers.source != "league_top_scorers.csv").all()


def test_f11_is_clipped_and_standardised_within_edition() -> None:
    score = pd.Series([0.0] * 28 + [1.0, 40.0])
    editions = pd.Series([2026] * 30)
    out = f11_values(score, editions, MAIN_AWARDS, zscore_within_edition)
    assert out.abs().max() <= F11_CLIP + TOLERANCE


def test_loaded_study_carries_the_weighted_f11_and_leaves_other_features_unchanged() -> None:
    v2: StudyData = load_study_v2(ROOT)
    v21 = load_study_v2_1(ROOT, V2_SPEC, MAIN_V2_1_OPTIONS, V2_FEATURES)
    assert (v21.frame_t1.player_key.to_numpy() == v2.frame_t1.player_key.to_numpy()).all()
    assert v21.frame_t1.F11_individual_awards.abs().max() <= F11_CLIP + TOLERANCE
    assert not np.allclose(v21.frame_t1.F11_individual_awards, v2.frame_t1.F11_individual_awards)
    for col in ("F1_attack", "F9_fame", "F12_market_value", "F6_team_strength"):
        np.testing.assert_allclose(v21.frame_t1[col], v2.frame_t1[col], atol=TOLERANCE)
    assert BEST_IS_WC_COLUMN in v21.frame_t1.columns
