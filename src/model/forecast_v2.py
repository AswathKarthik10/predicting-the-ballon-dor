"""2026 forecast from the ten-edition model (experiments spec section 9.1-9.2).

The forecast uses features and ballots only. `refuse_if_result_present` stops the run if a file holding the 2026 result is in the repo."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ballots as bl
from src.model import ballots_v2 as b2
from src.model import jury_v2 as jv
from src.model import link
from src.model.data import EDITION_COL, T1_FILE, V2_DIR, StudyData
from src.model.explain import relative_contributions
from src.model.forecast_ballots import BallotParams
from src.model.jurors import (COUNTRY_FILE, LEAGUE_FILE, NATIONS_FILE, PLAYER_FILE, PlayerAttributes, RealJuror, SimInputs,
                              build_real_sim_inputs, edition_players, league_country_map, load_juror_inputs_from)

RESULT_FILE_PATTERNS: tuple[str, ...] = ("*bdo_2026_result*", "*2026_result*")
HOLDOUT_DIR = Path("data") / "holdout"
EXCLUDED_DIRS: tuple[str, ...] = (".venv", ".git")
FEATURE_LABELS_V2: dict[str, str] = {
    "F1_attack": "attacking output (non-penalty goals + assists per 90)", "F2_form_rating": "match ratings",
    "F3_big_stage": "big-stage output (UCL knockout, tournament, finals)", "F4_major_titles": "major titles (UCL / World Cup / continental)",
    "F5_other_honors": "other honours (league, cups)", "F6_team_strength": "team strength (club rating, UCL and tournament stage)",
    "F7_prior_standing": "prior Ballon d'Or standing", "F8_defending_winner": "being the defending winner",
    "F9_fame": "baseline fame (pre-season Wikipedia attention)", "F10_season_attention": "season attention beyond what performance explains",
    "F11_individual_awards": "individual awards (player of the year, golden ball, ...)", "F12_market_value": "market value",
    "is_defender": "playing as a defender", "is_goalkeeper": "playing as a goalkeeper",
}


def refuse_if_result_present(root: Path) -> None:
    """Lock-in guard (spec 9.2): the forecast must be committed before any file with the 2026 result enters the repo."""
    found = [p for pattern in RESULT_FILE_PATTERNS for p in root.rglob(pattern) if not any(part in EXCLUDED_DIRS for part in p.parts)]
    if found or (root / HOLDOUT_DIR).exists():
        raise RuntimeError(f"2026 result file present, refusing to forecast: {[str(p) for p in found]} {(root / HOLDOUT_DIR)}")


def real_jury_v2(root: Path, edition: int) -> list[RealJuror]:
    nations = pd.read_parquet(root / V2_DIR / NATIONS_FILE)
    rows = nations[(nations["edition"] == edition) & (nations["voted"] != False)].sort_values("country_key")  # noqa: E712
    return [RealJuror(country_key=c, confederation=f) for c, f in zip(rows["country_key"], rows["confederation"])]


def player_attributes_v2(root: Path, edition: int) -> PlayerAttributes:
    rows = edition_players(load_juror_inputs_from(root / V2_DIR / T1_FILE), edition)
    players = pd.read_parquet(root / V2_DIR / PLAYER_FILE).set_index("player_key")["nationality_country_key"]
    confed = pd.read_parquet(root / V2_DIR / COUNTRY_FILE).set_index("country_key")["confederation"]
    league_map = league_country_map(pd.read_parquet(root / V2_DIR / LEAGUE_FILE))
    nationality = rows["player_key"].map(players)
    league = rows["league_key"].astype(int).astype(str).map(league_map)
    if nationality.isna().any() or league.isna().any():
        raise ValueError(f"edition {edition}: nominees without a nationality or league country")
    return PlayerAttributes(player_keys=tuple(rows["player_key"]), nationality_country=tuple(nationality),
                            confederation=tuple(nationality.map(confed)), league_country=tuple(league))


def inputs_2026_v2(root: Path, study: StudyData) -> SimInputs:
    attributes = player_attributes_v2(root, study.forecast_edition)
    expected = tuple(study.frame_t1.loc[study.frame_t1[EDITION_COL] == study.forecast_edition, "player_key"])
    if attributes.player_keys != expected:
        raise ValueError("2026 player order differs between the juror inputs and the study frame")
    return build_real_sim_inputs(real_jury_v2(root, study.forecast_edition), attributes)


def era_alpha_table(ballots: pd.DataFrame) -> tuple[pd.DataFrame, bl.BiasFit]:
    """Ballot-model player strengths for all nine editions, each from the fit of its own jury era, and the 2022-2025 fit
    whose bias terms drive the simulation (spec 9.2)."""
    pieces: list[pd.DataFrame] = []
    season_fit: bl.BiasFit | None = None
    for editions in (jv.CALENDAR_ERA, jv.SEASON_ERA):
        data = jv.build_data(ballots, editions, b2.BASE_INDICATORS)
        fit = bl.fit_bias_model(data, bl.RIDGE_ALPHA)
        pieces.append(link.alpha_by_player(fit, data))
        season_fit = fit if editions == jv.SEASON_ERA else season_fit
    if season_fit is None:
        raise ValueError("no 2022-2025 fit")
    return pd.concat(pieces, ignore_index=True), season_fit


def pooled_alpha_table(ballots: pd.DataFrame) -> tuple[pd.DataFrame, bl.BiasFit]:
    data = jv.build_data(ballots, b2.BALLOT_EDITIONS_V2, b2.BASE_INDICATORS)
    fit = bl.fit_bias_model(data, bl.RIDGE_ALPHA)
    return link.alpha_by_player(fit, data), fit


def ballot_params(alphas: pd.DataFrame, bias_fit: bl.BiasFit, strengths: pd.DataFrame) -> BallotParams:
    return BallotParams(beta=bias_fit.beta, beta_covariance=bias_fit.covariance,
                        link_fit=link.fit_link(alphas, strengths, "T1", b2.BALLOT_EDITIONS_V2))


def explanation_lines(table: pd.DataFrame, feature_columns: list[str], top_n: int, n_terms: int) -> list[str]:
    rel = relative_contributions(table)
    lines: list[str] = []
    for i in range(top_n):
        row = table.iloc[i]
        contrib = rel.iloc[i][feature_columns].sort_values()
        up = contrib[contrib > 0].sort_values(ascending=False).head(n_terms)
        down = contrib[contrib < 0].head(n_terms)
        fmt = lambda s: "; ".join(f"{FEATURE_LABELS_V2.get(f, f)} ({v:+.2f})" for f, v in s.items()) or "none"  # noqa: E731
        lines.append(f"- **#{int(row['rank_by_utility'])} {row['player_name']}**: utility {row['utility']:+.2f} "
                     f"(plain first-pick probability {row['first_pick_probability']:.1%}). Pushing up: {fmt(up)}. Pulling down: {fmt(down)}.")
    return lines


def compare_with_v1(v2: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    columns = ["player_key", "p_win", "p_top3", "expected_rank", "forecast_rank"]
    merged = v2[["player_key", "player_name", *columns[1:]]].merge(v1[columns], on="player_key", suffixes=("_v2", "_v1"), validate="1:1")
    for metric in ("p_win", "p_top3", "expected_rank", "forecast_rank"):
        merged[f"{metric}_change"] = merged[f"{metric}_v2"] - merged[f"{metric}_v1"]
    return merged.sort_values("forecast_rank_v2").reset_index(drop=True)


def top_probability(table: pd.DataFrame) -> float:
    return float(np.max(table["p_win"]))
