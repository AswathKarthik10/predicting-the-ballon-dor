"""Ten-edition feature set (experiments spec section 3).

Pure functions only; nothing here reads the outcome columns. Differences from the v1 features:
F1 uses non-penalty goals + assists (every edition has them, xG does not), F3 and F5 lose their award terms,
F11 (individual awards) and F12 (market value) are new, and F10 is residualised on F1-F7 and F11.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model.awards_v2_1 import BEST_IS_WC_COLUMN, SCORE_COLUMN, AwardVariant, f11_values
from src.model.features import (DEFENDER, EDITION_COL, GOALKEEPER, POST_FEATURES, POST_RESIDUALISER_REGRESSORS, PRIOR_RANK_BASE,
                                FeatureParams, FeatureSet, apply_residualiser, build_post_features_for, fit_residualiser,
                                impute_club_rating, require_columns, stage_ordinal, zscore_with_reference, zscore_within_edition)

CALENDAR_FORMAT = "calendar_year"
GOLDEN_BALL_PLACES_COUNTED_HALF: tuple[int, ...] = (2, 3)  # gold is already inside major_individual_awards_n
RUNNER_UP_AWARD_WEIGHT = 0.5

WORLD_CUP = "World Cup"
MAJOR_TOURNAMENTS: tuple[str, ...] = (WORLD_CUP, "EURO", "Copa America")
GOLDEN_BALL_GOLD = 1
POST_FORM_COLUMN_V2 = "post_npga_p90_opp_adj"

REQUIRED_V2_COLUMNS: tuple[str, ...] = (
    "edition", "tourn_name", "position_group", "is_goalkeeper", "eligibility_format", "prev_bdo_rank", "career_bdo_podiums_before",
    "is_defending_winner", "club_minutes", "club_avg_rating", "big_game_rating",
    "ucl_ko_goal_contribs", "intl_minutes", "tourn_ko_goal_contribs", "final_goal_contribs",
    "scored_winner_in_final", "ucl_stage_reached", "club_rating_window_end", "nt_tourn_stage_reached", "won_ucl",
    "won_league", "won_domestic_cup", "won_world_cup", "won_continental", "won_other_uefa", "ucl_back_to_back",
    "major_individual_awards_n", "uefa_player_award", "award_wc_golden_ball", "ea_overall_t1",
    "tm_value_eur_t1", "tm_value_eur_window_start", "wiki_views_pre_juror_w", "wiki_views_in_juror_w",
    "news_vol_pre_juror_w", "news_vol_in_juror_w",
)

V2_CORE_FEATURES: tuple[str, ...] = (
    "F1_attack", "F2_form_rating", "F3_big_stage", "F4_major_titles", "F5_other_honors", "F6_team_strength",
    "F7_prior_standing", "F8_defending_winner", "F9_fame", "F10_season_attention", "F11_individual_awards",
    "F12_market_value",
)
POSITION_FEATURES: tuple[str, ...] = ("is_defender", "is_goalkeeper")
BLOCKS_V2: dict[str, tuple[str, ...]] = {
    "Season output": ("F1_attack", "F2_form_rating"),
    "Big stage": ("F3_big_stage",),
    "Trophies and team": ("F4_major_titles", "F5_other_honors", "F6_team_strength"),
    "Individual awards": ("F11_individual_awards",),
    "Prior standing": ("F7_prior_standing", "F8_defending_winner"),
    "Fame and attention": ("F9_fame", "F10_season_attention"),
    "Market value": ("F12_market_value",),
    "Position": POSITION_FEATURES,
}
RESIDUALISER_REGRESSORS: tuple[str, ...] = V2_CORE_FEATURES[:7] + ("F11_individual_awards",)
FORMAT_INTERACTIONS: dict[str, str] = {
    "calendar_x_fame": "F9_fame",
    "calendar_x_attention": "F10_season_attention",
    "calendar_x_awards": "F11_individual_awards",
}
@dataclass(frozen=True)
class FeatureOptions:
    """Source columns that the robustness runs swap: attacking output (non-penalty goals + assists, or xG-based) and the
    league player-of-the-season award (league-run, or Transfermarkt's own)."""
    club_attack_column: str
    intl_attack_column: str
    league_award_column: str
    awards: AwardVariant | None  # None: the v2 award count; otherwise the weighted score of awards_v2_1


MAIN_OPTIONS = FeatureOptions("club_npga_p90_opp_adj", "intl_npga_p90_opp_adj", "league_player_of_season", None)
XG_OPTIONS = FeatureOptions("club_npxg_xa_p90_opp_adj", "intl_npxg_xa_p90_opp_adj", "league_player_of_season", None)
LEAGUE_POTY_OPTIONS = FeatureOptions("club_npga_p90_opp_adj", "intl_npga_p90_opp_adj", "league_poty", None)
EA_COLUMN = "ea_rating_z"
GATE_COLUMN = "trophy_gate"
TOURNAMENT_INTERACTION = "tourn_ko_x_world_cup"


def attack_raw_npga(df: pd.DataFrame) -> pd.Series:
    return attack_raw_with(df, MAIN_OPTIONS)


def attack_raw_with(df: pd.DataFrame, options: FeatureOptions) -> pd.Series:
    """Minutes-weighted mean of club and international attacking output per 90 (opponent-adjusted)."""
    club = df[options.club_attack_column]
    intl = df[options.intl_attack_column]
    club_weight = df["club_minutes"]
    intl_weight = df["intl_minutes"]
    blended = (club * club_weight + intl * intl_weight) / (club_weight + intl_weight)
    return blended.where(intl.notna(), club)


def market_value_at_t1(df: pd.DataFrame) -> pd.Series:
    """Transfermarkt value at T1; a retired player is valued at 0, so his window-start value stands in."""
    value = df["tm_value_eur_t1"].astype(float)
    return value.where(value > 0, df["tm_value_eur_window_start"].astype(float))


def individual_awards_raw(df: pd.DataFrame) -> pd.Series:
    return individual_awards_raw_with(df, MAIN_OPTIONS)


def individual_awards_raw_with(df: pd.DataFrame, options: FeatureOptions) -> pd.Series:
    runner_up_ball = df["award_wc_golden_ball"].isin(GOLDEN_BALL_PLACES_COUNTED_HALF).astype(float)
    return (df["major_individual_awards_n"].astype(float) + df[options.league_award_column].astype(float)
            + df["uefa_player_award"].astype(float) + RUNNER_UP_AWARD_WEIGHT * runner_up_ball)


def build_base_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    return build_base_features_with(df, MAIN_OPTIONS)


def build_base_features_with(df: pd.DataFrame, options: FeatureOptions) -> pd.DataFrame:
    """Edition-level features F1-F9, F11, F12 plus the unresidualised attention change. Each row is standardised
    only against its own edition."""
    variable_columns = (options.club_attack_column, options.intl_attack_column, options.league_award_column)
    require_columns(df, REQUIRED_V2_COLUMNS + variable_columns, "build_base_features_with")
    ed = df[EDITION_COL]
    is_gk = df["position_group"] == GOALKEEPER
    if not (df["is_goalkeeper"].astype(bool) == is_gk).all():
        raise ValueError("is_goalkeeper disagrees with position_group")

    out = pd.DataFrame(index=df.index)
    out[EDITION_COL] = ed
    out["F1_attack"] = zscore_with_reference(attack_raw_with(df, options), ed, ~is_gk, "attack")

    big_game = df["big_game_rating"].where(df["big_game_rating"].notna(), df["club_avg_rating"])
    form = (zscore_within_edition(df["club_avg_rating"], ed, "club_avg_rating")
            + zscore_within_edition(big_game, ed, "big_game_rating")) / 2
    out["F2_form_rating"] = zscore_within_edition(form, ed, "form_rating")

    big_stage_sum = (
        zscore_within_edition(df["ucl_ko_goal_contribs"].astype(float), ed, "ucl_ko_goal_contribs")
        + zscore_within_edition(df["tourn_ko_goal_contribs"].fillna(0).astype(float), ed, "tourn_ko_goal_contribs")
        + zscore_within_edition(df["final_goal_contribs"].fillna(0).astype(float), ed, "final_goal_contribs")
        + df["scored_winner_in_final"].fillna(False).astype(float)
    )
    out["F3_big_stage"] = zscore_within_edition(big_stage_sum, ed, "big_stage")

    major_titles = df[["won_ucl", "won_world_cup", "won_continental"]].astype(int).sum(axis=1).astype(float)
    other_honors = df[["won_league", "won_domestic_cup", "won_other_uefa", "ucl_back_to_back"]].astype(int).sum(axis=1).astype(float)
    prior_standing = (PRIOR_RANK_BASE - df["prev_bdo_rank"]).fillna(0.0)
    out["major_titles_raw"] = major_titles
    out["other_honors_raw"] = other_honors
    out["prior_standing_raw"] = prior_standing
    out["F4_major_titles"] = zscore_within_edition(major_titles, ed, "major_titles")
    out["F5_other_honors"] = zscore_within_edition(other_honors, ed, "other_honors")

    club_rating = impute_club_rating(df)
    team = (zscore_within_edition(club_rating, ed, "club_rating")
            + zscore_within_edition(stage_ordinal(df["ucl_stage_reached"], "ucl_stage"), ed, "ucl_stage")
            + zscore_within_edition(stage_ordinal(df["nt_tourn_stage_reached"], "nt_stage"), ed, "nt_stage")) / 3
    out["F6_team_strength"] = zscore_within_edition(team, ed, "team_strength")
    out["F7_prior_standing"] = zscore_within_edition(prior_standing, ed, "prior_standing")
    out["F8_defending_winner"] = df["is_defending_winner"].astype(float)
    out["F9_fame"] = zscore_within_edition(np.log1p(df["wiki_views_pre_juror_w"]), ed, "fame")
    if options.awards is None:
        out["F11_individual_awards"] = zscore_within_edition(individual_awards_raw_with(df, options), ed, "individual_awards")
    else:
        require_columns(df, (SCORE_COLUMN, BEST_IS_WC_COLUMN), "build_base_features_with")
        out["F11_individual_awards"] = f11_values(df[SCORE_COLUMN], ed, options.awards, zscore_within_edition)
        out[BEST_IS_WC_COLUMN] = df[BEST_IS_WC_COLUMN]
    out["F12_market_value"] = zscore_within_edition(np.log(market_value_at_t1(df)), ed, "market_value")
    ea = df["ea_overall_t1"].astype(float)
    out[EA_COLUMN] = zscore_within_edition(ea.where(ea.notna(), ea.groupby(ed).transform("mean")), ed, "ea_rating")

    wiki_change = np.log1p(df["wiki_views_in_juror_w"]) - np.log1p(df["wiki_views_pre_juror_w"])
    news_change = np.log1p(df["news_vol_in_juror_w"]) - np.log1p(df["news_vol_pre_juror_w"])
    out["season_attention_raw"] = (zscore_within_edition(wiki_change, ed, "wiki_change")
                                   + zscore_within_edition(news_change, ed, "news_change")) / 2

    out["is_defender"] = (df["position_group"] == DEFENDER).astype(float)
    out["is_goalkeeper"] = is_gk.astype(float)
    out["club_rating_imputed"] = df["club_rating_window_end"].isna().astype(float)
    out["first_time"] = (df["career_bdo_podiums_before"] == 0).astype(float)
    out[GATE_COLUMN] = (major_titles == 0).astype(float)
    out["is_calendar"] = (df["eligibility_format"] == CALENDAR_FORMAT).astype(float)
    out["tourn_ko_z"] = zscore_within_edition(df["tourn_ko_goal_contribs"].fillna(0).astype(float), ed, "tourn_ko_z")
    out["is_world_cup"] = (df["tourn_name"] == WORLD_CUP).astype(float)
    out["is_major_tournament"] = df["tourn_name"].isin(MAJOR_TOURNAMENTS).astype(float)
    out["golden_ball_score"] = np.where(df["award_wc_golden_ball"] == GOLDEN_BALL_GOLD, 1.0,
                                        np.where(df["award_wc_golden_ball"].isin(GOLDEN_BALL_PLACES_COUNTED_HALF), RUNNER_UP_AWARD_WEIGHT, 0.0))
    return out


def fit_feature_params_with(train: pd.DataFrame, include_post: bool, regressors: tuple[str, ...]) -> FeatureParams:
    """Residualisation coefficients for F10, learned from training-edition rows only."""
    if include_post:
        raise ValueError("the ten-edition feature set has no post-window block; M2 uses the 2022-2026 study")
    return FeatureParams(season_attention=fit_residualiser(train, "season_attention_raw", regressors), post_attention=None)


def fit_feature_params_v2(train: pd.DataFrame, include_post: bool) -> FeatureParams:
    return fit_feature_params_with(train, include_post, RESIDUALISER_REGRESSORS)


def make_v2_features(name: str, residualiser_regressors: tuple[str, ...], m1_columns: tuple[str, ...]) -> FeatureSet:
    """A robustness variant of the v2 features: its own residualiser regressors and its own M1 columns."""
    return FeatureSet(name=name, fit_params=lambda train, include_post: fit_feature_params_with(train, include_post, residualiser_regressors),
                      assemble=assemble_features_v2, base_columns=m1_columns, model_columns={"M1": m1_columns})


def assemble_features_v2(frame: pd.DataFrame, params: FeatureParams, include_post: bool) -> pd.DataFrame:
    """Model matrix: the 12 core features, position flags, and the extras used by the gate and format models."""
    if include_post:
        raise ValueError("the ten-edition feature set has no post-window block")
    ed = frame[EDITION_COL]
    out = frame[[c for c in V2_CORE_FEATURES if c != "F10_season_attention"]].copy()
    residual = apply_residualiser(frame, params.season_attention, "season_attention_raw")
    out["F10_season_attention"] = zscore_within_edition(residual, ed, "season_attention")
    out = out[list(V2_CORE_FEATURES)]
    for col in POSITION_FEATURES + (GATE_COLUMN, "first_time", "club_rating_imputed", "is_calendar"):
        out[col] = frame[col]
    out["season_attention_x_first_time"] = out["F10_season_attention"] * frame["first_time"]
    for name, source in FORMAT_INTERACTIONS.items():
        out[name] = out[source] * frame["is_calendar"]
    out[TOURNAMENT_INTERACTION] = frame["tourn_ko_z"] * frame["is_world_cup"]
    out["golden_ball_score"] = frame["golden_ball_score"]
    out[EA_COLUMN] = frame[EA_COLUMN]
    if BEST_IS_WC_COLUMN in frame.columns:
        out[BEST_IS_WC_COLUMN] = frame[BEST_IS_WC_COLUMN]
    return out


def m1_columns_v2() -> tuple[str, ...]:
    return V2_CORE_FEATURES + POSITION_FEATURES


V2_FEATURES = FeatureSet(
    name="v2",
    fit_params=fit_feature_params_v2,
    assemble=assemble_features_v2,
    base_columns=m1_columns_v2(),
    model_columns={
        "M1": m1_columns_v2(),
        "M1-G": m1_columns_v2() + (GATE_COLUMN,),
        "M1-F": m1_columns_v2() + tuple(FORMAT_INTERACTIONS),
    },
)


POST_REGRESSORS_V2: tuple[str, ...] = ("P1_post_form",) + RESIDUALISER_REGRESSORS


def build_post_features_v2(t2: pd.DataFrame) -> pd.DataFrame:
    return build_post_features_for(t2, POST_FORM_COLUMN_V2)


def fit_feature_params_v2_post(train: pd.DataFrame, include_post: bool) -> FeatureParams:
    season = fit_residualiser(train, "season_attention_raw", RESIDUALISER_REGRESSORS)
    if not include_post:
        return FeatureParams(season_attention=season, post_attention=None)
    return FeatureParams(season_attention=season, post_attention=fit_residualiser(train, "post_attention_raw", POST_REGRESSORS_V2))


def assemble_features_v2_post(frame: pd.DataFrame, params: FeatureParams, include_post: bool) -> pd.DataFrame:
    """The v2 matrix, plus P1-P3 for M2 (2022-2026 study only)."""
    out = assemble_features_v2(frame, params, False)
    if not include_post:
        return out
    if params.post_attention is None:
        raise ValueError("include_post requires params fitted with include_post=True")
    out["P1_post_form"] = frame["P1_post_form"]
    out["P2_post_attention"] = zscore_within_edition(apply_residualiser(frame, params.post_attention, "post_attention_raw"),
                                                     frame[EDITION_COL], "post_attention")
    out["P3_final_spike"] = frame["P3_final_spike"]
    return out


V2_POST_FEATURES = FeatureSet(
    name="v2-post",
    fit_params=fit_feature_params_v2_post,
    assemble=assemble_features_v2_post,
    base_columns=m1_columns_v2(),
    model_columns={"M1": m1_columns_v2(), "M2": m1_columns_v2() + tuple(POST_FEATURES)},
)
