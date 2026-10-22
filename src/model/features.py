"""Feature engineering for the Ballon d'Or study (spec section 3).

Pure functions only. Nothing in this module reads the outcome columns, so leakage from the
vote result into the features is impossible by construction.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

EDITION_COL = "edition"

STAGE_ORDINAL: dict[str, int] = {
    "group": 1,
    "league_phase": 1,
    "R32": 2,
    "playoff": 2,
    "R16": 3,
    "QF": 4,
    "SF": 5,
    "final": 6,
    "winner": 7,
}
WINDOW_END: dict[int, pd.Timestamp] = {
    2022: pd.Timestamp("2022-07-31"),
    2023: pd.Timestamp("2023-07-31"),
    2024: pd.Timestamp("2024-07-31"),
    2025: pd.Timestamp("2025-07-13"),
    2026: pd.Timestamp("2026-07-19"),
}
GOALKEEPER = "GK"
DEFENDER = "DEF"
POST_SHRINK_MINUTES = 450.0
RATING_IMPUTE_QUANTILE = 0.05
FINAL_WINDOW_DAYS = 14.0
PRIOR_RANK_BASE = 31
SD_DDOF = 1

REQUIRED_BASE_COLUMNS: tuple[str, ...] = (
    "edition", "position_group", "is_goalkeeper", "prev_bdo_rank", "career_bdo_podiums_before",
    "is_defending_winner", "club_minutes", "club_npxg_xa_p90_opp_adj", "club_avg_rating", "big_game_rating",
    "ucl_ko_goal_contribs", "intl_minutes", "intl_npxg_xa_p90_opp_adj", "intl_xg_missing",
    "tourn_ko_goal_contribs", "tourn_individual_award", "final_goal_contribs", "scored_winner_in_final",
    "ucl_stage_reached", "club_rating_window_end", "nt_tourn_stage_reached", "won_ucl", "won_league",
    "won_domestic_cup", "won_world_cup", "won_continental", "won_other_uefa", "ucl_back_to_back", "league_poty",
    "wiki_views_pre_juror_w", "wiki_views_in_juror_w", "news_vol_pre_juror_w", "news_vol_in_juror_w",
)
REQUIRED_POST_COLUMNS: tuple[str, ...] = (
    "snapshot_date", "wiki_views_post_juror_w", "wiki_views_final14_juror_w", "news_vol_post_juror_w",
    "post_minutes", "post_npxg_xa_p90_opp_adj", "wiki_views_in_juror_w", "news_vol_in_juror_w",
)

CORE_FEATURES: tuple[str, ...] = (
    "F1_attack", "F2_form_rating", "F3_big_stage", "F4_major_titles", "F5_other_honors",
    "F6_team_strength", "F7_prior_standing", "F8_defending_winner", "F9_fame", "F10_season_attention",
)
POSITION_FEATURES: tuple[str, ...] = ("is_defender", "is_goalkeeper")
POST_FEATURES: tuple[str, ...] = ("P1_post_form", "P2_post_attention", "P3_final_spike")
RESIDUALISER_REGRESSORS: tuple[str, ...] = CORE_FEATURES[:7]
POST_RESIDUALISER_REGRESSORS: tuple[str, ...] = ("P1_post_form",) + RESIDUALISER_REGRESSORS


@dataclass(frozen=True)
class Residualiser:
    regressors: tuple[str, ...]
    coefficients: np.ndarray  # intercept first, then one weight per regressor


@dataclass(frozen=True)
class FeatureParams:
    season_attention: Residualiser
    post_attention: Residualiser | None


def require_columns(df: pd.DataFrame, required: tuple[str, ...], context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{context}: missing columns {missing}")


def require_finite(values: pd.Series, context: str) -> None:
    bad = int((~np.isfinite(values.astype(float))).sum())
    if bad:
        raise ValueError(f"{context}: {bad} non-finite values")


def zscore_within_edition(values: pd.Series, editions: pd.Series, context: str) -> pd.Series:
    require_finite(values, context)
    grouped = values.astype(float).groupby(editions)
    mean = grouped.transform("mean")
    sd = grouped.transform("std", ddof=SD_DDOF)
    return ((values - mean) / sd).where(sd > 0, 0.0)


def zscore_with_reference(values: pd.Series, editions: pd.Series, reference: pd.Series, context: str) -> pd.Series:
    """Standardise using only the reference rows of each edition; rows outside the reference get 0."""
    require_finite(values, context)
    grouped = values.astype(float).where(reference).groupby(editions)
    mean = grouped.transform("mean")
    sd = grouped.transform("std", ddof=SD_DDOF)
    return ((values - mean) / sd).where(sd > 0, 0.0).where(reference, 0.0)


def stage_ordinal(stage: pd.Series, context: str) -> pd.Series:
    unknown = sorted(set(stage.dropna()) - set(STAGE_ORDINAL))
    if unknown:
        raise ValueError(f"{context}: unknown stage values {unknown}")
    return stage.map(STAGE_ORDINAL).fillna(0).astype(float)


def attack_raw(df: pd.DataFrame) -> pd.Series:
    club = df["club_npxg_xa_p90_opp_adj"]
    intl = df["intl_npxg_xa_p90_opp_adj"]
    club_weight = df["club_minutes"]
    intl_weight = df["intl_minutes"]
    blended = (club * club_weight + intl * intl_weight) / (club_weight + intl_weight)
    return blended.where(intl.notna(), club)


def impute_club_rating(df: pd.DataFrame) -> pd.Series:
    rating = df["club_rating_window_end"]
    floor = rating.groupby(df[EDITION_COL]).transform(lambda s: s.quantile(RATING_IMPUTE_QUANTILE))
    return rating.where(rating.notna(), floor)


def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Edition-level features F1-F9 plus the unresidualised attention change. Leak-free: each row is standardised
    only against its own edition."""
    require_columns(df, REQUIRED_BASE_COLUMNS, "build_base_features")
    ed = df[EDITION_COL]
    is_gk = df["position_group"] == GOALKEEPER
    if not (df["is_goalkeeper"].astype(bool) == is_gk).all():
        raise ValueError("is_goalkeeper disagrees with position_group")

    out = pd.DataFrame(index=df.index)
    out[EDITION_COL] = ed

    attack = zscore_with_reference(attack_raw(df), ed, ~is_gk, "attack")
    out["F1_attack"] = attack

    big_game = df["big_game_rating"].where(df["big_game_rating"].notna(), df["club_avg_rating"])
    form = (zscore_within_edition(df["club_avg_rating"], ed, "club_avg_rating")
            + zscore_within_edition(big_game, ed, "big_game_rating")) / 2
    out["F2_form_rating"] = zscore_within_edition(form, ed, "form_rating")

    big_stage_sum = (
        zscore_within_edition(df["ucl_ko_goal_contribs"].astype(float), ed, "ucl_ko_goal_contribs")
        + zscore_within_edition(df["tourn_ko_goal_contribs"].fillna(0).astype(float), ed, "tourn_ko_goal_contribs")
        + zscore_within_edition(df["final_goal_contribs"].fillna(0).astype(float), ed, "final_goal_contribs")
        + df["scored_winner_in_final"].fillna(False).astype(float)
        + df["tourn_individual_award"].notna().astype(float)
    )
    out["F3_big_stage"] = zscore_within_edition(big_stage_sum, ed, "big_stage")

    major_titles = df[["won_ucl", "won_world_cup", "won_continental"]].astype(int).sum(axis=1).astype(float)
    other_honors = df[["won_league", "won_domestic_cup", "won_other_uefa", "league_poty", "ucl_back_to_back"]
                      ].astype(int).sum(axis=1).astype(float)
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

    wiki_change = np.log1p(df["wiki_views_in_juror_w"]) - np.log1p(df["wiki_views_pre_juror_w"])
    news_change = np.log1p(df["news_vol_in_juror_w"]) - np.log1p(df["news_vol_pre_juror_w"])
    out["season_attention_raw"] = (zscore_within_edition(wiki_change, ed, "wiki_change")
                                   + zscore_within_edition(news_change, ed, "news_change")) / 2

    out["is_defender"] = (df["position_group"] == DEFENDER).astype(float)
    out["is_goalkeeper"] = is_gk.astype(float)
    out["club_rating_imputed"] = df["club_rating_window_end"].isna().astype(float)
    out["intl_xg_missing"] = df["intl_xg_missing"].astype(float)
    out["first_time"] = (df["career_bdo_podiums_before"] == 0).astype(float)
    out["trophy_gate"] = (major_titles == 0).astype(float)
    return out


def days_since_window_end(df: pd.DataFrame) -> pd.Series:
    days = (pd.to_datetime(df["snapshot_date"]) - df[EDITION_COL].map(WINDOW_END)).dt.days.astype(float)
    if (days <= 0).any():
        raise ValueError("snapshot_date must be after the window end for every row")
    return days


def build_post_features(df: pd.DataFrame) -> pd.DataFrame:
    """Post-window block (T2 rows only): P1, P3 and the unresidualised post-window attention change."""
    require_columns(df, REQUIRED_POST_COLUMNS + (EDITION_COL,), "build_post_features")
    ed = df[EDITION_COL]
    days_post = days_since_window_end(df)

    out = pd.DataFrame(index=df.index)
    value = df["post_npxg_xa_p90_opp_adj"]
    edition_mean = value.groupby(ed).transform("mean")
    if edition_mean.isna().any():
        raise ValueError("an edition has no non-null post_npxg_xa_p90_opp_adj values")
    weight = (df["post_minutes"] / (df["post_minutes"] + POST_SHRINK_MINUTES)).where(value.notna(), 0.0)
    shrunk = weight * value.fillna(0.0) + (1 - weight) * edition_mean
    out["P1_post_form"] = zscore_within_edition(shrunk, ed, "post_form")

    wiki_post = np.log(df["wiki_views_post_juror_w"] / days_post)
    news_post = np.log(df["news_vol_post_juror_w"] / days_post)
    # Dividing the in-window level by its unknown length only shifts the log by a constant within an edition,
    # and the within-edition z-score removes that constant.
    wiki_shift = wiki_post - np.log(df["wiki_views_in_juror_w"])
    news_shift = news_post - np.log(df["news_vol_in_juror_w"])
    out["post_attention_raw"] = (zscore_within_edition(wiki_shift, ed, "wiki_post_shift")
                                 + zscore_within_edition(news_shift, ed, "news_post_shift")) / 2

    spike = np.log(df["wiki_views_final14_juror_w"] / FINAL_WINDOW_DAYS) - wiki_post
    out["P3_final_spike"] = zscore_within_edition(spike, ed, "final_spike")
    return out


def fit_residualiser(train: pd.DataFrame, target_col: str, regressors: tuple[str, ...]) -> Residualiser:
    design = np.column_stack([np.ones(len(train)), train[list(regressors)].to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design, train[target_col].to_numpy(dtype=float), rcond=None)
    return Residualiser(regressors=regressors, coefficients=coefficients)


def apply_residualiser(df: pd.DataFrame, model: Residualiser, target_col: str) -> pd.Series:
    design = np.column_stack([np.ones(len(df)), df[list(model.regressors)].to_numpy(dtype=float)])
    return pd.Series(df[target_col].to_numpy(dtype=float) - design @ model.coefficients, index=df.index)


def fit_feature_params(train: pd.DataFrame, include_post: bool) -> FeatureParams:
    """Learn residualisation coefficients from training-edition rows only.

    `train` must hold the base features (and the post features when include_post is set)."""
    season = fit_residualiser(train, "season_attention_raw", RESIDUALISER_REGRESSORS)
    if not include_post:
        return FeatureParams(season_attention=season, post_attention=None)
    post = fit_residualiser(train, "post_attention_raw", POST_RESIDUALISER_REGRESSORS)
    return FeatureParams(season_attention=season, post_attention=post)


def assemble_features(frame: pd.DataFrame, params: FeatureParams, include_post: bool) -> pd.DataFrame:
    """Final model matrix: the 10 core features and position flags (plus P1-P3 for M2), plus extras for the
    gate and interaction models. F10 and P2 use coefficients from `params` and are re-standardised per edition."""
    ed = frame[EDITION_COL]
    out = frame[list(CORE_FEATURES[:6]) + ["F7_prior_standing", "F8_defending_winner", "F9_fame"]].copy()
    residual = apply_residualiser(frame, params.season_attention, "season_attention_raw")
    out["F10_season_attention"] = zscore_within_edition(residual, ed, "season_attention")
    out = out[list(CORE_FEATURES)]
    for col in POSITION_FEATURES:
        out[col] = frame[col]
    for col in ("trophy_gate", "first_time", "club_rating_imputed", "intl_xg_missing"):
        out[col] = frame[col]
    out["season_attention_x_first_time"] = out["F10_season_attention"] * frame["first_time"]
    if not include_post:
        return out
    if params.post_attention is None:
        raise ValueError("include_post requires params fitted with include_post=True")
    out["P1_post_form"] = frame["P1_post_form"]
    post_residual = apply_residualiser(frame, params.post_attention, "post_attention_raw")
    out["P2_post_attention"] = zscore_within_edition(post_residual, ed, "post_attention")
    out["P3_final_spike"] = frame["P3_final_spike"]
    return out


def m1_columns() -> list[str]:
    return list(CORE_FEATURES) + list(POSITION_FEATURES)


def m2_columns() -> list[str]:
    return m1_columns() + list(POST_FEATURES)
