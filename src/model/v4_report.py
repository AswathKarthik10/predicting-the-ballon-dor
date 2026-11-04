"""Experiment 4 reporting helpers (spec sections 7 and 8): coefficient intervals, permutation importance, v3 block overlap
and per-player feature contributions. Descriptive only."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.model.pl import build_ranking, edition_hessian_of_loglik, edition_loglik_per_voted
from src.model.v4_models import ModelSpec, fit_pl_beta
from src.model.v4_select import TrainSet, make_train_set

SEED = 7
N_BOOTSTRAP = 200
N_PERMUTATIONS = 20
INTERVAL_PCT = (5, 95)
NO_BLOCK = "none"
BLOCK_BY_COLUMN: dict[str, str] = {
    **{c: "Season output" for c in ("club_npga_p90_opp_adj", "intl_npga_p90_opp_adj", "club_avg_rating", "big_game_rating")},
    **{c: "Big stage" for c in ("ucl_ko_goal_contribs", "tourn_ko_goal_contribs", "final_goal_contribs", "scored_winner_in_final")},
    **{c: "Trophies and team" for c in ("won_ucl", "won_world_cup", "won_continental", "won_league", "won_domestic_cup",
                                         "won_other_uefa", "ucl_back_to_back", "club_rating_window_end", "ucl_stage_reached",
                                         "nt_tourn_stage_reached")},
    **{c: "Individual awards" for c in ("major_individual_awards_n", "league_player_of_season", "league_poty", "uefa_player_award",
                                         "award_wc_golden_ball")},
    **{c: "Prior standing" for c in ("prev_bdo_rank", "career_bdo_podiums_before", "is_defending_winner")},
    **{c: "Fame and attention" for c in ("wiki_views_pre_juror_w", "wiki_views_in_juror_w", "news_vol_pre_juror_w", "news_vol_in_juror_w")},
    **{c: "Market value" for c in ("tm_value_eur_window_start", "tm_value_eur_t1")},
    **{c: "Position" for c in ("position_group", "is_goalkeeper")},
}
V3_BLOCKS: tuple[str, ...] = ("Season output", "Big stage", "Trophies and team", "Individual awards", "Prior standing",
                              "Fame and attention", "Market value", "Position", NO_BLOCK)


def block_of(source_column: str) -> str:
    return BLOCK_BY_COLUMN.get(source_column, NO_BLOCK)


def block_overlap(selected: dict[str, list[str]], source_of: dict[str, str]) -> pd.DataFrame:
    """Per scope (fold or final fit), the number and share of selected encoded columns that map to each v3 block."""
    rows: list[dict[str, object]] = []
    for scope, features in selected.items():
        blocks = pd.Series([block_of(source_of[f]) for f in features], dtype=str)
        for block in V3_BLOCKS:
            n = int((blocks == block).sum())
            rows.append({"scope": scope, "block": block, "n_selected": n, "share": n / len(features) if features else float("nan")})
    return pd.DataFrame(rows)


def laplace_covariance(train: TrainSet, beta: np.ndarray, l2: float) -> np.ndarray:
    """Inverse Hessian of the penalised negative log-likelihood on the active columns (the L1 part is ignored: the interval is
    conditional on the selected set)."""
    hessian = 2.0 * l2 * np.eye(len(beta))
    for x, ranking in zip(train.xs, train.rankings):
        hessian -= edition_hessian_of_loglik(x, beta, ranking)
    return np.linalg.inv(hessian)


def laplace_draws(train: TrainSet, beta: np.ndarray, l2: float, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(beta, laplace_covariance(train, beta, l2), size=n_draws)


def bootstrap_betas(spec: ModelSpec, train: TrainSet, n_draws: int, seed: int) -> np.ndarray:
    """Refit on within-edition player resamples (columns and preprocessing held fixed). Rows: draws, columns: features."""
    rng = np.random.default_rng(seed)
    draws = np.zeros((n_draws, train.xs[0].shape[1]))
    for b in range(n_draws):
        rows = [rng.integers(0, len(p), size=len(p)) for p in train.points]
        sample = make_train_set(train.editions, [x[r] for x, r in zip(train.xs, rows)], [p[r] for p, r in zip(train.points, rows)])
        draws[b] = fit_pl_beta(spec, sample).beta
    return draws


def mean_loglik(train: TrainSet, utility_fn: Callable[[np.ndarray], np.ndarray]) -> float:
    return float(np.mean([edition_loglik_per_voted(utility_fn(x), build_ranking(p)) for x, p in zip(train.xs, train.points)]))


def permutation_importance(train: TrainSet, utility_fn: Callable[[np.ndarray], np.ndarray], columns: np.ndarray, n_repeats: int,
                           seed: int) -> np.ndarray:
    """Mean drop in training log-likelihood per voted player when a column is shuffled within each edition."""
    rng = np.random.default_rng(seed)
    baseline = mean_loglik(train, utility_fn)
    drops = np.zeros(len(columns))
    for k, col in enumerate(columns):
        for _ in range(n_repeats):
            shuffled = []
            for x in train.xs:
                copy = x.copy()
                copy[:, col] = rng.permutation(copy[:, col])
                shuffled.append(copy)
            permuted = make_train_set(train.editions, shuffled, train.points)
            drops[k] += baseline - mean_loglik(permuted, utility_fn)
    return drops / n_repeats


def coefficient_contributions(z_apply: np.ndarray, beta: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Per-player, per-feature utility contribution relative to the field mean (beta_j * (z_ij - mean_j))."""
    return pd.DataFrame((z_apply - z_apply.mean(axis=0)) * beta, columns=names)
