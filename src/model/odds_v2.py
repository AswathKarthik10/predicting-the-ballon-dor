"""Betting-odds baseline (experiments spec section 8). The market is only ever compared with the model; odds are never a feature."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import baselines
from src.model.data import EDITION_COL, T1_FILE, V2_DIR, StudyData
from src.model.pl import build_ranking, edition_loglik

ODDS_FILE = "fact_bdo_odds.parquet"
MAX_DAYS_BEFORE_T1 = 60
PHASE_BEFORE_T1 = "before_t1"
PHASE_T1_TO_DEADLINE = "t1_to_deadline"
CAPTURE_KEYS: tuple[str, ...] = (EDITION_COL, "captured_at")


def read_odds(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / V2_DIR / ODDS_FILE)


def usable_captures(odds: pd.DataFrame, t1_dates: pd.Series, labelled_editions: tuple[int, ...]) -> pd.DataFrame:
    """Nominee prices from captures that are before T1 and within 60 days of it, or between T1 and the deadline.
    A capture is one (edition, date); after-deadline captures and the forecast edition are excluded."""
    frame = odds[odds[EDITION_COL].isin(labelled_editions) & odds.is_nominee].copy()
    frame["days_before_t1"] = (pd.to_datetime(frame[EDITION_COL].map(t1_dates)) - pd.to_datetime(frame.captured_at)).dt.days
    before = (frame.capture_phase == PHASE_BEFORE_T1) & (frame.days_before_t1 >= 0) & (frame.days_before_t1 <= MAX_DAYS_BEFORE_T1)
    return frame[before | (frame.capture_phase == PHASE_T1_TO_DEADLINE)].reset_index(drop=True)


def loglik_on_quoted(utility: np.ndarray, points: np.ndarray) -> tuple[float, int]:
    """Plackett-Luce log-likelihood of the actual order restricted to the quoted players; zero-point players stay in the pool."""
    ranking = build_ranking(points)
    return edition_loglik(utility, ranking), ranking.n_voted


def capture_rows(study: StudyData, captures: pd.DataFrame, model_utilities: dict[int, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (edition, captured_at), group in captures.groupby(list(CAPTURE_KEYS)):
        frame = study.frame_t1[study.frame_t1.edition == edition].reset_index(drop=True)
        points = baselines.points_of(study, int(edition))
        index = {k: i for i, k in enumerate(frame.player_key)}
        quoted = group.groupby("player_key").implied_prob.max()  # best price per player across bookmakers
        keep = [index[k] for k in quoted.index]
        market = np.log((quoted / quoted.sum()).to_numpy())
        model = model_utilities[int(edition)][keep]
        model = model - np.log(np.exp(model).sum())
        quoted_points = points[keep]
        market_ll, n_voted = loglik_on_quoted(market, quoted_points)
        model_ll, _ = loglik_on_quoted(model, quoted_points)
        names = frame.player_name.to_numpy()
        winner = int(np.argmax(points))
        model_pick = int(np.argmax(model_utilities[int(edition)]))
        rows.append({"edition": int(edition), "captured_at": captured_at, "capture_phase": group.capture_phase.iloc[0],
                     "days_before_t1": int(group.days_before_t1.iloc[0]), "n_quoted": len(keep), "n_quoted_with_votes": n_voted,
                     "market_favourite": names[keep[int(np.argmax(market))]], "actual_winner": names[winner],
                     "favourite_is_winner": bool(keep[int(np.argmax(market))] == winner),
                     "model_pick_all_30": names[model_pick], "model_pick_is_winner": bool(model_pick == winner),
                     "model_pick_among_quoted": names[keep[int(np.argmax(model))]],
                     "market_loglik": market_ll, "model_loglik": model_ll, "market_loglik_per_voted": market_ll / n_voted,
                     "model_loglik_per_voted": model_ll / n_voted, "model_minus_market_per_voted": (model_ll - market_ll) / n_voted})
    return pd.DataFrame(rows).sort_values(["edition", "captured_at"]).reset_index(drop=True)


def t1_dates(root: Path) -> pd.Series:
    t1 = pd.read_csv(root / V2_DIR / T1_FILE, usecols=[EDITION_COL, "snapshot_date"])
    return t1.groupby(EDITION_COL).snapshot_date.first()
