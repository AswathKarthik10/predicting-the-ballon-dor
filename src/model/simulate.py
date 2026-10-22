"""Juror ballot simulation (spec 6.3). Bias parameters are placeholders unless estimated from ballots (S8)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model.jurors import BiasParams, SimInputs

SEED = 7
CHUNK = 500
BALLOT_POINTS = np.array([15, 12, 10, 8, 7, 5, 4, 3, 2, 1], dtype=float)
POINTS_PER_BALLOT = float(BALLOT_POINTS.sum())  # 67
BALLOT_SIZE = len(BALLOT_POINTS)
POINTS_WEIGHT = 1e6
FIRST_WEIGHT = 1e3
TIE_JITTER = 0.5
RANK_INTERVAL = (10.0, 90.0)


@dataclass(frozen=True)
class SimConfig:
    tau_shared: float
    tau_juror: float
    bias: BiasParams


@dataclass(frozen=True)
class SimResult:
    points: np.ndarray  # simulations x players
    ranks: np.ndarray  # 1 = winner
    n_jurors: int


def rank_players(points: np.ndarray, first: np.ndarray, second: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    """Rank 1..N per simulation; ties on points are broken by first-place then second-place votes, then by jitter."""
    composite = points * POINTS_WEIGHT + first * FIRST_WEIGHT + second + jitter
    order = np.argsort(-composite, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, points.shape[1] + 1), order.shape), axis=1)
    return ranks


def ballot_tallies(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """votes: sims x jurors x players juror valuations. Returns summed points, first-place and second-place counts."""
    top = np.argsort(-votes, axis=2)[:, :, :BALLOT_SIZE]
    shape = votes.shape
    pts = np.zeros(shape)
    np.put_along_axis(pts, top, np.broadcast_to(BALLOT_POINTS, top.shape), axis=2)
    first = np.zeros(shape)
    np.put_along_axis(first, top[:, :, :1], 1.0, axis=2)
    second = np.zeros(shape)
    np.put_along_axis(second, top[:, :, 1:2], 1.0, axis=2)
    return pts.sum(axis=1), first.sum(axis=1), second.sum(axis=1)


def bias_matrix(inputs: SimInputs, bias: BiasParams) -> np.ndarray:
    return (bias.b_home * inputs.nation_match + bias.b_confed * inputs.confed_match + bias.b_league * inputs.league_match).astype(float)


def simulate(strengths: np.ndarray, inputs: SimInputs, config: SimConfig, rng: np.random.Generator) -> SimResult:
    """strengths: simulations x players (one posterior draw of utilities per simulation)."""
    n_sims, n_players = strengths.shape
    bias = bias_matrix(inputs, config.bias)
    points_out = np.empty((n_sims, n_players))
    ranks_out = np.empty((n_sims, n_players), dtype=int)
    for start in range(0, n_sims, CHUNK):
        s = strengths[start:start + CHUNK]
        c = s.shape[0]
        shared = config.tau_shared * rng.gumbel(size=(c, n_players))
        noise = config.tau_juror * rng.gumbel(size=(c, inputs.n_jurors, n_players))
        votes = s[:, None, :] + shared[:, None, :] + bias[None, :, :] + noise
        pts, first, second = ballot_tallies(votes)
        points_out[start:start + c] = pts
        ranks_out[start:start + c] = rank_players(pts, first, second, TIE_JITTER * rng.random((c, n_players)))
    return SimResult(points=points_out, ranks=ranks_out, n_jurors=inputs.n_jurors)


def summarise(result: SimResult) -> pd.DataFrame:
    total_points = result.n_jurors * POINTS_PER_BALLOT
    lo, hi = RANK_INTERVAL
    return pd.DataFrame({
        "p_win": (result.ranks == 1).mean(axis=0),
        "p_top3": (result.ranks <= 3).mean(axis=0),
        "p_top10": (result.ranks <= 10).mean(axis=0),
        "expected_rank": result.ranks.mean(axis=0),
        "rank_p10": np.percentile(result.ranks, lo, axis=0),
        "rank_p90": np.percentile(result.ranks, hi, axis=0),
        "expected_points": result.points.mean(axis=0),
        "expected_vote_share": (result.points / total_points).mean(axis=0),
    })


def concentration_stats(points: np.ndarray) -> dict[str, float]:
    """Winner's vote share, number of players with any points and Herfindahl index, for one points vector."""
    total = points.sum()
    shares = points / total
    return {"winner_share": float(shares.max()), "n_with_points": float((points > 0).sum()), "herfindahl": float((shares ** 2).sum())}


def simulated_concentration(result: SimResult) -> dict[str, float]:
    total = result.points.sum(axis=1, keepdims=True)
    shares = result.points / total
    return {"winner_share": float(shares.max(axis=1).mean()), "n_with_points": float((result.points > 0).sum(axis=1).mean()),
            "herfindahl": float((shares ** 2).sum(axis=1).mean())}
