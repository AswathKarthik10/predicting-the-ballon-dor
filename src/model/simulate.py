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
    points_per_ballot: float
    home_first_share: np.ndarray  # per simulation: share of (juror, compatriot nominee) pairs where the juror ranked him 1st


def rank_players(points: np.ndarray, first: np.ndarray, second: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    """Rank 1..N per simulation; ties on points are broken by first-place then second-place votes, then by jitter."""
    composite = points * POINTS_WEIGHT + first * FIRST_WEIGHT + second + jitter
    order = np.argsort(-composite, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(1, points.shape[1] + 1), order.shape), axis=1)
    return ranks


def weighted_tallies(votes: np.ndarray, scheme: np.ndarray, juror_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """votes: sims x jurors x players valuations; juror_weights: sims x jurors (0 = juror did not vote).
    Returns summed points, first-place and second-place counts under the given points scheme (best first)."""
    top = np.argsort(-votes, axis=2)[:, :, :len(scheme)]
    shape = votes.shape
    weights = juror_weights[:, :, None]
    pts = np.zeros(shape)
    np.put_along_axis(pts, top, np.broadcast_to(scheme, top.shape), axis=2)
    first = np.zeros(shape)
    np.put_along_axis(first, top[:, :, :1], 1.0, axis=2)
    second = np.zeros(shape)
    np.put_along_axis(second, top[:, :, 1:2], 1.0, axis=2)
    return (pts * weights).sum(axis=1), (first * weights).sum(axis=1), (second * weights).sum(axis=1)


def home_first_share(votes: np.ndarray, nation_match: np.ndarray, juror_weights: np.ndarray) -> np.ndarray:
    """Per simulation: weighted share of (juror, compatriot nominee) pairs where the nominee is the juror's top choice."""
    top = np.argmax(votes, axis=2)
    hits = nation_match[np.arange(nation_match.shape[0])[None, :], top]
    pairs = (juror_weights * nation_match.sum(axis=1)[None, :]).sum(axis=1)
    return np.divide((juror_weights * hits).sum(axis=1), pairs, out=np.full(pairs.shape, np.nan), where=pairs > 0)


def ballot_tallies(votes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """votes: sims x jurors x players juror valuations, 10-pick scheme, every juror votes (original behaviour)."""
    return weighted_tallies(votes, BALLOT_POINTS, np.ones(votes.shape[:2]))


def bias_matrix(inputs: SimInputs, bias: BiasParams) -> np.ndarray:
    return (bias.b_home * inputs.nation_match + bias.b_confed * inputs.confed_match + bias.b_league * inputs.league_match).astype(float)


def simulate(strengths: np.ndarray, inputs: SimInputs, config: SimConfig, rng: np.random.Generator) -> SimResult:
    """strengths: simulations x players (one posterior draw of utilities per simulation); 10-pick scheme (original behaviour)."""
    return simulate_scheme(strengths, inputs, config, BALLOT_POINTS, rng)


def simulate_scheme(strengths: np.ndarray, inputs: SimInputs, config: SimConfig, scheme: np.ndarray, rng: np.random.Generator) -> SimResult:
    """Placeholder-bias simulation under an explicit points scheme (best first)."""
    n_sims, n_players = strengths.shape
    bias = bias_matrix(inputs, config.bias)
    points_out = np.empty((n_sims, n_players))
    ranks_out = np.empty((n_sims, n_players), dtype=int)
    home_out = np.empty(n_sims)
    for start in range(0, n_sims, CHUNK):
        s = strengths[start:start + CHUNK]
        c = s.shape[0]
        shared = config.tau_shared * rng.gumbel(size=(c, n_players))
        noise = config.tau_juror * rng.gumbel(size=(c, inputs.n_jurors, n_players))
        votes = s[:, None, :] + shared[:, None, :] + bias[None, :, :] + noise
        weights = np.ones((c, inputs.n_jurors))
        pts, first, second = weighted_tallies(votes, scheme, weights)
        home_out[start:start + c] = home_first_share(votes, inputs.nation_match, weights)
        points_out[start:start + c] = pts
        ranks_out[start:start + c] = rank_players(pts, first, second, TIE_JITTER * rng.random((c, n_players)))
    return SimResult(points=points_out, ranks=ranks_out, n_jurors=inputs.n_jurors, points_per_ballot=float(scheme.sum()),
                     home_first_share=home_out)


def summarise(result: SimResult) -> pd.DataFrame:
    total_points = result.n_jurors * result.points_per_ballot
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


@dataclass(frozen=True)
class BallotSimConfig:
    """Ballot-estimated simulation: v = k*u + Normal(0, sigma_shared) shared per player + beta . match + Gumbel(0, 1)."""
    k: float
    sigma_shared: float
    scheme: tuple[int, ...]
    n_dropped: int  # jurors who do not vote, drawn at random per simulation


def match_tensor(inputs: SimInputs) -> np.ndarray:
    """jurors x players x 3 indicators in the order (home, confed, league)."""
    return np.stack([inputs.nation_match, inputs.confed_match, inputs.league_match], axis=2).astype(float)


def voting_weights(n_sims: int, n_jurors: int, n_dropped: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.ones((n_sims, n_jurors))
    if n_dropped == 0:
        return weights
    dropped = np.argsort(rng.random((n_sims, n_jurors)), axis=1)[:, :n_dropped]
    np.put_along_axis(weights, dropped, 0.0, axis=1)
    return weights


def simulate_ballots(utilities: np.ndarray, betas: np.ndarray, inputs: SimInputs, config: BallotSimConfig,
                     rng: np.random.Generator) -> SimResult:
    """utilities: sims x players model strengths u (one draw per simulation); betas: sims x 3 bias draws."""
    n_sims, n_players = utilities.shape
    if betas.shape != (n_sims, len(match_tensor(inputs)[0, 0])):
        raise ValueError(f"betas shape {betas.shape} does not match {n_sims} simulations x 3 indicators")
    scheme = np.array(config.scheme, dtype=float)
    matches = match_tensor(inputs)
    points_out = np.empty((n_sims, n_players))
    ranks_out = np.empty((n_sims, n_players), dtype=int)
    home_out = np.empty(n_sims)
    for start in range(0, n_sims, CHUNK):
        u = utilities[start:start + CHUNK]
        c = u.shape[0]
        shared = config.sigma_shared * rng.standard_normal((c, n_players))
        noise = rng.gumbel(size=(c, inputs.n_jurors, n_players))
        bias = np.einsum("cb,jpb->cjp", betas[start:start + CHUNK], matches)
        votes = config.k * u[:, None, :] + shared[:, None, :] + bias + noise
        weights = voting_weights(c, inputs.n_jurors, config.n_dropped, rng)
        pts, first, second = weighted_tallies(votes, scheme, weights)
        home_out[start:start + c] = home_first_share(votes, inputs.nation_match, weights)
        points_out[start:start + c] = pts
        ranks_out[start:start + c] = rank_players(pts, first, second, TIE_JITTER * rng.random((c, n_players)))
    return SimResult(points=points_out, ranks=ranks_out, n_jurors=inputs.n_jurors - config.n_dropped,
                     points_per_ballot=float(scheme.sum()), home_first_share=home_out)
