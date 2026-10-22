"""Real juror ballots: ballot table, validation checks and the ballot-level Plackett-Luce bias model.

Implements juror_simulation_ballots_spec.md sections 2-3. Ballots exist for 2022-2025 only; 2026 targets are never read.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.model.jurors import (CLUB_FILE, COUNTRY_FILE, JURY_DIR, JURY_FILES, LEAGUE_FILE, NATIONS_FILE, PLAYER_FILE, RESULTS_FILE,
                              VOTES_FILE, league_country_map, load_juror_inputs)

BALLOT_EDITIONS: tuple[int, ...] = (2022, 2023, 2024, 2025)
SCHEME_5: tuple[int, ...] = (6, 4, 3, 2, 1)
SCHEME_10: tuple[int, ...] = (15, 12, 10, 8, 7, 5, 4, 3, 2, 1)
SCHEME_BY_EDITION: dict[int, tuple[int, ...]] = {2022: SCHEME_5, 2023: SCHEME_5, 2024: SCHEME_10, 2025: SCHEME_10, 2026: SCHEME_10}
BALLOTS_BY_EDITION: dict[int, int] = {2022: 93, 2023: 92, 2024: 99, 2025: 100}
BALLOT_KEYS: tuple[str, ...] = ("edition", "juror_country_key", "player_key")
INDICATORS: tuple[str, ...] = ("home", "confed", "league")


@dataclass(frozen=True)
class JuryTables:
    votes: pd.DataFrame
    nations: pd.DataFrame
    countries: pd.DataFrame
    players: pd.DataFrame
    results: pd.DataFrame
    leagues: pd.DataFrame


def read_jury_tables(root: Path) -> JuryTables:
    read = lambda name: pd.read_parquet(root / JURY_DIR / name)  # noqa: E731
    return JuryTables(votes=read(VOTES_FILE), nations=read(NATIONS_FILE), countries=read(COUNTRY_FILE),
                      players=read(PLAYER_FILE), results=read(RESULTS_FILE), leagues=read(LEAGUE_FILE))


def player_league_country(juror_inputs: pd.DataFrame, league_to_country: dict[str, str]) -> pd.DataFrame:
    """(edition, player_key) -> league country, from the snapshot's integer league_key."""
    frame = juror_inputs[["edition", "player_key", "league_key"]].drop_duplicates().copy()
    frame["league_country_key"] = frame.league_key.astype(int).astype(str).map(league_to_country)
    unmapped = frame[frame.league_country_key.isna()]
    if len(unmapped):
        raise ValueError(f"league keys without a country: {sorted(unmapped.league_key.unique())}")
    return frame[["edition", "player_key", "league_country_key"]]


def attach_pick_rank(frame: pd.DataFrame) -> pd.DataFrame:
    """1 for the highest points on a ballot, 2 for the next, ...; NaN for unpicked nominees."""
    ranked = frame.copy()
    ranked["pick_rank"] = ranked.groupby(["edition", "juror_country_key"]).points.rank(method="first", ascending=False)
    ranked.loc[ranked.points == 0, "pick_rank"] = np.nan
    return ranked


def build_ballot_table(tables: JuryTables, juror_inputs: pd.DataFrame) -> pd.DataFrame:
    votes = tables.votes.rename(columns={"country_key": "juror_country_key"})
    players = tables.players[["player_key", "wiki_title", "nationality_country_key"]]
    frame = votes[["edition", "juror_country_key", "wiki_title", "points"]].merge(players, on="wiki_title", how="left", validate="m:1")
    if frame.player_key.isna().any():
        raise ValueError(f"ballot rows without a player_key: {sorted(frame[frame.player_key.isna()].wiki_title.unique())}")
    confed = tables.countries.set_index("country_key").confederation
    frame["player_confed"] = frame.nationality_country_key.map(confed)
    frame["juror_confed"] = frame.juror_country_key.map(confed)
    league = player_league_country(juror_inputs, league_country_map(tables.leagues))
    frame = frame.merge(league, on=["edition", "player_key"], how="left", validate="m:1")
    if frame.league_country_key.isna().any():
        raise ValueError("ballot rows for a player-edition missing from the snapshot")
    frame["home"] = (frame.nationality_country_key == frame.juror_country_key).astype(int)
    frame["confed"] = ((frame.player_confed == frame.juror_confed) & (frame.home == 0)).astype(int)
    frame["league"] = (frame.league_country_key == frame.juror_country_key).astype(int)
    frame["points"] = frame.points.astype(int)
    frame = attach_pick_rank(frame)
    columns = [*BALLOT_KEYS, "points", "pick_rank", *INDICATORS]
    return frame[columns].sort_values(list(BALLOT_KEYS)).reset_index(drop=True)


def check_points_match_results(ballots: pd.DataFrame, results: pd.DataFrame) -> None:
    summed = ballots.groupby(["edition", "player_key"]).points.sum()
    official = results[results.edition.isin(BALLOT_EDITIONS)].set_index(["edition", "player_key"]).points
    joined = pd.concat([summed.rename("ballots"), official.rename("official")], axis=1)
    bad = joined[(joined.ballots.fillna(-1) != joined.official.fillna(-1))]
    if len(bad):
        raise ValueError(f"ballot totals differ from fact_bdo_results for {len(bad)} player-editions:\n{bad.head(10)}")


def check_scheme_usage(ballots: pd.DataFrame) -> None:
    for (edition, juror), group in ballots.groupby(["edition", "juror_country_key"]):
        picked = sorted(group.points[group.points > 0], reverse=True)
        if picked != sorted(SCHEME_BY_EDITION[int(edition)], reverse=True):
            raise ValueError(f"ballot {edition}/{juror} does not use the {len(SCHEME_BY_EDITION[int(edition)])}-pick scheme exactly once: {picked}")


def check_ballot_counts(ballots: pd.DataFrame) -> None:
    counts = ballots.groupby("edition").juror_country_key.nunique().to_dict()
    if counts != BALLOTS_BY_EDITION:
        raise ValueError(f"ballot counts per edition {counts} != expected {BALLOTS_BY_EDITION}")


def check_voters_are_jurors(ballots: pd.DataFrame, nations: pd.DataFrame) -> None:
    voters = set(map(tuple, ballots[["edition", "juror_country_key"]].drop_duplicates().to_numpy()))
    listed = nations[nations.voted == True]  # noqa: E712 - voted is an object column holding None for 2026
    allowed = set(zip(listed.edition.astype(int), listed.country_key))
    missing = voters - allowed
    if missing:
        raise ValueError(f"voting countries not marked voted in fact_juror_nations: {sorted(missing)[:10]}")


def check_indicators_exclusive(ballots: pd.DataFrame) -> None:
    if ((ballots.home == 1) & (ballots.confed == 1)).any():
        raise ValueError("home and confed are both 1 on some rows")


def validate_ballots(ballots: pd.DataFrame, tables: JuryTables) -> None:
    check_points_match_results(ballots, tables.results)
    check_scheme_usage(ballots)
    check_ballot_counts(ballots)
    check_voters_are_jurors(ballots, tables.nations)
    check_indicators_exclusive(ballots)


def load_ballots(root: Path) -> pd.DataFrame:
    tables = read_jury_tables(root)
    ballots = build_ballot_table(tables, load_juror_inputs(root))
    validate_ballots(ballots, tables)
    return ballots


# ---------------------------------------------------------------------------------------------------------------------
# Ballot-level Plackett-Luce: v_ij = alpha_(i,e) + beta . indicators_ij + Gumbel(0, 1), top-k partial rankings.
# ---------------------------------------------------------------------------------------------------------------------

RIDGE_ALPHA = 0.01
N_BETA = len(INDICATORS)
FIT_MAX_ITER = 2000
FIT_FTOL = 1e-12
FIT_GTOL = 1e-8
N_BOOTSTRAP = 500
BOOTSTRAP_SEED = 7
Z_90 = 1.6448536269514722
MIN_INDICATOR_PAIRS = 20


@dataclass(frozen=True)
class EditionBallots:
    edition: int
    player_keys: tuple[str, ...]
    juror_keys: tuple[str, ...]
    indicators: np.ndarray  # ballots x players x N_BETA
    picks: np.ndarray  # ballots x k, player indices in pick order (best first)


@dataclass(frozen=True)
class BiasFit:
    editions: tuple[int, ...]
    alpha: dict[int, np.ndarray]
    beta: np.ndarray
    covariance: np.ndarray  # N_BETA x N_BETA, inverse penalised Hessian restricted to beta
    log_likelihood: float


def edition_ballots(ballots: pd.DataFrame, edition: int) -> EditionBallots:
    frame = ballots[ballots.edition == edition]
    players = tuple(sorted(frame.player_key.unique()))
    jurors = tuple(sorted(frame.juror_country_key.unique()))
    player_index = {p: i for i, p in enumerate(players)}
    juror_index = {j: i for i, j in enumerate(jurors)}
    rows = frame.juror_country_key.map(juror_index).to_numpy()
    cols = frame.player_key.map(player_index).to_numpy()
    indicators = np.zeros((len(jurors), len(players), N_BETA))
    indicators[rows, cols] = frame[list(INDICATORS)].to_numpy(dtype=float)
    picked = frame[frame.pick_rank.notna()].sort_values(["juror_country_key", "pick_rank"])
    k = len(SCHEME_BY_EDITION[edition])
    picks = picked.player_key.map(player_index).to_numpy().reshape(len(jurors), k)
    return EditionBallots(edition=edition, player_keys=players, juror_keys=jurors, indicators=indicators, picks=picks)


def build_edition_ballots(ballots: pd.DataFrame, editions: tuple[int, ...]) -> list[EditionBallots]:
    return [edition_ballots(ballots, e) for e in editions]


def resample_ballots(data: EditionBallots, rng: np.random.Generator) -> EditionBallots:
    """Bootstrap: whole ballots drawn with replacement within the edition."""
    rows = rng.integers(0, len(data.juror_keys), size=len(data.juror_keys))
    return EditionBallots(edition=data.edition, player_keys=data.player_keys, juror_keys=tuple(data.juror_keys[r] for r in rows),
                          indicators=data.indicators[rows], picks=data.picks[rows])


def utilities(data: EditionBallots, alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return alpha[None, :] + data.indicators @ beta


def softmax_steps(u: np.ndarray, picks: np.ndarray):
    """Yield (step, chosen index, choice probabilities over the not-yet-chosen players, log-sum-exp) for each pick."""
    n_ballots = u.shape[0]
    rows = np.arange(n_ballots)
    available = np.ones(u.shape, dtype=bool)
    for step in range(picks.shape[1]):
        logits = np.where(available, u, -np.inf)
        top = logits.max(axis=1)
        shifted = np.exp(logits - top[:, None])
        total = shifted.sum(axis=1)
        probs = shifted / total[:, None]
        yield step, picks[:, step], probs, top + np.log(total)
        available[rows, picks[:, step]] = False


def edition_loglik_and_grad_u(u: np.ndarray, picks: np.ndarray) -> tuple[float, np.ndarray]:
    rows = np.arange(u.shape[0])
    loglik = 0.0
    grad_u = np.zeros_like(u)
    for _, chosen, probs, lse in softmax_steps(u, picks):
        loglik += float((u[rows, chosen] - lse).sum())
        grad_u -= probs
        grad_u[rows, chosen] += 1.0
    return loglik, grad_u


def split_theta(theta: np.ndarray, data: list[EditionBallots]) -> tuple[list[np.ndarray], np.ndarray]:
    sizes = [len(d.player_keys) for d in data]
    bounds = np.cumsum([0, *sizes])
    alphas = [theta[bounds[i]:bounds[i + 1]] for i in range(len(data))]
    return alphas, theta[bounds[-1]:]


def penalised_negative_loglik(theta: np.ndarray, data: list[EditionBallots], lam: float) -> tuple[float, np.ndarray]:
    alphas, beta = split_theta(theta, data)
    value = 0.0
    grad_alphas: list[np.ndarray] = []
    grad_beta = np.zeros(N_BETA)
    for d, alpha in zip(data, alphas):
        loglik, grad_u = edition_loglik_and_grad_u(utilities(d, alpha, beta), d.picks)
        value -= loglik
        grad_alphas.append(-grad_u.sum(axis=0))
        grad_beta -= np.einsum("bp,bpk->k", grad_u, d.indicators)
    alpha_all = np.concatenate(alphas)
    value += lam * float(alpha_all @ alpha_all)
    grad = np.concatenate([np.concatenate(grad_alphas) + 2.0 * lam * alpha_all, grad_beta])
    return value, grad


def penalised_hessian(theta: np.ndarray, data: list[EditionBallots], lam: float) -> np.ndarray:
    """Hessian of the penalised negative log-likelihood over theta = [alpha per edition..., beta]."""
    alphas, beta = split_theta(theta, data)
    sizes = [len(d.player_keys) for d in data]
    bounds = np.cumsum([0, *sizes])
    n_alpha = int(bounds[-1])
    hess = np.zeros((n_alpha + N_BETA, n_alpha + N_BETA))
    for i, (d, alpha) in enumerate(zip(data, alphas)):
        u = utilities(d, alpha, beta)
        curvature = np.zeros((u.shape[0], u.shape[1], u.shape[1]))
        for _, _, probs, _ in softmax_steps(u, d.picks):
            curvature += np.einsum("bp,pq->bpq", probs, np.eye(u.shape[1])) - np.einsum("bp,bq->bpq", probs, probs)
        lo, hi = bounds[i], bounds[i + 1]
        cross = np.einsum("bpq,bqk->pk", curvature, d.indicators)
        hess[lo:hi, lo:hi] += curvature.sum(axis=0)
        hess[lo:hi, n_alpha:] += cross
        hess[n_alpha:, lo:hi] += cross.T
        hess[n_alpha:, n_alpha:] += np.einsum("bpk,bpq,bql->kl", d.indicators, curvature, d.indicators)
    hess[:n_alpha, :n_alpha] += 2.0 * lam * np.eye(n_alpha)
    return hess


def optimise_theta(data: list[EditionBallots], lam: float, start: np.ndarray) -> np.ndarray:
    result = minimize(penalised_negative_loglik, start, args=(data, lam), jac=True, method="L-BFGS-B",
                      options={"maxiter": FIT_MAX_ITER, "ftol": FIT_FTOL, "gtol": FIT_GTOL})
    if not result.success:
        raise RuntimeError(f"ballot PL fit did not converge: {result.message}")
    return result.x


def zero_theta(data: list[EditionBallots]) -> np.ndarray:
    return np.zeros(sum(len(d.player_keys) for d in data) + N_BETA)


def fit_bias_model(data: list[EditionBallots], lam: float) -> BiasFit:
    theta = optimise_theta(data, lam, zero_theta(data))
    alphas, beta = split_theta(theta, data)
    covariance = np.linalg.inv(penalised_hessian(theta, data, lam))[-N_BETA:, -N_BETA:]
    loglik = -penalised_negative_loglik(theta, data, 0.0)[0]
    return BiasFit(editions=tuple(d.edition for d in data), alpha={d.edition: a.copy() for d, a in zip(data, alphas)},
                   beta=beta.copy(), covariance=covariance, log_likelihood=float(loglik))


def bootstrap_beta(data: list[EditionBallots], lam: float, fit: BiasFit, n_boot: int, seed: int) -> np.ndarray:
    """Refit on ballots resampled within edition; returns n_boot x N_BETA, warm-started from the main fit."""
    rng = np.random.default_rng(seed)
    start = np.concatenate([fit.alpha[d.edition] for d in data] + [fit.beta])
    draws = np.zeros((n_boot, N_BETA))
    for b in range(n_boot):
        resampled = [resample_ballots(d, rng) for d in data]
        draws[b] = split_theta(optimise_theta(resampled, lam, start), resampled)[1]
    return draws
