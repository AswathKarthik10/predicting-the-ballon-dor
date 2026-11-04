"""2026 forecast from the ballot-estimated juror simulation, and its comparison with the original placeholder forecast."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ballots as bl
from src.model import forecast, inference, jurors, link, simulate
from src.model.data import FORECAST_EDITION, StudyData

HEADLINE_SIMS = 10_000
SEED = 7
JURORS_DROPPED_SENSITIVITY = 5
FORECAST_FILES: dict[str, str] = {"T1": "forecast_2026_T1", "T2": "forecast_2026_T2"}
COMPARED_METRICS: tuple[str, ...] = ("p_win", "p_top3", "expected_rank")


@dataclass(frozen=True)
class BallotParams:
    beta: np.ndarray
    beta_covariance: np.ndarray
    link_fit: link.LinkFit


def full_params(ballot_table: pd.DataFrame, strengths: pd.DataFrame, snapshot: str) -> BallotParams:
    data = bl.build_edition_ballots(ballot_table, bl.BALLOT_EDITIONS)
    fit = bl.fit_bias_model(data, bl.RIDGE_ALPHA)
    return BallotParams(beta=fit.beta, beta_covariance=fit.covariance,
                        link_fit=link.fit_link(link.alpha_by_player(fit, data), strengths, snapshot, bl.BALLOT_EDITIONS))


def inputs_2026(root: Path, study: StudyData) -> jurors.SimInputs:
    attributes = jurors.player_attributes(root, FORECAST_EDITION)
    expected = tuple(study.frame_t1.loc[study.frame_t1.edition == FORECAST_EDITION, "player_key"])
    if attributes.player_keys != expected:
        raise ValueError("2026 player order differs between the juror inputs and the study frame")
    return jurors.build_real_sim_inputs(jurors.real_jury(root, FORECAST_EDITION), attributes)


def simulate_snapshot_ballots(fit: inference.FinalFit, inputs: jurors.SimInputs, params: BallotParams, n_dropped: int,
                              n_sims: int, seed: int) -> pd.DataFrame:
    """u from the Laplace draws of the published model, beta from the ballot model's Laplace distribution."""
    rng = np.random.default_rng(seed)
    utilities = inference.laplace_draws(fit, n_sims, rng) @ fit.matrices[FORECAST_EDITION].T
    betas = rng.multivariate_normal(params.beta, params.beta_covariance, size=n_sims)
    config = simulate.BallotSimConfig(k=params.link_fit.k, sigma_shared=params.link_fit.sigma_shared,
                                      scheme=bl.SCHEME_BY_EDITION[FORECAST_EDITION], n_dropped=n_dropped)
    return simulate.summarise(simulate.simulate_ballots(utilities, betas, inputs, config, rng))


def forecast_table_ballots(study: StudyData, fit: inference.FinalFit, inputs: jurors.SimInputs, params: BallotParams, n_dropped: int,
                           n_sims: int, seed: int) -> pd.DataFrame:
    return forecast.assemble_forecast_table(study, fit, simulate_snapshot_ballots(fit, inputs, params, n_dropped, n_sims, seed))


def read_original(root: Path, snapshot: str) -> pd.DataFrame:
    return pd.read_csv(root / "results" / f"{FORECAST_FILES[snapshot]}.csv", comment="#")


def comparison_table(original: dict[str, pd.DataFrame], ballots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for snapshot in original:
        merged = original[snapshot][["player_key", "player_name", *COMPARED_METRICS]].merge(
            ballots[snapshot][["player_key", *COMPARED_METRICS]], on="player_key", suffixes=("_original", "_ballots"), validate="1:1")
        merged.insert(0, "snapshot", snapshot)
        for metric in COMPARED_METRICS:
            merged[f"{metric}_change"] = merged[f"{metric}_ballots"] - merged[f"{metric}_original"]
        rows.append(merged.sort_values("p_win_ballots", ascending=False))
    return pd.concat(rows, ignore_index=True)


def sensitivity_table(full: dict[str, pd.DataFrame], dropped: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for snapshot in full:
        merged = full[snapshot][["player_key", "player_name", *COMPARED_METRICS]].merge(
            dropped[snapshot][["player_key", *COMPARED_METRICS]], on="player_key", suffixes=("_100_jurors", "_95_jurors"), validate="1:1")
        merged.insert(0, "snapshot", snapshot)
        rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def forecast_without_player(study: StudyData, fit: inference.FinalFit, inputs: jurors.SimInputs, params: BallotParams, player_key: str,
                            n_sims: int, seed: int) -> pd.DataFrame:
    """Counterfactual: the player is removed from every ballot, so the remaining nominees compete. Same draws as the main forecast."""
    keys = study.frame_t1[study.frame_t1["edition"] == FORECAST_EDITION][["player_key", "player_name", "nationality"]].reset_index(drop=True)
    keep = np.flatnonzero(keys["player_key"].to_numpy() != player_key)
    if len(keep) != len(keys) - 1:
        raise ValueError(f"player_key {player_key!r} is not a unique 2026 nominee")
    reduced = jurors.SimInputs(nation_match=inputs.nation_match[:, keep], confed_match=inputs.confed_match[:, keep],
                               league_match=inputs.league_match[:, keep], n_jurors=inputs.n_jurors)
    rng = np.random.default_rng(seed)
    utilities = (inference.laplace_draws(fit, n_sims, rng) @ fit.matrices[FORECAST_EDITION].T)[:, keep]
    betas = rng.multivariate_normal(params.beta, params.beta_covariance, size=n_sims)
    config = simulate.BallotSimConfig(k=params.link_fit.k, sigma_shared=params.link_fit.sigma_shared,
                                      scheme=bl.SCHEME_BY_EDITION[FORECAST_EDITION], n_dropped=0)
    summary = simulate.summarise(simulate.simulate_ballots(utilities, betas, reduced, config, rng))
    table = pd.concat([keys.iloc[keep].reset_index(drop=True), summary], axis=1)
    table = table.sort_values(["expected_rank", "p_win"], ascending=[True, False]).reset_index(drop=True)
    table.insert(0, "forecast_rank", np.arange(1, len(table) + 1))
    return table
