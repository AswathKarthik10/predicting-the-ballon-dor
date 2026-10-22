"""Calibration, sensitivity and the 2026 forecast tables built on the juror simulation (spec 6.3)."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.model import baselines, cv, explain, inference, jurors, simulate
from src.model.data import FORECAST_EDITION, TRAIN_EDITIONS, StudyData
from src.model.jurors import BiasParams, PLACEHOLDER_BIAS

SEED = 7
HEADLINE_SIMS = 10_000
GRID_SIMS = 2_000
CALIBRATION_EDITIONS: tuple[int, ...] = (2024, 2025)
TAU_JUROR_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TAU_SHARED_DEFAULT = 1.0
TAU_SHARED_GRID: tuple[float, ...] = (0.5, 1.0, 1.5)
HOME_GRID: tuple[float, ...] = (0.0, 0.8, 1.6)
CONFED_GRID: tuple[float, ...] = (0.0, 0.2, 0.4)
STAT_NAMES: tuple[str, ...] = ("winner_share", "n_with_points", "herfindahl")


@dataclass(frozen=True)
class Calibration:
    table: pd.DataFrame
    best_tau_juror: float


def oof_utilities(study: StudyData, edition: int) -> np.ndarray:
    """M1 utilities for a labelled edition from a model that never saw that edition."""
    train = [ed for ed in TRAIN_EDITIONS if ed != edition]
    utilities, _ = cv.predict_model(study, "M1", train, [edition])
    return utilities[edition]


def oof_utilities_for(study: StudyData, model: str, edition: int) -> np.ndarray:
    """Out-of-fold utilities of any model (M1 on T1, M2 on T2) for a labelled edition; rows follow study.frame_t1."""
    train = [ed for ed in TRAIN_EDITIONS if ed != edition]
    utilities, _ = cv.predict_model(study, model, train, [edition])
    return utilities[edition]


def calibrate_tau_juror(study: StudyData, juror_inputs: pd.DataFrame, n_sims: int, seed: int) -> Calibration:
    rows: list[dict[str, object]] = []
    for edition in CALIBRATION_EDITIONS:
        players = jurors.edition_players(juror_inputs, edition)
        inputs = jurors.build_sim_inputs(players, jurors.JURORS_BY_EDITION[edition])
        observed = simulate.concentration_stats(baselines.points_of(study, edition))
        strengths = np.tile(oof_utilities(study, edition), (n_sims, 1))
        for tau in TAU_JUROR_GRID:
            config = simulate.SimConfig(tau_shared=TAU_SHARED_DEFAULT, tau_juror=tau, bias=PLACEHOLDER_BIAS)
            result = simulate.simulate(strengths, inputs, config, np.random.default_rng(seed))
            sim = simulate.simulated_concentration(result)
            for stat in STAT_NAMES:
                rows.append({"edition": edition, "tau_juror": tau, "statistic": stat, "observed": observed[stat], "simulated": sim[stat],
                             "squared_relative_error": ((sim[stat] - observed[stat]) / observed[stat]) ** 2})
    table = pd.DataFrame(rows)
    loss = table.groupby("tau_juror")["squared_relative_error"].sum()
    return Calibration(table=table, best_tau_juror=float(loss.idxmin()))


def simulate_snapshot(fit: inference.FinalFit, inputs: jurors.SimInputs, config: simulate.SimConfig, n_sims: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    draws = inference.laplace_draws(fit, n_sims, rng)
    strengths = draws @ fit.matrices[FORECAST_EDITION].T
    return simulate.summarise(simulate.simulate(strengths, inputs, config, rng))


def assemble_forecast_table(study: StudyData, fit: inference.FinalFit, sim: pd.DataFrame) -> pd.DataFrame:
    """Attach player identity and utility contributions to a simulation summary and rank by expected rank."""
    keys = study.frame_t1[study.frame_t1["edition"] == FORECAST_EDITION][["player_key", "player_name", "nationality"]].reset_index(drop=True)
    contrib = explain.contribution_table(study, fit, FORECAST_EDITION).drop(columns="player_name")
    table = pd.concat([keys, sim], axis=1).merge(contrib, on="player_key")
    table = table.sort_values(["expected_rank", "p_win"], ascending=[True, False]).reset_index(drop=True)
    table.insert(0, "forecast_rank", np.arange(1, len(table) + 1))
    return table


def forecast_table(study: StudyData, fit: inference.FinalFit, inputs: jurors.SimInputs, config: simulate.SimConfig,
                   n_sims: int, seed: int) -> pd.DataFrame:
    return assemble_forecast_table(study, fit, simulate_snapshot(fit, inputs, config, n_sims, seed))


def sensitivity_scenarios(best_tau_juror: float) -> list[tuple[str, simulate.SimConfig]]:
    scenarios: list[tuple[str, simulate.SimConfig]] = []
    for home in HOME_GRID:
        for confed in CONFED_GRID:
            bias = BiasParams(b_home=home, b_confed=confed, b_league=PLACEHOLDER_BIAS.b_league)
            scenarios.append(("bias_grid", simulate.SimConfig(TAU_SHARED_DEFAULT, best_tau_juror, bias)))
    for tau_shared in TAU_SHARED_GRID:
        scenarios.append(("tau_shared", simulate.SimConfig(tau_shared, best_tau_juror, PLACEHOLDER_BIAS)))
    for tau_juror in TAU_JUROR_GRID:
        scenarios.append(("tau_juror", simulate.SimConfig(TAU_SHARED_DEFAULT, tau_juror, PLACEHOLDER_BIAS)))
    return scenarios


def run_sensitivity(snapshots: dict[str, inference.FinalFit], inputs: jurors.SimInputs, best_tau_juror: float, n_sims: int,
                    seed: int, names: dict[str, pd.Series]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for kind, config in sensitivity_scenarios(best_tau_juror):
        for snapshot, fit in snapshots.items():
            sim = simulate_snapshot(fit, inputs, config, n_sims, seed)[["p_win", "p_top3", "expected_rank"]]
            sim.insert(0, "player_name", names[snapshot].to_numpy())
            sim.insert(0, "snapshot", snapshot)
            sim.insert(0, "tau_juror", config.tau_juror)
            sim.insert(0, "tau_shared", config.tau_shared)
            sim.insert(0, "b_league", config.bias.b_league)
            sim.insert(0, "b_confed", config.bias.b_confed)
            sim.insert(0, "b_home", config.bias.b_home)
            sim.insert(0, "scenario", kind)
            rows.append(sim)
    out = pd.concat(rows, ignore_index=True)
    out["bias_label"] = jurors.PLACEHOLDER_LABEL
    return out


def with_tau(config: simulate.SimConfig, tau_juror: float) -> simulate.SimConfig:
    return replace(config, tau_juror=tau_juror)
