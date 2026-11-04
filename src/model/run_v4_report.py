"""Experiment 4, spec sections 7 and 8 (EXPLORATORY, post-hoc): what the pipeline chose, the 2026 forecast and its comparison.

Needs results/v4/chosen_final_v4.csv from run_v4_final_fit. The final model is a PL ridge on lasso-selected columns, which is the only
case implemented here; any other final model raises. Never the forecast of record (tag forecast-v2). Writes results/v4/."""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.model import ballots_v2 as b2
from src.model import forecast_v2 as f2
from src.model import link, simulate, v4_candidates as vcand, v4_report as vr, v4_search as vs
from src.model.ballots import SCHEME_BY_EDITION
from src.model.compare_2026_v3 import compare_one
from src.model.data import FORECAST_EDITION, V2_SPEC, load_study_v2_1
from src.model.features_v2 import V2_FEATURES
from src.model.forecast_ballots import HEADLINE_SIMS, BallotParams
from src.model.report import git_commit, write_csv_with_header
from src.model.run_v2_1 import v2_1_options
from src.model.run_v4_backtest import describe, refit_chosen
from src.model.run_v4_final_fit import LABELLED, build_panel
from src.model.v4_models import ModelKind, fit_pl_beta
from src.model.v4_select import STABILITY_PENALTY, make_train_set, stability_frequencies
from src.model.validation_v2 import SEASON_FOLDS

OUT_DIR = Path("results") / "v4"
V3_COMPARISON = Path("results") / "v3" / "comparison_2026_all_versions.csv"
FLAG = "# EXPLORATORY, post-hoc: the 2026 result was known when this was written. Not the forecast of record (tag forecast-v2)."
SEED = 7
TOP_PLAYERS = 5
TOP_FEATURES = 10
VERSION_LABEL = "v4 (hypothesis-free)"
L1_MODELS = ("P-L", "P-EN")


def median_l1_penalty(chosen: pd.DataFrame) -> float:
    """Median of the lasso-type penalties chosen anywhere (selector FS2 lambda, P-L / P-EN lambda); STABILITY_PENALTY if none."""
    penalties = list(chosen.loc[chosen.selector == "FS2", "selector_lambda"]) + list(chosen.loc[chosen.model.isin(L1_MODELS), "model_lambda"])
    return float(np.median(penalties)) if penalties else STABILITY_PENALTY


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    f2.refuse_if_result_present(root)
    panel = build_panel(root)
    configs = vs.sample_configs(vs.full_grid(), vs.SEARCH_BUDGET, vs.SEARCH_SEED)
    final_row = pd.read_csv(out / "chosen_final_v4.csv").iloc[0]
    config = configs[int(final_row.config_index)]
    if config.model.kind is not ModelKind.RIDGE:
        raise NotImplementedError(f"report implemented for a ridge final model, got {config.model.kind.value}")
    if any(describe(config)[k] != final_row[k] for k in ("preprocessing", "selector", "model", "model_lambda", "selector_lambda")):
        raise ValueError("chosen_final_v4.csv does not match the regenerated configuration list")

    cache = vs.FitCache()
    prepared = vs.prepared_fold(panel, LABELLED, config.preprocess, cache)
    columns = vs.selected_columns(prepared, LABELLED, config.preprocess, config.selector, cache)
    names = [panel.columns[c] for c in columns]
    train = make_train_set(LABELLED, [x[:, columns] for x in prepared.train.xs], prepared.train.points)
    z_2026 = prepared.z[panel.edition == FORECAST_EDITION][:, columns]
    fit = fit_pl_beta(config.model, train)
    beta = fit.beta
    print(f"final model: {config.name}; {len(columns)} features; converged={fit.converged}", flush=True)

    candidates = vcand.build_candidates(vcand.load_raw_t1(root))
    source_of = dict(zip(candidates.table.column, candidates.table.source_column))
    write_importance(out, train, config, beta, names, source_of)
    print(f"importance done in {time.time() - started:.0f}s", flush=True)
    write_stability(out, prepared.train, panel.columns, final_row)
    write_block_overlap(out, panel.columns, source_of, names)

    table = forecast_table(root, panel, configs, config, train, beta, z_2026, names)
    write_csv_with_header(table, out / "forecast_2026_v4_T1.csv", [
        FLAG, f"# git_commit={git_commit(root)}", f"# model={config.name}", f"# features={len(columns)} seed={SEED} n_simulations={HEADLINE_SIMS}"])
    comparison = pd.concat([pd.read_csv(root / V3_COMPARISON), pd.DataFrame([compare_one(VERSION_LABEL, table)])], ignore_index=True)
    comparison.to_csv(out / "comparison_2026_all_versions.csv", index=False)
    write_top_features(out, table, candidates.rows.loc[candidates.rows.edition == FORECAST_EDITION, "player_key"].to_numpy(), z_2026, beta, names, source_of)
    print(comparison.round(3).to_string(index=False), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


def write_importance(out: Path, train, config: vs.Config, beta: np.ndarray, names: list[str], source_of: dict[str, str]) -> None:
    rng = np.random.default_rng(SEED)
    laplace = vr.laplace_draws(train, beta, config.model.lam, 2000, rng)
    boot = vr.bootstrap_betas(config.model, train, vr.N_BOOTSTRAP, SEED)
    permutation = vr.permutation_importance(train, lambda x: x @ beta, np.arange(len(names)), vr.N_PERMUTATIONS, SEED)
    lo, hi = vr.INTERVAL_PCT
    table = pd.DataFrame({"feature": names, "source_column": [source_of[n] for n in names], "v3_block": [vr.block_of(source_of[n]) for n in names],
                          "coefficient": beta, "laplace_lo90": np.percentile(laplace, lo, axis=0), "laplace_hi90": np.percentile(laplace, hi, axis=0),
                          "bootstrap_lo90": np.percentile(boot, lo, axis=0), "bootstrap_hi90": np.percentile(boot, hi, axis=0),
                          "permutation_drop_ll": permutation})
    table.iloc[np.argsort(-np.abs(beta))].to_csv(out / "importance_final_v4.csv", index=False)


def write_stability(out: Path, prepared_train, all_columns: tuple[str, ...], final_row: pd.Series) -> None:
    chosen = pd.concat([pd.read_csv(out / "chosen_by_fold_v4.csv"), final_row.to_frame().T], ignore_index=True)
    penalty = median_l1_penalty(chosen)
    frequencies = stability_frequencies(prepared_train, penalty)
    by_fold = pd.read_csv(out / "selected_features_by_fold.csv").set_index("feature")
    fold_share = by_fold.reindex(list(all_columns)).fillna(0).mean(axis=1)
    pd.DataFrame({"feature": list(all_columns), "stability_frequency": frequencies, "outer_fold_share": fold_share.to_numpy(),
                  "lasso_penalty": penalty}).sort_values("stability_frequency", ascending=False).to_csv(out / "stability_selection_v4.csv", index=False)


def write_block_overlap(out: Path, all_columns: tuple[str, ...], source_of: dict[str, str], final_names: list[str]) -> None:
    by_fold = pd.read_csv(out / "selected_features_by_fold.csv").set_index("feature")
    selected = {f"fold_{fold}": list(by_fold.index[by_fold[fold] == 1]) for fold in by_fold.columns}
    selected["final"] = final_names
    vr.block_overlap(selected, source_of).to_csv(out / "block_overlap_v4.csv", index=False)


def forecast_table(root: Path, panel: vs.Panel, configs: list[vs.Config], config: vs.Config, train, beta: np.ndarray, z_2026: np.ndarray,
                   names: list[str]) -> pd.DataFrame:
    study = replace(load_study_v2_1(root, V2_SPEC, v2_1_options(), V2_FEATURES), labelled_editions=SEASON_FOLDS)
    identity = study.frame_t1[study.frame_t1.edition == FORECAST_EDITION][["player_key", "player_name", "nationality"]].reset_index(drop=True)
    candidates = vcand.build_candidates(vcand.load_raw_t1(root))
    panel_keys = candidates.rows.loc[candidates.rows.edition == FORECAST_EDITION, "player_key"].reset_index(drop=True)
    if not identity.player_key.equals(panel_keys):
        raise ValueError("2026 player order differs between the v4 panel and the study frame")

    log = pd.read_csv(root / OUT_DIR / "search_log_v4.csv")
    oof = [refit_chosen(panel, configs, log, held) for held in LABELLED]
    strengths = pd.concat([pd.DataFrame({"edition": f.held_out, "player_key": candidates.rows.loc[candidates.rows.edition == f.held_out, "player_key"].to_numpy(),
                                         "u": f.utility}) for f in oof], ignore_index=True)
    ballots = b2.load_ballots_v2(root)
    era_alphas, season_fit = f2.era_alpha_table(ballots)
    params = BallotParams(beta=season_fit.beta, beta_covariance=season_fit.covariance,
                          link_fit=link.fit_link(era_alphas, strengths, "T1", SEASON_FOLDS))
    inputs = f2.inputs_2026_v2(root, study)

    rng = np.random.default_rng(SEED)
    utilities = vr.laplace_draws(train, beta, config.model.lam, HEADLINE_SIMS, rng) @ z_2026.T
    betas = rng.multivariate_normal(params.beta, params.beta_covariance, size=HEADLINE_SIMS)
    sim_config = simulate.BallotSimConfig(k=params.link_fit.k, sigma_shared=params.link_fit.sigma_shared,
                                          scheme=SCHEME_BY_EDITION[FORECAST_EDITION], n_dropped=0)
    sim = simulate.summarise(simulate.simulate_ballots(utilities, betas, inputs, sim_config, rng))
    utility = z_2026 @ beta
    table = pd.concat([identity, pd.DataFrame({"utility": utility, "p_first_plain_pl": softmax(utility)}), sim], axis=1)
    table = table.sort_values(["expected_rank", "p_win"], ascending=[True, False]).reset_index(drop=True)
    table.insert(0, "forecast_rank", np.arange(1, len(table) + 1))
    return table


def write_top_features(out: Path, table: pd.DataFrame, keys_2026: np.ndarray, z_2026: np.ndarray, beta: np.ndarray, names: list[str],
                       source_of: dict[str, str]) -> None:
    """Features that push the forecast top five above the 2026 field: summed utility contribution beta * (z - field mean)."""
    contributions = vr.coefficient_contributions(z_2026, beta, names)
    contributions.index = keys_2026
    top = table.head(TOP_PLAYERS)
    per_player = contributions.loc[top.player_key].set_axis(list(top.player_name), axis=0).T
    per_player.insert(0, "total_top5", per_player.sum(axis=1))
    per_player.insert(0, "v3_block", [vr.block_of(source_of[n]) for n in per_player.index])
    result = per_player.reindex(per_player.total_top5.abs().sort_values(ascending=False).index).head(TOP_FEATURES)
    result.rename_axis("feature").reset_index().to_csv(out / "top_features_2026_v4.csv", index=False)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
