"""Stage 6: final fit and 2026 forecast of the selected model (experiments spec 9.1-9.2). Written before the 2026 result is added.

The model was chosen from the 2016-2025 back-test only (results/v2/selection.md). No 2026 target is read."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import ballots_v2 as b2
from src.model import forecast_ballots as fb
from src.model import forecast_v2 as f2
from src.model import inference, link
from src.model.data import EDITION_COL, StudyData, load_study_v2, sha256_of
from src.model.explain import contribution_table, feature_columns
from src.model.features_v2 import m1_columns_v2
from src.model.report import git_commit, write_csv_with_header
from src.model.research_v2 import pl_spec_for

OUT_DIR = Path("results") / "v2"
SELECTED_MODEL = "M1"
N_BOOT = 500
N_SIMS = fb.HEADLINE_SIMS
SEED = fb.SEED
TOP_EXPLAINED = 8
N_TERMS = 3
V1_FORECAST = Path("results") / "forecast_2026_T1_ballots.csv"


def log(message: str, started: float) -> None:
    print(f"[{time.time() - started:6.0f}s] {message}", flush=True)


def assert_targets_null(study: StudyData) -> None:
    if not study.targets.loc[study.frame_t1[EDITION_COL] == study.forecast_edition].isna().all().all():
        raise ValueError("2026 targets must be null")


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    f2.refuse_if_result_present(root)
    study = load_study_v2(root)
    assert_targets_null(study)
    header = [f"# sha256_t1={study.sha256_t1}", f"# git_commit={git_commit(root)}", f"# model={SELECTED_MODEL} (ten-edition PL, v2 features)",
              f"# seed={SEED} n_simulations={N_SIMS} n_bootstrap={N_BOOT}"]

    fit = inference.final_fit(study, pl_spec_for(SELECTED_MODEL, m1_columns_v2()))
    fold_coefs = pd.read_csv(out / "coefficients_by_fold.csv")
    write_csv_with_header(inference.coefficient_table(fit, study, fold_coefs, N_BOOT, SEED), out / "coefficients_M1.csv", header + [f"# lambda={fit.lam}"])
    ranking = contribution_table(study, fit, study.forecast_edition)
    write_csv_with_header(ranking, out / "utility_ranking_2026_T1.csv", header)
    log(f"final fit done (lambda {fit.lam})", started)

    ballots = b2.load_ballots_v2(root)
    strengths = link.oof_strengths(study, "T1", b2.BALLOT_EDITIONS_V2)
    era_alphas, season_fit = f2.era_alpha_table(ballots)
    params = f2.ballot_params(era_alphas, season_fit, strengths)
    pooled_alphas, pooled_fit = f2.pooled_alpha_table(ballots)
    pooled_params = f2.ballot_params(pooled_alphas, pooled_fit, strengths)
    link_table = pd.DataFrame([
        {"alpha_source": "own jury era (main)", "bias_source": "2022-2025 fit", "k": params.link_fit.k, "sigma_shared": params.link_fit.sigma_shared,
         "r2_within_edition": params.link_fit.r2_within, "n_player_editions": params.link_fit.n_players,
         "b_home": params.beta[0], "b_confed": params.beta[1], "b_league": params.beta[2]},
        {"alpha_source": "pooled nine-edition fit (sensitivity)", "bias_source": "pooled 2016-2025 fit", "k": pooled_params.link_fit.k,
         "sigma_shared": pooled_params.link_fit.sigma_shared, "r2_within_edition": pooled_params.link_fit.r2_within,
         "n_player_editions": pooled_params.link_fit.n_players, "b_home": pooled_params.beta[0], "b_confed": pooled_params.beta[1],
         "b_league": pooled_params.beta[2]}])
    link_table.to_csv(out / "juror_link_v2.csv", index=False)
    log(f"link re-estimated on nine editions: k={params.link_fit.k:.3f} sigma_shared={params.link_fit.sigma_shared:.3f}", started)

    inputs = f2.inputs_2026_v2(root, study)
    if inputs.n_jurors != 100:
        raise ValueError(f"expected 100 juror nations for 2026, got {inputs.n_jurors}")
    main_table = fb.forecast_table_ballots(study, fit, inputs, params, 0, N_SIMS, SEED)
    lines = header + [f"# juror_bias=ESTIMATED from 2022-2025 ballots: b_home={params.beta[0]:.4f} b_confed={params.beta[1]:.4f} b_league={params.beta[2]:.4f}",
                      f"# link (nine editions): k={params.link_fit.k:.4f} sigma_shared={params.link_fit.sigma_shared:.4f} n_jurors={inputs.n_jurors}"]
    write_csv_with_header(main_table, out / "forecast_2026_v2_T1.csv", lines)
    log("main forecast done", started)

    pooled = fb.forecast_table_ballots(study, fit, inputs, pooled_params, 0, N_SIMS, SEED)
    dropped = fb.forecast_table_ballots(study, fit, inputs, params, fb.JURORS_DROPPED_SENSITIVITY, N_SIMS, SEED)
    keep = ["player_key", "p_win", "p_top3", "expected_rank"]
    sensitivity = main_table[["player_key", "player_name", *keep[1:]]].merge(
        pooled[keep].rename(columns=lambda c: c if c == "player_key" else f"{c}_pooled_bias"), on="player_key").merge(
        dropped[keep].rename(columns=lambda c: c if c == "player_key" else f"{c}_95_jurors"), on="player_key")
    write_csv_with_header(sensitivity, out / "forecast_2026_v2_sensitivity.csv", header + ["# main = 2022-2025 bias; pooled_bias = bias from all nine editions; 95_jurors = 5 of 100 jurors absent"])
    v1 = pd.read_csv(root / V1_FORECAST, comment="#")
    f2.compare_with_v1(main_table, v1).to_csv(out / "forecast_2026_v2_vs_v1.csv", index=False)

    explained = ranking.copy()
    text = ["# Why the v2 model ranks 2026 this way (T1, M1)", "", *[line.replace("# ", "<!-- ", 1) + " -->" for line in header], "",
            "These are the model's arithmetic, not causes. Every feature is a within-edition z-score, so a contribution is measured against the average "
            "2026 nominee. Nothing here is used to tune or change the model.", "",
            *f2.explanation_lines(explained, feature_columns(explained), TOP_EXPLAINED, N_TERMS), ""]
    (out / "explanations_2026_v2.md").write_text("\n".join(text))
    (out / "forecast_inputs.txt").write_text("\n".join(f"sha256 {n}={sha256_of(root / 'data' / 'v2' / n)}" for n in
                                                       ("player_snapshot_t1.csv", "fact_juror_nations.parquet", "fact_bdo_country_votes.parquet")) + "\n")
    log(f"done; top forecast: {main_table.player_name.iloc[0]} (P(win) {np.max(main_table.p_win):.1%})", started)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
