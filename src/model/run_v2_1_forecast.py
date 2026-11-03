"""Awards fix v2.1, B5: EXPLORATORY post-hoc 2026 forecast (awards fix spec B5). Never the forecast of record (that is tag forecast-v2).

Final fit of M1 with the weighted F11 on the nine labelled editions, then the v2 juror simulation unchanged. No 2026 target is read."""

import time
from pathlib import Path

import pandas as pd

from src.model import ballots_v2 as b2
from src.model import forecast_ballots as fb
from src.model import forecast_v2 as f2
from src.model import inference, link
from src.model.data import EDITION_COL, StudyData
from src.model.explain import contribution_table, feature_columns
from src.model.report import git_commit, write_csv_with_header
from src.model.research_v2 import pl_spec_for
from src.model.run_v2_1_sensitivity import RUNS, study_for
from src.model.run_stage6 import N_BOOT, N_SIMS, SEED, TOP_EXPLAINED, N_TERMS, assert_targets_null, log

OUT_DIR = Path("results") / "v2_1"
FLAG = "# EXPLORATORY, post-hoc: F11 redefined after the 2026 result was known (awards fix spec v2.1). Not the forecast of record (tag forecast-v2)."
TOP_N = 5


def top_players(table: pd.DataFrame, run_id: str) -> pd.DataFrame:
    top = table.sort_values("expected_rank").head(TOP_N)
    return top[["player_name", "p_win", "p_top3", "expected_rank"]].assign(run=run_id).reset_index(drop=True)


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    f2.refuse_if_result_present(root)
    runs = {run_id: study_for(root, variant, with_wc) for run_id, _, variant, with_wc in RUNS}
    study, columns = runs["v2.1"]
    assert_targets_null(study)
    header = [FLAG, f"# sha256_t1={study.sha256_t1}", f"# git_commit={git_commit(root)}", "# model=M1 (ten-edition PL, v2 features with F11 v2.1)",
              f"# seed={SEED} n_simulations={N_SIMS} n_bootstrap={N_BOOT}"]

    fit = inference.final_fit(study, pl_spec_for("M1", columns))
    fold_coefs = pd.read_csv(out / "coefficients_by_fold.csv")
    write_csv_with_header(inference.coefficient_table(fit, study, fold_coefs, N_BOOT, SEED), out / "coefficients_M1_final_v2_1.csv", header + [f"# lambda={fit.lam}"])
    ranking = contribution_table(study, fit, study.forecast_edition)
    write_csv_with_header(ranking, out / "utility_ranking_2026_v2_1_T1.csv", header)
    log(f"final fit done (lambda {fit.lam})", started)

    ballots = b2.load_ballots_v2(root)
    strengths = link.oof_strengths(study, "T1", b2.BALLOT_EDITIONS_V2)
    era_alphas, season_fit = f2.era_alpha_table(ballots)
    params = f2.ballot_params(era_alphas, season_fit, strengths)
    inputs = f2.inputs_2026_v2(root, study)
    if inputs.n_jurors != 100:
        raise ValueError(f"expected 100 juror nations for 2026, got {inputs.n_jurors}")
    main_table = fb.forecast_table_ballots(study, fit, inputs, params, 0, N_SIMS, SEED)
    write_csv_with_header(main_table, out / "forecast_2026_v2_1_T1.csv", header + [
        f"# juror_bias=ESTIMATED from 2022-2025 ballots: b_home={params.beta[0]:.4f} b_confed={params.beta[1]:.4f} b_league={params.beta[2]:.4f}",
        f"# link (nine editions, v2.1 out-of-fold strengths): k={params.link_fit.k:.4f} sigma_shared={params.link_fit.sigma_shared:.4f}"])
    log(f"main forecast done; top: {main_table.player_name.iloc[0]}", started)

    tops = [top_players(main_table, "v2.1")]
    for run_id, (run_study, run_columns) in runs.items():
        if run_id == "v2.1":
            continue
        assert_targets_null(run_study)
        run_fit = inference.final_fit(run_study, pl_spec_for(f"M1-{run_id}", run_columns))
        run_inputs = f2.inputs_2026_v2(root, run_study)
        tops.append(top_players(fb.forecast_table_ballots(run_study, run_fit, run_inputs, params, 0, N_SIMS, SEED), run_id))
        log(f"{run_id} done", started)
    write_csv_with_header(pd.concat(tops, ignore_index=True), out / "forecast_2026_v2_1_sensitivity_top5.csv",
                          header + ["# top 5 by expected rank for each run; S1-S4 reuse the main run's juror bias and link"])
    text = ["# Why the v2.1 model ranks 2026 this way (T1, M1) - EXPLORATORY", "", *[line.replace("# ", "<!-- ", 1) + " -->" for line in header], "",
            *f2.explanation_lines(ranking, feature_columns(ranking), TOP_EXPLAINED, N_TERMS), ""]
    (out / "explanations_2026_v2_1.md").write_text("\n".join(text))
    log("done", started)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
