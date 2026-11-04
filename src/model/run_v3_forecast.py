"""Experiment 3, section 4: EXPLORATORY post-hoc 2026 forecast from M1 trained on the four season editions only.

Reads no 2026 outcome (the comparison with the actual result lives in compare_2026_v3.py and runs afterwards). Never the forecast of
record (tag forecast-v2)."""

import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.model import ballots_v2 as b2
from src.model import forecast_ballots as fb
from src.model import forecast_v2 as f2
from src.model import inference, link
from src.model.data import V2_SPEC, load_study_v2_1
from src.model.features_v2 import V2_FEATURES, m1_columns_v2
from src.model.forecast_ballots import BallotParams
from src.model.report import git_commit, write_csv_with_header
from src.model.research_v2 import pl_spec_for
from src.model.run_stage6 import N_BOOT, N_SIMS, SEED, assert_targets_null, log
from src.model.run_v2_1 import v2_1_options
from src.model.validation_v2 import SEASON_FOLDS

OUT_DIR = Path("results") / "v3"
FLAG = "# EXPLORATORY, post-hoc: the 2026 result was known when this was written. Not the forecast of record (tag forecast-v2)."
SEASON_ERA_LABEL = "2022-2025"


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    f2.refuse_if_result_present(root)
    full_study = load_study_v2_1(root, V2_SPEC, v2_1_options(), V2_FEATURES)
    study = replace(full_study, labelled_editions=SEASON_FOLDS)
    assert_targets_null(study)
    header = [FLAG, f"# sha256_t1={study.sha256_t1}", f"# git_commit={git_commit(root)}",
              "# model=M1 (PL, v2.1 features) trained on the four season editions 2022-2025 only (E3-S setting)",
              f"# seed={SEED} n_simulations={N_SIMS} n_bootstrap={N_BOOT}"]

    fit = inference.final_fit(study, pl_spec_for("M1", m1_columns_v2()))
    log(f"final fit on {SEASON_ERA_LABEL} done (lambda {fit.lam})", started)
    coefficients = inference.laplace_summary(fit).assign(**{"lambda": fit.lam})
    write_csv_with_header(coefficients, out / "coefficients_M1_final_v3.csv", header)

    ballots = b2.load_ballots_v2(root)
    strengths = link.oof_strengths(study, "T1", SEASON_FOLDS)
    era_alphas, season_fit = f2.era_alpha_table(ballots)
    params = BallotParams(beta=season_fit.beta, beta_covariance=season_fit.covariance,
                          link_fit=link.fit_link(era_alphas, strengths, "T1", SEASON_FOLDS))
    log(f"link re-estimated on the season editions: k={params.link_fit.k:.4f} sigma_shared={params.link_fit.sigma_shared:.4f}", started)

    inputs = f2.inputs_2026_v2(root, study)
    if inputs.n_jurors != 100:
        raise ValueError(f"expected 100 juror nations for 2026, got {inputs.n_jurors}")
    table = fb.forecast_table_ballots(study, fit, inputs, params, 0, N_SIMS, SEED)
    write_csv_with_header(table, out / "forecast_2026_v3_T1.csv", header + [
        f"# juror_bias=ESTIMATED from 2022-2025 ballots: b_home={params.beta[0]:.4f} b_confed={params.beta[1]:.4f} b_league={params.beta[2]:.4f}",
        f"# link (season editions, out-of-fold strengths from 3-edition fits): k={params.link_fit.k:.4f} sigma_shared={params.link_fit.sigma_shared:.4f} "
        f"n_jurors={inputs.n_jurors} lambda={fit.lam}"])
    log(f"done; top: {table.player_name.iloc[0]} (P(win) {table.p_win.max():.1%})", started)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
