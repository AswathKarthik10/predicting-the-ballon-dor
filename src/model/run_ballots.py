"""End-to-end run of the ballot-based juror simulation (spec section 10, steps 1-5). Writes only `_ballots` outputs."""

from pathlib import Path

import pandas as pd

from src.model import ballots as bl
from src.model import ballots_report as br
from src.model import cv, figures_ballots, forecast_ballots as fb, inference, link, validation_ballots as vb
from src.model.data import FORECAST_EDITION, StudyData, load_study_data
from src.model.report import write_csv_with_header

SNAPSHOTS: tuple[str, ...] = ("T1", "T2")
MODEL_BY_SNAPSHOT = link.MODEL_BY_SNAPSHOT


def assert_2026_targets_null(study: StudyData) -> None:
    if not study.targets.loc[study.frame_t1.edition == FORECAST_EDITION].isna().all().all():
        raise ValueError("2026 targets must be null")


def run_bias_estimates(root: Path, ballot_table: pd.DataFrame, header: list[str]) -> pd.DataFrame:
    estimates, fit, boot = br.estimate_all_specs(ballot_table, bl.N_BOOTSTRAP)
    spec_header = header + [f"# ridge_alpha={bl.RIDGE_ALPHA} n_bootstrap={bl.N_BOOTSTRAP} spec=ballot Plackett-Luce, alpha per player-edition"]
    write_csv_with_header(estimates, root / "results" / "juror_bias_estimates_ballots.csv", spec_header)
    br.write_bias_summary(estimates, br.rank_shift_table(fit, boot), br.descriptive_summary(ballot_table), spec_header,
                          root / "results" / "juror_bias_summary_ballots.md")
    figures_ballots.plot_bias(estimates, root / "figures" / "juror_bias_ballots.png")
    return estimates


def run_link(root: Path, ballot_table: pd.DataFrame, strengths: dict[str, pd.DataFrame], header: list[str]) -> None:
    data = bl.build_edition_ballots(ballot_table, bl.BALLOT_EDITIONS)
    alphas = link.alpha_by_player(bl.fit_bias_model(data, bl.RIDGE_ALPHA), data)
    fits = [link.fit_link(alphas, strengths[s], s, bl.BALLOT_EDITIONS) for s in SNAPSHOTS]
    write_csv_with_header(link.link_table(fits), root / "results" / "juror_link_ballots.csv",
                          header + ["# alpha from the main ballot fit; strengths are out-of-fold"])


def run_validation(root: Path, study: StudyData, ballot_table: pd.DataFrame, strengths: dict[str, pd.DataFrame], header: list[str]) -> pd.DataFrame:
    table = vb.validate(root, study, ballot_table, strengths, vb.VALIDATION_SIMS)
    val_header = header + [f"# n_simulations={vb.VALIDATION_SIMS} seed={vb.SEED} leave_one_edition_out=true"]
    vb.write_validation_report(table, val_header, root / "results" / "juror_validation_ballots.md")
    write_csv_with_header(table, root / "results" / "juror_validation_ballots.csv", val_header)
    return table


def run_forecast(root: Path, study: StudyData, ballot_table: pd.DataFrame, strengths: dict[str, pd.DataFrame], header: list[str]) -> None:
    inputs = fb.inputs_2026(root, study)
    full: dict[str, pd.DataFrame] = {}
    dropped: dict[str, pd.DataFrame] = {}
    for snapshot in SNAPSHOTS:
        model = MODEL_BY_SNAPSHOT[snapshot]
        fit = inference.final_fit(study, cv.PL_SPECS[model])
        params = fb.full_params(ballot_table, strengths[snapshot], snapshot)
        full[snapshot] = fb.forecast_table_ballots(study, fit, inputs, params, 0, fb.HEADLINE_SIMS, fb.SEED)
        dropped[snapshot] = fb.forecast_table_ballots(study, fit, inputs, params, fb.JURORS_DROPPED_SENSITIVITY, fb.HEADLINE_SIMS, fb.SEED)
        lines = header + [
            f"# juror_bias=ESTIMATED from 2022-2025 ballots: b_home={params.beta[0]:.4f} b_confed={params.beta[1]:.4f} b_league={params.beta[2]:.4f}",
            f"# link: k={params.link_fit.k:.4f} sigma_shared={params.link_fit.sigma_shared:.4f} gumbel_scale=1 n_jurors=100 "
            f"n_simulations={fb.HEADLINE_SIMS} seed={fb.SEED}",
            f"# snapshot={snapshot} model={model} ballot-based simulation on the real 2026 jury"]
        write_csv_with_header(full[snapshot], root / "results" / f"forecast_2026_{snapshot}_ballots.csv", lines)
    original = {s: fb.read_original(root, s) for s in SNAPSHOTS}
    comparison = fb.comparison_table(original, full)
    write_csv_with_header(comparison, root / "results" / "forecast_2026_comparison_ballots.csv",
                          header + ["# original = placeholder simulation (results/forecast_2026_T*.csv), unchanged"])
    write_csv_with_header(fb.sensitivity_table(full, dropped), root / "results" / "forecast_2026_sensitivity_95jurors_ballots.csv",
                          header + [f"# {fb.JURORS_DROPPED_SENSITIVITY} of 100 jurors dropped at random per simulation"])
    figures_ballots.plot_win_probability(comparison, root / "figures" / "forecast_2026_win_prob_ballots.png")


def run(root: Path, include_forecast: bool) -> pd.DataFrame:
    """Steps 1-4 always; step 5 (the 2026 forecast) only when include_forecast, after the validation result has been read."""
    study = load_study_data(root)
    assert_2026_targets_null(study)
    ballot_table = bl.load_ballots(root)
    header = br.jury_header_lines(root)
    strengths = {s: link.oof_strengths(study, s, bl.BALLOT_EDITIONS) for s in SNAPSHOTS}
    run_bias_estimates(root, ballot_table, header)
    run_link(root, ballot_table, strengths, header)
    validation = run_validation(root, study, ballot_table, strengths, header)
    if include_forecast:
        run_forecast(root, study, ballot_table, strengths, header)
    return validation
