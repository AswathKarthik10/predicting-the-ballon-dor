"""Robustness runs R1-R9 (experiments spec section 10): the headline model M1 re-run under one change at a time.

Each run repeats the leave-one-edition-out back-test and the final fit with its 2026 utilities. The 2026 utilities are only
compared with the base model's; no decision depends on them."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.model import inference
from src.model.data import EDITION_COL, V2_SPEC, StudySpec, StudyData, load_study_v2, load_study_v2_variant
from src.model.features_v2 import (EA_COLUMN, LEAGUE_POTY_OPTIONS, MAIN_OPTIONS, RESIDUALISER_REGRESSORS, V2_FEATURES, XG_OPTIONS,
                                   m1_columns_v2, make_v2_features)
from src.model.research_v2 import fold_metrics, loeo_predictions, pl_spec_for

RUN_NAME = "M1"
SEASON_EDITIONS: tuple[int, ...] = (2022, 2023, 2024, 2025)
CALENDAR_EDITIONS: tuple[int, ...] = (2016, 2017, 2018, 2019, 2021)
XG_EDITIONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025, 2026)
TOP_N_LIST = 5
OVERLAP_N = 10
SEED = 7


@dataclass(frozen=True)
class RobustnessRun:
    run_id: str
    description: str
    study: StudyData
    columns: tuple[str, ...]


def without(columns: tuple[str, ...], dropped: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(c for c in columns if c not in dropped)


def build_runs(root: Path, base: StudyData) -> list[RobustnessRun]:
    m1 = m1_columns_v2()
    labelled = base.labelled_editions

    def subset(editions: tuple[int, ...]) -> StudyData:
        return replace(base, labelled_editions=tuple(e for e in labelled if e in editions))

    def variant(name: str, dropped: tuple[str, ...]) -> tuple[StudyData, tuple[str, ...]]:
        columns = without(m1, dropped)
        return replace(base, features=make_v2_features(name, without(RESIDUALISER_REGRESSORS, dropped), columns)), columns

    r5, r5_cols = variant("no-ratings", ("F2_form_rating",))
    r7a, r7a_cols = variant("no-market-value", ("F12_market_value",))
    r8, r8_cols = variant("no-fame-attention", ("F9_fame", "F10_season_attention"))
    ea_columns = m1 + (EA_COLUMN,)
    r6 = replace(base, features=make_v2_features("with-ea", RESIDUALISER_REGRESSORS, ea_columns))
    calendar_columns = m1 + ("is_calendar",)
    r9 = replace(base, features=make_v2_features("format-fixed-effect", RESIDUALISER_REGRESSORS, calendar_columns))
    xg_spec = StudySpec(t1_editions=XG_EDITIONS, t2_editions=V2_SPEC.t2_editions, forecast_edition=V2_SPEC.forecast_edition)
    r4 = load_study_v2_variant(root, xg_spec, XG_OPTIONS, V2_FEATURES)
    r7b = load_study_v2_variant(root, V2_SPEC, LEAGUE_POTY_OPTIONS, V2_FEATURES)
    return [
        RobustnessRun("R1", "Season editions only (2022-2025), v2 features", subset(SEASON_EDITIONS), m1),
        RobustnessRun("R2", "Drop 2016 (3-pick ballot, half-covered fame baseline)", subset(tuple(e for e in labelled if e != 2016)), m1),
        RobustnessRun("R3", "Calendar editions only (2016-2021), five folds", subset(CALENDAR_EDITIONS), m1),
        RobustnessRun("R4", "2021-2025 with xG (npxG + xA instead of npG + A in F1)", r4, m1),
        RobustnessRun("R5", "Without ratings (F2)", r5, r5_cols),
        RobustnessRun("R6", "Add EA rating (z of ea_overall_t1)", r6, ea_columns),
        RobustnessRun("R7a", "Without market value (F12)", r7a, r7a_cols),
        RobustnessRun("R7b", "league_poty instead of league_player_of_season in F11", r7b, m1),
        RobustnessRun("R8", "Without fame and attention (F9, F10)", r8, r8_cols),
        RobustnessRun("R9", "Format fixed effect only (is_calendar main effect, no M1-F interactions)", r9, calendar_columns),
    ]


def forecast_utilities(study: StudyData, columns: tuple[str, ...]) -> tuple[np.ndarray, float]:
    """2026 utilities from the final fit (features only; no target is read) and the lambda chosen by leave-one-edition-out."""
    fit = inference.final_fit(study, pl_spec_for(RUN_NAME, columns))
    return fit.matrices[study.forecast_edition] @ fit.beta, fit.lam


def forecast_comparison(utilities: np.ndarray, names: np.ndarray, base_utilities: np.ndarray) -> dict[str, object]:
    top = np.argsort(-utilities)
    base_top = np.argsort(-base_utilities)
    probabilities = np.exp(utilities - utilities.max()) / np.exp(utilities - utilities.max()).sum()
    return {"top5_2026": "; ".join(f"{names[i]} ({probabilities[i]:.0%})" for i in top[:TOP_N_LIST]),
            "top5_overlap_with_base": len(set(top[:TOP_N_LIST]) & set(base_top[:TOP_N_LIST])),
            "top10_overlap_with_base": len(set(top[:OVERLAP_N]) & set(base_top[:OVERLAP_N])),
            "spearman_with_base_2026": float(spearmanr(utilities, base_utilities)[0]),
            "same_top_pick_as_base": bool(top[0] == base_top[0])}


def run_one(run: RobustnessRun, base_by_fold: pd.DataFrame, base_utilities: np.ndarray) -> tuple[dict[str, object], pd.DataFrame]:
    spec = pl_spec_for(RUN_NAME, run.columns)
    metrics = fold_metrics(run.study, loeo_predictions(run.study, spec))
    folds = list(run.study.labelled_editions)
    base_same = base_by_fold.loc[folds]
    utilities, lam = forecast_utilities(run.study, run.columns)
    names = run.study.frame_t1.loc[run.study.frame_t1[EDITION_COL] == run.study.forecast_edition, "player_name"].to_numpy()
    summary = {"run": run.run_id, "description": run.description, "n_folds": len(folds), "folds": f"{folds[0]}-{folds[-1]}",
               "mean_ll": float(metrics.heldout_ll.mean()), "base_mean_ll_same_folds": float(base_same.heldout_ll.mean()),
               "mean_winner_rank": float(metrics.winner_rank.mean()), "base_winner_rank_same_folds": float(base_same.winner_rank.mean()),
               "mean_top3_overlap": float(metrics.top3_overlap.mean()), "mean_top10_acc": float(metrics.top10_acc.mean()),
               "mean_ndcg10": float(metrics.ndcg10.mean()), "final_lambda": lam,
               **forecast_comparison(utilities, names, base_utilities)}
    summary["delta_ll_vs_base"] = summary["mean_ll"] - summary["base_mean_ll_same_folds"]
    per_fold = metrics.reset_index().assign(run=run.run_id)
    per_fold["base_heldout_ll"] = base_same.heldout_ll.to_numpy()
    return summary, per_fold


def load_base(root: Path) -> StudyData:
    return load_study_v2(root)


def base_forecast_utilities(base: StudyData) -> np.ndarray:
    return forecast_utilities(base, m1_columns_v2())[0]
