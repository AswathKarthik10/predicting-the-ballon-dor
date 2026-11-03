"""Stage 3 of the experiments (spec section 14, step 2): ten-edition back-test, comparison table and model selection.

Reads no 2026 output: cross-validation only touches the labelled editions."""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import baselines, cv, report, validation_v2
from src.model.data import StudyData, load_study_v2
from src.model.validation_v2 import SEASON_FOLDS

OUT_DIR = Path("results") / "v2"
FULL_MODELS: list[str] = ["U0", "B0", "B1", "B2", "X1", "L1", "O1", "D1", "M1", "M1-G", "M1-F"]
SEASON4_MODEL = "M1-season4"
TABLE1_MODELS: tuple[str, ...] = ("B0", "B1", "X1", "L1", "M1", "M1-G", "M1-F")
TABLE1_METRICS: tuple[str, ...] = ("heldout_ll", "winner_rank", "top3_overlap", "top10_acc", "ndcg10")
V1_M1_MEAN_LL = -2.3348405384671667  # results/metrics_summary.csv, v1 features, four folds
SUMMARY_DIGITS = 3
INTERVAL_METRICS = validation_v2.BOOTSTRAP_METRICS


def format_cell(mean: float, lo: float | None, hi: float | None, digits: int) -> str:
    if lo is None or hi is None or np.isnan(lo):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def table1(summary: pd.DataFrame, intervals: pd.DataFrame, subset: str, mean_column: str) -> pd.DataFrame:
    rows: dict[str, dict[str, str]] = {}
    for model in TABLE1_MODELS:
        row: dict[str, str] = {}
        for metric in TABLE1_METRICS:
            mean = summary[(summary["model"] == model) & (summary["metric"] == metric)][mean_column].iloc[0]
            hit = intervals[(intervals["model"] == model) & (intervals["metric"] == metric) & (intervals["subset"] == subset)]
            lo, hi = (hit["lo90"].iloc[0], hit["hi90"].iloc[0]) if len(hit) else (None, None)
            row[metric] = format_cell(mean, lo, hi, SUMMARY_DIGITS)
        rows[model] = row
    out = pd.DataFrame(rows).T
    out.index.name = "model"
    return out


def run_season4(study: StudyData) -> pd.DataFrame:
    """M1 with the v2 features trained on the season editions only, scored on the same four folds (spec 6.1)."""
    season_study = replace(study, labelled_editions=SEASON_FOLDS)
    metrics = cv.run_cross_validation(season_study, ["M1"]).metrics
    metrics["model"] = SEASON4_MODEL
    return metrics


def per_fold_ll(summary: pd.DataFrame, folds: list[int]) -> pd.DataFrame:
    ll = summary[summary["metric"] == "heldout_ll"].set_index("model")
    cols = ["mean_all", "mean_2022_25"] + [f"fold_{f}" for f in folds]
    order = [m for m in FULL_MODELS + [SEASON4_MODEL] if m in ll.index]
    return ll.loc[order, cols].sort_values("mean_all", ascending=False, na_position="last")


def write_selection(path: Path, header: list[str], summary: pd.DataFrame, intervals: pd.DataFrame, paired: pd.DataFrame,
                    selection: cv.Selection, folds: list[int], tuning: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    all_t1 = table1(summary, intervals, "all", "mean_all")
    season_t1 = table1(summary, intervals, "season4", "mean_2022_25")
    ll_folds = per_fold_ll(summary, folds)
    ll = summary[summary["metric"] == "heldout_ll"].set_index("model")
    season_m1 = ll.loc["M1", "mean_2022_25"]
    season4 = ll.loc[SEASON4_MODEL, "mean_2022_25"]
    paired_all = paired[paired["subset"] == "all"].set_index("model").drop(columns=["subset", "vs"])
    lam = tuning[tuning["model"].isin(["M1", "M1-G", "M1-F"])].pivot(index="fold", columns="model", values="lambda_or_alpha")
    lines = header + ["", "# Model selection, ten-edition back-test (experiments spec 5.4)", "",
        "Primary metric: mean held-out Plackett-Luce log-likelihood per voted player over nine leave-one-edition-out folds "
        "(higher is better). Intervals are 90% within-edition bootstrap intervals (1,000 draws). No 2026 output was used.", "",
        "## Table 1. Nine-fold means with 90% intervals", "", report.markdown_table(all_t1, SUMMARY_DIGITS), "",
        "## Same table over the four v1 editions (2022-2025 folds of the nine-edition models)", "",
        report.markdown_table(season_t1, SUMMARY_DIGITS), "",
        "## Held-out log-likelihood per fold", "", report.markdown_table(ll_folds, SUMMARY_DIGITS), "",
        "## Paired difference in mean log-likelihood versus M1 (nine folds)", "", report.markdown_table(paired_all, SUMMARY_DIGITS), "",
        "## What the extra editions buy (2022-2025 folds)", "",
        f"- v1: M1 with v1 features, trained on 3 season editions per fold: {V1_M1_MEAN_LL:.4f}",
        f"- v2 features, trained on the same 3 season editions ({SEASON4_MODEL}): {season4:.4f}",
        f"- v2 features, trained on the 8 other editions (M1, 2022-2025 folds): {season_m1:.4f}", "",
        "## Selection outcome", "", f"**Published model: {selection.published}.** Reported alongside: {selection.reported_alongside}.", ""]
    lines += [f"- {note}" for note in selection.notes]
    lines += ["", "## Selected ridge penalty per fold (PL models)", "", report.markdown_table(lam, SUMMARY_DIGITS), "",
              "## Back-test coefficients of M1 (mean over the nine fold fits)", "",
              report.markdown_table(report.coefficient_summary(coefficients, "M1"), SUMMARY_DIGITS)]
    path.write_text("\n".join(lines) + "\n")


def backtest(root: Path, study: StudyData, out: Path, suffix: str, header_note: str) -> cv.CvResults:
    """Stage 3 for any ten-edition study; `suffix` is appended to the metrics file names (v2: none)."""
    started = time.time()
    folds = list(study.labelled_editions)
    out.mkdir(parents=True, exist_ok=True)
    results = cv.run_cross_validation(study, FULL_MODELS)
    metrics = pd.concat([results.metrics, run_season4(study)], ignore_index=True)
    print(f"cross-validation done in {time.time() - started:.0f}s", flush=True)

    summary = validation_v2.summarise_with_subsets(metrics, folds)
    boot_models = [m for m in FULL_MODELS if m != "U0"]
    points = {ed: baselines.points_of(study, ed) for ed in folds}
    draws = validation_v2.bootstrap_fold_metrics(results.heldout_utilities, points, boot_models, folds,
                                                 validation_v2.N_BOOTSTRAP, validation_v2.BOOTSTRAP_SEED)
    subsets = {"all": folds, "season4": [f for f in folds if f in SEASON_FOLDS]}
    intervals = validation_v2.interval_table(draws, folds, subsets)
    paired = validation_v2.paired_differences(draws, folds, subsets, boot_models)
    print(f"bootstrap done in {time.time() - started:.0f}s", flush=True)

    mean_ll = summary[summary["metric"] == "heldout_ll"].set_index("model")["mean_all"].to_dict()
    selection = validation_v2.apply_selection_rule({m: v for m, v in mean_ll.items() if m in FULL_MODELS})
    header = [f"<!-- sha256_t1={study.sha256_t1} git_commit={report.git_commit(root)} -->"] + ([header_note] if header_note else [])

    metrics.to_csv(out / f"metrics_by_fold{suffix}.csv", index=False)
    joined = summary.merge(intervals.pivot_table(index=["model", "metric"], columns="subset", values=["lo90", "hi90"]).pipe(
        lambda d: d.set_axis([f"{a}_{b}" for a, b in d.columns], axis=1)).reset_index(), on=["model", "metric"], how="left")
    joined.to_csv(out / f"metrics_summary{suffix}.csv", index=False)
    paired.to_csv(out / "paired_vs_M1.csv", index=False)
    results.tuning.to_csv(out / "tuning_by_fold.csv", index=False)
    results.coefficients.to_csv(out / "coefficients_by_fold.csv", index=False)
    write_selection(out / "selection.md", header, summary, intervals, paired, selection, folds, results.tuning, results.coefficients)
    print(f"done in {time.time() - started:.0f}s; published model {selection.published}", flush=True)
    return results


def main(root: Path) -> None:
    backtest(root, load_study_v2(root), root / OUT_DIR, "", "")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
