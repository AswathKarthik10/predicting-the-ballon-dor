"""Experiment 3: does the 2016-2021 calendar format help or hurt predictions of the 2022-2025 season editions?

Only the training set changes between runs; features, grids and seeds are the v2.1 ones. Reads no 2026 outcome."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model import baselines, cv, validation_v2
from src.model.data import StudyData
from src.model.validation_v2 import BOOTSTRAP_METRICS, INTERVAL_HIGH_PCT, INTERVAL_LOW_PCT, SEASON_FOLDS

CALENDAR_FOLDS: tuple[int, ...] = (2016, 2017, 2018, 2019, 2021)
RUN_ALL, RUN_SEASON, RUN_CALENDAR = "E3-A", "E3-S", "E3-C"
RUN_CALENDAR_SELF, RUN_SEASON_ON_CALENDAR = "E3-Cself", "E3-Scal"
SEASON_RUNS: tuple[str, ...] = (RUN_SEASON, RUN_ALL, RUN_CALENDAR)
MIRROR_RUNS: tuple[str, ...] = (RUN_CALENDAR_SELF, RUN_ALL, RUN_SEASON_ON_CALENDAR)
MODELS: tuple[str, ...] = ("M1", "X1", "B1")
SEASON_CONTRASTS: tuple[tuple[str, str], ...] = ((RUN_SEASON, RUN_ALL), (RUN_CALENDAR, RUN_SEASON), (RUN_CALENDAR, RUN_ALL))
MIRROR_CONTRASTS: tuple[tuple[str, str], ...] = ((RUN_CALENDAR_SELF, RUN_ALL), (RUN_SEASON_ON_CALENDAR, RUN_CALENDAR_SELF),
                                                 (RUN_SEASON_ON_CALENDAR, RUN_ALL))
FIVE_PICK_FOLDS: tuple[int, ...] = (2022, 2023)
TEN_PICK_FOLDS: tuple[int, ...] = (2024, 2025)
NOISE_FLOOR = 0.03  # spec section 5: log-likelihood differences below this are within noise at four editions
PRIMARY_METRIC = "heldout_ll"
POINT_METRICS: tuple[str, ...] = cv.METRIC_NAMES


def held_out_folds(run: str) -> tuple[int, ...]:
    """Folds a run is scored on in its own comparison (E3-A is scored on both groups, see `JOBS`)."""
    if run in (RUN_SEASON, RUN_CALENDAR):
        return SEASON_FOLDS
    if run in (RUN_CALENDAR_SELF, RUN_SEASON_ON_CALENDAR):
        return CALENDAR_FOLDS
    raise ValueError(f"run {run!r} has no fixed fold group; pass the folds explicitly")


JOBS: tuple[tuple[str, tuple[int, ...]], ...] = (
    (RUN_SEASON, SEASON_FOLDS), (RUN_CALENDAR, SEASON_FOLDS), (RUN_ALL, SEASON_FOLDS),
    (RUN_CALENDAR_SELF, CALENDAR_FOLDS), (RUN_SEASON_ON_CALENDAR, CALENDAR_FOLDS), (RUN_ALL, CALENDAR_FOLDS))


def train_set(run: str, held: int, labelled: tuple[int, ...]) -> list[int]:
    """Training editions of one fold. E3-A is the v2.1 leave-one-edition-out set."""
    if run == RUN_ALL:
        return [ed for ed in labelled if ed != held]
    if run == RUN_SEASON:
        return [ed for ed in SEASON_FOLDS if ed != held]
    if run == RUN_CALENDAR_SELF:
        return [ed for ed in CALENDAR_FOLDS if ed != held]
    if run == RUN_CALENDAR:
        return list(CALENDAR_FOLDS)
    if run == RUN_SEASON_ON_CALENDAR:
        return list(SEASON_FOLDS)
    raise ValueError(f"unknown run {run!r}; expected one of {SEASON_RUNS + MIRROR_RUNS}")


@dataclass(frozen=True)
class RunResult:
    utilities: dict[tuple[str, str, int], np.ndarray]  # (run, model, fold) to held-out utilities
    metrics: pd.DataFrame  # run, model, fold, metric, value
    tuning: pd.DataFrame  # run, model, fold, lambda_or_alpha, temperature


def run_one(study: StudyData, run: str, model: str, folds: tuple[int, ...]) -> RunResult:
    utilities: dict[tuple[str, str, int], np.ndarray] = {}
    metric_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    for held in folds:
        train = train_set(run, held, study.labelled_editions)
        scores, info = cv.predict_model(study, model, train, [held])
        utilities[(run, model, held)] = scores[held]
        for metric, value in cv.edition_metrics(scores[held], baselines.points_of(study, held)).items():
            metric_rows.append({"run": run, "model": model, "fold": held, "metric": metric, "value": value})
        tuning_rows.append({"run": run, "model": model, "fold": held, "train_editions": " ".join(map(str, train)),
                            "lambda_or_alpha": info.lam, "temperature": info.temperature})
    return RunResult(utilities, pd.DataFrame(metric_rows), pd.DataFrame(tuning_rows))


def run_many(study: StudyData, jobs: tuple[tuple[str, tuple[int, ...]], ...], models: tuple[str, ...]) -> RunResult:
    parts = [run_one(study, run, model, folds) for run, folds in jobs for model in models]
    utilities: dict[tuple[str, str, int], np.ndarray] = {}
    for part in parts:
        utilities |= part.utilities
    return RunResult(utilities, pd.concat([p.metrics for p in parts], ignore_index=True),
                     pd.concat([p.tuning for p in parts], ignore_index=True))


def bootstrap_runs(study: StudyData, result: RunResult, runs: tuple[str, ...], model: str, folds: tuple[int, ...]) -> validation_v2.BootstrapDraws:
    """Paired within-edition bootstrap: every run sees the same resampled players in a given fold."""
    utilities = {(run, fold): result.utilities[(run, model, fold)] for run in runs for fold in folds}
    points = {fold: baselines.points_of(study, fold) for fold in folds}
    return validation_v2.bootstrap_fold_metrics(utilities, points, list(runs), list(folds), validation_v2.N_BOOTSTRAP, validation_v2.BOOTSTRAP_SEED)


def count_better(metrics: pd.DataFrame, model: str, metric: str, first: str, second: str, folds: tuple[int, ...]) -> tuple[int, int]:
    """Number of folds where `first` beats `second` and where `second` beats `first` (higher is better)."""
    table = metrics[(metrics.model == model) & (metrics.metric == metric) & (metrics.fold.isin(folds))].pivot(index="fold", columns="run", values="value")
    return int((table[first] > table[second]).sum()), int((table[second] > table[first]).sum())


def paired_contrasts(draws: validation_v2.BootstrapDraws, metrics: pd.DataFrame, model: str, folds: tuple[int, ...], subset: str,
                     contrasts: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    """first minus second on the paired draws (mean over `folds`), with 90% interval and fold counts."""
    rows: list[dict[str, object]] = []
    fold_list = list(folds)
    for first, second in contrasts:
        for metric in BOOTSTRAP_METRICS:
            diff = np.nanmean(draws[(first, metric)], axis=1) - np.nanmean(draws[(second, metric)], axis=1)
            first_wins, second_wins = count_better(metrics, model, metric, first, second, folds)
            rows.append({"model": model, "subset": subset, "contrast": f"{first} - {second}", "metric": metric,
                         "mean_diff": float(diff.mean()), "lo90": float(np.percentile(diff, INTERVAL_LOW_PCT)),
                         "hi90": float(np.percentile(diff, INTERVAL_HIGH_PCT)), "excludes_zero": bool(np.percentile(diff, INTERVAL_LOW_PCT) > 0
                                                                                                    or np.percentile(diff, INTERVAL_HIGH_PCT) < 0),
                         "folds_first_better": first_wins, "folds_second_better": second_wins, "n_folds": len(fold_list)})
    return pd.DataFrame(rows)


def summarise_runs(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per run, model, metric and fold group: mean over the group's folds, the 5-pick / 10-pick split (descriptive) and every fold.
    E3-A is scored on both groups, so each group is summarised on its own."""
    parts: list[pd.DataFrame] = []
    for subset, group_folds in (("season4", SEASON_FOLDS), ("calendar5", CALENDAR_FOLDS)):
        scored = metrics[metrics.fold.isin(group_folds)]
        if scored.empty:
            continue
        wide = scored.pivot_table(index=["run", "model", "metric"], columns="fold", values="value")
        out = pd.DataFrame(index=wide.index)
        out["subset"] = subset
        out["mean"] = wide.mean(axis=1)
        for label, picks in (("mean_5pick_2022_23", FIVE_PICK_FOLDS), ("mean_10pick_2024_25", TEN_PICK_FOLDS)):
            present = [f for f in picks if f in wide.columns]
            out[label] = wide[present].mean(axis=1) if present else np.nan
        for fold in wide.columns:
            out[f"fold_{fold}"] = wide[fold]
        parts.append(out.reset_index())
    return pd.concat(parts, ignore_index=True)


def max_abs_gap(metrics: pd.DataFrame, reference: pd.DataFrame, run: str, models: tuple[str, ...], folds: tuple[int, ...]) -> float:
    """Largest absolute difference between a run's metrics and a reference table (model, fold, metric, value)."""
    left = metrics[(metrics.run == run) & metrics.model.isin(models) & metrics.fold.isin(folds)].set_index(["model", "fold", "metric"]).value
    right = reference[reference.model.isin(models) & reference.fold.isin(folds)].set_index(["model", "fold", "metric"]).value
    joined = pd.concat([left.rename("run"), right.rename("reference")], axis=1, join="inner")
    if len(joined) != len(left):
        raise ValueError(f"reference covers {len(joined)} of {len(left)} metric cells for run {run}")
    if (joined["run"].isna() != joined["reference"].isna()).any():
        raise ValueError(f"run {run} and the reference disagree on which metric cells are undefined")
    return float((joined["run"] - joined["reference"]).abs().max())
