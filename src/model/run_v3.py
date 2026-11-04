"""Experiment 3, stage 1: back-test comparison of E3-S, E3-A and E3-C (spec section 2) plus the mirror check.

Reads no 2026 outcome. Writes results/v3/."""

import time
from pathlib import Path

import pandas as pd

from src.model import experiment3 as e3
from src.model.data import V2_SPEC, load_study_v2_1
from src.model.features_v2 import V2_FEATURES
from src.model.run_stage3 import V1_M1_MEAN_LL
from src.model.run_v2_1 import v2_1_options

OUT_DIR = Path("results") / "v3"
V2_DIR = Path("results") / "v2"
V2_1_DIR = Path("results") / "v2_1"
SEASON_SUBSET, MIRROR_SUBSET = "season4", "calendar5"
V2_SEASON_ONLY_MODEL = "M1-season4"


def feature_vs_data_grid(season_metrics: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Spec 2.3: M1 held-out log-likelihood on 2022-2025 by feature set and training data, plus the v2 reference pair."""
    v2 = pd.read_csv(root / V2_DIR / "metrics_by_fold.csv")
    v2 = v2[(v2.metric == e3.PRIMARY_METRIC) & v2.fold.isin(e3.SEASON_FOLDS)]
    v2_mean = v2.groupby("model").value.mean()
    m1 = season_metrics[(season_metrics.model == "M1") & (season_metrics.metric == e3.PRIMARY_METRIC)].groupby("run").value.mean()
    rows = [
        ("v1 features", "season editions only", V1_M1_MEAN_LL, "results/metrics_summary.csv (v1 M1)"),
        ("v1 features", "all editions", float("nan"), "not possible: no xG before 2021"),
        ("v2.1 features", "season editions only", m1[e3.RUN_SEASON], "E3-S"),
        ("v2.1 features", "all editions", m1[e3.RUN_ALL], "E3-A (v2.1, reproduced)"),
        ("v2.1 features", "calendar editions only", m1[e3.RUN_CALENDAR], "E3-C"),
        ("v2 features", "season editions only", v2_mean[V2_SEASON_ONLY_MODEL], "results/v2 M1-season4"),
        ("v2 features", "all editions", v2_mean["M1"], "results/v2 M1, 2022-2025 folds"),
    ]
    return pd.DataFrame(rows, columns=["features", "training_data", "heldout_ll_2022_25", "source"])


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    study = load_study_v2_1(root, V2_SPEC, v2_1_options(), V2_FEATURES)
    result = e3.run_many(study, e3.JOBS, e3.MODELS)
    print(f"runs done in {time.time() - started:.0f}s", flush=True)

    reference = pd.read_csv(root / V2_1_DIR / "metrics_by_fold_v2_1.csv")
    for folds in (e3.SEASON_FOLDS, e3.CALENDAR_FOLDS):
        gap = e3.max_abs_gap(result.metrics, reference, e3.RUN_ALL, e3.MODELS, folds)
        print(f"E3-A vs v2.1 on {folds[0]}-{folds[-1]}: max abs gap {gap:.2e}", flush=True)
        if gap >= 1e-6:
            raise ValueError(f"E3-A does not reproduce v2.1 on folds {folds}: max abs gap {gap}")

    paired_parts: list[pd.DataFrame] = []
    for model in e3.MODELS:
        for runs_in_group, folds, subset, contrasts in ((e3.SEASON_RUNS, e3.SEASON_FOLDS, SEASON_SUBSET, e3.SEASON_CONTRASTS),
                                                        (e3.MIRROR_RUNS, e3.CALENDAR_FOLDS, MIRROR_SUBSET, e3.MIRROR_CONTRASTS)):
            draws = e3.bootstrap_runs(study, result, runs_in_group, model, folds)
            paired_parts.append(e3.paired_contrasts(draws, result.metrics, model, folds, subset, contrasts))
        print(f"bootstrap {model} done in {time.time() - started:.0f}s", flush=True)

    season_metrics = result.metrics[result.metrics.run.isin(e3.SEASON_RUNS) & result.metrics.fold.isin(e3.SEASON_FOLDS)]
    mirror_metrics = result.metrics[result.metrics.run.isin(e3.MIRROR_RUNS) & result.metrics.fold.isin(e3.CALENDAR_FOLDS)]
    kept = pd.concat([season_metrics, mirror_metrics]).drop_duplicates(subset=["run", "model", "fold", "metric"])
    kept.to_csv(out / "metrics_by_fold_v3.csv", index=False)
    e3.summarise_runs(kept).to_csv(out / "metrics_summary_v3.csv", index=False)
    pd.concat(paired_parts, ignore_index=True).to_csv(out / "paired_runs_v3.csv", index=False)
    result.tuning.drop_duplicates(subset=["run", "model", "fold"]).to_csv(out / "tuning_by_fold_v3.csv", index=False)
    feature_vs_data_grid(season_metrics, root).to_csv(out / "feature_vs_data_grid.csv", index=False)
    print(f"all stage-1 outputs written in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
