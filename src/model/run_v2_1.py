"""Awards fix v2.1, B3 back-test (awards fix spec section B3). Post-hoc (written after the 2026 result was known): EXPLORATORY.

Reads no 2026 outcome. Writes results/v2_1/; nothing under results/v2/ is touched."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.model import baselines, cv, run_stage3, validation_v2
from src.model import research_v2 as rs
from src.model.awards_v2_1 import MAIN_AWARDS, SCORE_COLUMN, best_per_competition_season, event_scores
from src.model.data import V2_SPEC, load_study_v2, load_study_v2_1
from src.model.features_v2 import MAIN_OPTIONS, V2_FEATURES, FeatureOptions, m1_columns_v2
from src.model.run_stage3 import SEASON4_MODEL
from src.model.validation_v2 import SEASON_FOLDS

OUT_DIR = Path("results") / "v2_1"
V2_DIR = Path("results") / "v2"
SUFFIX = "_v2_1"
LABEL = "<!-- EXPLORATORY, post-hoc: F11 redefined after the 2026 result was known (awards fix spec v2.1). -->"
N_BOOT_COEF = 500
SEED = 7
NEW, OLD = "M1_v2_1", "M1_v2"


def v2_1_options() -> FeatureOptions:
    return FeatureOptions(MAIN_OPTIONS.club_attack_column, MAIN_OPTIONS.intl_attack_column, MAIN_OPTIONS.league_award_column, MAIN_AWARDS)


def paired_v2_1_vs_v2(new_utilities: dict[int, np.ndarray], old_utilities: dict[int, np.ndarray], points: dict[int, np.ndarray],
                      folds: list[int]) -> pd.DataFrame:
    """M1 (v2.1 features) minus M1 (v2 features) on paired within-edition bootstrap draws, 90% intervals."""
    utilities = {(NEW, f): new_utilities[f] for f in folds} | {(OLD, f): old_utilities[f] for f in folds}
    draws = validation_v2.bootstrap_fold_metrics(utilities, points, [NEW, OLD], folds, validation_v2.N_BOOTSTRAP, validation_v2.BOOTSTRAP_SEED)
    rows = []
    for subset, members in {"all": folds, "season4": [f for f in folds if f in SEASON_FOLDS]}.items():
        cols = [folds.index(f) for f in members]
        for metric in validation_v2.BOOTSTRAP_METRICS:
            diff = np.nanmean(draws[(NEW, metric)][:, cols], axis=1) - np.nanmean(draws[(OLD, metric)][:, cols], axis=1)
            rows.append({"metric": metric, "subset": subset, "mean_diff": float(diff.mean()),
                         "lo90": float(np.percentile(diff, validation_v2.INTERVAL_LOW_PCT)),
                         "hi90": float(np.percentile(diff, validation_v2.INTERVAL_HIGH_PCT))})
    return pd.DataFrame(rows)


def table1_side_by_side(root: Path, out: Path) -> pd.DataFrame:
    old = pd.read_csv(root / V2_DIR / "metrics_by_fold.csv").assign(version="v2")
    new = pd.read_csv(out / f"metrics_by_fold{SUFFIX}.csv").assign(version="v2.1")
    both = pd.concat([old, new], ignore_index=True)
    both = both[both.metric.isin(run_stage3.TABLE1_METRICS) & both.model.isin(run_stage3.TABLE1_MODELS)]
    wide = both.pivot_table(index=["model", "metric", "version"], columns="fold", values="value")
    wide["mean_all"] = wide.mean(axis=1)
    return wide.reset_index()


def award_events_check(study_events: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    scored = event_scores(study_events, MAIN_AWARDS)
    kept = best_per_competition_season(scored)
    counts = pd.DataFrame({"n_events": scored.groupby(["player_key", "edition"]).size(),
                           "n_after_dedupe": kept.groupby(["player_key", "edition"]).size(),
                           "mean_difficulty": scored.groupby(["player_key", "edition"]).difficulty.mean()}).reset_index()
    base = frame[["edition", "player_key", "player_name", "F11_individual_awards"]]
    return base.merge(counts, on=["player_key", "edition"], how="left").fillna({"n_events": 0, "n_after_dedupe": 0})


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    study = load_study_v2_1(root, V2_SPEC, v2_1_options(), V2_FEATURES)
    folds = list(study.labelled_editions)
    results = run_stage3.backtest(root, study, out, SUFFIX, LABEL)
    print(f"back-test done in {time.time() - started:.0f}s", flush=True)

    old_study = load_study_v2(root)
    old_utilities = cv.run_cross_validation(old_study, ["M1"]).heldout_utilities
    new_utilities = {f: results.heldout_utilities[("M1", f)] for f in folds}
    points = {f: baselines.points_of(study, f) for f in folds}
    paired = paired_v2_1_vs_v2(new_utilities, {f: old_utilities[("M1", f)] for f in folds}, points, folds)
    paired.to_csv(out / "paired_v2_1_vs_v2.csv", index=False)
    table1_side_by_side(root, out).to_csv(out / "table1_v2_vs_v2_1.csv", index=False)
    print("paired difference done", flush=True)

    inputs = rs.build_share_inputs(study)
    rs.variance_shares(inputs, rs.N_BOOT_SHARES, SEED).to_csv(out / "variance_shares_v2_1.csv", index=False)
    by_fold = pd.read_csv(out / f"metrics_by_fold{SUFFIX}.csv")
    m1 = by_fold[by_fold.model == "M1"].pivot(index="fold", columns="metric", values="value")
    rs.ablation_table(study, m1).to_csv(out / "ablations_v2_1.csv", index=False)
    rs.final_coefficients(study, m1_columns_v2(), "M1", N_BOOT_COEF, SEED).table.to_csv(out / "coefficients_M1_v2_1.csv", index=False)
    events = pd.read_parquet(root / "data" / "v2_1" / "fact_award_events.parquet")
    award_events_check(events, study.frame_t1).to_csv(out / "award_events_check.csv", index=False)
    print(f"all B3 outputs done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
