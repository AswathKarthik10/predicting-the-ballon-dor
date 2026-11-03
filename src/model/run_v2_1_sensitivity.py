"""Awards fix v2.1, B4 sensitivity runs S1-S4 (back-test only; no 2026 output). EXPLORATORY, post-hoc.

S1 without league top scorers; S2 uniform difficulty; S3 main rule plus a World Cup indicator; S4 no log1p and no clipping."""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.model import research_v2 as rs
from src.model.awards_v2_1 import BEST_IS_WC_COLUMN, MAIN_AWARDS, S1_NO_LEAGUE_SCORERS, S2_UNIFORM_DIFFICULTY, S4_RAW_SCORE, AwardVariant
from src.model.data import V2_SPEC, StudyData, load_study_v2_1
from src.model.features_v2 import (MAIN_OPTIONS, RESIDUALISER_REGRESSORS, V2_FEATURES, FeatureOptions, m1_columns_v2, make_v2_features)

OUT_FILE = Path("results") / "v2_1" / "sensitivity_awards.csv"
RUNS: tuple[tuple[str, str, AwardVariant, bool], ...] = (
    ("v2.1", "main rule", MAIN_AWARDS, False),
    ("S1", "without league top scorers", S1_NO_LEAGUE_SCORERS, False),
    ("S2", "uniform difficulty (all 1.0)", S2_UNIFORM_DIFFICULTY, False),
    ("S3", "main rule plus World Cup indicator", MAIN_AWARDS, True),
    ("S4", "no log1p, no clipping", S4_RAW_SCORE, False),
)


def options_for(variant: AwardVariant) -> FeatureOptions:
    return FeatureOptions(MAIN_OPTIONS.club_attack_column, MAIN_OPTIONS.intl_attack_column, MAIN_OPTIONS.league_award_column, variant)


def study_for(root: Path, variant: AwardVariant, with_wc: bool) -> tuple[StudyData, tuple[str, ...]]:
    columns = m1_columns_v2() + ((BEST_IS_WC_COLUMN,) if with_wc else ())
    features = make_v2_features(variant.name + ("-wc" if with_wc else ""), RESIDUALISER_REGRESSORS, columns) if with_wc else V2_FEATURES
    return load_study_v2_1(root, V2_SPEC, options_for(variant), features), columns


def main(root: Path) -> None:
    rows = []
    for run_id, description, variant, with_wc in RUNS:
        study, columns = study_for(root, variant, with_wc)
        metrics = rs.fold_metrics(study, rs.loeo_predictions(study, rs.pl_spec_for(f"M1-{run_id}", columns)))
        rows.append({"run": run_id, "description": description, "mean_heldout_ll": metrics.heldout_ll.mean(),
                     "mean_ll_2022_25": metrics.heldout_ll.loc[[2022, 2023, 2024, 2025]].mean(),
                     "mean_winner_rank": metrics.winner_rank.mean(), "mean_top3_overlap": metrics.top3_overlap.mean(),
                     "mean_top10_acc": metrics.top10_acc.mean(), "mean_ndcg10": metrics.ndcg10.mean(),
                     **{f"ll_{f}": v for f, v in metrics.heldout_ll.items()}})
        print(run_id, round(rows[-1]["mean_heldout_ll"], 4), flush=True)
        pd.DataFrame(rows).to_csv(root / OUT_FILE, index=False)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
