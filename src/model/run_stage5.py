"""Stage 5: betting-odds baseline (spec section 8) and robustness runs R1-R9 (section 10). Odds are never a model input."""

import time
from pathlib import Path

import pandas as pd

from src.model import odds_v2, robustness_v2 as rb
from src.model.data import load_study_v2
from src.model.features_v2 import m1_columns_v2
from src.model.research_v2 import loeo_predictions, pl_spec_for

OUT_DIR = Path("results") / "v2"


def log(message: str, started: float) -> None:
    print(f"[{time.time() - started:6.0f}s] {message}", flush=True)


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    study = load_study_v2(root)

    m1_utilities = loeo_predictions(study, pl_spec_for("M1", m1_columns_v2()))
    odds = odds_v2.read_odds(root)
    captures = odds_v2.usable_captures(odds, odds_v2.t1_dates(root), study.labelled_editions)
    baseline = odds_v2.capture_rows(study, captures, m1_utilities)
    baseline.to_csv(out / "odds_baseline.csv", index=False)
    log(f"odds baseline done: {len(baseline)} captures", started)

    metrics = pd.read_csv(out / "metrics_by_fold.csv")
    base_by_fold = metrics[metrics.model == "M1"].pivot(index="fold", columns="metric", values="value")
    base_utilities = rb.base_forecast_utilities(study)
    summaries: list[dict[str, object]] = []
    per_fold: list[pd.DataFrame] = []
    for run in rb.build_runs(root, study):
        summary, folds = rb.run_one(run, base_by_fold, base_utilities)
        summaries.append(summary)
        per_fold.append(folds)
        pd.DataFrame(summaries).to_csv(out / "robustness.csv", index=False)
        pd.concat(per_fold, ignore_index=True).to_csv(out / "robustness_by_fold.csv", index=False)
        log(f"{run.run_id} done: mean ll {summary['mean_ll']:.4f} (base on same folds {summary['base_mean_ll_same_folds']:.4f})", started)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
