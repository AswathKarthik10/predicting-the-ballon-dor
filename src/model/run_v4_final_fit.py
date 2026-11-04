"""Experiment 4, spec 8.1 (EXPLORATORY, post-hoc): choose the configuration once on all four labelled editions.

Leave-one-edition-out over 2022-2025 with the same 300 configurations as the back-test and the same one-standard-error rule.
Reads no 2026 outcome. Writes results/v4/search_log_final_v4.csv and results/v4/chosen_final_v4.csv."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.model import v4_candidates as vcand, v4_search as vs
from src.model.run_v4_backtest import describe

OUT_DIR = Path("results") / "v4"
LABELLED = vcand.LABELLED_EDITIONS
N_JOBS = 8


def build_panel(root: Path) -> vs.Panel:
    candidates = vcand.build_candidates(vcand.load_raw_t1(root))
    panel = vs.Panel(x=candidates.features.to_numpy(), edition=candidates.rows["edition"].to_numpy(),
                     points=candidates.rows[vcand.POINTS_COL].to_numpy(), columns=tuple(candidates.features.columns))
    if not np.isnan(panel.points[panel.edition == 2026]).all():
        raise ValueError("2026 targets must be null")
    return panel


def score_chunk(panel: vs.Panel, configs: list[vs.Config], indices: list[int]) -> list[vs.InnerResult]:
    cache = vs.FitCache()
    return [vs.score_config(i, configs[i], panel, LABELLED, cache) for i in indices]


def main(root: Path) -> None:
    started = time.time()
    panel = build_panel(root)
    configs = vs.sample_configs(vs.full_grid(), vs.SEARCH_BUDGET, vs.SEARCH_SEED)
    chunks = [list(range(j, len(configs), N_JOBS)) for j in range(N_JOBS)]
    parts = Parallel(n_jobs=N_JOBS)(delayed(score_chunk)(panel, configs, chunk) for chunk in chunks)
    results = sorted((r for part in parts for r in part), key=lambda r: r.config_index)
    best = vs.one_standard_error_choice(results)
    log = pd.DataFrame([{"config_index": r.config_index, **describe(configs[r.config_index]),
                         **{f"inner_ll_{ed}": s for ed, s in zip(LABELLED, r.scores)}, "inner_mean": r.mean_score,
                         "inner_se": r.standard_error, "mean_n_features": r.mean_n_features, "converged": r.converged,
                         "chosen": r.config_index == best.config_index} for r in results])
    log.to_csv(root / OUT_DIR / "search_log_final_v4.csv", index=False)
    log[log.chosen].assign(inner_mean_best=max(r.mean_score for r in results)).to_csv(root / OUT_DIR / "chosen_final_v4.csv", index=False)
    print(f"final search done in {time.time() - started:.0f}s; chosen config {best.config_index}: {configs[best.config_index].name}", flush=True)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
