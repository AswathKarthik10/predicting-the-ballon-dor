"""Experiment 4, stage 1: fully nested leave-one-edition-out back-test of the hypothesis-free pipeline (spec section 6).

Reads no 2026 outcome. Stops here: the 2026 forecast (spec section 8) is a separate step after review. Writes results/v4/."""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.model import experiment3 as e3, v4_candidates as vcand, v4_compare as vcmp, v4_search as vs, validation_v2
from src.model.cv import edition_metrics
from src.model.pl import build_ranking, edition_loglik_per_voted

OUT_DIR = Path("results") / "v4"
LABELLED = vcand.LABELLED_EDITIONS
N_JOBS = len(LABELLED)


@dataclass(frozen=True)
class FoldResult:
    held_out: int
    log: list[dict[str, object]]
    chosen: dict[str, object]
    selected: list[str]
    utility: np.ndarray
    inner_all: list[vs.InnerResult]


def describe(config: vs.Config) -> dict[str, object]:
    return {"preprocessing": config.preprocess.name, "selector": config.selector.kind.value, "selector_k": config.selector.k,
            "selector_lambda": config.selector.lam, "selector_pi": config.selector.pi, "model": config.model.kind.value,
            "model_lambda": config.model.lam, "model_mix": config.model.mix, "model_size": config.model.size,
            "model_n_estimators": config.model.n_estimators, "model_min_leaf": config.model.min_leaf}


def run_outer_fold(panel: vs.Panel, held_out: int, configs: list[vs.Config]) -> FoldResult:
    train = tuple(ed for ed in LABELLED if ed != held_out)
    cache = vs.FitCache()
    started = time.time()
    results: list[vs.InnerResult] = []
    for index, config in enumerate(configs):
        results.append(vs.score_config(index, config, panel, train, cache))
        if (index + 1) % 50 == 0:
            print(f"fold {held_out}: {index + 1}/{len(configs)} configs in {time.time() - started:.0f}s", flush=True)
    best = vs.one_standard_error_choice(results)
    config = configs[best.config_index]
    prediction, columns = vs.fit_and_predict(config, panel, train, held_out, vs.FitCache())
    utility = prediction.utilities[0]
    points = panel.points[panel.edition == held_out]
    outer_score = edition_loglik_per_voted(utility, build_ranking(points))
    log = [{"outer_fold": held_out, "config_index": r.config_index, **describe(configs[r.config_index]),
            **{f"inner_ll_{ed}": s for ed, s in zip(train, r.scores)}, "inner_mean": r.mean_score, "inner_se": r.standard_error,
            "mean_n_features": r.mean_n_features, "converged": r.converged, "chosen": r.config_index == best.config_index}
           for r in results]
    chosen = {"outer_fold": held_out, **describe(config), "n_features": len(columns), "inner_mean": best.mean_score,
              "inner_se": best.standard_error, "outer_ll": outer_score, "gap_inner_minus_outer": best.mean_score - outer_score,
              "temperature": prediction.temperature, "converged": prediction.converged, "configs_searched": len(configs)}
    return FoldResult(held_out, log, chosen, [panel.columns[c] for c in columns], utility, results)


def selected_matrix(folds: list[FoldResult], columns: tuple[str, ...]) -> pd.DataFrame:
    table = pd.DataFrame({"feature": list(columns)})
    for fold in folds:
        table[str(fold.held_out)] = table["feature"].isin(fold.selected).astype(int)
    return table[table.drop(columns="feature").sum(axis=1) > 0].reset_index(drop=True)


def main(root: Path) -> None:
    started = time.time()
    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    candidates = vcand.build_candidates(vcand.load_raw_t1(root))
    vcand.write_candidates(candidates, out / "candidates.csv")
    panel = vs.Panel(x=candidates.features.to_numpy(), edition=candidates.rows["edition"].to_numpy(),
                     points=candidates.rows[vcand.POINTS_COL].to_numpy(), columns=tuple(candidates.features.columns))
    if not np.isnan(panel.points[panel.edition == 2026]).all():
        raise ValueError("2026 targets must be null before the forecast step")

    grid = vs.full_grid()
    configs = vs.sample_configs(grid, vs.SEARCH_BUDGET, vs.SEARCH_SEED)
    print(f"full grid {len(grid)} configurations; searching {len(configs)} (random, seed {vs.SEARCH_SEED})", flush=True)
    folds = Parallel(n_jobs=N_JOBS)(delayed(run_outer_fold)(panel, held, configs) for held in LABELLED)
    print(f"search done in {time.time() - started:.0f}s", flush=True)

    pd.DataFrame([row for f in folds for row in f.log]).to_csv(out / "search_log_v4.csv", index=False)
    chosen = pd.DataFrame([f.chosen for f in folds])
    chosen.to_csv(out / "chosen_by_fold_v4.csv", index=False)
    selected_matrix(folds, panel.columns).to_csv(out / "selected_features_by_fold.csv", index=False)

    write_comparison(root, out, panel, folds, chosen)
    print(f"back-test outputs written in {time.time() - started:.0f}s", flush=True)


def write_comparison(root: Path, out: Path, panel: vs.Panel, folds: list[FoldResult], chosen: pd.DataFrame) -> None:
    keys = {ed: candidates_keys(root, panel, ed) for ed in LABELLED}
    v4 = {f.held_out: f.utility for f in folds}
    v3, v3_inner, v3_study = vcmp.v3_utilities(root)
    v1, v1_study = vcmp.v1_utilities(root)
    utilities: dict[tuple[str, int], np.ndarray] = {}
    for ed in LABELLED:
        utilities[(vcmp.LABEL_V4, ed)] = v4[ed]
        utilities[(vcmp.LABEL_V3, ed)] = vcmp.align_to(v3[ed], vcmp.keys_of(v3_study, ed), keys[ed])
        utilities[(vcmp.LABEL_V1, ed)] = vcmp.align_to(v1[ed], vcmp.keys_of(v1_study, ed), keys[ed])
        utilities[(vcmp.LABEL_U0, ed)] = np.zeros(len(keys[ed]))
    points = {ed: panel.points[panel.edition == ed] for ed in LABELLED}
    rows = [r for (label, ed), u in utilities.items() for r in vcmp.metric_rows(label, ed, u, points[ed])]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics_by_fold_v4.csv", index=False)
    e3.summarise_runs(metrics).to_csv(out / "metrics_summary_v4.csv", index=False)

    labels = [vcmp.LABEL_V4, vcmp.LABEL_V3, vcmp.LABEL_V1]
    draws = validation_v2.bootstrap_fold_metrics(utilities, points, labels, list(LABELLED), validation_v2.N_BOOTSTRAP, validation_v2.BOOTSTRAP_SEED)
    contrasts = ((vcmp.LABEL_V4, vcmp.LABEL_V3), (vcmp.LABEL_V4, vcmp.LABEL_V1))
    paired = e3.paired_contrasts(draws, metrics, "PL", LABELLED, "all4", contrasts)
    paired.to_csv(out / "paired_v4.csv", index=False)

    v3_outer = metrics[(metrics.run == vcmp.LABEL_V3) & (metrics.metric == vcmp.PRIMARY)].set_index("fold")["value"]
    gap = pd.DataFrame({"fold": list(LABELLED), "v4_inner": chosen["inner_mean"].to_numpy(), "v4_outer": chosen["outer_ll"].to_numpy(),
                        "v3_inner": [v3_inner[ed] for ed in LABELLED], "v3_outer": [v3_outer[ed] for ed in LABELLED]})
    gap["v4_gap"] = gap["v4_inner"] - gap["v4_outer"]
    gap["v3_gap"] = gap["v3_inner"] - gap["v3_outer"]
    gap.to_csv(out / "overfitting_gap_v4.csv", index=False)


def refit_chosen(panel: vs.Panel, configs: list[vs.Config], log: pd.DataFrame, held_out: int) -> FoldResult:
    """Rebuild a fold's held-out utility by refitting its logged choice (used to finish a run whose search already completed)."""
    train = tuple(ed for ed in LABELLED if ed != held_out)
    rows = log[(log.outer_fold == held_out) & log.chosen]
    config = configs[int(rows.config_index.iloc[0])]
    prediction, columns = vs.fit_and_predict(config, panel, train, held_out, vs.FitCache())
    return FoldResult(held_out, [], {}, [panel.columns[c] for c in columns], prediction.utilities[0], [])


def finish_from_saved(root: Path) -> None:
    out = root / OUT_DIR
    candidates = vcand.build_candidates(vcand.load_raw_t1(root))
    panel = vs.Panel(x=candidates.features.to_numpy(), edition=candidates.rows["edition"].to_numpy(),
                     points=candidates.rows[vcand.POINTS_COL].to_numpy(), columns=tuple(candidates.features.columns))
    configs = vs.sample_configs(vs.full_grid(), vs.SEARCH_BUDGET, vs.SEARCH_SEED)
    log = pd.read_csv(out / "search_log_v4.csv")
    folds = [refit_chosen(panel, configs, log, held) for held in LABELLED]
    chosen = pd.read_csv(out / "chosen_by_fold_v4.csv")
    for fold, outer_ll in zip(folds, chosen["outer_ll"]):
        refit_ll = edition_loglik_per_voted(fold.utility, build_ranking(panel.points[panel.edition == fold.held_out]))
        if abs(refit_ll - outer_ll) > 1e-9:
            raise ValueError(f"refit of fold {fold.held_out} does not match the logged outer score: {refit_ll} vs {outer_ll}")
    write_comparison(root, out, panel, folds, chosen)


def candidates_keys(root: Path, panel: vs.Panel, edition: int) -> np.ndarray:
    raw = vcand.load_raw_t1(root)
    return raw.loc[raw[vcand.EDITION_COL] == edition, "player_key"].to_numpy()


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[2])
