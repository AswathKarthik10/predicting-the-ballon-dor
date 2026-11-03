"""Uncertainty and selection for the ten-edition back-test (experiments spec sections 5.3 and 5.4)."""

import numpy as np
import pandas as pd

from src.model.cv import SELECTION_MARGIN, Selection, edition_metrics

BOOTSTRAP_METRICS: tuple[str, ...] = ("heldout_ll", "spearman", "pairwise_acc", "top3_overlap", "top10_acc", "ndcg10")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 7
INTERVAL_LOW_PCT = 5
INTERVAL_HIGH_PCT = 95
SEASON_FOLDS: tuple[int, ...] = (2022, 2023, 2024, 2025)
PL_SIMPLICITY_ORDER: tuple[str, ...] = ("M1", "M1-G", "M1-F")
REFERENCE_MODEL = "M1"
PAIRED_METRIC = "heldout_ll"

BootstrapDraws = dict[tuple[str, str], np.ndarray]  # (model, metric) to draws x folds


def bootstrap_fold_metrics(utilities: dict[tuple[str, int], np.ndarray], points: dict[int, np.ndarray],
                           models: list[str], folds: list[int], n_boot: int, seed: int) -> BootstrapDraws:
    """Within-edition player bootstrap of the held-out metrics. The resampled players are the same for every model
    (seeded per fold), so differences between models are paired."""
    draws: BootstrapDraws = {(m, k): np.full((n_boot, len(folds)), np.nan) for m in models for k in BOOTSTRAP_METRICS}
    for j, fold in enumerate(folds):
        n_players = len(points[fold])
        resamples = np.random.default_rng(seed + fold).integers(0, n_players, size=(n_boot, n_players))
        for b in range(n_boot):
            idx = resamples[b]
            for model in models:
                values = edition_metrics(utilities[(model, fold)][idx], points[fold][idx])
                for metric in BOOTSTRAP_METRICS:
                    draws[(model, metric)][b, j] = values[metric]
    return draws


def interval_table(draws: BootstrapDraws, folds: list[int], subsets: dict[str, list[int]]) -> pd.DataFrame:
    """90% bootstrap interval of the mean over each fold subset."""
    rows: list[dict[str, object]] = []
    for (model, metric), array in draws.items():
        for subset, members in subsets.items():
            columns = [folds.index(f) for f in members]
            means = np.nanmean(array[:, columns], axis=1)
            rows.append({"model": model, "metric": metric, "subset": subset,
                         "lo90": float(np.nanpercentile(means, INTERVAL_LOW_PCT)),
                         "hi90": float(np.nanpercentile(means, INTERVAL_HIGH_PCT))})
    return pd.DataFrame(rows)


def paired_differences(draws: BootstrapDraws, folds: list[int], subsets: dict[str, list[int]], models: list[str]) -> pd.DataFrame:
    """Each model minus the reference model on the paired bootstrap draws (held-out log-likelihood)."""
    rows: list[dict[str, object]] = []
    for subset, members in subsets.items():
        columns = [folds.index(f) for f in members]
        reference = np.nanmean(draws[(REFERENCE_MODEL, PAIRED_METRIC)][:, columns], axis=1)
        for model in models:
            if model == REFERENCE_MODEL:
                continue
            diff = np.nanmean(draws[(model, PAIRED_METRIC)][:, columns], axis=1) - reference
            rows.append({"model": model, "subset": subset, "vs": REFERENCE_MODEL, "mean_diff": float(diff.mean()),
                         "lo90": float(np.percentile(diff, INTERVAL_LOW_PCT)), "hi90": float(np.percentile(diff, INTERVAL_HIGH_PCT)),
                         "share_better": float((diff > 0).mean())})
    return pd.DataFrame(rows)


def summarise_with_subsets(metrics: pd.DataFrame, folds: list[int]) -> pd.DataFrame:
    """Per model and metric: mean over all folds, mean over the 2022-2025 folds, then every fold."""
    wide = metrics.pivot_table(index=["model", "metric"], columns="fold", values="value")
    out = pd.DataFrame(index=wide.index)
    out["mean_all"] = wide[folds].mean(axis=1, skipna=False)
    out["mean_2022_25"] = wide[[f for f in folds if f in SEASON_FOLDS]].mean(axis=1)
    for fold in folds:
        out[f"fold_{fold}"] = wide[fold]
    return out.reset_index()


def apply_selection_rule(mean_ll: dict[str, float]) -> Selection:
    """Rule 5.4: publish the simplest Plackett-Luce model within the margin of the best one. A non-PL winner is
    reported alongside and logged."""
    best_pl = max(mean_ll[m] for m in PL_SIMPLICITY_ORDER)
    contenders = [m for m in PL_SIMPLICITY_ORDER if best_pl - mean_ll[m] < SELECTION_MARGIN]
    published = contenders[0]
    notes = [f"Best PL model by mean held-out log-likelihood: {max(PL_SIMPLICITY_ORDER, key=lambda m: mean_ll[m])} ({best_pl:.4f}). "
             f"Models within {SELECTION_MARGIN} of it: {', '.join(contenders)}; the simplest, {published}, is published."]
    others = {m: v for m, v in mean_ll.items() if m not in PL_SIMPLICITY_ORDER and m != "U0"}
    rival = max(others, key=lambda m: others[m])
    if others[rival] > mean_ll[published]:
        notes.append(f"{rival} beats the published model ({others[rival]:.4f} vs {mean_ll[published]:.4f}); publish {published} anyway per "
                     f"rule 5.4(3), report {rival} alongside and log it in deviations.md.")
    return Selection(published=published, reported_alongside=rival, notes=tuple(notes), mean_ll=mean_ll)
