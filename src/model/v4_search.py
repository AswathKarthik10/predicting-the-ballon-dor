"""Experiment 4 configuration search (spec section 6.1).

A configuration is (preprocessing, selector, model). Inner scores are leave-one-edition-out over the training editions only;
the one-standard-error rule then picks the configuration with the fewest selected features among those within one standard
error of the best inner score. Caches only hold fits made from the training editions named in their key.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np

from src.model.pl import build_ranking, edition_loglik_per_voted
from src.model.v4_models import ModelSpec, Prediction, model_grid, predict_utilities
from src.model.v4_preprocess import PREPROCESS_OPTIONS, PreprocessOption, Preprocessor, fit_preprocessor, transform
from src.model.v4_select import (LAMBDA_FS2, NOT_USED, PI_GRID, STABILITY_PENALTY, TOP_K_GRID, SelectorKind, SelectorSpec, TrainSet,
                                 make_train_set, select_features, stability_frequencies)

SEARCH_SEED = 7
SEARCH_BUDGET = 300


@dataclass(frozen=True)
class Panel:
    x: np.ndarray  # encoded candidates, NaN kept
    edition: np.ndarray
    points: np.ndarray  # NaN for unlabelled editions
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    preprocess: PreprocessOption
    selector: SelectorSpec
    model: ModelSpec

    @property
    def name(self) -> str:
        return f"{self.preprocess.name}|{self.selector.name}|{self.model.name}"


@dataclass(frozen=True)
class InnerResult:
    config_index: int
    scores: tuple[float, ...]  # held-out PL log-likelihood per voted player, one per inner fold
    mean_score: float
    standard_error: float
    mean_n_features: float
    converged: bool


@dataclass(frozen=True)
class Prepared:
    preprocessor: Preprocessor
    z: np.ndarray  # every panel row, transformed with the training-fold preprocessor
    train: TrainSet


@dataclass
class FitCache:
    """Memo of fits keyed by the training editions they were fitted on."""
    prepared: dict[tuple[tuple[int, ...], str], Prepared] = field(default_factory=dict)
    frequencies: dict[tuple[tuple[int, ...], str], np.ndarray] = field(default_factory=dict)
    selections: dict[tuple[tuple[int, ...], str, str], np.ndarray] = field(default_factory=dict)


def selector_grid() -> list[SelectorSpec]:
    specs = [SelectorSpec(SelectorKind.NONE, NOT_USED, NOT_USED, NOT_USED)]
    specs += [SelectorSpec(SelectorKind.FILTER, k, NOT_USED, NOT_USED) for k in TOP_K_GRID]
    specs += [SelectorSpec(SelectorKind.LASSO, NOT_USED, lam, NOT_USED) for lam in LAMBDA_FS2]
    specs += [SelectorSpec(SelectorKind.STABILITY, NOT_USED, NOT_USED, pi) for pi in PI_GRID]
    specs += [SelectorSpec(SelectorKind.TREE, k, NOT_USED, NOT_USED) for k in TOP_K_GRID]
    specs += [SelectorSpec(SelectorKind.PRUNED_FILTER, k, NOT_USED, NOT_USED) for k in TOP_K_GRID]
    return specs


def full_grid() -> list[Config]:
    return [Config(p, s, m) for p, s, m in product(PREPROCESS_OPTIONS, selector_grid(), model_grid())]


def sample_configs(grid: list[Config], budget: int, seed: int) -> list[Config]:
    """The same random subset (without replacement) in every outer fold; the full grid when it fits in the budget."""
    if len(grid) <= budget:
        return grid
    chosen = np.sort(np.random.default_rng(seed).choice(len(grid), size=budget, replace=False))
    return [grid[i] for i in chosen]


def prepared_fold(panel: Panel, train_editions: tuple[int, ...], option: PreprocessOption, cache: FitCache) -> Prepared:
    key = (train_editions, option.name)
    if key in cache.prepared:
        return cache.prepared[key]
    in_train = np.isin(panel.edition, train_editions)
    preprocessor = fit_preprocessor(panel.x[in_train], option)
    z = transform(preprocessor, panel.x, panel.edition)
    rows = [panel.edition == ed for ed in train_editions]
    train = make_train_set(train_editions, [z[r] for r in rows], [panel.points[r] for r in rows])
    cache.prepared[key] = Prepared(preprocessor=preprocessor, z=z, train=train)
    return cache.prepared[key]


def selected_columns(prepared: Prepared, train_editions: tuple[int, ...], option: PreprocessOption, selector: SelectorSpec,
                     cache: FitCache) -> np.ndarray:
    key = (train_editions, option.name, selector.name)
    if key in cache.selections:
        return cache.selections[key]
    frequency_key = (train_editions, option.name)
    if selector.kind is SelectorKind.STABILITY and frequency_key not in cache.frequencies:
        cache.frequencies[frequency_key] = stability_frequencies(prepared.train, STABILITY_PENALTY)
    frequencies = cache.frequencies.get(frequency_key, np.zeros(0))
    cache.selections[key] = select_features(prepared.train, selector, frequencies)
    return cache.selections[key]


def fit_and_predict(config: Config, panel: Panel, train_editions: tuple[int, ...], apply_edition: int,
                    cache: FitCache) -> tuple[Prediction, np.ndarray]:
    prepared = prepared_fold(panel, train_editions, config.preprocess, cache)
    columns = selected_columns(prepared, train_editions, config.preprocess, config.selector, cache)
    apply_x = prepared.z[panel.edition == apply_edition]
    return predict_utilities(config.model, prepared.train, columns, [apply_x]), columns


def score_config(index: int, config: Config, panel: Panel, train_editions: tuple[int, ...], cache: FitCache) -> InnerResult:
    scores: list[float] = []
    n_features: list[int] = []
    converged = True
    for held in train_editions:
        inner_train = tuple(ed for ed in train_editions if ed != held)
        prediction, columns = fit_and_predict(config, panel, inner_train, held, cache)
        ranking = build_ranking(panel.points[panel.edition == held])
        scores.append(edition_loglik_per_voted(prediction.utilities[0], ranking))
        n_features.append(len(columns))
        converged = converged and prediction.converged
    values = np.array(scores)
    return InnerResult(config_index=index, scores=tuple(scores), mean_score=float(values.mean()),
                       standard_error=float(values.std(ddof=1) / np.sqrt(len(values))), mean_n_features=float(np.mean(n_features)),
                       converged=converged)


def one_standard_error_choice(results: list[InnerResult]) -> InnerResult:
    """Among configurations within one SE of the best mean score, the one with the fewest selected features.
    Ties go to the higher mean score, then to the lower configuration index."""
    best = max(results, key=lambda r: (r.mean_score, -r.config_index))
    threshold = best.mean_score - best.standard_error
    eligible = [r for r in results if r.mean_score >= threshold]
    return min(eligible, key=lambda r: (r.mean_n_features, -r.mean_score, r.config_index))
