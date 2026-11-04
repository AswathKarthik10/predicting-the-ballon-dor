from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import pl, v4_candidates as vc, v4_models as vm, v4_pl_fast as vf, v4_pl_penalised as vpen
from src.model import v4_preprocess as vp, v4_search as vs
from src.model.v4_select import SelectorKind, SelectorSpec

ROOT = Path(__file__).resolve().parents[1]
TARGET_DERIVED_PREFIXES = ("bdo_", "vote_share")


@pytest.fixture(scope="module")
def t1() -> pd.DataFrame:
    return vc.load_raw_t1(ROOT)


@pytest.fixture(scope="module")
def candidates(t1: pd.DataFrame) -> vc.Candidates:
    return vc.build_candidates(t1)


def test_t1_has_five_editions_of_thirty_rows(t1: pd.DataFrame) -> None:
    assert len(t1) == 150
    assert t1.groupby("edition").size().eq(30).all()


def test_candidates_exclude_every_listed_column_and_no_target_derived_column(candidates: vc.Candidates) -> None:
    listed = set(vc.listed_exclusions())
    sources = set(candidates.table.loc[candidates.table["status"] == "candidate", "source_column"])
    assert not sources & listed
    assert not [c for c in candidates.features.columns if c.startswith(TARGET_DERIVED_PREFIXES)]


def test_known_zero_variance_and_edition_constant_columns_are_excluded(t1: pd.DataFrame) -> None:
    reasons = vc.exclusion_reasons(t1)
    assert reasons["jurors_same_nation"] == vc.REASON_ZERO_VARIANCE
    assert reasons["prev_bdo_rank_partial"] == vc.REASON_ZERO_VARIANCE
    assert reasons["ballot_picks"] == vc.REASON_EDITION_CONSTANT


def test_encoding_is_numeric_with_missing_indicator_for_every_column_with_nulls(t1: pd.DataFrame, candidates: vc.Candidates) -> None:
    assert (candidates.features.dtypes == float).all()
    labelled = t1["edition"].isin(vc.LABELLED_EDITIONS)
    for col in ("prev_bdo_rank", "intl_npxg_xa_p90", "ea_overall_t1"):
        assert f"{col}{vc.MISSING_SUFFIX}" in candidates.features.columns
        assert col in candidates.features.columns
    indicators = candidates.features.loc[labelled, [c for c in candidates.features if c.endswith(vc.MISSING_SUFFIX)]]
    assert (indicators.nunique() > 1).all()


def test_rare_categories_merge_into_other_and_each_level_has_enough_labelled_rows(t1: pd.DataFrame, candidates: vc.Candidates) -> None:
    labelled = t1["edition"].isin(vc.LABELLED_EDITIONS)
    levels = [c for c in candidates.features.columns if c.startswith("nationality_")]
    assert "nationality_Netherlands" not in levels
    assert "nationality_other" in levels
    kept = [c for c in levels if c != "nationality_other"]
    assert (candidates.features.loc[labelled, kept].sum() >= vc.MIN_CATEGORY_ROWS).all()


def test_candidate_features_keep_row_alignment_with_t1(t1: pd.DataFrame, candidates: vc.Candidates) -> None:
    assert candidates.features.index.equals(t1.index)
    assert candidates.rows["player_key"].equals(t1["player_key"])


def test_forecast_edition_targets_are_null_in_rows(candidates: vc.Candidates) -> None:
    assert candidates.rows.loc[candidates.rows["edition"] == 2026, vc.POINTS_COL].isna().all()


# ---- pipeline tests (spec section 10) ----
TRAIN = (2022, 2023, 2024)
HELD_OUT = 2025


@pytest.fixture(scope="module")
def panel(candidates: vc.Candidates) -> vs.Panel:
    return vs.Panel(x=candidates.features.to_numpy(), edition=candidates.rows["edition"].to_numpy(),
                    points=candidates.rows[vc.POINTS_COL].to_numpy(), columns=tuple(candidates.features.columns))


def perturbed_held_out(panel: vs.Panel, edition: int) -> vs.Panel:
    """Same panel with the held-out edition's features replaced by noise and its points shuffled."""
    rng = np.random.default_rng(3)
    rows = panel.edition == edition
    x = panel.x.copy()
    x[rows] = rng.normal(size=x[rows].shape) * 50
    points = panel.points.copy()
    points[rows] = rng.permutation(points[rows])
    return vs.Panel(x=x, edition=panel.edition, points=points, columns=panel.columns)


def config_of(option_index: int, selector: SelectorSpec, model: vm.ModelSpec) -> vs.Config:
    return vs.Config(vp.PREPROCESS_OPTIONS[option_index], selector, model)


def spec_of(kind: vm.ModelKind, lam: float, mix: float, size: int, n: int, leaf: int) -> vm.ModelSpec:
    return vm.ModelSpec(kind, lam, mix, size, n, leaf)


SELECTOR_CASES = [
    SelectorSpec(SelectorKind.NONE, 0, 0, 0),
    SelectorSpec(SelectorKind.FILTER, 10, 0, 0),
    SelectorSpec(SelectorKind.LASSO, 0, 0.1, 0),
    SelectorSpec(SelectorKind.STABILITY, 0, 0, 0.6),
    SelectorSpec(SelectorKind.TREE, 10, 0, 0),
    SelectorSpec(SelectorKind.PRUNED_FILTER, 10, 0, 0),
]
MODEL_CASES = [
    spec_of(vm.ModelKind.RIDGE, 1.0, 0, 0, 0, 0),
    spec_of(vm.ModelKind.ELASTIC, 0.5, 0.5, 0, 0, 0),
    spec_of(vm.ModelKind.LGB, 0, 0, 4, 100, 5),
    spec_of(vm.ModelKind.XGB, 0, 0, 2, 100, 0),
    spec_of(vm.ModelKind.RF, 0, 0, 3, 500, 3),
]


@pytest.mark.parametrize("option_index", [0, 3])
@pytest.mark.parametrize("selector", SELECTOR_CASES, ids=lambda s: s.kind.value)
def test_outer_held_out_edition_never_enters_any_inner_fit(panel: vs.Panel, option_index: int, selector: SelectorSpec) -> None:
    model = MODEL_CASES[SELECTOR_CASES.index(selector) % len(MODEL_CASES)]
    config = config_of(option_index, selector, model)
    altered = perturbed_held_out(panel, HELD_OUT)
    cache_a, cache_b = vs.FitCache(), vs.FitCache()
    a = vs.score_config(0, config, panel, TRAIN, cache_a)
    b = vs.score_config(0, config, altered, TRAIN, cache_b)
    assert a.scores == b.scores
    for key, prepared in cache_a.prepared.items():
        np.testing.assert_array_equal(prepared.preprocessor.medians, cache_b.prepared[key].preprocessor.medians)
        np.testing.assert_array_equal(prepared.preprocessor.mean, cache_b.prepared[key].preprocessor.mean)
    for key, columns in cache_a.selections.items():
        np.testing.assert_array_equal(columns, cache_b.selections[key])


@pytest.mark.parametrize("model", MODEL_CASES, ids=lambda m: m.kind.value)
def test_refit_prediction_for_held_out_edition_ignores_its_points(panel: vs.Panel, model: vm.ModelSpec) -> None:
    config = config_of(1, SelectorSpec(SelectorKind.FILTER, 10, 0, 0), model)
    shuffled_points = panel.points.copy()
    rows = panel.edition == HELD_OUT
    shuffled_points[rows] = np.random.default_rng(5).permutation(shuffled_points[rows])
    other = vs.Panel(x=panel.x, edition=panel.edition, points=shuffled_points, columns=panel.columns)
    a, _ = vs.fit_and_predict(config, panel, TRAIN, HELD_OUT, vs.FitCache())
    b, _ = vs.fit_and_predict(config, other, TRAIN, HELD_OUT, vs.FitCache())
    np.testing.assert_allclose(a.utilities[0], b.utilities[0], rtol=0, atol=1e-12)


def test_within_edition_scaling_uses_only_the_rows_own_edition(panel: vs.Panel) -> None:
    train_rows = np.isin(panel.edition, TRAIN)
    for option in (vp.PREPROCESS_OPTIONS[0], vp.PREPROCESS_OPTIONS[1], vp.PREPROCESS_OPTIONS[4]):
        prep = vp.fit_preprocessor(panel.x[train_rows], option)
        before = vp.transform(prep, panel.x, panel.edition)
        altered = perturbed_held_out(panel, 2023)
        after = vp.transform(prep, altered.x, altered.edition)
        for edition in (2022, 2024, 2025, 2026):
            rows = panel.edition == edition
            np.testing.assert_allclose(before[rows], after[rows], rtol=0, atol=1e-12)


def test_global_scaling_is_fitted_on_training_rows_only(panel: vs.Panel) -> None:
    option = vp.PREPROCESS_OPTIONS[2]
    train_rows = np.isin(panel.edition, TRAIN)
    prep = vp.fit_preprocessor(panel.x[train_rows], option)
    altered = perturbed_held_out(panel, HELD_OUT)
    other = vp.fit_preprocessor(altered.x[np.isin(altered.edition, TRAIN)], option)
    np.testing.assert_array_equal(prep.mean, other.mean)
    np.testing.assert_array_equal(prep.medians, other.medians)


def result(index: int, mean: float, se: float, n_features: float) -> vs.InnerResult:
    return vs.InnerResult(config_index=index, scores=(mean,), mean_score=mean, standard_error=se, mean_n_features=n_features, converged=True)


def test_one_standard_error_rule_picks_fewest_features_among_tied_configurations() -> None:
    results = [result(0, -2.00, 0.10, 40), result(1, -2.05, 0.10, 8), result(2, -2.09, 0.10, 3), result(3, -2.30, 0.10, 1)]
    assert vs.one_standard_error_choice(results).config_index == 2


def test_one_standard_error_rule_breaks_feature_ties_by_score_then_index() -> None:
    results = [result(0, -2.00, 0.10, 5), result(1, -2.02, 0.10, 5), result(2, -2.00, 0.10, 5)]
    assert vs.one_standard_error_choice(results).config_index == 0


def test_forecast_edition_targets_are_never_read_during_the_search(panel: vs.Panel) -> None:
    assert np.isnan(panel.points[panel.edition == 2026]).all()
    config = config_of(0, SelectorSpec(SelectorKind.FILTER, 5, 0, 0), spec_of(vm.ModelKind.RIDGE, 1.0, 0, 0, 0, 0))
    vs.score_config(0, config, panel, TRAIN, vs.FitCache())  # would raise on NaN points if 2026 were in any training set


def test_two_runs_with_the_same_seed_give_identical_scores_and_samples(panel: vs.Panel) -> None:
    grid = vs.full_grid()
    assert [c.name for c in vs.sample_configs(grid, 20, vs.SEARCH_SEED)] == [c.name for c in vs.sample_configs(grid, 20, vs.SEARCH_SEED)]
    for model in (MODEL_CASES[2], MODEL_CASES[4]):
        config = config_of(1, SelectorSpec(SelectorKind.TREE, 10, 0, 0), model)
        a = vs.score_config(0, config, panel, TRAIN, vs.FitCache())
        b = vs.score_config(0, config, panel, TRAIN, vs.FitCache())
        assert a == b


def test_vectorised_pl_matches_the_reference_loss_and_gradient(panel: vs.Panel) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 6))
    points = panel.points[panel.edition == 2024]
    ranking = pl.build_ranking(points)
    beta = rng.normal(size=6)
    value, grad = vf.loglik_and_gradient(vf.sort_edition(x, ranking), beta)
    assert value == pytest.approx(pl.edition_loglik(x @ beta, ranking), abs=1e-9)
    np.testing.assert_allclose(grad, pl.edition_gradient(x, beta, ranking), atol=1e-9)


def test_ridge_via_proximal_solver_matches_reference_ridge(panel: vs.Panel) -> None:
    rng = np.random.default_rng(1)
    rankings = [pl.build_ranking(panel.points[panel.edition == ed]) for ed in TRAIN]
    xs = [rng.normal(size=(30, 5)) for _ in TRAIN]
    reference = pl.fit_pl(xs, rankings, 1.0)
    fitted = vpen.fit_elastic_net_sorted(vpen.sort_editions(xs, rankings), 0.0, 1.0, np.zeros(5))
    assert fitted.converged
    np.testing.assert_allclose(fitted.beta, reference, atol=1e-3)


def test_lasso_selects_fewer_columns_as_the_penalty_grows(panel: vs.Panel) -> None:
    rng = np.random.default_rng(2)
    rankings = [pl.build_ranking(panel.points[panel.edition == ed]) for ed in TRAIN]
    xs = [rng.normal(size=(30, 12)) for _ in TRAIN]
    editions = vpen.sort_editions(xs, rankings)
    counts = [int((vpen.fit_pl_lasso(editions, lam, np.zeros(12)).beta != 0).sum()) for lam in (0.01, 1.0, 100.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0
