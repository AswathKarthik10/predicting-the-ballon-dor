from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import experiment3 as e3
from src.model.data import V2_SPEC, load_study_v2_1
from src.model.features_v2 import V2_FEATURES
from src.model.run_v2_1 import v2_1_options

ROOT = Path(__file__).resolve().parents[1]
LABELLED = V2_SPEC.labelled_editions
REPRODUCTION_TOLERANCE = 1e-6


def test_labelled_editions_split_into_the_two_formats() -> None:
    assert set(LABELLED) == set(e3.CALENDAR_FOLDS) | set(e3.SEASON_FOLDS)
    assert not set(e3.CALENDAR_FOLDS) & set(e3.SEASON_FOLDS)


@pytest.mark.parametrize("held", e3.SEASON_FOLDS)
def test_season_fold_training_sets(held: int) -> None:
    others = [ed for ed in e3.SEASON_FOLDS if ed != held]
    assert e3.train_set(e3.RUN_SEASON, held, LABELLED) == others
    assert e3.train_set(e3.RUN_CALENDAR, held, LABELLED) == list(e3.CALENDAR_FOLDS)
    assert sorted(e3.train_set(e3.RUN_ALL, held, LABELLED)) == sorted(set(LABELLED) - {held})
    assert len(e3.train_set(e3.RUN_SEASON, held, LABELLED)) == 3
    assert len(e3.train_set(e3.RUN_ALL, held, LABELLED)) == 8


@pytest.mark.parametrize("held", e3.CALENDAR_FOLDS)
def test_mirror_fold_training_sets(held: int) -> None:
    assert e3.train_set(e3.RUN_CALENDAR_SELF, held, LABELLED) == [ed for ed in e3.CALENDAR_FOLDS if ed != held]
    assert e3.train_set(e3.RUN_SEASON_ON_CALENDAR, held, LABELLED) == list(e3.SEASON_FOLDS)
    assert len(e3.train_set(e3.RUN_ALL, held, LABELLED)) == 8


@pytest.mark.parametrize("job", e3.JOBS)
def test_held_out_edition_never_in_its_own_training_set(job: tuple[str, tuple[int, ...]]) -> None:
    run, folds = job
    for held in folds:
        assert held not in e3.train_set(run, held, LABELLED)


def test_unknown_run_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown run"):
        e3.train_set("E3-X", 2022, LABELLED)


def test_count_better_counts_folds_in_each_direction() -> None:
    rows = [{"run": r, "model": "M1", "fold": f, "metric": "heldout_ll", "value": v}
            for f, (a, b) in {2022: (1.0, 2.0), 2023: (3.0, 2.0), 2024: (5.0, 4.0)}.items() for r, v in (("A", a), ("B", b))]
    assert e3.count_better(pd.DataFrame(rows), "M1", "heldout_ll", "A", "B", (2022, 2023, 2024)) == (2, 1)


@pytest.fixture(scope="module")
def study():
    return load_study_v2_1(ROOT, V2_SPEC, v2_1_options(), V2_FEATURES)


def test_features_do_not_depend_on_the_run(study) -> None:
    """The feature frame is built once; a run only selects training editions, so rows are shared by construction."""
    before = study.frame_t1.copy()
    e3.run_one(study, e3.RUN_SEASON, "B1", e3.SEASON_FOLDS)
    pd.testing.assert_frame_equal(before, study.frame_t1)


def test_no_2026_target_is_read(study) -> None:
    assert 2026 not in study.labelled_editions
    assert study.targets.loc[study.frame_t1["edition"] == 2026, "bdo_points"].isna().all()


def test_e3_a_reproduces_v2_1_per_fold_numbers(study) -> None:
    result = e3.run_one(study, e3.RUN_ALL, "M1", e3.SEASON_FOLDS)
    reference = pd.read_csv(ROOT / "results" / "v2_1" / "metrics_by_fold_v2_1.csv")
    gap = e3.max_abs_gap(result.metrics, reference, e3.RUN_ALL, ("M1",), e3.SEASON_FOLDS)
    assert gap < REPRODUCTION_TOLERANCE


def test_season_interaction_columns_vanish_on_calendar_rows(study) -> None:
    from src.model import format_effects as fe
    variant, columns = fe.with_season_interactions(study, ("F4_major_titles",))
    params = variant.features.fit_params(variant.frame_t1, False)
    matrix = variant.features.assemble(variant.frame_t1, params, False)
    calendar = variant.frame_t1["is_calendar"] == 1
    assert (matrix.loc[calendar, "F4_major_titles_x_season"] == 0).all()
    assert np.allclose(matrix.loc[~calendar, "F4_major_titles_x_season"], matrix.loc[~calendar, "F4_major_titles"])
    assert columns[-1] == "F4_major_titles_x_season"


def test_median_lambda_needs_nine_folds() -> None:
    from src.model import format_effects as fe
    tuning = pd.DataFrame({"run": [e3.RUN_ALL] * 3, "model": ["M1"] * 3, "lambda_or_alpha": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="expected 9"):
        fe.median_e3a_lambda(tuning)


def test_published_top_ten_names_each_match_one_nominee() -> None:
    from src.model import compare_2026_v3 as cmp
    table = pd.read_csv(ROOT / "results" / "v3" / "forecast_2026_v3_T1.csv", comment="#")
    ranks = cmp.actual_rank_by_player(table)
    assert sorted(ranks.to_numpy()) == list(range(1, 11))


def test_unmatched_published_name_is_rejected() -> None:
    from src.model import compare_2026_v3 as cmp
    table = pd.DataFrame({"player_key": ["a"], "player_name": ["Somebody Else"]})
    with pytest.raises(ValueError, match="matched 0"):
        cmp.actual_rank_by_player(table)
