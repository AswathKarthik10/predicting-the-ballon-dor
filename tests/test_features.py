from pathlib import Path

import re

import numpy as np
import pandas as pd
import pytest

from src.model import features
from src.model.data import (FORECAST_EDITION, TARGET_COLS, TRAIN_EDITIONS, StudyData, check_snapshots,
                            load_study_data)
from src.model.features import (CORE_FEATURES, POST_FEATURES, POSITION_FEATURES, apply_residualiser,
                                assemble_features, fit_feature_params, m1_columns, m2_columns,
                                zscore_within_edition)

ROOT = Path(__file__).resolve().parents[1]
FEATURES_SOURCE = ROOT / "src" / "model" / "features.py"
CONTINUOUS = [c for c in CORE_FEATURES if c != "F8_defending_winner"]


@pytest.fixture(scope="module")
def study() -> StudyData:
    return load_study_data(ROOT)


def train_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["edition"].isin(TRAIN_EDITIONS)]


def test_feature_module_never_mentions_target_columns() -> None:
    source = FEATURES_SOURCE.read_text()
    # prev_bdo_rank and career_bdo_podiums_before are legitimate pre-window inputs, so match whole column names only
    for name in TARGET_COLS:
        assert not re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", source), f"features.py must not reference {name}"


def test_features_ignore_target_columns() -> None:
    raw = pd.read_csv(ROOT / "player_snapshot_t1.csv")
    shuffled = raw.copy()
    rng = np.random.default_rng(7)
    for col in TARGET_COLS:
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
    dropped = raw.drop(columns=list(TARGET_COLS))
    base = features.build_base_features(raw)
    pd.testing.assert_frame_equal(base, features.build_base_features(shuffled))
    pd.testing.assert_frame_equal(base, features.build_base_features(dropped))


def test_forecast_targets_are_null_and_enforced(study: StudyData) -> None:
    assert study.targets.loc[study.frame_t1["edition"] == FORECAST_EDITION].isna().all().all()
    t1 = pd.read_csv(ROOT / "player_snapshot_t1.csv")
    t2 = pd.read_csv(ROOT / "player_snapshot_t2.csv")
    t1.loc[t1["edition"] == FORECAST_EDITION, "bdo_points"] = 5.0
    t2.loc[t2["edition"] == FORECAST_EDITION, "bdo_points"] = 5.0
    with pytest.raises(ValueError, match="2026 target"):
        check_snapshots(t1, t2)


def test_feature_frames_contain_no_target_columns(study: StudyData) -> None:
    for frame in (study.frame_t1, study.frame_t2):
        assert not set(TARGET_COLS) & set(frame.columns)


def test_all_features_are_finite(study: StudyData) -> None:
    for frame, include_post in ((study.frame_t1, False), (study.frame_t2, True)):
        params = fit_feature_params(train_rows(frame), include_post)
        matrix = assemble_features(frame, params, include_post)
        assert np.isfinite(matrix.to_numpy(dtype=float)).all()


def test_continuous_features_are_standardised_within_edition(study: StudyData) -> None:
    params = fit_feature_params(train_rows(study.frame_t2), True)
    matrix = assemble_features(study.frame_t2, params, True)
    matrix["edition"] = study.frame_t2["edition"]
    for col in CONTINUOUS + list(POST_FEATURES):
        for ed, group in matrix.groupby("edition"):
            sd = group[col].std(ddof=1)
            if sd == 0:
                assert (group[col] == 0).all()
                continue
            assert abs(group[col].mean()) < 1e-9, (col, ed)
            if col == "F1_attack":
                continue  # goalkeepers are pinned to 0 after standardising the others
            assert abs(sd - 1) < 1e-9, (col, ed)


def test_goalkeepers_get_zero_attack(study: StudyData) -> None:
    gk = study.frame_t1["is_goalkeeper"] == 1
    assert gk.sum() > 0
    assert (study.frame_t1.loc[gk, "F1_attack"] == 0).all()


def test_zero_variance_columns_become_zero(study: StudyData) -> None:
    ed2025 = study.frame_t2[study.frame_t2["edition"] == 2025]
    assert (ed2025["P1_post_form"] == 0).all()
    values = pd.Series([3.0, 3.0, 3.0, 1.0, 2.0, 3.0], index=range(6))
    editions = pd.Series([1, 1, 1, 2, 2, 2], index=range(6))
    z = zscore_within_edition(values, editions, "test")
    assert (z.iloc[:3] == 0).all()
    assert abs(z.iloc[3:].mean()) < 1e-12


def test_playoff_stage_maps_to_round_of_32() -> None:
    stage = pd.Series(["playoff", "R32", None, "winner"])
    ordinal = features.stage_ordinal(stage, "test")
    assert ordinal.tolist() == [2.0, 2.0, 0.0, 7.0]
    with pytest.raises(ValueError, match="unknown stage"):
        features.stage_ordinal(pd.Series(["semi-ish"]), "test")


def test_imputation_flags_match_spec_counts(study: StudyData) -> None:
    assert int(study.frame_t1["club_rating_imputed"].sum()) == 2
    assert (study.frame_t1.loc[study.frame_t1["club_rating_imputed"] == 1, "edition"] == FORECAST_EDITION).all()


def test_residualiser_uses_training_rows_only(study: StudyData) -> None:
    frame = study.frame_t2
    train = train_rows(frame)
    base_params = fit_feature_params(train, True)

    altered = frame.copy()
    held_out = altered["edition"] == FORECAST_EDITION
    altered.loc[held_out, "season_attention_raw"] += 5.0
    altered.loc[held_out, "post_attention_raw"] -= 3.0
    unchanged_params = fit_feature_params(train_rows(altered), True)
    np.testing.assert_array_equal(base_params.season_attention.coefficients, unchanged_params.season_attention.coefficients)
    np.testing.assert_array_equal(base_params.post_attention.coefficients, unchanged_params.post_attention.coefficients)

    shifted = train.copy()
    shifted["season_attention_raw"] = shifted["season_attention_raw"] + 2.0 * shifted["F1_attack"]
    changed = fit_feature_params(shifted, True)
    assert not np.allclose(base_params.season_attention.coefficients, changed.season_attention.coefficients)


def test_season_attention_is_uncorrelated_with_regressors_in_training_fit(study: StudyData) -> None:
    train = train_rows(study.frame_t1)
    params = fit_feature_params(train, False)
    residual = apply_residualiser(train, params.season_attention, "season_attention_raw")
    for col in features.RESIDUALISER_REGRESSORS:
        assert abs(np.corrcoef(residual, train[col])[0, 1]) < 1e-9, col


def test_held_out_edition_features_use_training_params_only(study: StudyData) -> None:
    frame = study.frame_t1
    train = frame[frame["edition"].isin([2022, 2023, 2024])]
    params = fit_feature_params(train, False)
    first = assemble_features(frame, params, False)
    altered = frame.copy()
    altered.loc[altered["edition"] == 2025, "season_attention_raw"] += 10.0
    second = assemble_features(altered, params, False)
    other = frame["edition"] != 2025
    pd.testing.assert_frame_equal(first[other], second[other])


def test_column_sets_match_spec(study: StudyData) -> None:
    assert len(CORE_FEATURES) == 10
    assert m1_columns() == list(CORE_FEATURES) + list(POSITION_FEATURES)
    assert m2_columns()[-3:] == list(POST_FEATURES)
    params = fit_feature_params(train_rows(study.frame_t2), True)
    matrix = assemble_features(study.frame_t2, params, True)
    assert set(m2_columns()) <= set(matrix.columns)

