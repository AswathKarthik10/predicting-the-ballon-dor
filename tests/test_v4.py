from pathlib import Path

import pandas as pd
import pytest

from src.model import v4_candidates as vc

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
