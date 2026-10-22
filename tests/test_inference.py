from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.model import cv, inference
from src.model.data import FORECAST_EDITION, StudyData, load_study_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def study() -> StudyData:
    return load_study_data(ROOT)


@pytest.fixture(scope="module")
def m1_fit(study: StudyData) -> inference.FinalFit:
    return inference.final_fit(study, cv.PL_SPECS["M1"])


def test_covariance_is_symmetric_positive_definite(m1_fit: inference.FinalFit) -> None:
    cov = m1_fit.covariance
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_laplace_draws_match_the_fitted_normal_and_are_reproducible(m1_fit: inference.FinalFit) -> None:
    first = inference.laplace_draws(m1_fit, 20000, np.random.default_rng(7))
    second = inference.laplace_draws(m1_fit, 20000, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.mean(axis=0), m1_fit.beta, atol=0.01)
    np.testing.assert_allclose(np.cov(first.T), m1_fit.covariance, atol=0.002)


def test_bootstrap_is_seeded_and_centred_near_the_fit(study: StudyData, m1_fit: inference.FinalFit) -> None:
    a = inference.bootstrap_betas(m1_fit, study, 30, np.random.default_rng(7))
    b = inference.bootstrap_betas(m1_fit, study, 30, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (30, len(m1_fit.beta))
    se = np.sqrt(np.diag(m1_fit.covariance))
    assert np.all(np.abs(a.mean(axis=0) - m1_fit.beta) < 4 * se)


def test_final_fit_never_uses_forecast_targets(study: StudyData, m1_fit: inference.FinalFit) -> None:
    targets = study.targets.copy()
    mask = (study.frame_t1["edition"] == FORECAST_EDITION).to_numpy()
    targets.loc[mask, "bdo_points"] = np.arange(mask.sum(), dtype=float)
    targets.loc[mask, "vote_share"] = 0.5
    other = inference.final_fit(replace(study, targets=targets), cv.PL_SPECS["M1"])
    np.testing.assert_array_equal(m1_fit.beta, other.beta)
    np.testing.assert_array_equal(m1_fit.matrices[FORECAST_EDITION], other.matrices[FORECAST_EDITION])


def test_signs_match_the_backtest(m1_fit: inference.FinalFit) -> None:
    weights = dict(zip(m1_fit.columns, m1_fit.beta))
    for feature in ("F3_big_stage", "F9_fame", "F4_major_titles", "F5_other_honors"):
        assert weights[feature] > 0
