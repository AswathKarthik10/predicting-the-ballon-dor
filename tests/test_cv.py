from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model import baselines, cv
from src.model.data import StudyData, load_study_data
from src.model.pl import build_ranking

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = 2025
TRAIN = [2022, 2023, 2024]


@pytest.fixture(scope="module")
def study() -> StudyData:
    return load_study_data(ROOT)


def with_scrambled_targets(study: StudyData, edition: int) -> StudyData:
    targets = study.targets.copy()
    mask = (study.frame_t1["edition"] == edition).to_numpy()
    rng = np.random.default_rng(1)
    for col in ("bdo_points", "vote_share"):
        targets.loc[mask, col] = rng.permutation(targets.loc[mask, col].to_numpy())
    return replace(study, targets=targets)


@pytest.mark.parametrize("name", ["M1", "M2", "B0", "B1", "B2"])
def test_predictions_for_held_out_edition_do_not_depend_on_its_targets(study: StudyData, name: str) -> None:
    before, _ = cv.predict_model(study, name, TRAIN, [HELD_OUT])
    after, _ = cv.predict_model(with_scrambled_targets(study, HELD_OUT), name, TRAIN, [HELD_OUT])
    np.testing.assert_allclose(before[HELD_OUT], after[HELD_OUT], rtol=0, atol=1e-12)


def test_predictions_do_change_when_training_targets_change(study: StudyData) -> None:
    before, _ = cv.predict_model(study, "M1", TRAIN, [HELD_OUT])
    after, _ = cv.predict_model(with_scrambled_targets(study, 2023), "M1", TRAIN, [HELD_OUT])
    assert not np.allclose(before[HELD_OUT], after[HELD_OUT])


def test_edition_metrics_on_known_case() -> None:
    points = np.array([10.0, 6.0, 3.0, 1.0, 0.0, 0.0])
    perfect = cv.edition_metrics(np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), points)
    assert perfect["top1_hit"] == 1.0 and perfect["top3_overlap"] == 3.0 and perfect["winner_rank"] == 1.0
    assert perfect["pairwise_acc"] == 1.0 and perfect["spearman"] > 0.9
    reversed_ = cv.edition_metrics(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), points)
    assert reversed_["top1_hit"] == 0.0 and reversed_["winner_rank"] == 6.0 and reversed_["pairwise_acc"] == 0.0


def test_uniform_utility_log_likelihood_is_the_reference(study: StudyData) -> None:
    utilities, _ = cv.predict_model(study, "U0", TRAIN, [HELD_OUT])
    points = baselines.points_of(study, HELD_OUT)
    metrics = cv.edition_metrics(utilities[HELD_OUT], points)
    ranking = build_ranking(points)
    # each tied group of size m chosen from the n - start players still available costs m * log(n - start)
    expected = -sum(m * np.log(len(points) - s) for s, m in zip(ranking.group_starts, ranking.group_sizes)) / ranking.n_voted
    assert metrics["heldout_ll"] == pytest.approx(expected, abs=1e-12)


def test_selection_rule_prefers_m1_inside_the_margin() -> None:
    summary = pd.DataFrame([{"model": m, "metric": "heldout_ll", "mean": v} for m, v in
                            {"M1": -2.00, "M2": -1.99, "B1": -2.2, "B2": -2.3, "X1": -2.1}.items()])
    assert cv.apply_selection_rule(summary).published == "M1"
    summary.loc[summary["model"] == "M2", "mean"] = -1.97
    assert cv.apply_selection_rule(summary).published == "M2"
    summary.loc[summary["model"] == "B1", "mean"] = -1.0
    assert any("B1 beats both" in n for n in cv.apply_selection_rule(summary).notes)
