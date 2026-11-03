from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

from src.model import features
from src.model.data import (EDITION_FACTS, TARGET_COLS, V1_SPEC, V2_DIR, V2_SPEC, StudyData, check_edition_facts,
                            check_snapshots, load_bridge_study)
from src.model.features import fit_feature_params, zscore_within_edition
from src.model.pl import build_ranking, edition_loglik, edition_loglik_per_voted

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
T1_V2 = ROOT / V2_DIR / "player_snapshot_t1.csv"
T2_V2 = ROOT / V2_DIR / "player_snapshot_t2.csv"
BRIDGE_LABELLED = (2022, 2023, 2024, 2025)
FORBIDDEN_FEATURE_WORDS = ("odds", "implied_prob")
HOLDOUT_PATH_FRAGMENT = "holdout"
TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def t1_v2() -> pd.DataFrame:
    return pd.read_csv(T1_V2)


@pytest.fixture(scope="module")
def bridge() -> StudyData:
    return load_bridge_study(ROOT)


def test_pl_log_likelihood_hand_checked_with_tie_and_zero_point_player() -> None:
    utility = np.array([0.5, 0.2, -0.1, 0.3])
    points = np.array([5.0, 3.0, 3.0, 0.0])  # players 1 and 2 tie; player 3 has no points
    ranking = build_ranking(points)
    first = utility[0] - logsumexp(utility)  # player 0 chosen from all four
    tied = utility[1] + utility[2] - 2 * logsumexp(utility[[1, 2, 3]])  # tied pair, zero-point player still in the pool
    assert edition_loglik(utility, ranking) == pytest.approx(first + tied, abs=TOLERANCE)
    assert ranking.n_voted == 3
    assert edition_loglik_per_voted(utility, ranking) == pytest.approx((first + tied) / 3, abs=TOLERANCE)


def test_pl_log_likelihood_for_a_three_pick_ballot_edition() -> None:
    rng = np.random.default_rng(7)
    utility = rng.normal(size=8)
    points = np.zeros(8)
    points[[4, 1, 6]] = [5.0, 3.0, 1.0]  # the 2016 scheme: only three players receive points
    ranking = build_ranking(points)
    expected = 0.0
    remaining = list(range(8))
    for chosen in (4, 1, 6):
        expected += utility[chosen] - logsumexp(utility[remaining])
        remaining.remove(chosen)
    assert ranking.n_voted == 3
    assert edition_loglik(utility, ranking) == pytest.approx(expected, abs=TOLERANCE)


def test_edition_facts_cover_every_edition_in_the_expanded_file(t1_v2: pd.DataFrame) -> None:
    assert set(t1_v2["edition"]) == set(EDITION_FACTS) == set(V2_SPEC.t1_editions)


def test_expanded_file_satisfies_the_edition_facts(t1_v2: pd.DataFrame) -> None:
    check_edition_facts(t1_v2)


def test_point_sum_assertion_fails_when_a_point_is_moved(t1_v2: pd.DataFrame) -> None:
    broken = t1_v2.copy()
    first = broken.index[(broken["edition"] == 2018) & (broken["bdo_points"] > 0)][0]
    broken.loc[first, "bdo_points"] += 1
    with pytest.raises(ValueError, match="2018"):
        check_edition_facts(broken)


def test_zero_point_count_assertion_fails_when_a_nominee_gains_points(t1_v2: pd.DataFrame) -> None:
    broken = t1_v2.copy()
    zero_rows = broken.index[(broken["edition"] == 2016) & (broken["bdo_points"] == 0)]
    donor = broken.index[(broken["edition"] == 2016) & (broken["bdo_points"] > 1)][0]
    broken.loc[zero_rows[0], "bdo_points"] = 1
    broken.loc[donor, "bdo_points"] -= 1
    with pytest.raises(ValueError, match="zero-point"):
        check_edition_facts(broken)


def test_2026_targets_are_null_in_the_expanded_file(t1_v2: pd.DataFrame) -> None:
    forecast = t1_v2[t1_v2["edition"] == V2_SPEC.forecast_edition]
    assert forecast[list(TARGET_COLS)].isna().all().all()


def test_2026_targets_are_null_in_every_loaded_frame(bridge: StudyData) -> None:
    mask = (bridge.frame_t1["edition"] == bridge.forecast_edition).to_numpy()
    assert bridge.targets.loc[mask].isna().all().all()
    assert not bridge.targets.loc[~mask].isna().any().any()


def test_snapshot_check_rejects_a_non_null_forecast_target() -> None:
    t1 = pd.read_csv(T1_V2)
    t1 = t1[t1["edition"].isin(V1_SPEC.t1_editions)].reset_index(drop=True)
    t2 = pd.read_csv(T2_V2)
    t1.loc[t1["edition"] == V1_SPEC.forecast_edition, "bdo_points"] = 5.0
    t2.loc[t2["edition"] == V1_SPEC.forecast_edition, "bdo_points"] = 5.0
    with pytest.raises(ValueError, match="null"):
        check_snapshots(t1, t2, V1_SPEC)


def test_no_source_file_reads_the_2026_result() -> None:
    guard_file = "forecast_v2.py"  # names the holdout path only to refuse to run when it exists
    mentions = {p.name: [line for line in p.read_text().splitlines() if HOLDOUT_PATH_FRAGMENT in line.lower()] for p in SRC.rglob("*.py")}
    offenders = [name for name, lines in mentions.items() if lines and name != guard_file]
    assert offenders == []
    reads = [line for line in mentions[guard_file] if any(call in line for call in ("read_csv", "read_parquet", "open(", "read_text"))]
    assert reads == []


def test_odds_never_enter_the_feature_code_or_the_feature_frames(bridge: StudyData) -> None:
    source = (SRC / "model" / "features.py").read_text().lower()
    assert not any(word in source for word in FORBIDDEN_FEATURE_WORDS)
    columns = [c.lower() for c in bridge.frame_t2.columns]
    assert not any(word in col for col in columns for word in FORBIDDEN_FEATURE_WORDS)


def test_bridge_study_has_the_v1_shape(bridge: StudyData) -> None:
    assert bridge.labelled_editions == BRIDGE_LABELLED
    assert bridge.forecast_edition == 2026
    assert len(bridge.frame_t1) == 150


def test_within_edition_standardisation_ignores_other_editions(bridge: StudyData) -> None:
    raw = bridge.frame_t1["F9_fame"].copy()
    editions = bridge.frame_t1["edition"]
    before = zscore_within_edition(raw, editions, "fame")
    perturbed = raw.copy()
    perturbed[editions == 2025] = perturbed[editions == 2025] * 10 + 3
    after = zscore_within_edition(perturbed, editions, "fame")
    np.testing.assert_allclose(before[editions != 2025], after[editions != 2025], rtol=0, atol=TOLERANCE)


def test_residualiser_fit_on_training_editions_ignores_the_held_out_edition(bridge: StudyData) -> None:
    frame = bridge.frame_t1
    train = frame[frame["edition"].isin(BRIDGE_LABELLED[:-1])]
    altered = frame.copy()
    held_out = altered["edition"] == BRIDGE_LABELLED[-1]
    altered.loc[held_out, "season_attention_raw"] = altered.loc[held_out, "season_attention_raw"] * -5 + 2
    train_altered = altered[altered["edition"].isin(BRIDGE_LABELLED[:-1])]
    before = fit_feature_params(train, include_post=False).season_attention.coefficients
    after = fit_feature_params(train_altered, include_post=False).season_attention.coefficients
    np.testing.assert_array_equal(before, after)


def test_study_data_carries_its_own_edition_lists(bridge: StudyData) -> None:
    changed = replace(bridge, labelled_editions=(2022, 2023))
    assert changed.labelled_editions == (2022, 2023) and bridge.labelled_editions == BRIDGE_LABELLED
    assert features.WINDOW_END.keys() >= set(V2_SPEC.t1_editions)
