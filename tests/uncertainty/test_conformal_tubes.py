import json
import math

import pytest

from scripts.uncertainty.calibrate_tubes import CalibrationConfig
from scripts.uncertainty.conformal_tubes import (
    conformal_rank,
    evaluate_trajectory_coverage,
    fit_trajectory_split_conformal,
    trajectory_nonconformity_score,
    validate_seed_splits,
)
from scripts.uncertainty.scenarios import ScenarioConfig


def _scene(frame_count: int = 40) -> dict:
    positions = [
        [0.01 * frame, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
        for frame in range(frame_count)
    ]
    return {
        "frame_count": frame_count,
        "fps": 30,
        "obstacles": [
            {
                "name": "moving-a",
                "type": "dynamic_sphere",
                "radius": 0.05,
                "positions": positions,
            },
            {
                "name": "moving-b",
                "type": "dynamic_sphere",
                "radius": 0.05,
                "positions": positions,
            },
        ],
    }


def test_conformal_rank_uses_one_based_finite_sample_quantile() -> None:
    assert conformal_rank(50, 0.05) == 49
    with pytest.raises(ValueError, match="no finite conformal threshold"):
        conformal_rank(10, 0.05)


def test_trajectory_score_is_maximum_over_obstacles_and_leads() -> None:
    score = trajectory_nonconformity_score(
        [(1, 0.10), (2, 0.30), (1, 0.20), (3, 0.18)],
        [0.0, 0.10, 0.15, 0.30],
    )
    assert score == pytest.approx(2.0)

    with pytest.raises(ValueError, match="cannot represent"):
        trajectory_nonconformity_score([(1, 0.01)], [0.0, 0.0])


def test_seed_splits_are_rejected_before_scenario_generation(monkeypatch) -> None:
    def fail_if_generated(*_args, **_kwargs):
        raise AssertionError("scenario generation must not run before split validation")

    monkeypatch.setattr(
        "scripts.uncertainty.conformal_tubes.build_scenario_truth",
        fail_if_generated,
    )
    with pytest.raises(ValueError, match="overlap"):
        fit_trajectory_split_conformal(
            _scene(),
            ScenarioConfig(kind="random_acceleration", event_frame_ratio=0.3),
            proper_training_seeds=(1, 2, 3),
            calibration_seeds=(3, 4, 5),
            config=CalibrationConfig(
                seeds=(3, 4, 5),
                issue_frame=8,
                horizon_frames=12,
                target_coverage=0.8,
            ),
        )

    with pytest.raises(ValueError, match="overlap"):
        validate_seed_splits((1, 2), (3, 4), (4, 5))


def test_schema_v2_fit_and_held_out_trajectory_coverage() -> None:
    scenario = ScenarioConfig(
        kind="random_acceleration",
        event_frame_ratio=9.0 / 39.0,
        affected_obstacle_fraction=1.0,
        acceleration_std=0.0002,
    )
    profile = fit_trajectory_split_conformal(
        _scene(),
        scenario,
        proper_training_seeds=tuple(range(1, 13)),
        calibration_seeds=tuple(range(101, 120)),
        config=CalibrationConfig(
            seeds=tuple(range(101, 120)),
            issue_frame=8,
            horizon_frames=12,
            target_coverage=0.90,
            base_margin=0.0,
        ),
    )

    assert profile["schema_version"] == 2
    assert profile["method"] == "trajectory_split_conformal"
    assert profile["coverage_scope"] == "simultaneous_affected_obstacles_and_leads"
    assert profile["proper_training_seeds"] == list(range(1, 13))
    assert profile["calibration_seeds"] == list(range(101, 120))
    assert len(profile["trajectory_calibration_scores"]) == 19
    assert profile["conformal_rank"] == 18
    assert all(
        right + 1e-12 >= left
        for left, right in zip(
            profile["adaptive_lead_shape"],
            profile["adaptive_lead_shape"][1:],
        )
    )
    assert math.isfinite(profile["conformal_threshold"])
    assert len(profile["provenance"]["base_scene_sha256"]) == 64

    coverage = evaluate_trajectory_coverage(
        profile,
        _scene(),
        scenario,
        test_seeds=tuple(range(201, 221)),
    )
    assert coverage["test_seeds"] == list(range(201, 221))
    assert coverage["total_trajectories"] == 20
    assert 0 <= coverage["covered_trajectories"] <= 20
    assert 0.0 <= coverage["trajectory_coverage"] <= 1.0
    assert len(coverage["wilson_95_interval"]) == 2
    assert len(coverage["per_lead_marginal_coverage"]) == 12
    assert coverage["maximum_violation"] >= 0.0


def test_serialized_profile_preserves_scenario_signature_compatibility() -> None:
    scenario = ScenarioConfig(
        kind="position_noise",
        affected_obstacle_fraction=1.0,
    )
    profile = fit_trajectory_split_conformal(
        _scene(),
        scenario,
        proper_training_seeds=tuple(range(1, 13)),
        calibration_seeds=tuple(range(101, 120)),
        config=CalibrationConfig(
            seeds=tuple(range(101, 120)),
            issue_frame=8,
            horizon_frames=12,
            target_coverage=0.90,
            base_margin=0.0,
        ),
    )

    serialized_profile = json.loads(json.dumps(profile))
    coverage = evaluate_trajectory_coverage(
        serialized_profile,
        _scene(),
        scenario,
        test_seeds=tuple(range(201, 211)),
    )

    assert coverage["total_trajectories"] == 10


def test_coverage_rejects_training_or_calibration_seed() -> None:
    profile = {
        "proper_training_seeds": [1, 2],
        "calibration_seeds": [3, 4],
    }
    with pytest.raises(ValueError, match="overlap"):
        evaluate_trajectory_coverage(
            profile,
            _scene(),
            ScenarioConfig(kind="position_noise"),
            test_seeds=(4, 5),
        )
