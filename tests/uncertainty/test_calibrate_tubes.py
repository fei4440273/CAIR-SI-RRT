import json
from pathlib import Path

import pytest

from scripts.uncertainty.calibrate_tubes import (
    CalibrationConfig,
    build_calibration_profile,
    load_calibration_profile,
)
from scripts.uncertainty.conformal_tubes import fit_trajectory_split_conformal
from scripts.uncertainty.generate_trial import PredictionConfig, build_prediction_snapshot
from scripts.uncertainty.run_pilot import _validated_profile
from scripts.uncertainty.scenarios import ScenarioConfig, build_scenario_truth


def make_scene(frame_count: int = 40) -> dict:
    return {
        "frame_count": frame_count,
        "fps": 30,
        "obstacles": [
            {
                "name": "moving",
                "type": "dynamic_sphere",
                "radius": 0.05,
                "positions": [
                    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
                    for _ in range(frame_count)
                ],
            }
        ],
    }


def calibration_profile(source: Path | None = None) -> tuple[dict, ScenarioConfig]:
    scenario = ScenarioConfig(
        kind="milestone_intrusion",
        event_frame_ratio=6.0 / 39.0,
        intrusion_approach_frames=5,
        intrusion_hold_frames=4,
    )
    return (
        build_calibration_profile(
            make_scene(),
            scenario,
            CalibrationConfig(
                seeds=(1, 2, 3, 4, 5),
                issue_frame=5,
                horizon_frames=12,
                target_coverage=0.8,
                base_margin=0.01,
            ),
            source_scene=source,
        ),
        scenario,
    )


def test_fixed_and_adaptive_are_calibrated_to_same_target() -> None:
    profile, _ = calibration_profile()

    bounds = profile["adaptive_residual_bounds"]
    metrics = profile["calibration_metrics"]
    assert bounds[-1] > bounds[1]
    assert max(bounds) > bounds[-1]
    assert metrics["adaptive_empirical_coverage"] >= profile["target_coverage"]
    assert metrics["fixed_empirical_coverage"] >= profile["target_coverage"]
    assert metrics["sample_count"] == 5 * 12


def test_calibration_uses_worst_obstacle_per_seed_and_lead() -> None:
    single_profile, scenario = calibration_profile()
    dense_scene = make_scene()
    template = dense_scene["obstacles"][0]
    for index in range(1, 20):
        obstacle = json.loads(json.dumps(template))
        obstacle["name"] = f"moving-{index}"
        dense_scene["obstacles"].append(obstacle)
    dense_profile = build_calibration_profile(
        dense_scene,
        scenario,
        CalibrationConfig(
            seeds=(1, 2, 3, 4, 5),
            issue_frame=5,
            horizon_frames=12,
            target_coverage=0.8,
            base_margin=0.01,
        ),
    )

    assert dense_profile["coverage_scope"] == "simultaneous_obstacle_max_per_seed_lead"
    assert dense_profile["calibration_metrics"]["sample_count"] == 5 * 12
    assert dense_profile["fixed_radius"] == pytest.approx(single_profile["fixed_radius"])
    assert dense_profile["adaptive_residual_bounds"] == pytest.approx(
        single_profile["adaptive_residual_bounds"]
    )


def test_position_noise_calibration_uses_constant_offset_model() -> None:
    profile = build_calibration_profile(
        make_scene(frame_count=80),
        ScenarioConfig(kind="position_noise", event_frame_ratio=0.2),
        CalibrationConfig(
            seeds=(1, 2, 3, 4, 5),
            issue_frame=20,
            horizon_frames=40,
            target_coverage=0.8,
            observation_noise_std=0.01,
            residual_motion_model="auto",
        ),
    )

    assert profile["runtime_parameters"]["residual_motion_model"] == "constant_offset"
    assert profile["adaptive_residual_bounds"][-1] < 0.05


def test_offline_bounds_are_used_without_reading_test_future() -> None:
    profile, scenario_config = calibration_profile()
    truth = build_scenario_truth(
        make_scene(),
        ScenarioConfig(**{**scenario_config.__dict__, "seed": 101}),
    )
    snapshot = build_prediction_snapshot(
        truth,
        issue_frame=5,
        config=PredictionConfig(
            seed=101,
            base_margin=0.01,
            growth_per_second=0.0,
            max_tube_radius=1.0,
            offline_residual_bounds=tuple(profile["adaptive_residual_bounds"]),
        ),
        tube_mode="adaptive",
    )

    radii = snapshot["obstacles"][0]["uncertainty_radii"]
    assert radii[10] == pytest.approx(0.01 + profile["adaptive_residual_bounds"][5])
    assert snapshot["_uncertainty_experiment"]["calibration_model"].startswith(
        "offline_seed_split"
    )


def test_profile_loader_and_runner_reject_calibration_seed(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(make_scene()), encoding="utf-8")
    profile, scenario_config = calibration_profile(source)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    loaded, digest = load_calibration_profile(profile_path)
    assert loaded["fixed_radius"] == profile["fixed_radius"]
    assert len(digest) == 64

    truth = build_scenario_truth(
        make_scene(),
        ScenarioConfig(**{**scenario_config.__dict__, "seed": 1}),
    )
    truth["_uncertainty_scenario"]["source_scene"] = str(source.resolve())
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "calibration_profile": profile_path,
            "seed": 1,
            "truth_scene": truth_path,
            "initial_issue_frame": 5,
        },
    )()
    with pytest.raises(ValueError, match="calibration seed split"):
        _validated_profile(args)


def test_runner_accepts_independent_test_seed_and_matching_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(make_scene()), encoding="utf-8")
    profile, scenario_config = calibration_profile(source)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    truth = build_scenario_truth(
        make_scene(),
        ScenarioConfig(**{**scenario_config.__dict__, "seed": 101}),
    )
    truth["_uncertainty_scenario"]["source_scene"] = str(source.resolve())
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "calibration_profile": profile_path,
            "seed": 101,
            "truth_scene": truth_path,
            "initial_issue_frame": 5,
        },
    )()

    loaded, digest = _validated_profile(args)
    assert loaded is not None
    assert len(digest) == 64


def test_runner_accepts_legacy_profile_without_irrelevant_temporal_gate_defaults(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(make_scene()), encoding="utf-8")
    profile, scenario_config = calibration_profile(source)
    legacy_config = profile["scenario_config_without_seed"]
    newly_added_defaults = {
        "affected_obstacle_fraction",
        "event_frame_jitter_fraction",
        "occlusion_reappearance_offset_std",
        "occlusion_reappearance_maximum_offset",
    }
    for key in list(legacy_config):
        if key.startswith("temporal_gate_") or key in newly_added_defaults:
            legacy_config.pop(key)
    profile_path = tmp_path / "legacy-profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    truth = build_scenario_truth(
        make_scene(),
        ScenarioConfig(**{**scenario_config.__dict__, "seed": 101}),
    )
    truth["_uncertainty_scenario"]["source_scene"] = str(source.resolve())
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "calibration_profile": profile_path,
            "seed": 101,
            "truth_scene": truth_path,
            "initial_issue_frame": 5,
        },
    )()

    loaded, _ = _validated_profile(args)

    assert loaded is not None


def test_profile_loader_accepts_schema_v2_and_rejects_overlap(tmp_path: Path) -> None:
    scenario = ScenarioConfig(
        kind="random_acceleration",
        event_frame_ratio=9.0 / 39.0,
        affected_obstacle_fraction=1.0,
        acceleration_std=0.0002,
    )
    config = CalibrationConfig(
        seeds=tuple(range(101, 120)),
        issue_frame=8,
        horizon_frames=12,
        target_coverage=0.9,
        base_margin=0.0,
    )
    profile = fit_trajectory_split_conformal(
        make_scene(),
        scenario,
        proper_training_seeds=tuple(range(1, 13)),
        calibration_seeds=config.seeds,
        config=config,
    )
    profile_path = tmp_path / "profile-v2.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    loaded, digest = load_calibration_profile(profile_path)

    assert loaded["schema_version"] == 2
    assert loaded["calibration_method"] == "trajectory_split_conformal"
    assert len(digest) == 64

    profile["proper_training_seeds"] = [101]
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_calibration_profile(profile_path)


def test_runner_rejects_schema_v2_proper_training_seed(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(make_scene()), encoding="utf-8")
    scenario = ScenarioConfig(
        kind="random_acceleration",
        event_frame_ratio=9.0 / 39.0,
        affected_obstacle_fraction=1.0,
        acceleration_std=0.0002,
    )
    config = CalibrationConfig(
        seeds=tuple(range(101, 120)),
        issue_frame=8,
        horizon_frames=12,
        target_coverage=0.9,
        base_margin=0.0,
    )
    profile = fit_trajectory_split_conformal(
        make_scene(),
        scenario,
        proper_training_seeds=tuple(range(1, 13)),
        calibration_seeds=config.seeds,
        config=config,
        source_scene=source,
    )
    profile_path = tmp_path / "profile-v2.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    truth = build_scenario_truth(
        make_scene(), ScenarioConfig(**{**scenario.__dict__, "seed": 1})
    )
    truth["_uncertainty_scenario"]["source_scene"] = str(source.resolve())
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "calibration_profile": profile_path,
            "seed": 1,
            "truth_scene": truth_path,
            "initial_issue_frame": 8,
        },
    )()

    with pytest.raises(ValueError, match="profile fitting seed split"):
        _validated_profile(args)
