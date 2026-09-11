import copy
import json
from pathlib import Path

import pytest

from scripts.uncertainty.generate_trial import (
    PredictionConfig,
    _historical_horizon_residual_bounds,
    build_prediction_snapshot,
    generate_trial,
)


def make_scene(frame_count: int = 8, fps: int = 2) -> dict:
    positions = [
        [float(frame), 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        for frame in range(frame_count)
    ]
    return {
        "frame_count": frame_count,
        "fps": fps,
        "obstacles": [
            {
                "name": "moving",
                "type": "dynamic_sphere",
                "radius": 0.1,
                "positions": positions,
            },
            {
                "name": "table",
                "type": "static_box",
                "dimensions": [1.0, 1.0, 0.1],
                "positions": [[0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]],
            },
        ],
    }


def config(**overrides) -> PredictionConfig:
    values = {
        "seed": 7,
        "observation_noise_std": 0.0,
        "base_margin": 0.01,
        "growth_per_second": 0.02,
        "fixed_radius": 0.05,
        "max_tube_radius": 0.08,
        "max_prediction_horizon_frames": 20,
        "residual_quantile": 0.95,
    }
    values.update(overrides)
    return PredictionConfig(**values)


def test_snapshot_does_not_use_future_truth() -> None:
    original = make_scene()
    changed_future = copy.deepcopy(original)
    for frame in range(4, changed_future["frame_count"]):
        changed_future["obstacles"][0]["positions"][frame][0] += 100.0

    first = build_prediction_snapshot(original, issue_frame=3, config=config())
    second = build_prediction_snapshot(changed_future, issue_frame=3, config=config())

    assert first["obstacles"][0]["positions"] == second["obstacles"][0]["positions"]
    assert first["obstacles"][0]["uncertainty_radii"] == second["obstacles"][0]["uncertainty_radii"]


def test_occluded_current_frame_does_not_leak_truth() -> None:
    original = make_scene()
    original["_uncertainty_scenario"] = {"occlusion_ranges": [[3, 4]]}
    changed_hidden = copy.deepcopy(original)
    changed_hidden["obstacles"][0]["positions"][3][0] += 100.0

    first = build_prediction_snapshot(original, issue_frame=3, config=config())
    second = build_prediction_snapshot(changed_hidden, issue_frame=3, config=config())

    assert first["obstacles"][0]["positions"] == second["obstacles"][0]["positions"]


def test_nominal_prior_predicts_nominal_motion_without_reading_future_truth() -> None:
    original = make_scene()
    nominal = copy.deepcopy(original["obstacles"][0]["positions"])
    original["obstacles"][0]["_uncertainty_nominal_positions"] = nominal
    changed_future = copy.deepcopy(original)
    for frame in range(4, changed_future["frame_count"]):
        changed_future["obstacles"][0]["positions"][frame][0] += 100.0

    first = build_prediction_snapshot(original, issue_frame=3, config=config())
    second = build_prediction_snapshot(changed_future, issue_frame=3, config=config())

    assert first["obstacles"][0]["positions"] == second["obstacles"][0]["positions"]
    assert first["obstacles"][0]["positions"] == nominal
    assert first["_uncertainty_experiment"]["prediction_model"].startswith(
        "nominal_trajectory"
    )


def test_adaptive_radius_and_reliable_horizon() -> None:
    snapshot = build_prediction_snapshot(make_scene(), issue_frame=2, config=config())
    obstacle = snapshot["obstacles"][0]
    radii = obstacle["uncertainty_radii"]

    assert radii[:3] == [pytest.approx(0.01)] * 3
    assert radii[2:] == sorted(radii[2:])
    metadata = snapshot["_uncertainty_experiment"]
    assert metadata["issue_frame"] == 2
    assert metadata["reliable_until_frame"] == 7


def test_seed_is_deterministic_and_changes_noise() -> None:
    noisy = config(observation_noise_std=0.1)
    first = build_prediction_snapshot(make_scene(), issue_frame=3, config=noisy)
    second = build_prediction_snapshot(make_scene(), issue_frame=3, config=noisy)
    other_seed = build_prediction_snapshot(
        make_scene(), issue_frame=3, config=config(seed=8, observation_noise_std=0.1)
    )

    assert first == second
    assert first["obstacles"][0]["positions"] != other_seed["obstacles"][0]["positions"]


def test_position_noise_auto_model_does_not_extrapolate_residual_velocity() -> None:
    truth = make_scene(frame_count=80, fps=10)
    nominal = copy.deepcopy(truth["obstacles"][0]["positions"])
    truth["obstacles"][0]["_uncertainty_nominal_positions"] = nominal
    truth["_uncertainty_scenario"] = {"kind": "position_noise"}

    snapshot = build_prediction_snapshot(
        truth,
        issue_frame=20,
        config=config(
            seed=7,
            observation_noise_std=0.01,
            max_prediction_horizon_frames=59,
            residual_motion_model="auto",
        ),
    )

    assert snapshot["_uncertainty_experiment"]["residual_motion_model"] == "constant_offset"
    far_error = [
        snapshot["obstacles"][0]["positions"][79][axis] - nominal[79][axis]
        for axis in range(3)
    ]
    assert sum(value * value for value in far_error) ** 0.5 < 0.02


def test_dynamic_deviation_auto_model_keeps_residual_velocity() -> None:
    truth = make_scene(frame_count=40, fps=10)
    truth["obstacles"][0]["_uncertainty_nominal_positions"] = copy.deepcopy(
        truth["obstacles"][0]["positions"]
    )
    truth["_uncertainty_scenario"] = {"kind": "random_acceleration"}

    snapshot = build_prediction_snapshot(
        truth,
        issue_frame=20,
        config=config(max_prediction_horizon_frames=19, residual_motion_model="auto"),
    )

    assert snapshot["_uncertainty_experiment"]["residual_motion_model"] == "constant_velocity"


def test_invalid_residual_motion_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="residual_motion_model"):
        config(residual_motion_model="quadratic").validate()


def test_fixed_and_center_tubes_share_prediction_centers() -> None:
    adaptive = build_prediction_snapshot(make_scene(), issue_frame=3, config=config(), tube_mode="adaptive")
    fixed = build_prediction_snapshot(make_scene(), issue_frame=3, config=config(), tube_mode="fixed")
    center = build_prediction_snapshot(make_scene(), issue_frame=3, config=config(), tube_mode="center")

    assert adaptive["obstacles"][0]["positions"] == fixed["obstacles"][0]["positions"]
    assert fixed["obstacles"][0]["positions"] == center["obstacles"][0]["positions"]
    assert fixed["obstacles"][0]["uncertainty_radii"] == [pytest.approx(0.05)] * 8
    assert center["obstacles"][0]["uncertainty_radii"] == [0.0] * 8


def test_oracle_uses_truth_and_full_horizon() -> None:
    truth = make_scene()
    oracle = build_prediction_snapshot(truth, issue_frame=2, config=config(), tube_mode="oracle")

    assert oracle["obstacles"][0]["positions"] == truth["obstacles"][0]["positions"]
    assert oracle["obstacles"][0]["uncertainty_radii"] == [0.0] * 8
    assert oracle["_uncertainty_experiment"]["reliable_until_frame"] == 7
    assert oracle["_uncertainty_experiment"]["prediction_model"] == "oracle_truth"


def test_invalid_issue_frame_is_rejected() -> None:
    with pytest.raises(ValueError, match="issue_frame"):
        build_prediction_snapshot(make_scene(), issue_frame=8, config=config())


def test_horizon_residual_bounds_backtest_each_lead_without_decreasing() -> None:
    observed = [
        (frame, [position, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        for frame, position in enumerate([0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0])
    ]

    bounds = _historical_horizon_residual_bounds(observed, max_lead=4, quantile=0.95)

    assert bounds[0] == 0.0
    assert bounds[1] > 0.0
    assert bounds == sorted(bounds)


def test_offline_residual_bounds_must_be_nonnegative() -> None:
    config(offline_residual_bounds=(0.0, 0.2, 0.1)).validate()
    with pytest.raises(ValueError, match="nonnegative"):
        config(offline_residual_bounds=(0.0, -0.1)).validate()


def test_trial_can_start_after_observation_warmup(tmp_path: Path) -> None:
    source = tmp_path / "truth.json"
    output = tmp_path / "trial"
    source.write_text(json.dumps(make_scene()), encoding="utf-8")

    generate_trial(source, output, config(), 2, "adaptive", initial_issue_frame=3)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["initial_issue_frame"] == 3
    assert manifest["prediction_files"] == [
        "predictions/update_000003.json",
        "predictions/update_000005.json",
        "predictions/update_000007.json",
    ]
    first = json.loads((output / manifest["prediction_files"][0]).read_text(encoding="utf-8"))
    assert first["_uncertainty_experiment"]["issue_frame"] == 3


def test_trial_only_materializes_requested_updates(tmp_path: Path) -> None:
    source = tmp_path / "truth.json"
    output = tmp_path / "trial"
    source.write_text(json.dumps(make_scene()), encoding="utf-8")

    generate_trial(
        source,
        output,
        config(),
        1,
        "center",
        initial_issue_frame=1,
        max_updates=2,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["max_updates"] == 2
    assert manifest["prediction_files"] == [
        "predictions/update_000001.json",
        "predictions/update_000002.json",
    ]
    assert len(list((output / "predictions").glob("*.json"))) == 2
