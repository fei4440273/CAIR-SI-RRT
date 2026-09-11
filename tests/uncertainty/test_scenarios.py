import json
from pathlib import Path

import pytest

from scripts.uncertainty.generate_trial import PredictionConfig, build_prediction_snapshot
from scripts.uncertainty.scenarios import ScenarioConfig, build_scenario_truth, generate_scenario


def make_scene(frame_count: int = 20) -> dict:
    return {
        "frame_count": frame_count,
        "fps": 10,
        "obstacles": [
            {
                "name": "moving",
                "type": "dynamic_sphere",
                "radius": 0.1,
                "positions": [
                    [0.01 * frame, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
                    for frame in range(frame_count)
                ],
            }
        ],
    }


def make_multi_obstacle_scene(obstacle_count: int = 20, frame_count: int = 100) -> dict:
    scene = make_scene(frame_count)
    template = scene["obstacles"][0]
    scene["obstacles"] = []
    for index in range(obstacle_count):
        obstacle = json.loads(json.dumps(template))
        obstacle["name"] = f"moving-{index}"
        obstacle["positions"] = [
            [position[0], 0.02 * index, *position[2:]]
            for position in obstacle["positions"]
        ]
        scene["obstacles"].append(obstacle)
    return scene


def prediction_config(**overrides) -> PredictionConfig:
    values = {
        "seed": 42,
        "observation_noise_std": 0.0,
        "base_margin": 0.01,
        "growth_per_second": 0.02,
        "fixed_radius": 0.05,
        "max_tube_radius": 0.25,
        "max_prediction_horizon_frames": 10,
        "residual_quantile": 0.95,
    }
    values.update(overrides)
    return PredictionConfig(**values)


@pytest.mark.parametrize(
    "kind",
    [
        "position_noise",
        "random_acceleration",
        "sudden_turn",
        "occlusion",
        "complete_failure",
        "milestone_intrusion",
    ],
)
def test_scenario_generation_is_deterministic(kind: str) -> None:
    config = ScenarioConfig(kind=kind, seed=9, event_frame_ratio=0.4)
    assert build_scenario_truth(make_scene(), config) == build_scenario_truth(make_scene(), config)


def test_random_acceleration_depends_on_seed() -> None:
    first = build_scenario_truth(make_scene(), ScenarioConfig(kind="random_acceleration", seed=1))
    second = build_scenario_truth(make_scene(), ScenarioConfig(kind="random_acceleration", seed=2))

    assert first["obstacles"][0]["positions"] != second["obstacles"][0]["positions"]


def test_scenario_preserves_nominal_obstacle_trajectory() -> None:
    base = make_scene()
    scenario = build_scenario_truth(
        base, ScenarioConfig(kind="random_acceleration", seed=1)
    )

    assert scenario["obstacles"][0]["_uncertainty_nominal_positions"] == base["obstacles"][0]["positions"]


def test_multi_obstacle_events_are_deterministic_partial_and_staggered() -> None:
    base = make_multi_obstacle_scene()
    config = ScenarioConfig(
        kind="sudden_turn",
        seed=11,
        affected_obstacle_fraction=0.35,
        event_frame_jitter_fraction=0.03,
    )

    first = build_scenario_truth(base, config)
    second = build_scenario_truth(base, config)
    events = first["_uncertainty_scenario"]["obstacle_events"]
    affected = [event for event in events.values() if event["affected"]]

    assert first == second
    assert len(affected) == 7
    assert len(affected) < len(base["obstacles"])
    assert len({event["event_frame"] for event in affected}) > 1
    for obstacle in first["obstacles"]:
        if not obstacle["_uncertainty_affected"]:
            assert obstacle["positions"] == obstacle["_uncertainty_nominal_positions"]


def test_sudden_turn_preserves_history_and_changes_future() -> None:
    base = make_scene()
    scenario = build_scenario_truth(
        base,
        ScenarioConfig(kind="sudden_turn", seed=1, event_frame_ratio=0.5, turn_degrees=90.0),
    )
    event_frame = scenario["_uncertainty_scenario"]["event_frame"]

    assert scenario["obstacles"][0]["positions"][: event_frame + 1] == base["obstacles"][0]["positions"][: event_frame + 1]
    assert scenario["obstacles"][0]["positions"][event_frame + 1] != base["obstacles"][0]["positions"][event_frame + 1]


def test_occlusion_uses_last_visible_observation() -> None:
    truth = build_scenario_truth(
        make_scene(),
        ScenarioConfig(kind="occlusion", seed=1, event_frame_ratio=0.4, occlusion_fraction=0.3),
    )
    start, end = truth["_uncertainty_scenario"]["occlusion_ranges"][0]
    issue_frame = min(start + 2, end)
    prediction = build_prediction_snapshot(truth, issue_frame, prediction_config())
    metadata = prediction["_uncertainty_experiment"]

    assert metadata["prediction_status"] == "occluded"
    assert metadata["last_observation_frame"] == start - 1


def test_occlusion_reappears_with_bounded_offset() -> None:
    truth = build_scenario_truth(
        make_multi_obstacle_scene(obstacle_count=8, frame_count=100),
        ScenarioConfig(
            kind="occlusion",
            seed=13,
            affected_obstacle_fraction=0.5,
            event_frame_ratio=0.3,
            occlusion_fraction=0.2,
            occlusion_reappearance_offset_std=0.03,
            occlusion_reappearance_maximum_offset=0.10,
        ),
    )
    affected = next(
        obstacle for obstacle in truth["obstacles"] if obstacle["_uncertainty_affected"]
    )
    start, end = affected["_uncertainty_occlusion_ranges"][0]
    nominal = affected["_uncertainty_nominal_positions"]
    offset = [
        affected["positions"][end + 1][axis] - nominal[end + 1][axis]
        for axis in range(3)
    ]

    assert start <= end < truth["frame_count"] - 1
    assert sum(value * value for value in offset) ** 0.5 > 0.0
    assert sum(value * value for value in offset) ** 0.5 <= 0.10 + 1e-12
    assert affected["positions"][:start] == nominal[:start]


def test_prediction_applies_occlusion_per_obstacle() -> None:
    truth = build_scenario_truth(
        make_multi_obstacle_scene(obstacle_count=4, frame_count=100),
        ScenarioConfig(
            kind="occlusion",
            seed=17,
            affected_obstacle_fraction=0.5,
            event_frame_jitter_fraction=0.03,
        ),
    )
    affected = next(
        obstacle for obstacle in truth["obstacles"] if obstacle["_uncertainty_affected"]
    )
    issue_frame = affected["_uncertainty_occlusion_ranges"][0][0]

    prediction = build_prediction_snapshot(truth, issue_frame, prediction_config())
    by_name = {obstacle["name"]: obstacle for obstacle in prediction["obstacles"]}

    assert by_name[affected["name"]]["prediction_last_observation_frame"] < issue_frame
    for obstacle in truth["obstacles"]:
        if not obstacle["_uncertainty_affected"]:
            assert by_name[obstacle["name"]]["prediction_last_observation_frame"] == issue_frame


def test_position_noise_only_affects_selected_obstacles() -> None:
    truth = build_scenario_truth(
        make_multi_obstacle_scene(obstacle_count=4, frame_count=40),
        ScenarioConfig(
            kind="position_noise",
            seed=19,
            affected_obstacle_fraction=0.5,
        ),
    )
    prediction = build_prediction_snapshot(
        truth,
        issue_frame=10,
        config=prediction_config(observation_noise_std=0.02),
    )

    for truth_obstacle, predicted_obstacle in zip(
        truth["obstacles"], prediction["obstacles"], strict=True
    ):
        if truth_obstacle["_uncertainty_affected"]:
            assert predicted_obstacle["positions"][10][:3] != truth_obstacle["positions"][10][:3]
        else:
            assert predicted_obstacle["positions"] == truth_obstacle["positions"]


def test_complete_failure_disables_future_planning_after_event() -> None:
    truth = build_scenario_truth(
        make_scene(), ScenarioConfig(kind="complete_failure", seed=1, event_frame_ratio=0.4)
    )
    failure_frame = truth["_uncertainty_scenario"]["failure_frame"]
    prediction = build_prediction_snapshot(truth, failure_frame, prediction_config())
    metadata = prediction["_uncertainty_experiment"]

    assert metadata["prediction_status"] == "unavailable"
    assert metadata["reliable_until_frame"] == failure_frame


def test_milestone_intrusion_is_hidden_until_event_and_reaches_target() -> None:
    truth = build_scenario_truth(
        make_scene(),
        ScenarioConfig(
            kind="milestone_intrusion",
            seed=7,
            event_frame_ratio=0.25,
            intrusion_approach_frames=4,
            intrusion_hold_frames=3,
        ),
    )
    metadata = truth["_uncertainty_scenario"]
    event = metadata["event_frame"]
    intrusion = metadata["intrusion"]
    obstacle = truth["obstacles"][0]
    nominal = obstacle["_uncertainty_nominal_positions"]

    assert obstacle["positions"][: event + 1] == nominal[: event + 1]
    assert obstacle["positions"][event + 1][:3] != nominal[event + 1][:3]
    assert obstacle["positions"][intrusion["collision_frame"]][:3] == pytest.approx(
        intrusion["target"]
    )
    assert intrusion["offset_norm"] == pytest.approx(
        sum(value * value for value in intrusion["offset"]) ** 0.5
    )


def test_adaptive_components_sum_to_radius() -> None:
    prediction = build_prediction_snapshot(
        make_scene(), 4, prediction_config(observation_noise_std=0.01)
    )
    obstacle = prediction["obstacles"][0]
    components = obstacle["uncertainty_radius_components"]

    for frame, radius in enumerate(obstacle["uncertainty_radii"]):
        assert radius == pytest.approx(sum(values[frame] for values in components.values()))


def test_generate_scenario_writes_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "scenario.json"
    source.write_text(json.dumps(make_scene()), encoding="utf-8")

    generate_scenario(source, output, ScenarioConfig(kind="random_acceleration", seed=8))

    generated = json.loads(output.read_text(encoding="utf-8"))
    assert generated["_uncertainty_scenario"]["kind"] == "random_acceleration"
    assert generated["_uncertainty_scenario"]["source_scene"] == str(source.resolve())
