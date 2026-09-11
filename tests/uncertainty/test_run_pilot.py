import argparse
import json
import os
from pathlib import Path

from scripts.uncertainty.run_pilot import (
    _attach_calibration_provenance,
    _resolved_planner_environment,
    _selected_methods,
    _write_initial_prediction_rejection,
)


def test_default_method_scope_only_runs_adaptive_repair() -> None:
    assert _selected_methods(argparse.Namespace()) == ["adaptive_tube_repair"]


def test_explicit_method_scope_is_deduplicated_in_requested_order() -> None:
    args = argparse.Namespace(
        methods=[
            "adaptive_tube_full_replan",
            "adaptive_tube_repair",
            "adaptive_tube_full_replan",
        ]
    )
    assert _selected_methods(args) == [
        "adaptive_tube_full_replan",
        "adaptive_tube_repair",
    ]


def test_planner_environment_overrides_are_resolved_without_global_mutation(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MSIRRT_ADAPTIVE_REPAIR_POLICY", "legacy")
    args = argparse.Namespace(
        planner_environment={
            "MSIRRT_ADAPTIVE_REPAIR_POLICY": "optimized",
            "MSIRRT_FEASIBILITY_AWARE_HORIZON": "1",
            "MSIRRT_HORIZON_FEASIBILITY_SLACK_FRAMES": "3",
            "MSIRRT_SKIPPED_STAGE_BUDGET_REUSE_FRACTION": "0.5",
            "MSIRRT_REUSE_STORED_PREVIOUS_INTERVALS": "1",
            "MSIRRT_REUSE_FIRST_PREDICTION_SCENE": "1",
        }
    )

    resolved = _resolved_planner_environment(args)

    assert resolved["MSIRRT_ADAPTIVE_REPAIR_POLICY"] == "optimized"
    assert resolved["MSIRRT_FEASIBILITY_AWARE_HORIZON"] == "1"
    assert resolved["MSIRRT_HORIZON_FEASIBILITY_SLACK_FRAMES"] == "3"
    assert resolved["MSIRRT_SKIPPED_STAGE_BUDGET_REUSE_FRACTION"] == "0.5"
    assert resolved["MSIRRT_REUSE_STORED_PREVIOUS_INTERVALS"] == "1"
    assert resolved["MSIRRT_REUSE_FIRST_PREDICTION_SCENE"] == "1"
    assert os.environ["MSIRRT_ADAPTIVE_REPAIR_POLICY"] == "legacy"


def test_initial_prediction_rejection_is_a_structured_safe_stop(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.json"
    prediction.write_text(
        json.dumps(
            {
                "_uncertainty_experiment": {
                    "issue_frame": 15,
                    "reliable_until_frame": 105,
                    "prediction_status": "available",
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"prediction_files": [prediction.name]}), encoding="utf-8"
    )
    result = tmp_path / "result.json"

    _write_initial_prediction_rejection(
        manifest,
        result,
        seed=101,
        mode="rolling_repair",
        deadline_ms=1000.0,
        braking_frames=2,
        elapsed_ms=3.5,
    )

    value = json.loads(result.read_text(encoding="utf-8"))
    assert value["emergency_stop"]
    assert not value["executed_truth_collision"]
    assert value["termination_reason"] == "initial_configuration_unsafe_in_prediction"
    update = value["updates"][0]
    assert update["control_state"] == "safe_stop_initial_configuration_unsafe"
    assert update["issue_frame"] == 15
    assert update["planning_horizon_frame"] == 105
    assert update["repair_invariants_hold"]


def test_calibration_provenance_is_written_into_each_result(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"seed": 1401}), encoding="utf-8")

    _attach_calibration_provenance(
        result,
        profile={
            "schema_version": 2,
            "calibration_method": "trajectory_split_conformal",
        },
        profile_sha256="a" * 64,
    )

    value = json.loads(result.read_text(encoding="utf-8"))
    assert value["calibration_profile_sha256"] == "a" * 64
    assert value["calibration_profile_schema_version"] == 2
    assert value["calibration_method"] == "trajectory_split_conformal"
