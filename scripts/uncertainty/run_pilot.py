#!/usr/bin/env python3
"""Run paired uncertainty-aware SI-RRT pilot experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from scripts.uncertainty.calibrate_tubes import load_calibration_profile
from scripts.uncertainty.generate_trial import PredictionConfig, generate_trial
from scripts.uncertainty.scenarios import ScenarioConfig


METHODS = {
    "oracle": ("oracle", "repair"),
    "center": ("center", "repair"),
    "fixed_tube": ("fixed", "repair"),
    "adaptive_tube_full_replan": ("adaptive", "full_replan"),
    "adaptive_tube_repair": ("adaptive", "repair"),
}
DEFAULT_METHODS = ("adaptive_tube_repair",)
PLANNER_ENVIRONMENT_KEYS = (
    "MSIRRT_MAX_PLANNING_TIME",
    "MSIRRT_CONDITIONAL_SAMPLING",
    "MSIRRT_CONDITIONAL_EXPLORATION_RATE",
    "MSIRRT_INITIAL_HORIZON_FRAMES",
    "MSIRRT_HORIZON_STEP_FRAMES",
    "MSIRRT_HORIZON_STAGE_BUDGET_MS",
    "MSIRRT_SKIPPED_STAGE_BUDGET_REUSE_FRACTION",
    "MSIRRT_ADAPTIVE_REPAIR_POLICY",
    "MSIRRT_FEASIBILITY_AWARE_HORIZON",
    "MSIRRT_HORIZON_FEASIBILITY_SLACK_FRAMES",
    "MSIRRT_REUSE_STORED_PREVIOUS_INTERVALS",
    "MSIRRT_REUSE_FIRST_PREDICTION_SCENE",
    "MSIRRT_REUSE_PARSED_PREDICTION_METADATA",
    "MSIRRT_COOPERATIVE_DEADLINE_CHECKS",
    "MSIRRT_REACHABLE_ELLIPSOID_SAMPLING",
    "MSIRRT_FALLBACK_ENABLED",
    "MSIRRT_TUBE_ABLATION_MODE",
    "MSIRRT_UPDATE_ZERO_FINAL_GUARD_MS",
    "MSIRRT_REJECTION_FEEDBACK",
    "MSIRRT_REJECTION_WINDOW",
    "MSIRRT_REJECTION_THRESHOLD",
    "MSIRRT_REJECTION_EXPLORATION_BOOST",
)

INITIAL_PREDICTION_REJECTION = (
    "start configuration is not safe at prediction issue frame"
)
TEMPORAL_GATE_CONFIG_FIELDS = {
    "temporal_gate_target",
    "temporal_gate_minimum_offset",
    "temporal_gate_maximum_offset",
    "temporal_gate_approach_frames",
    "temporal_gate_hold_frames",
    "temporal_gate_departure_frames",
    "temporal_gate_sphere_radius",
}
SCENARIO_HARDENING_DEFAULTS = {
    name: asdict(ScenarioConfig(kind="position_noise"))[name]
    for name in (
        "affected_obstacle_fraction",
        "event_frame_jitter_fraction",
        "occlusion_reappearance_offset_std",
        "occlusion_reappearance_maximum_offset",
    )
}


def _selected_methods(args: argparse.Namespace) -> list[str]:
    requested = getattr(args, "methods", None) or DEFAULT_METHODS
    selected: list[str] = []
    for method in requested:
        if method not in METHODS:
            raise ValueError(f"unsupported uncertainty method: {method}")
        if method not in selected:
            selected.append(method)
    return selected


def _resolved_planner_environment(args: argparse.Namespace) -> dict[str, str | None]:
    resolved = {key: os.environ.get(key) for key in PLANNER_ENVIRONMENT_KEYS}
    overrides = getattr(args, "planner_environment", None) or {}
    unknown = set(overrides) - set(PLANNER_ENVIRONMENT_KEYS)
    if unknown:
        raise ValueError(f"unsupported planner environment keys: {sorted(unknown)}")
    resolved.update(
        {key: None if value is None else str(value) for key, value in overrides.items()}
    )
    return resolved


def git_value(*arguments: str) -> str:
    try:
        process = subprocess.run(
            ["git", *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return process.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if arguments == ("rev-parse", "HEAD"):
            return os.environ.get("SIRRT_SOURCE_COMMIT", "unavailable-in-container")
        if arguments == ("status", "--porcelain"):
            return "unavailable-in-container"
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_profile(args: argparse.Namespace) -> tuple[dict | None, str | None]:
    profile_path = getattr(args, "calibration_profile", None)
    if profile_path is None:
        return None, None
    profile, profile_sha256 = load_calibration_profile(profile_path)
    calibration_seeds = {int(seed) for seed in profile["calibration_seeds"]}
    proper_training_seeds = {
        int(seed) for seed in profile.get("proper_training_seeds", [])
    }
    if args.seed in proper_training_seeds:
        raise ValueError(
            f"test seed {args.seed} is present in the profile fitting seed split"
        )
    if args.seed in calibration_seeds:
        raise ValueError(
            f"test seed {args.seed} is present in the calibration seed split"
        )
    truth = json.loads(args.truth_scene.resolve().read_text(encoding="utf-8"))
    scenario = truth.get("_uncertainty_scenario", {})
    if scenario.get("kind") != profile["scenario_kind"]:
        raise ValueError("truth scenario kind does not match calibration profile")
    if getattr(args, "initial_issue_frame", 0) != int(profile["issue_frame"]):
        raise ValueError("initial issue frame does not match calibration profile")
    expected_config = profile.get("scenario_config_without_seed")
    actual_config = dict(scenario.get("config", {}))
    actual_config.pop("seed", None)
    if expected_config is not None:
        mismatched = any(
            actual_config.get(key) != value for key, value in expected_config.items()
        )
        extra = set(actual_config) - set(expected_config)
        allowed_extra = (
            TEMPORAL_GATE_CONFIG_FIELDS | set(SCENARIO_HARDENING_DEFAULTS)
            if scenario.get("kind") != "temporal_gate"
            else set(SCENARIO_HARDENING_DEFAULTS)
        )
        hardened_extra_mismatch = any(
            actual_config.get(key) != SCENARIO_HARDENING_DEFAULTS[key]
            for key in extra & set(SCENARIO_HARDENING_DEFAULTS)
        )
        if mismatched or extra - allowed_extra or hardened_extra_mismatch:
            raise ValueError(
                "truth scenario parameters do not match calibration profile"
            )
    source_scene = scenario.get("source_scene")
    expected_source_hash = profile.get("source_scene_sha256")
    if source_scene and expected_source_hash:
        source_path = Path(source_scene)
        if not source_path.exists() or _sha256(source_path) != expected_source_hash:
            raise ValueError("truth source scene does not match calibration profile")
    return profile, profile_sha256


def _attach_calibration_provenance(
    result_path: Path,
    *,
    profile: dict | None,
    profile_sha256: str | None,
) -> None:
    if profile is None:
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["calibration_profile_sha256"] = profile_sha256
    result["calibration_profile_schema_version"] = int(profile["schema_version"])
    result["calibration_method"] = profile.get("calibration_method") or profile.get(
        "method"
    )
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_initial_prediction_rejection(
    manifest_path: Path,
    result_path: Path,
    *,
    seed: int,
    mode: str,
    deadline_ms: float,
    braking_frames: int,
    elapsed_ms: float,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = manifest_path.parent / manifest["prediction_files"][0]
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    experiment = prediction.get("_uncertainty_experiment", {})
    issue_frame = int(
        experiment.get("issue_frame", manifest.get("initial_issue_frame", 0))
    )
    horizon = int(experiment.get("reliable_until_frame", issue_frame))
    result = {
        "schema_version": 1,
        "seed": seed,
        "mode": mode,
        "rolling_execution": mode.startswith("rolling_"),
        "deadline_ms": deadline_ms,
        "braking_frames": braking_frames,
        "updates": [
            {
                "update_index": 0,
                "issue_frame": issue_frame,
                "planning_start_frame": issue_frame,
                "planning_horizon_frame": horizon,
                "maximum_reliable_horizon_frame": horizon,
                "initial_planning_horizon_frame": horizon,
                "horizon_expansions": 0,
                "solved": False,
                "accepted_plan": False,
                "solve_seconds": 0.0,
                "online_latency_ms": elapsed_ms,
                "deadline_missed": elapsed_ms > deadline_ms,
                "control_state": "safe_stop_initial_configuration_unsafe",
                "prediction_status": experiment.get("prediction_status", "available"),
                "truth_collision": False,
                "truth_checked_frames": 0,
                "truth_min_distance": None,
                "repair_invariants_hold": True,
                "emergency_stop": True,
                "braking_truth_collision": False,
                "braking_truth_checked_frames": 0,
                "braking_truth_min_distance": None,
                "braking_trajectory": [],
                "final_path": [],
            }
        ],
        "task_completed": False,
        "emergency_stop": True,
        "braking_collision": False,
        "executed_truth_collision": False,
        "executed_truth_checked_frames": 0,
        "executed_truth_min_distance": None,
        "completion_frame": None,
        "termination_reason": "initial_configuration_unsafe_in_prediction",
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _run_planner(
    command: list[str],
    *,
    manifest_path: Path,
    result_path: Path,
    seed: int,
    mode: str,
    deadline_ms: float,
    braking_frames: int,
    environment: dict[str, str | None] | None = None,
) -> None:
    start = time.perf_counter()
    process_environment = os.environ.copy()
    for key, value in (environment or {}).items():
        if value is None:
            process_environment.pop(key, None)
        else:
            process_environment[key] = value
    process = subprocess.run(
        command,
        cwd=Path.cwd(),
        env=process_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    if process.stdout:
        print(process.stdout, end="")
    if process.returncode == 0:
        if process.stderr:
            print(process.stderr, end="", file=sys.stderr)
        return
    if process.stderr.strip() == INITIAL_PREDICTION_REJECTION:
        _write_initial_prediction_rejection(
            manifest_path,
            result_path,
            seed=seed,
            mode=mode,
            deadline_ms=deadline_ms,
            braking_frames=braking_frames,
            elapsed_ms=elapsed_ms,
        )
        return
    raise subprocess.CalledProcessError(
        process.returncode,
        command,
        output=process.stdout,
        stderr=process.stderr,
    )


def _build_prediction_config(
    args: argparse.Namespace,
    calibration_profile: dict | None,
) -> PredictionConfig:
    runtime = (
        calibration_profile["runtime_parameters"]
        if calibration_profile is not None
        else {}
    )
    return PredictionConfig(
        seed=args.seed,
        observation_noise_std=runtime.get(
            "observation_noise_std", args.observation_noise_std
        ),
        base_margin=runtime.get("base_margin", args.base_margin),
        growth_per_second=runtime.get("growth_per_second", args.growth_per_second),
        fixed_radius=(
            float(calibration_profile["fixed_radius"])
            if calibration_profile is not None
            else args.fixed_radius
        ),
        max_tube_radius=args.max_tube_radius,
        max_prediction_horizon_frames=(
            int(calibration_profile["horizon_frames"])
            if calibration_profile is not None
            else args.max_prediction_horizon_frames
        ),
        residual_quantile=(
            float(calibration_profile["target_coverage"])
            if calibration_profile is not None
            else args.residual_quantile
        ),
        observation_confidence_multiplier=runtime.get(
            "observation_confidence_multiplier",
            args.observation_confidence_multiplier,
        ),
        control_error_margin=runtime.get(
            "control_error_margin", args.control_error_margin
        ),
        velocity_fit_window_frames=runtime.get(
            "velocity_fit_window_frames",
            getattr(args, "velocity_fit_window_frames", 15),
        ),
        velocity_fit_min_samples=runtime.get(
            "velocity_fit_min_samples",
            getattr(args, "velocity_fit_min_samples", 5),
        ),
        residual_motion_model=runtime.get(
            "residual_motion_model",
            getattr(args, "residual_motion_model", "auto"),
        ),
        scenario_label_independent=runtime.get(
            "scenario_label_independent",
            getattr(args, "scenario_label_independent", False),
        ),
        velocity_model_improvement_ratio=runtime.get(
            "velocity_model_improvement_ratio",
            getattr(args, "velocity_model_improvement_ratio", 0.8),
        ),
        offline_residual_bounds=(
            tuple(
                float(value)
                for value in calibration_profile["adaptive_residual_bounds"]
            )
            if calibration_profile is not None
            else ()
        ),
        historical_residual_bounds_enabled=runtime.get(
            "historical_residual_bounds_enabled",
            getattr(args, "historical_residual_bounds_enabled", True),
        ),
        mode_change_threshold=runtime.get(
            "mode_change_threshold", getattr(args, "mode_change_threshold", None)
        ),
        mode_change_short_horizon_frames=runtime.get(
            "mode_change_short_horizon_frames",
            getattr(args, "mode_change_short_horizon_frames", 15),
        ),
        mode_change_recovery_frames=runtime.get(
            "mode_change_recovery_frames",
            getattr(args, "mode_change_recovery_frames", 30),
        ),
    )


def run_pilot(args: argparse.Namespace) -> Path:
    selected_methods = _selected_methods(args)
    planner_environment = _resolved_planner_environment(args)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    output_root.mkdir(parents=True)
    trials_root = output_root / "trials"
    results_root = output_root / "results"
    trials_root.mkdir()
    results_root.mkdir()

    calibration_profile, calibration_profile_sha256 = _validated_profile(args)
    prediction_config = _build_prediction_config(args, calibration_profile)

    manifests: dict[str, Path] = {}
    for tube_mode in sorted({METHODS[method][0] for method in selected_methods}):
        trial_dir = trials_root / tube_mode
        generate_trial(
            args.truth_scene.resolve(),
            trial_dir,
            prediction_config,
            args.update_stride,
            tube_mode,
            getattr(args, "initial_issue_frame", 0),
            args.max_updates,
        )
        manifests[tube_mode] = trial_dir / "manifest.json"

    result_files: dict[str, str] = {}
    for method in selected_methods:
        tube_mode, planner_mode = METHODS[method]
        method_dir = results_root / method
        method_dir.mkdir()
        result_path = method_dir / f"seed_{args.seed}.json"
        effective_planner_mode = (
            f"rolling_{planner_mode}" if args.rolling else planner_mode
        )
        command = [
            str(args.planner_binary.resolve()),
            str(manifests[tube_mode]),
            str(result_path),
            str(args.seed),
            str(args.max_updates),
            effective_planner_mode,
            str(args.deadline_ms),
            str(args.braking_frames),
            str(args.reactive_stop_distance),
        ]
        _run_planner(
            command,
            manifest_path=manifests[tube_mode],
            result_path=result_path,
            seed=args.seed,
            mode=effective_planner_mode,
            deadline_ms=args.deadline_ms,
            braking_frames=args.braking_frames,
            environment=planner_environment,
        )
        _attach_calibration_provenance(
            result_path,
            profile=calibration_profile,
            profile_sha256=calibration_profile_sha256,
        )
        result_files[method] = str(result_path.relative_to(output_root))

    run_manifest = {
        "schema_version": 1,
        "source_commit": git_value("rev-parse", "HEAD"),
        "source_dirty": bool(git_value("status", "--porcelain")),
        "truth_scene": str(args.truth_scene.resolve()),
        "prediction_config": asdict(prediction_config),
        "calibration_profile": (
            str(Path(args.calibration_profile).resolve())
            if calibration_profile is not None
            else None
        ),
        "calibration_profile_sha256": calibration_profile_sha256,
        "calibration_profile_schema_version": (
            calibration_profile["schema_version"]
            if calibration_profile is not None
            else None
        ),
        "calibration_method": (
            calibration_profile.get("calibration_method")
            or calibration_profile.get("method")
            if calibration_profile is not None
            else None
        ),
        "proper_training_seeds": (
            calibration_profile.get("proper_training_seeds", [])
            if calibration_profile is not None
            else []
        ),
        "calibration_seeds": (
            calibration_profile["calibration_seeds"]
            if calibration_profile is not None
            else []
        ),
        "test_seed": args.seed,
        "update_stride": args.update_stride,
        "initial_issue_frame": getattr(args, "initial_issue_frame", 0),
        "max_updates": args.max_updates,
        "rolling_execution": args.rolling,
        "deadline_ms": args.deadline_ms,
        "braking_frames": args.braking_frames,
        "reactive_stop_distance": args.reactive_stop_distance,
        "planner_binary": str(args.planner_binary.resolve()),
        "planner_environment": planner_environment,
        "results": result_files,
        "methods": selected_methods,
    }
    manifest_path = output_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth_scene", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--planner-binary",
        type=Path,
        default=Path("reproduction/build/msirrt/MSIRRT_RepairSequence"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-profile", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(METHODS),
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--update-stride", type=int, default=150)
    parser.add_argument("--initial-issue-frame", type=int, default=15)
    parser.add_argument("--max-updates", type=int, default=2)
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--deadline-ms", type=float, default=100.0)
    parser.add_argument("--braking-frames", type=int, default=2)
    parser.add_argument(
        "--reactive-stop-distance",
        type=float,
        default=-1.0,
        help="optional clearance-stop diagnostic; -1 disables it",
    )
    parser.add_argument("--observation-noise-std", type=float, default=0.0)
    parser.add_argument("--base-margin", type=float, default=0.01)
    parser.add_argument("--growth-per-second", type=float, default=0.02)
    parser.add_argument("--fixed-radius", type=float, default=0.05)
    parser.add_argument("--max-tube-radius", type=float, default=0.25)
    parser.add_argument("--max-prediction-horizon-frames", type=int, default=150)
    parser.add_argument("--residual-quantile", type=float, default=0.95)
    parser.add_argument("--observation-confidence-multiplier", type=float, default=3.0)
    parser.add_argument("--control-error-margin", type=float, default=0.0)
    parser.add_argument("--velocity-fit-window-frames", type=int, default=15)
    parser.add_argument("--velocity-fit-min-samples", type=int, default=5)
    parser.add_argument(
        "--residual-motion-model",
        choices=(
            "auto",
            "constant_offset",
            "constant_velocity",
            "history_selected",
        ),
        default="auto",
    )
    parser.add_argument("--scenario-label-independent", action="store_true")
    parser.add_argument("--velocity-model-improvement-ratio", type=float, default=0.8)
    parser.add_argument(
        "--disable-historical-residual-bounds",
        action="store_false",
        dest="historical_residual_bounds_enabled",
    )
    parser.add_argument("--mode-change-threshold", type=float)
    parser.add_argument("--mode-change-short-horizon-frames", type=int, default=15)
    parser.add_argument("--mode-change-recovery-frames", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    manifest = run_pilot(parse_args())
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
