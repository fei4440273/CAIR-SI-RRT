#!/usr/bin/env python3
"""Calibrate fixed and lead-wise uncertainty tubes on independent seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from scripts.uncertainty.generate_trial import (
    PredictionConfig,
    _resolve_residual_motion_model,
    build_prediction_snapshot,
)
from scripts.uncertainty.scenarios import (
    SCENARIO_KINDS,
    ScenarioConfig,
    build_scenario_truth,
)


@dataclass(frozen=True)
class CalibrationConfig:
    seeds: tuple[int, ...]
    issue_frame: int = 15
    horizon_frames: int = 15
    target_coverage: float = 0.95
    observation_noise_std: float = 0.0
    base_margin: float = 0.01
    growth_per_second: float = 0.0
    observation_confidence_multiplier: float = 3.0
    control_error_margin: float = 0.0
    velocity_fit_window_frames: int = 15
    velocity_fit_min_samples: int = 5
    residual_motion_model: str = "auto"

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("calibration seeds must be nonempty and unique")
        if self.issue_frame < 0 or self.horizon_frames < 1:
            raise ValueError("issue_frame must be nonnegative and horizon_frames positive")
        if not 0.0 < self.target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0, 1]")
        prediction_config = PredictionConfig(
            observation_noise_std=self.observation_noise_std,
            base_margin=self.base_margin,
            growth_per_second=self.growth_per_second,
            observation_confidence_multiplier=self.observation_confidence_multiplier,
            control_error_margin=self.control_error_margin,
            velocity_fit_window_frames=self.velocity_fit_window_frames,
            velocity_fit_min_samples=self.velocity_fit_min_samples,
            residual_motion_model=self.residual_motion_model,
        )
        prediction_config.validate()


def parse_seed_list(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            seed = int(item)
            if seed not in result:
                result.append(seed)
    if not result:
        raise ValueError("at least one calibration seed is required")
    return tuple(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((left[axis] - right[axis]) ** 2 for axis in range(3)))


def _upper_quantile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _dynamic_obstacles(scene: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for obstacle in scene.get("obstacles", []):
        if obstacle.get("type") in {"dynamic_sphere", "dynamic_box"}:
            name = str(obstacle.get("name", ""))
            if not name or name in result:
                raise ValueError("dynamic obstacles need unique nonempty names for calibration")
            result[name] = obstacle
    if not result:
        raise ValueError("calibration requires at least one dynamic obstacle")
    return result


def _scenario_signature(config: ScenarioConfig) -> dict:
    signature = asdict(config)
    signature.pop("seed")
    return signature


def _coverage(errors: Sequence[tuple[int, float]], radii: Sequence[float]) -> float:
    if not errors:
        return 0.0
    covered = sum(error <= radii[min(lead, len(radii) - 1)] + 1e-12 for lead, error in errors)
    return covered / len(errors)


def build_calibration_profile(
    base_scene: dict,
    scenario_template: ScenarioConfig,
    config: CalibrationConfig,
    *,
    source_scene: Path | None = None,
) -> dict:
    """Build a seed-split calibration artifact without using any test trial."""

    config.validate()
    frame_count = int(base_scene.get("frame_count", 0))
    fps = int(base_scene.get("fps", 0))
    if fps <= 0 or not 0 <= config.issue_frame < frame_count:
        raise ValueError("calibration issue_frame must be inside a valid scene")
    horizon = min(config.horizon_frames, frame_count - 1 - config.issue_frame)
    if horizon < 1:
        raise ValueError("calibration horizon has no future frames")

    errors_by_lead: list[list[float]] = [[] for _ in range(horizon + 1)]
    samples: list[tuple[int, float]] = []
    for seed in config.seeds:
        scenario_values = asdict(scenario_template)
        scenario_values["seed"] = seed
        truth = build_scenario_truth(base_scene, ScenarioConfig(**scenario_values))
        center = build_prediction_snapshot(
            truth,
            config.issue_frame,
            PredictionConfig(
                seed=seed,
                observation_noise_std=config.observation_noise_std,
                base_margin=config.base_margin,
                growth_per_second=config.growth_per_second,
                max_prediction_horizon_frames=horizon,
                residual_quantile=config.target_coverage,
                observation_confidence_multiplier=config.observation_confidence_multiplier,
                control_error_margin=config.control_error_margin,
                velocity_fit_window_frames=config.velocity_fit_window_frames,
                velocity_fit_min_samples=config.velocity_fit_min_samples,
                residual_motion_model=config.residual_motion_model,
            ),
            "center",
        )
        truth_obstacles = _dynamic_obstacles(truth)
        center_obstacles = _dynamic_obstacles(center)
        if truth_obstacles.keys() != center_obstacles.keys():
            raise ValueError("truth and center prediction obstacle IDs differ")
        for lead in range(1, horizon + 1):
            frame = config.issue_frame + lead
            maximum_error = max(
                _distance(
                    truth_obstacle["positions"][frame],
                    center_obstacles[name]["positions"][frame],
                )
                for name, truth_obstacle in truth_obstacles.items()
            )
            errors_by_lead[lead].append(maximum_error)
            samples.append((lead, maximum_error))

    common_margin = (
        config.base_margin
        + config.observation_confidence_multiplier * config.observation_noise_std
        + config.control_error_margin
    )
    required_by_lead: list[list[float]] = [[] for _ in range(horizon + 1)]
    for lead in range(1, horizon + 1):
        growth = config.growth_per_second * lead / fps
        required_by_lead[lead] = [
            max(0.0, error - common_margin - growth)
            for error in errors_by_lead[lead]
        ]

    # Learn only the lead-time shape here. A single pooled conformal scale below
    # calibrates that shape to the same marginal target used by the fixed tube.
    adaptive_shape = [0.0] * (horizon + 1)
    for lead in range(1, horizon + 1):
        values = required_by_lead[lead]
        adaptive_shape[lead] = sum(values) / len(values) if values else 0.0
    normalized_scores: list[float] = []
    for lead in range(1, horizon + 1):
        shape = adaptive_shape[lead]
        for required in required_by_lead[lead]:
            if shape <= 1e-15:
                normalized_scores.append(0.0 if required <= 1e-15 else math.inf)
            else:
                normalized_scores.append(required / shape)
    adaptive_scale_factor = _upper_quantile(
        normalized_scores, config.target_coverage
    )
    if not math.isfinite(adaptive_scale_factor):
        raise ValueError("adaptive lead shape cannot represent nonzero calibration error")
    adaptive_residual_bounds = [
        shape * adaptive_scale_factor for shape in adaptive_shape
    ]

    fixed_required = [max(0.0, error - common_margin) for _, error in samples]
    fixed_residual_bound = _upper_quantile(fixed_required, config.target_coverage)
    fixed_radius = common_margin + fixed_residual_bound
    adaptive_total_radii = [
        common_margin
        + config.growth_per_second * lead / fps
        + adaptive_residual_bounds[lead]
        for lead in range(horizon + 1)
    ]
    fixed_total_radii = [fixed_radius] * (horizon + 1)

    source_scene_resolved = source_scene.resolve() if source_scene is not None else None
    profile = {
        "schema_version": 1,
        "method": "independent_seed_split_equal_marginal_coverage",
        "coverage_scope": "simultaneous_obstacle_max_per_seed_lead",
        "source_scene": str(source_scene_resolved) if source_scene_resolved else None,
        "source_scene_sha256": _sha256(source_scene_resolved) if source_scene_resolved else None,
        "scenario_kind": scenario_template.kind,
        "scenario_config_without_seed": _scenario_signature(scenario_template),
        "calibration_seeds": list(config.seeds),
        "issue_frame": config.issue_frame,
        "horizon_frames": horizon,
        "target_coverage": config.target_coverage,
        "runtime_parameters": {
            "observation_noise_std": config.observation_noise_std,
            "base_margin": config.base_margin,
            "growth_per_second": config.growth_per_second,
            "observation_confidence_multiplier": config.observation_confidence_multiplier,
            "control_error_margin": config.control_error_margin,
            "velocity_fit_window_frames": config.velocity_fit_window_frames,
            "velocity_fit_min_samples": config.velocity_fit_min_samples,
            "residual_motion_model": _resolve_residual_motion_model(
                scenario_template.kind,
                config.residual_motion_model,
            ),
        },
        "adaptive_residual_bounds": adaptive_residual_bounds,
        "adaptive_lead_shape": adaptive_shape,
        "adaptive_scale_factor": adaptive_scale_factor,
        "fixed_residual_bound": fixed_residual_bound,
        "fixed_radius": fixed_radius,
        "calibration_metrics": {
            "sample_count": len(samples),
            "samples_per_lead": [len(values) for values in errors_by_lead],
            "adaptive_empirical_coverage": _coverage(samples, adaptive_total_radii),
            "fixed_empirical_coverage": _coverage(samples, fixed_total_radii),
            "adaptive_mean_radius": sum(adaptive_total_radii[1:]) / horizon,
            "fixed_radius": fixed_radius,
        },
    }
    return profile


def load_calibration_profile(path: Path) -> tuple[dict, str]:
    path = path.resolve()
    profile = json.loads(path.read_text(encoding="utf-8"))
    schema_version = profile.get("schema_version")
    if schema_version not in {1, 2}:
        raise ValueError("unsupported calibration profile schema")
    required = (
        "calibration_seeds",
        "scenario_kind",
        "issue_frame",
        "runtime_parameters",
        "adaptive_residual_bounds",
        "fixed_radius",
    )
    if any(key not in profile for key in required):
        raise ValueError("calibration profile is missing required fields")
    if schema_version == 2:
        from scripts.uncertainty.conformal_tubes import validate_seed_splits

        conformal_required = (
            "proper_training_seeds",
            "calibration_method",
            "alpha",
            "conformal_rank",
            "conformal_threshold",
            "trajectory_calibration_scores",
            "provenance",
        )
        if any(key not in profile for key in conformal_required):
            raise ValueError("schema-v2 calibration profile is missing required fields")
        validate_seed_splits(
            profile["proper_training_seeds"],
            profile["calibration_seeds"],
            profile.get("test_seeds", ()),
        )
        if profile["calibration_method"] != "trajectory_split_conformal":
            raise ValueError("unsupported schema-v2 calibration method")
    bounds = tuple(float(value) for value in profile["adaptive_residual_bounds"])
    if not bounds:
        raise ValueError("calibration profile has no adaptive bounds")
    PredictionConfig(offline_residual_bounds=bounds).validate()
    return profile, _sha256(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_scene", type=Path)
    parser.add_argument("output_profile", type=Path)
    parser.add_argument("--seeds", type=parse_seed_list, default=parse_seed_list("1,2,3,4,5,6,7,8,9,10"))
    parser.add_argument("--proper-training-seeds", type=parse_seed_list)
    parser.add_argument("--issue-frame", type=int, default=15)
    parser.add_argument("--horizon-frames", type=int, default=15)
    parser.add_argument("--target-coverage", type=float, default=0.95)
    parser.add_argument("--event-frame-ratio", type=float, default=16.0 / 599.0)
    parser.add_argument("--scenario-kind", choices=SCENARIO_KINDS, default="milestone_intrusion")
    parser.add_argument("--acceleration-std", type=float, default=0.0006)
    parser.add_argument("--turn-degrees", type=float, default=90.0)
    parser.add_argument("--occlusion-fraction", type=float, default=0.2)
    parser.add_argument("--unpredictable-step-std", type=float, default=0.015)
    parser.add_argument("--recommended-observation-noise-std", type=float, default=0.01)
    parser.add_argument("--observation-noise-std", type=float, default=0.0)
    parser.add_argument("--base-margin", type=float, default=0.01)
    parser.add_argument("--growth-per-second", type=float, default=0.0)
    parser.add_argument("--observation-confidence-multiplier", type=float, default=3.0)
    parser.add_argument("--control-error-margin", type=float, default=0.0)
    parser.add_argument("--velocity-fit-window-frames", type=int, default=15)
    parser.add_argument("--velocity-fit-min-samples", type=int, default=5)
    parser.add_argument(
        "--residual-motion-model",
        choices=("auto", "constant_offset", "constant_velocity"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_scene.resolve()
    base_scene = json.loads(source.read_text(encoding="utf-8"))
    scenario = ScenarioConfig(
        kind=args.scenario_kind,
        event_frame_ratio=args.event_frame_ratio,
        acceleration_std=args.acceleration_std,
        turn_degrees=args.turn_degrees,
        occlusion_fraction=args.occlusion_fraction,
        unpredictable_step_std=args.unpredictable_step_std,
        recommended_observation_noise_std=args.recommended_observation_noise_std,
    )
    config = CalibrationConfig(
        seeds=args.seeds,
        issue_frame=args.issue_frame,
        horizon_frames=args.horizon_frames,
        target_coverage=args.target_coverage,
        observation_noise_std=args.observation_noise_std,
        base_margin=args.base_margin,
        growth_per_second=args.growth_per_second,
        observation_confidence_multiplier=args.observation_confidence_multiplier,
        control_error_margin=args.control_error_margin,
        velocity_fit_window_frames=args.velocity_fit_window_frames,
        velocity_fit_min_samples=args.velocity_fit_min_samples,
        residual_motion_model=args.residual_motion_model,
    )
    if args.proper_training_seeds is None:
        profile = build_calibration_profile(
            base_scene,
            scenario,
            config,
            source_scene=source,
        )
    else:
        from scripts.uncertainty.conformal_tubes import (
            fit_trajectory_split_conformal,
        )

        profile = fit_trajectory_split_conformal(
            base_scene,
            scenario,
            args.proper_training_seeds,
            args.seeds,
            config,
            source_scene=source,
        )
    output = args.output_profile.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
