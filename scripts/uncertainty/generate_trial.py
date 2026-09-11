#!/usr/bin/env python3
"""Generate deterministic, no-future-leakage prediction snapshots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from scripts.uncertainty.mode_change import (
    detector_diagnostics,
    one_step_innovations,
)


@dataclass(frozen=True)
class PredictionConfig:
    seed: int = 42
    observation_noise_std: float = 0.0
    base_margin: float = 0.01
    growth_per_second: float = 0.02
    fixed_radius: float = 0.05
    max_tube_radius: float = 0.25
    max_prediction_horizon_frames: int = 150
    residual_quantile: float = 0.95
    observation_confidence_multiplier: float = 3.0
    control_error_margin: float = 0.0
    velocity_fit_window_frames: int = 15
    velocity_fit_min_samples: int = 5
    residual_motion_model: str = "auto"
    offline_residual_bounds: tuple[float, ...] = ()
    historical_residual_bounds_enabled: bool = True
    scenario_label_independent: bool = False
    velocity_model_improvement_ratio: float = 0.8
    mode_change_threshold: float | None = None
    mode_change_short_horizon_frames: int = 15
    mode_change_recovery_frames: int = 30

    def validate(self) -> None:
        nonnegative = {
            "observation_noise_std": self.observation_noise_std,
            "base_margin": self.base_margin,
            "growth_per_second": self.growth_per_second,
            "fixed_radius": self.fixed_radius,
            "max_tube_radius": self.max_tube_radius,
            "observation_confidence_multiplier": self.observation_confidence_multiplier,
            "control_error_margin": self.control_error_margin,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_prediction_horizon_frames < 0:
            raise ValueError("max_prediction_horizon_frames must be nonnegative")
        if self.velocity_fit_window_frames < 2:
            raise ValueError("velocity_fit_window_frames must be at least 2")
        if not 2 <= self.velocity_fit_min_samples <= self.velocity_fit_window_frames:
            raise ValueError(
                "velocity_fit_min_samples must be between 2 and velocity_fit_window_frames"
            )
        if self.residual_motion_model not in {
            "auto",
            "constant_offset",
            "constant_velocity",
            "history_selected",
        }:
            raise ValueError(
                "residual_motion_model must be auto, constant_offset, "
                "constant_velocity, or history_selected"
            )
        if type(self.scenario_label_independent) is not bool:
            raise ValueError("scenario_label_independent must be a boolean")
        if type(self.historical_residual_bounds_enabled) is not bool:
            raise ValueError("historical_residual_bounds_enabled must be a boolean")
        if (
            not math.isfinite(self.velocity_model_improvement_ratio)
            or not 0.0 < self.velocity_model_improvement_ratio <= 1.0
        ):
            raise ValueError("velocity_model_improvement_ratio must be in (0, 1]")
        if not 0 < self.residual_quantile <= 1:
            raise ValueError("residual_quantile must be in (0, 1]")
        for value in self.offline_residual_bounds:
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "offline_residual_bounds must contain finite nonnegative values"
                )
        if self.mode_change_threshold is not None and (
            not math.isfinite(self.mode_change_threshold)
            or self.mode_change_threshold <= 0.0
        ):
            raise ValueError("mode_change_threshold must be finite and positive")
        if self.mode_change_short_horizon_frames <= 0:
            raise ValueError("mode_change_short_horizon_frames must be positive")
        if self.mode_change_recovery_frames < 0:
            raise ValueError("mode_change_recovery_frames must be nonnegative")


def _seed_for_obstacle(seed: int, obstacle_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{obstacle_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _upper_quantile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _observed_positions(
    positions: Sequence[Sequence[float]],
    issue_frame: int,
    noise_std: float,
    rng: random.Random,
    occlusion_ranges: Sequence[Sequence[int]] = (),
) -> list[tuple[int, list[float]]]:
    observed: list[tuple[int, list[float]]] = []
    for frame in range(issue_frame + 1):
        if any(int(first) <= frame <= int(last) for first, last in occlusion_ranges):
            continue
        point = list(positions[frame])
        for axis in range(3):
            point[axis] += rng.gauss(0.0, noise_std)
        observed.append((frame, point))
    if not observed:
        raise ValueError("prediction snapshot has no visible obstacle observation")
    return observed


def _resolve_residual_motion_model(scenario_kind: str, requested: str) -> str:
    if requested != "auto":
        return requested
    if scenario_kind in {"position_noise", "occlusion"}:
        return "constant_offset"
    return "constant_velocity"


def _select_residual_motion_model(
    observed: Sequence[tuple[int, Sequence[float]]],
    *,
    nominal_positions: Sequence[Sequence[float]] | None,
    window_frames: int,
    minimum_samples: int,
    velocity_improvement_ratio: float = 0.8,
) -> str:
    """Select a future residual model from causal one-step history errors."""

    errors: dict[str, list[float]] = {
        "constant_offset": [],
        "constant_velocity": [],
    }
    for target_index in range(minimum_samples, len(observed)):
        target_frame, target = observed[target_index]
        prior = [
            item
            for item in observed[:target_index]
            if target_frame - window_frames <= item[0] < target_frame
        ]
        if len(prior) < minimum_samples:
            continue
        origin_frame = prior[-1][0]
        for model in errors:
            level, velocity = _fit_linear_samples(
                prior, origin_frame, nominal_positions, model
            )
            lead = target_frame - origin_frame
            predicted = [level[axis] + velocity[axis] * lead for axis in range(3)]
            if nominal_positions is not None:
                predicted = [
                    nominal_positions[target_frame][axis] + predicted[axis]
                    for axis in range(3)
                ]
            errors[model].append(_distance(predicted, target))
    if not errors["constant_offset"]:
        return "constant_offset"
    offset_error = sum(errors["constant_offset"]) / len(errors["constant_offset"])
    velocity_error = sum(errors["constant_velocity"]) / len(errors["constant_velocity"])
    return (
        "constant_velocity"
        if velocity_error < velocity_improvement_ratio * offset_error
        else "constant_offset"
    )


def _historical_horizon_residual_bounds(
    observed: Sequence[tuple[int, Sequence[float]]],
    max_lead: int,
    quantile: float,
    nominal_positions: Sequence[Sequence[float]] | None = None,
    velocity_fit_window_frames: int = 15,
    velocity_fit_min_samples: int = 5,
    residual_motion_model: str = "constant_velocity",
) -> list[float]:
    """Backtest the prediction model at each lead using visible history only."""

    if max_lead < 0:
        raise ValueError("max_lead must be nonnegative")
    by_frame = {frame: point for frame, point in observed}
    fitted_states: dict[int, tuple[list[float], list[float]]] = {}
    left = 0
    for right, (origin_frame, _) in enumerate(observed):
        first_frame = origin_frame - velocity_fit_window_frames + 1
        while observed[left][0] < first_frame:
            left += 1
        samples = observed[left : right + 1]
        if len(samples) >= velocity_fit_min_samples:
            fitted_states[origin_frame] = _fit_linear_samples(
                samples,
                origin_frame,
                nominal_positions,
                residual_motion_model,
            )
    bounds = [0.0] * (max_lead + 1)
    previous_bound = 0.0
    for lead in range(1, max_lead + 1):
        errors: list[float] = []
        for origin_frame, (level, velocity) in fitted_states.items():
            target = by_frame.get(origin_frame + lead)
            if target is None:
                continue
            predicted = [level[axis] + velocity[axis] * lead for axis in range(3)]
            if nominal_positions is not None:
                predicted = [
                    nominal_positions[origin_frame + lead][axis] + predicted[axis]
                    for axis in range(3)
                ]
            errors.append(_distance(predicted, target))
        if errors:
            previous_bound = max(previous_bound, _upper_quantile(errors, quantile))
        bounds[lead] = previous_bound
    return bounds


def _fit_linear_state(
    observed: Sequence[tuple[int, Sequence[float]]],
    origin_frame: int,
    window_frames: int,
    nominal_positions: Sequence[Sequence[float]] | None,
    residual_motion_model: str = "constant_velocity",
) -> tuple[list[float], list[float]]:
    samples = [
        (frame, point)
        for frame, point in observed
        if origin_frame - window_frames + 1 <= frame <= origin_frame
    ]
    if not samples:
        raise ValueError("linear state fit requires at least one observation")
    return _fit_linear_samples(
        samples,
        origin_frame,
        nominal_positions,
        residual_motion_model,
    )


def _fit_linear_samples(
    samples: Sequence[tuple[int, Sequence[float]]],
    origin_frame: int,
    nominal_positions: Sequence[Sequence[float]] | None,
    residual_motion_model: str = "constant_velocity",
) -> tuple[list[float], list[float]]:
    if residual_motion_model not in {"constant_offset", "constant_velocity"}:
        raise ValueError("resolved residual motion model is invalid")
    mean_frame = sum(frame for frame, _ in samples) / len(samples)
    denominator = sum((frame - mean_frame) ** 2 for frame, _ in samples)
    level: list[float] = []
    velocity: list[float] = []
    for axis in range(3):
        values = [
            (
                point[axis] - nominal_positions[frame][axis]
                if nominal_positions is not None
                else point[axis]
            )
            for frame, point in samples
        ]
        mean_value = sum(values) / len(values)
        slope = (
            0.0
            if residual_motion_model == "constant_offset"
            else (
                sum(
                    (frame - mean_frame) * (value - mean_value)
                    for (frame, _), value in zip(samples, values)
                )
                / denominator
                if denominator > 0.0
                else 0.0
            )
        )
        level.append(mean_value + slope * (origin_frame - mean_frame))
        velocity.append(slope)
    return level, velocity


def _predict_positions(
    truth_positions: Sequence[Sequence[float]],
    observed: Sequence[tuple[int, Sequence[float]]],
    issue_frame: int,
    nominal_positions: Sequence[Sequence[float]] | None = None,
    velocity_fit_window_frames: int = 15,
    residual_motion_model: str = "constant_velocity",
) -> list[list[float]]:
    current_frame, current = observed[-1]
    level, velocity = _fit_linear_state(
        observed,
        current_frame,
        velocity_fit_window_frames,
        nominal_positions,
        residual_motion_model,
    )

    observed_by_frame = {frame: point for frame, point in observed}
    result: list[list[float]] = []
    for frame in range(len(truth_positions)):
        known = observed_by_frame.get(frame)
        if known is not None:
            result.append(list(known))
            continue
        lead = frame - current_frame
        if nominal_positions is None:
            center = [level[axis] + velocity[axis] * lead for axis in range(3)]
            orientation = list(current[3:7])
        else:
            center = [
                nominal_positions[frame][axis] + level[axis] + velocity[axis] * lead
                for axis in range(3)
            ]
            orientation = list(nominal_positions[frame][3:7])
        result.append(center + orientation)
    return result


def _tube_radii(
    frame_count: int,
    fps: int,
    last_observation_frame: int,
    residual_bounds: Sequence[float],
    config: PredictionConfig,
    tube_mode: str,
) -> tuple[list[float], dict[str, list[float]]]:
    zero = [0.0] * frame_count
    if tube_mode in {"center", "oracle"}:
        return zero, {
            "safety_margin": list(zero),
            "observation_noise_bound": list(zero),
            "residual_error_bound": list(zero),
            "lead_time_growth": list(zero),
            "control_error_bound": list(zero),
        }
    if tube_mode == "fixed":
        fixed = [config.fixed_radius] * frame_count
        return fixed, {
            "fixed_margin": list(fixed),
        }
    if tube_mode != "adaptive":
        raise ValueError(f"unsupported tube_mode: {tube_mode}")

    safety_margin = [config.base_margin] * frame_count
    observation_bound = [
        config.observation_confidence_multiplier * config.observation_noise_std
    ] * frame_count
    residual_error: list[float] = []
    control_error = [config.control_error_margin] * frame_count
    lead_time_growth: list[float] = []
    for frame in range(frame_count):
        lead = max(0, frame - last_observation_frame)
        online_bound = (
            residual_bounds[min(lead, len(residual_bounds) - 1)]
            if residual_bounds
            else 0.0
        )
        offline_bound = (
            config.offline_residual_bounds[
                min(lead, len(config.offline_residual_bounds) - 1)
            ]
            if config.offline_residual_bounds
            else 0.0
        )
        residual_error.append(max(online_bound, offline_bound))
        growth = config.growth_per_second * lead / fps
        lead_time_growth.append(growth)
    components = {
        "safety_margin": safety_margin,
        "observation_noise_bound": observation_bound,
        "residual_error_bound": residual_error,
        "lead_time_growth": lead_time_growth,
        "control_error_bound": control_error,
    }
    result = [
        sum(values[frame] for values in components.values())
        for frame in range(frame_count)
    ]
    return result, components


def _validate_scene(scene: dict) -> tuple[int, int]:
    try:
        frame_count = int(scene["frame_count"])
        fps = int(scene["fps"])
        obstacles = scene["obstacles"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "scene must contain frame_count, fps, and obstacles"
        ) from error
    if frame_count <= 0 or fps <= 0 or not isinstance(obstacles, list):
        raise ValueError(
            "scene frame_count and fps must be positive and obstacles must be an array"
        )
    return frame_count, fps


def build_prediction_snapshot(
    truth_scene: dict,
    issue_frame: int,
    config: PredictionConfig,
    tube_mode: str = "adaptive",
) -> dict:
    """Build one snapshot using observations no later than ``issue_frame``."""

    config.validate()
    frame_count, fps = _validate_scene(truth_scene)
    if issue_frame < 0 or issue_frame >= frame_count:
        raise ValueError("issue_frame must be inside the scene frame range")

    snapshot = copy.deepcopy(truth_scene)
    scenario = truth_scene.get("_uncertainty_scenario", {})
    residual_motion_model = _resolve_residual_motion_model(
        str(scenario.get("kind", "official_truth")),
        config.residual_motion_model,
    )
    occlusion_ranges = scenario.get("occlusion_ranges", [])
    failure_frame = scenario.get("failure_frame")
    prediction_unavailable = (
        tube_mode != "oracle"
        and failure_frame is not None
        and issue_frame >= int(failure_frame)
    )
    reliable_until = (
        frame_count - 1
        if tube_mode == "oracle"
        else min(
            frame_count - 1,
            issue_frame + config.max_prediction_horizon_frames,
        )
    )
    if prediction_unavailable:
        reliable_until = issue_frame
    last_observation_frames: list[int] = []
    used_nominal_prior = False
    in_occlusion = False
    obstacle_innovations: list[tuple[str, list[tuple[int, float]]]] = []

    for obstacle in snapshot["obstacles"]:
        obstacle_type = obstacle.get("type")
        if obstacle_type not in {"dynamic_sphere", "dynamic_box"}:
            continue
        positions = obstacle.get("positions")
        if not isinstance(positions, list) or len(positions) < frame_count:
            raise ValueError(
                f"dynamic obstacle {obstacle.get('name', '<unnamed>')} must cover every declared frame"
            )
        positions = positions[:frame_count]
        nominal_positions = obstacle.get("_uncertainty_nominal_positions")
        if nominal_positions is not None:
            if (
                not isinstance(nominal_positions, list)
                or len(nominal_positions) < frame_count
            ):
                raise ValueError(
                    f"dynamic obstacle {obstacle.get('name', '<unnamed>')} has an invalid nominal trajectory"
                )
            nominal_positions = nominal_positions[:frame_count]
            used_nominal_prior = True

        obstacle_occlusion_ranges = obstacle.get(
            "_uncertainty_occlusion_ranges", occlusion_ranges
        )
        if not isinstance(obstacle_occlusion_ranges, list):
            raise ValueError("obstacle occlusion ranges must be an array")
        in_occlusion = in_occlusion or any(
            int(first) <= issue_frame <= int(last)
            for first, last in obstacle_occlusion_ranges
        )
        observation_noise_std = config.observation_noise_std
        if (
            not config.scenario_label_independent
            and scenario.get("kind") == "position_noise"
            and obstacle.get("_uncertainty_affected") is False
        ):
            observation_noise_std = 0.0
        rng = random.Random(_seed_for_obstacle(config.seed, obstacle.get("name", "")))
        observed = _observed_positions(
            positions,
            issue_frame,
            observation_noise_std,
            rng,
            obstacle_occlusion_ranges,
        )
        last_observation_frame = observed[-1][0]
        last_observation_frames.append(last_observation_frame)
        obstacle["prediction_last_observation_frame"] = last_observation_frame
        obstacle_motion_model = (
            _select_residual_motion_model(
                observed,
                nominal_positions=nominal_positions,
                window_frames=config.velocity_fit_window_frames,
                minimum_samples=config.velocity_fit_min_samples,
                velocity_improvement_ratio=config.velocity_model_improvement_ratio,
            )
            if residual_motion_model == "history_selected"
            else residual_motion_model
        )
        obstacle["prediction_residual_motion_model"] = obstacle_motion_model
        obstacle_innovations.append(
            (
                str(obstacle.get("name", "")),
                one_step_innovations(
                    observed,
                    fit_window_frames=config.velocity_fit_window_frames,
                    minimum_samples=config.velocity_fit_min_samples,
                    nominal_positions=nominal_positions,
                    residual_motion_model=residual_motion_model,
                ),
            )
        )
        residual_bounds = (
            _historical_horizon_residual_bounds(
                observed,
                min(config.max_prediction_horizon_frames, frame_count - 1),
                config.residual_quantile,
                nominal_positions,
                config.velocity_fit_window_frames,
                config.velocity_fit_min_samples,
                obstacle_motion_model,
            )
            if tube_mode == "adaptive" and config.historical_residual_bounds_enabled
            else []
        )
        obstacle["positions"] = (
            [list(position) for position in positions]
            if tube_mode == "oracle"
            else _predict_positions(
                positions,
                observed,
                issue_frame,
                nominal_positions,
                config.velocity_fit_window_frames,
                obstacle_motion_model,
            )
        )
        obstacle["prediction_source_id"] = obstacle.get("name", "")

        if obstacle_type == "dynamic_sphere":
            radii, components = _tube_radii(
                frame_count,
                fps,
                last_observation_frame,
                residual_bounds,
                config,
                tube_mode,
            )
            obstacle["uncertainty_radii"] = radii
            obstacle["uncertainty_radius_components"] = components
            for frame in range(issue_frame, reliable_until + 1):
                if radii[frame] > config.max_tube_radius:
                    reliable_until = frame - 1
                    break

    last_observation_frame = min(last_observation_frames, default=issue_frame)
    mode_change = detector_diagnostics(
        obstacle_innovations,
        issue_frame=issue_frame,
        threshold_m=config.mode_change_threshold,
        short_horizon_frames=config.mode_change_short_horizon_frames,
        recovery_frames=config.mode_change_recovery_frames,
    )
    if tube_mode != "oracle" and mode_change["active"]:
        reliable_until = min(
            reliable_until,
            issue_frame + config.mode_change_short_horizon_frames,
        )
    if tube_mode == "oracle":
        prediction_status = "oracle"
    elif prediction_unavailable:
        prediction_status = "unavailable"
    elif in_occlusion:
        prediction_status = "occluded"
    else:
        prediction_status = "available"

    snapshot["_uncertainty_experiment"] = {
        "schema_version": 1,
        "issue_frame": issue_frame,
        "reliable_until_frame": max(issue_frame, reliable_until),
        "tube_mode": tube_mode,
        "prediction_model": (
            "oracle_truth"
            if tube_mode == "oracle"
            else (
                "nominal_trajectory_plus_history_only_offset"
                if used_nominal_prior
                else "history_only_constant_velocity"
            )
        ),
        "calibration_model": (
            "none"
            if tube_mode in {"oracle", "center", "fixed"}
            else (
                "offline_seed_split_plus_history_only_leadwise_upper_quantile"
                if config.offline_residual_bounds
                else (
                    "history_only_leadwise_backtest_upper_quantile"
                    if config.historical_residual_bounds_enabled
                    else "prechange_base_growth_only"
                )
            )
        ),
        "prediction_status": prediction_status,
        "last_observation_frame": last_observation_frame,
        "scenario_kind": scenario.get("kind", "official_truth"),
        "residual_motion_model": residual_motion_model,
        "mode_change_detector": mode_change,
        "config": asdict(config),
    }
    return snapshot


def generate_trial(
    truth_scene_path: Path,
    output_dir: Path,
    config: PredictionConfig,
    update_stride: int,
    tube_mode: str,
    initial_issue_frame: int = 0,
    max_updates: int = 0,
) -> None:
    if update_stride <= 0:
        raise ValueError("update_stride must be positive")
    truth_scene = json.loads(truth_scene_path.read_text(encoding="utf-8"))
    frame_count, _ = _validate_scene(truth_scene)
    if initial_issue_frame < 0 or initial_issue_frame >= frame_count:
        raise ValueError("initial_issue_frame must be inside the scene frame range")
    if max_updates < 0:
        raise ValueError("max_updates must be nonnegative")

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir()
    (output_dir / "truth_scene.json").write_text(
        json.dumps(truth_scene, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    prediction_files: list[str] = []
    issue_frames = list(range(initial_issue_frame, frame_count, update_stride))
    if max_updates > 0:
        issue_frames = issue_frames[:max_updates]
    for issue_frame in issue_frames:
        prediction = build_prediction_snapshot(
            truth_scene, issue_frame, config, tube_mode
        )
        relative_path = f"predictions/update_{issue_frame:06d}.json"
        (output_dir / relative_path).write_text(
            json.dumps(prediction, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        prediction_files.append(relative_path)

    manifest = {
        "schema_version": 1,
        "truth_scene": "truth_scene.json",
        "prediction_files": prediction_files,
        "update_stride": update_stride,
        "initial_issue_frame": initial_issue_frame,
        "max_updates": max_updates,
        "tube_mode": tube_mode,
        "config": asdict(config),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("truth_scene", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update-stride", type=int, default=15)
    parser.add_argument("--initial-issue-frame", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument(
        "--tube-mode",
        choices=("oracle", "center", "fixed", "adaptive"),
        default="adaptive",
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
    args = parse_args()
    generate_trial(
        args.truth_scene,
        args.output_dir,
        PredictionConfig(
            seed=args.seed,
            observation_noise_std=args.observation_noise_std,
            base_margin=args.base_margin,
            growth_per_second=args.growth_per_second,
            fixed_radius=args.fixed_radius,
            max_tube_radius=args.max_tube_radius,
            max_prediction_horizon_frames=args.max_prediction_horizon_frames,
            residual_quantile=args.residual_quantile,
            observation_confidence_multiplier=args.observation_confidence_multiplier,
            control_error_margin=args.control_error_margin,
            velocity_fit_window_frames=args.velocity_fit_window_frames,
            velocity_fit_min_samples=args.velocity_fit_min_samples,
            residual_motion_model=args.residual_motion_model,
            historical_residual_bounds_enabled=(
                args.historical_residual_bounds_enabled
            ),
            scenario_label_independent=args.scenario_label_independent,
            velocity_model_improvement_ratio=args.velocity_model_improvement_ratio,
            mode_change_threshold=args.mode_change_threshold,
            mode_change_short_horizon_frames=args.mode_change_short_horizon_frames,
            mode_change_recovery_frames=args.mode_change_recovery_frames,
        ),
        args.update_stride,
        args.tube_mode,
        args.initial_issue_frame,
        args.max_updates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
