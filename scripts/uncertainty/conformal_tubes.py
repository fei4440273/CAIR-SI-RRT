#!/usr/bin/env python3
"""Trajectory-level split-conformal calibration for uncertainty tubes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from scripts.uncertainty.generate_trial import PredictionConfig, build_prediction_snapshot
from scripts.uncertainty.scenarios import ScenarioConfig, build_scenario_truth


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def conformal_rank(sample_count: int, alpha: float) -> int:
    """Return the one-based finite-sample split-conformal rank."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    rank = math.ceil((sample_count + 1) * (1.0 - alpha))
    if rank > sample_count:
        raise ValueError(
            "no finite conformal threshold: increase the calibration sample count"
        )
    return rank


def _normalized_seed_group(name: str, seeds: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)
    if not normalized:
        raise ValueError(f"{name} seeds must be nonempty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} seeds must be unique")
    return normalized


def validate_seed_splits(
    proper_training_seeds: Iterable[int],
    calibration_seeds: Iterable[int],
    test_seeds: Iterable[int] = (),
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    proper = _normalized_seed_group("proper-training", proper_training_seeds)
    calibration = _normalized_seed_group("calibration", calibration_seeds)
    test_values = tuple(int(seed) for seed in test_seeds)
    if test_values:
        test = _normalized_seed_group("test", test_values)
    else:
        test = ()
    overlaps = {
        "proper-training/calibration": sorted(set(proper) & set(calibration)),
        "proper-training/test": sorted(set(proper) & set(test)),
        "calibration/test": sorted(set(calibration) & set(test)),
    }
    present = {name: values for name, values in overlaps.items() if values}
    if present:
        raise ValueError(f"conformal seed split overlap: {present}")
    return proper, calibration, test


def trajectory_nonconformity_score(
    required_residuals: Sequence[tuple[int, float]],
    lead_shape: Sequence[float],
) -> float:
    """Return one seed's maximum normalized residual across obstacles and leads."""

    if not required_residuals:
        raise ValueError("trajectory score requires at least one residual")
    scores: list[float] = []
    for lead, required in required_residuals:
        if lead < 0 or lead >= len(lead_shape):
            raise ValueError("trajectory residual lead is outside the lead shape")
        if not math.isfinite(required) or required < 0.0:
            raise ValueError("required residuals must be finite and nonnegative")
        denominator = float(lead_shape[lead])
        if not math.isfinite(denominator) or denominator < 0.0:
            raise ValueError("lead shape must be finite and nonnegative")
        if denominator <= 1e-12:
            if required > 1e-12:
                raise ValueError(
                    "lead shape cannot represent a positive trajectory residual"
                )
            scores.append(0.0)
        else:
            scores.append(required / denominator)
    return max(scores)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((left[axis] - right[axis]) ** 2 for axis in range(3)))


def _scenario_signature(config: ScenarioConfig) -> dict:
    signature = asdict(config)
    signature.pop("seed")
    # Canonicalize tuples to their JSON representation so an in-memory profile
    # and the same profile reloaded from disk compare identically.
    return json.loads(json.dumps(signature, ensure_ascii=True, sort_keys=True))


def _runtime_parameters(config: object, scenario_kind: str) -> dict:
    requested_model = str(getattr(config, "residual_motion_model"))
    if requested_model == "auto":
        residual_model = (
            "constant_offset"
            if scenario_kind in {"position_noise", "occlusion"}
            else "constant_velocity"
        )
    else:
        residual_model = requested_model
    return {
        "observation_noise_std": float(getattr(config, "observation_noise_std")),
        "base_margin": float(getattr(config, "base_margin")),
        "growth_per_second": float(getattr(config, "growth_per_second")),
        "observation_confidence_multiplier": float(
            getattr(config, "observation_confidence_multiplier")
        ),
        "control_error_margin": float(getattr(config, "control_error_margin")),
        "velocity_fit_window_frames": int(
            getattr(config, "velocity_fit_window_frames")
        ),
        "velocity_fit_min_samples": int(
            getattr(config, "velocity_fit_min_samples")
        ),
        "residual_motion_model": residual_model,
    }


def _prediction_config(seed: int, config: object, runtime: dict, horizon: int) -> PredictionConfig:
    return PredictionConfig(
        seed=seed,
        observation_noise_std=runtime["observation_noise_std"],
        base_margin=runtime["base_margin"],
        growth_per_second=runtime["growth_per_second"],
        max_prediction_horizon_frames=horizon,
        residual_quantile=float(getattr(config, "target_coverage")),
        observation_confidence_multiplier=runtime[
            "observation_confidence_multiplier"
        ],
        control_error_margin=runtime["control_error_margin"],
        velocity_fit_window_frames=runtime["velocity_fit_window_frames"],
        velocity_fit_min_samples=runtime["velocity_fit_min_samples"],
        residual_motion_model=runtime["residual_motion_model"],
    )


def _dynamic_obstacles(scene: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for index, obstacle in enumerate(scene.get("obstacles", [])):
        if obstacle.get("type") not in {"dynamic_sphere", "dynamic_box"}:
            continue
        identifier = str(obstacle.get("name") or f"obstacle-{index}")
        if identifier in result:
            raise ValueError("dynamic obstacles need unique identifiers")
        result[identifier] = obstacle
    return result


def _trajectory_required_residuals(
    base_scene: dict,
    scenario_template: ScenarioConfig,
    seed: int,
    config: object,
    runtime: dict,
) -> list[tuple[int, float]]:
    scenario_values = asdict(scenario_template)
    scenario_values["seed"] = seed
    truth = build_scenario_truth(base_scene, ScenarioConfig(**scenario_values))
    issue_frame = int(getattr(config, "issue_frame"))
    requested_horizon = int(getattr(config, "horizon_frames"))
    frame_count = int(truth.get("frame_count", 0))
    fps = int(truth.get("fps", 0))
    horizon = min(requested_horizon, frame_count - 1 - issue_frame)
    if fps <= 0 or issue_frame < 0 or horizon < 1:
        raise ValueError("conformal calibration needs a valid future prediction horizon")
    center = build_prediction_snapshot(
        truth,
        issue_frame,
        _prediction_config(seed, config, runtime, horizon),
        "center",
    )
    truth_obstacles = _dynamic_obstacles(truth)
    center_obstacles = _dynamic_obstacles(center)
    if truth_obstacles.keys() != center_obstacles.keys():
        raise ValueError("truth and center prediction obstacle IDs differ")
    affected = [
        identifier
        for identifier, obstacle in truth_obstacles.items()
        if obstacle.get("_uncertainty_affected") is True
    ]
    if not affected:
        raise ValueError("trajectory conformal calibration requires affected obstacles")

    common_margin = (
        runtime["base_margin"]
        + runtime["observation_confidence_multiplier"]
        * runtime["observation_noise_std"]
        + runtime["control_error_margin"]
    )
    residuals: list[tuple[int, float]] = []
    for identifier in affected:
        truth_positions = truth_obstacles[identifier]["positions"]
        center_positions = center_obstacles[identifier]["positions"]
        for lead in range(1, horizon + 1):
            frame = issue_frame + lead
            error = _distance(truth_positions[frame], center_positions[frame])
            growth = runtime["growth_per_second"] * lead / fps
            residuals.append((lead, max(0.0, error - common_margin - growth)))
    return residuals


def _fit_lead_shape(
    trajectories: Sequence[Sequence[tuple[int, float]]],
    horizon: int,
) -> list[float]:
    values_by_lead: list[list[float]] = [[] for _ in range(horizon + 1)]
    for residuals in trajectories:
        for lead, residual in residuals:
            values_by_lead[lead].append(residual)
    shape = [0.0] * (horizon + 1)
    for lead in range(1, horizon + 1):
        values = values_by_lead[lead]
        if not values:
            raise ValueError(f"proper-training data has no samples for lead {lead}")
        shape[lead] = max(shape[lead - 1], sum(values) / len(values))
    return shape


def fit_trajectory_split_conformal(
    base_scene: dict,
    scenario_template: ScenarioConfig,
    proper_training_seeds: Iterable[int],
    calibration_seeds: Iterable[int],
    config: object,
    *,
    source_scene: Path | None = None,
) -> dict:
    """Fit a schema-v2 trajectory-level split-conformal tube profile."""

    proper, calibration, _ = validate_seed_splits(
        proper_training_seeds, calibration_seeds
    )
    config.validate()
    if tuple(int(seed) for seed in getattr(config, "seeds")) != calibration:
        raise ValueError("config seeds must exactly match calibration seeds")
    scenario_template.validate()
    alpha = 1.0 - float(getattr(config, "target_coverage"))
    rank = conformal_rank(len(calibration), alpha)
    issue_frame = int(getattr(config, "issue_frame"))
    horizon = min(
        int(getattr(config, "horizon_frames")),
        int(base_scene.get("frame_count", 0)) - 1 - issue_frame,
    )
    if horizon < 1:
        raise ValueError("conformal calibration has no future frames")
    runtime = _runtime_parameters(config, scenario_template.kind)

    proper_data = [
        {
            "seed": seed,
            "residuals": _trajectory_required_residuals(
                base_scene, scenario_template, seed, config, runtime
            ),
        }
        for seed in proper
    ]
    lead_shape = _fit_lead_shape(
        [entry["residuals"] for entry in proper_data], horizon
    )
    calibration_data = []
    for seed in calibration:
        residuals = _trajectory_required_residuals(
            base_scene, scenario_template, seed, config, runtime
        )
        score = trajectory_nonconformity_score(residuals, lead_shape)
        calibration_data.append({"seed": seed, "residuals": residuals, "score": score})
    ordered_scores = sorted(float(entry["score"]) for entry in calibration_data)
    threshold = ordered_scores[rank - 1]
    if not math.isfinite(threshold):
        raise ValueError("trajectory calibration produced a non-finite threshold")
    bounds = [shape * threshold for shape in lead_shape]
    common_margin = (
        runtime["base_margin"]
        + runtime["observation_confidence_multiplier"]
        * runtime["observation_noise_std"]
        + runtime["control_error_margin"]
    )
    source_scene_resolved = source_scene.resolve() if source_scene is not None else None
    scenario_signature = _scenario_signature(scenario_template)
    config_signature = asdict(config)
    profile = {
        "schema_version": 2,
        "method": "trajectory_split_conformal",
        "calibration_method": "trajectory_split_conformal",
        "coverage_scope": "simultaneous_affected_obstacles_and_leads",
        "source_scene": str(source_scene_resolved) if source_scene_resolved else None,
        "source_scene_sha256": (
            _file_sha256(source_scene_resolved) if source_scene_resolved else None
        ),
        "scenario_kind": scenario_template.kind,
        "scenario_config_without_seed": scenario_signature,
        "proper_training_seeds": list(proper),
        "calibration_seeds": list(calibration),
        "test_seeds": [],
        "issue_frame": issue_frame,
        "horizon_frames": horizon,
        "target_coverage": float(getattr(config, "target_coverage")),
        "alpha": alpha,
        "runtime_parameters": runtime,
        "adaptive_lead_shape": lead_shape,
        "trajectory_calibration_scores": [
            {"seed": entry["seed"], "score": entry["score"]}
            for entry in calibration_data
        ],
        "conformal_rank": rank,
        "conformal_threshold": threshold,
        "adaptive_residual_bounds": bounds,
        "fixed_residual_bound": max(bounds),
        "fixed_radius": common_margin + max(bounds),
        "provenance": {
            "base_scene_sha256": _canonical_sha256(base_scene),
            "scenario_template_sha256": _canonical_sha256(scenario_signature),
            "calibration_config_sha256": _canonical_sha256(config_signature),
            "proper_training_seeds_sha256": _canonical_sha256(list(proper)),
            "calibration_seeds_sha256": _canonical_sha256(list(calibration)),
            "proper_training_data_sha256": _canonical_sha256(proper_data),
            "calibration_data_sha256": _canonical_sha256(calibration_data),
        },
    }
    return profile


def _wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1:
        raise ValueError("Wilson interval requires at least one trial")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def evaluate_trajectory_coverage(
    profile: dict,
    base_scene: dict,
    scenario_template: ScenarioConfig,
    test_seeds: Iterable[int],
) -> dict:
    """Evaluate simultaneous trajectory coverage on a held-out seed split."""

    proper, calibration, test = validate_seed_splits(
        profile.get("proper_training_seeds", ()),
        profile.get("calibration_seeds", ()),
        test_seeds,
    )
    if profile.get("schema_version") != 2:
        raise ValueError("trajectory coverage requires a schema-v2 profile")
    if profile.get("scenario_kind") != scenario_template.kind:
        raise ValueError("scenario kind does not match conformal profile")
    if profile.get("scenario_config_without_seed") != _scenario_signature(
        scenario_template
    ):
        raise ValueError("scenario parameters do not match conformal profile")
    bounds = tuple(float(value) for value in profile["adaptive_residual_bounds"])
    horizon = int(profile["horizon_frames"])
    if len(bounds) < horizon + 1:
        raise ValueError("conformal profile has incomplete residual bounds")

    class _Config:
        pass

    config = _Config()
    config.issue_frame = int(profile["issue_frame"])
    config.horizon_frames = horizon
    config.target_coverage = float(profile["target_coverage"])
    for name, value in profile["runtime_parameters"].items():
        setattr(config, name, value)
    runtime = dict(profile["runtime_parameters"])
    test_data = []
    lead_covered = [0] * (horizon + 1)
    lead_total = [0] * (horizon + 1)
    covered_trajectories = 0
    maximum_violation = 0.0
    for seed in test:
        residuals = _trajectory_required_residuals(
            base_scene, scenario_template, seed, config, runtime
        )
        violations = []
        for lead, required in residuals:
            violation = max(0.0, required - bounds[lead])
            violations.append(violation)
            lead_total[lead] += 1
            lead_covered[lead] += int(violation <= 1e-12)
        seed_maximum = max(violations, default=0.0)
        covered = seed_maximum <= 1e-12
        covered_trajectories += int(covered)
        maximum_violation = max(maximum_violation, seed_maximum)
        test_data.append(
            {
                "seed": seed,
                "covered": covered,
                "maximum_violation": seed_maximum,
                "residuals": residuals,
            }
        )
    total = len(test)
    fps = int(base_scene["fps"])
    common_margin = (
        runtime["base_margin"]
        + runtime["observation_confidence_multiplier"]
        * runtime["observation_noise_std"]
        + runtime["control_error_margin"]
    )
    radii = [
        common_margin + runtime["growth_per_second"] * lead / fps + bounds[lead]
        for lead in range(1, horizon + 1)
    ]
    return {
        "schema_version": 1,
        "method": "held_out_trajectory_coverage",
        "profile_content_sha256": _canonical_sha256(profile),
        "scenario_kind": scenario_template.kind,
        "proper_training_seeds": list(proper),
        "calibration_seeds": list(calibration),
        "test_seeds": list(test),
        "covered_trajectories": covered_trajectories,
        "total_trajectories": total,
        "trajectory_coverage": covered_trajectories / total,
        "wilson_95_interval": _wilson_interval(covered_trajectories, total),
        "per_lead_marginal_coverage": [
            {
                "lead": lead,
                "covered": lead_covered[lead],
                "total": lead_total[lead],
                "coverage": lead_covered[lead] / lead_total[lead],
            }
            for lead in range(1, horizon + 1)
        ],
        "maximum_violation": maximum_violation,
        "mean_radius": sum(radii) / len(radii),
        "maximum_radius": max(radii),
        "test_data_sha256": _canonical_sha256(test_data),
        "trajectory_results": [
            {
                "seed": entry["seed"],
                "covered": entry["covered"],
                "maximum_violation": entry["maximum_violation"],
            }
            for entry in test_data
        ],
    }
