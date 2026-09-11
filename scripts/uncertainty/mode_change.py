"""History-only motion-mode change detection for prediction snapshots."""

from __future__ import annotations

import math
from typing import Sequence


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((left[axis] - right[axis]) ** 2 for axis in range(3)))


def _fit_state(
    samples: Sequence[tuple[int, Sequence[float]]],
    origin_frame: int,
    nominal_positions: Sequence[Sequence[float]] | None,
    residual_motion_model: str,
) -> tuple[list[float], list[float]]:
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


def one_step_innovations(
    observed: Sequence[tuple[int, Sequence[float]]],
    *,
    fit_window_frames: int,
    minimum_samples: int,
    nominal_positions: Sequence[Sequence[float]] | None = None,
    residual_motion_model: str = "constant_velocity",
) -> list[tuple[int, float]]:
    """Return causal one-step errors; every score excludes its target sample."""

    innovations: list[tuple[int, float]] = []
    for target_index in range(minimum_samples, len(observed)):
        target_frame, target = observed[target_index]
        prior = [
            item
            for item in observed[:target_index]
            if target_frame - fit_window_frames <= item[0] < target_frame
        ]
        if len(prior) < minimum_samples:
            continue
        origin_frame = prior[-1][0]
        models = (
            ("constant_offset", "constant_velocity")
            if residual_motion_model == "history_selected"
            else (residual_motion_model,)
        )
        if any(
            model not in {"constant_offset", "constant_velocity"} for model in models
        ):
            raise ValueError("innovation residual motion model is invalid")
        errors: list[float] = []
        for model in models:
            level, velocity = _fit_state(prior, origin_frame, nominal_positions, model)
            lead = target_frame - origin_frame
            predicted = [level[axis] + velocity[axis] * lead for axis in range(3)]
            if nominal_positions is not None:
                predicted = [
                    nominal_positions[target_frame][axis] + predicted[axis]
                    for axis in range(3)
                ]
            errors.append(_distance(predicted, target))
        innovations.append((target_frame, min(errors)))
    return innovations


def detector_diagnostics(
    obstacle_innovations: Sequence[tuple[str, Sequence[tuple[int, float]]]],
    *,
    issue_frame: int,
    threshold_m: float | None,
    short_horizon_frames: int,
    recovery_frames: int,
) -> dict:
    """Aggregate per-obstacle scores without consulting scenario metadata."""

    if threshold_m is None:
        return {
            "enabled": False,
            "active": False,
            "threshold_m": None,
            "maximum_innovation_m": None,
            "latest_alert_frame": None,
            "short_horizon_frames": short_horizon_frames,
            "recovery_frames": recovery_frames,
            "obstacle_latest_innovation_m": {},
        }

    all_scores = [
        (frame, score)
        for _, innovations in obstacle_innovations
        for frame, score in innovations
    ]
    alert_frames = [frame for frame, score in all_scores if score > threshold_m]
    latest_alert_frame = max(alert_frames, default=None)
    active = (
        latest_alert_frame is not None
        and issue_frame - latest_alert_frame < recovery_frames
    )
    return {
        "enabled": True,
        "active": active,
        "threshold_m": threshold_m,
        "maximum_innovation_m": max((score for _, score in all_scores), default=0.0),
        "latest_alert_frame": latest_alert_frame,
        "short_horizon_frames": short_horizon_frames,
        "recovery_frames": recovery_frames,
        "obstacle_latest_innovation_m": {
            name: (innovations[-1][1] if innovations else None)
            for name, innovations in obstacle_innovations
        },
    }
