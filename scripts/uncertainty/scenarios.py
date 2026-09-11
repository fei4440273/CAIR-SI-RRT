#!/usr/bin/env python3
"""Create deterministic truth scenarios for uncertainty experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


STANDARD_SCENARIO_KINDS = (
    "position_noise",
    "random_acceleration",
    "sudden_turn",
    "occlusion",
    "complete_failure",
)
SCENARIO_KINDS = (*STANDARD_SCENARIO_KINDS, "milestone_intrusion", "temporal_gate")


@dataclass(frozen=True)
class ScenarioConfig:
    kind: str
    seed: int = 42
    event_frame_ratio: float = 0.4
    acceleration_std: float = 0.0006
    acceleration_correlation: float = 0.8
    maximum_deviation: float = 0.25
    turn_degrees: float = 90.0
    occlusion_fraction: float = 0.2
    affected_obstacle_fraction: float = 0.35
    event_frame_jitter_fraction: float = 0.03
    occlusion_reappearance_offset_std: float = 0.03
    occlusion_reappearance_maximum_offset: float = 0.10
    unpredictable_step_std: float = 0.015
    recommended_observation_noise_std: float = 0.01
    intrusion_target: tuple[float, float, float] = (
        0.045593950496455385,
        0.0070001952526671345,
        0.8392593750847593,
    )
    intrusion_minimum_offset: float = 0.16
    intrusion_maximum_offset: float = 0.20
    intrusion_approach_frames: int = 18
    intrusion_hold_frames: int = 6
    intrusion_direction_spread_degrees: float = 30.0
    intrusion_sphere_radius: float = 0.075
    temporal_gate_target: tuple[float, float, float] = (
        0.045593950496455385,
        0.0070001952526671345,
        0.8392593750847593,
    )
    temporal_gate_minimum_offset: float = 0.32
    temporal_gate_maximum_offset: float = 0.40
    temporal_gate_approach_frames: int = 12
    temporal_gate_hold_frames: int = 60
    temporal_gate_departure_frames: int = 12
    temporal_gate_sphere_radius: float = 0.10

    def validate(self) -> None:
        if self.kind not in SCENARIO_KINDS:
            raise ValueError(f"unsupported scenario kind: {self.kind}")
        if not 0.0 < self.event_frame_ratio < 1.0:
            raise ValueError("event_frame_ratio must be in (0, 1)")
        if not 0.0 < self.occlusion_fraction < 1.0:
            raise ValueError("occlusion_fraction must be in (0, 1)")
        if not 0.0 < self.affected_obstacle_fraction <= 1.0:
            raise ValueError("affected_obstacle_fraction must be in (0, 1]")
        if not 0.0 <= self.event_frame_jitter_fraction < 0.25:
            raise ValueError("event_frame_jitter_fraction must be in [0, 0.25)")
        if not 0.0 <= self.acceleration_correlation < 1.0:
            raise ValueError("acceleration_correlation must be in [0, 1)")
        for name in (
            "acceleration_std",
            "maximum_deviation",
            "unpredictable_step_std",
            "recommended_observation_noise_std",
            "occlusion_reappearance_offset_std",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not math.isfinite(self.occlusion_reappearance_maximum_offset)
            or self.occlusion_reappearance_maximum_offset <= 0.0
        ):
            raise ValueError(
                "occlusion_reappearance_maximum_offset must be finite and positive"
            )
        if not math.isfinite(self.turn_degrees):
            raise ValueError("turn_degrees must be finite")
        if len(self.intrusion_target) != 3 or not all(
            math.isfinite(value) for value in self.intrusion_target
        ):
            raise ValueError("intrusion_target must contain three finite values")
        if not 0.0 < self.intrusion_minimum_offset <= self.intrusion_maximum_offset:
            raise ValueError("intrusion offsets must be positive and ordered")
        if self.intrusion_approach_frames < 1 or self.intrusion_hold_frames < 1:
            raise ValueError("intrusion approach and hold durations must be positive")
        if not 0.0 <= self.intrusion_direction_spread_degrees <= 75.0:
            raise ValueError("intrusion direction spread must be in [0, 75] degrees")
        if not math.isfinite(self.intrusion_sphere_radius) or self.intrusion_sphere_radius <= 0.0:
            raise ValueError("intrusion sphere radius must be finite and positive")
        if len(self.temporal_gate_target) != 3 or not all(
            math.isfinite(value) for value in self.temporal_gate_target
        ):
            raise ValueError("temporal_gate_target must contain three finite values")
        if not 0.0 < self.temporal_gate_minimum_offset <= self.temporal_gate_maximum_offset:
            raise ValueError("temporal gate offsets must be positive and ordered")
        if min(
            self.temporal_gate_approach_frames,
            self.temporal_gate_hold_frames,
            self.temporal_gate_departure_frames,
        ) < 1:
            raise ValueError("temporal gate durations must be positive")
        if not math.isfinite(self.temporal_gate_sphere_radius) or self.temporal_gate_sphere_radius <= 0.0:
            raise ValueError("temporal gate sphere radius must be finite and positive")


def _obstacle_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"scenario:{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _event_frame(frame_count: int, ratio: float) -> int:
    if frame_count < 4:
        return max(0, frame_count // 2)
    return min(frame_count - 2, max(1, int(round((frame_count - 1) * ratio))))


def _staggered_event_frames(
    frame_count: int,
    base_event_frame: int,
    count: int,
    jitter_fraction: float,
) -> list[int]:
    if count <= 1 or frame_count < 4:
        return [base_event_frame] * count
    maximum_offset = int(round((frame_count - 1) * jitter_fraction))
    if maximum_offset == 0:
        return [base_event_frame] * count
    result: list[int] = []
    for index in range(count):
        relative = -maximum_offset + 2.0 * maximum_offset * index / (count - 1)
        result.append(
            min(frame_count - 2, max(1, base_event_frame + int(round(relative))))
        )
    return result


def _bounded(vector: Sequence[float], maximum_norm: float) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0 or norm <= maximum_norm:
        return list(vector)
    scale = maximum_norm / norm
    return [value * scale for value in vector]


def _random_acceleration_positions(
    positions: Sequence[Sequence[float]],
    event_frame: int,
    config: ScenarioConfig,
    rng: random.Random,
) -> list[list[float]]:
    result = [list(position) for position in positions]
    acceleration = [0.0, 0.0, 0.0]
    velocity_offset = [0.0, 0.0, 0.0]
    position_offset = [0.0, 0.0, 0.0]
    for frame in range(event_frame + 1, len(result)):
        for axis in range(3):
            acceleration[axis] = (
                config.acceleration_correlation * acceleration[axis]
                + rng.gauss(0.0, config.acceleration_std)
            )
            velocity_offset[axis] += acceleration[axis]
            position_offset[axis] += velocity_offset[axis]
        position_offset = _bounded(position_offset, config.maximum_deviation)
        for axis in range(3):
            result[frame][axis] += position_offset[axis]
    return result


def _sudden_turn_positions(
    positions: Sequence[Sequence[float]], event_frame: int, turn_degrees: float
) -> list[list[float]]:
    result = [list(position) for position in positions]
    if event_frame == 0:
        return result
    previous = positions[event_frame - 1]
    current = positions[event_frame]
    velocity_x = current[0] - previous[0]
    velocity_y = current[1] - previous[1]
    angle = math.radians(turn_degrees)
    turned_x = math.cos(angle) * velocity_x - math.sin(angle) * velocity_y
    turned_y = math.sin(angle) * velocity_x + math.cos(angle) * velocity_y
    for frame in range(event_frame + 1, len(result)):
        lead = frame - event_frame
        result[frame][0] = current[0] + lead * turned_x
        result[frame][1] = current[1] + lead * turned_y
        result[frame][2] = current[2] + positions[frame][2] - current[2]
    return result


def _unpredictable_positions(
    positions: Sequence[Sequence[float]],
    failure_frame: int,
    config: ScenarioConfig,
    rng: random.Random,
) -> list[list[float]]:
    result = [list(position) for position in positions]
    current = list(result[failure_frame][:3])
    maximum_step = 3.0 * config.unpredictable_step_std
    for frame in range(failure_frame + 1, len(result)):
        step = _bounded(
            [rng.gauss(0.0, config.unpredictable_step_std) for _ in range(3)],
            maximum_step,
        )
        current = [current[axis] + step[axis] for axis in range(3)]
        result[frame][:3] = current
    return result


def _occlusion_positions(
    positions: Sequence[Sequence[float]],
    first_hidden_frame: int,
    last_hidden_frame: int,
    config: ScenarioConfig,
    rng: random.Random,
) -> tuple[list[list[float]], list[float]]:
    result = [list(position) for position in positions]
    offset = _bounded(
        [rng.gauss(0.0, config.occlusion_reappearance_offset_std) for _ in range(3)],
        config.occlusion_reappearance_maximum_offset,
    )
    hidden_duration = max(1, last_hidden_frame - first_hidden_frame + 1)
    for frame in range(first_hidden_frame, len(result)):
        fraction = (
            _smoothstep((frame - first_hidden_frame + 1) / hidden_duration)
            if frame <= last_hidden_frame
            else 1.0
        )
        for axis in range(3):
            result[frame][axis] += fraction * offset[axis]
    return result, offset


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _milestone_intrusion_positions(
    frame_count: int,
    event_frame: int,
    config: ScenarioConfig,
    rng: random.Random,
) -> tuple[list[list[float]], list[list[float]], dict]:
    """Create a hidden between-updates intrusion into a frozen robot sweep point."""

    angle = math.radians(
        rng.uniform(
            -config.intrusion_direction_spread_degrees,
            config.intrusion_direction_spread_degrees,
        )
    )
    offset_norm = rng.uniform(
        config.intrusion_minimum_offset, config.intrusion_maximum_offset
    )
    offset = [0.0, offset_norm * math.cos(angle), offset_norm * math.sin(angle)]
    target = list(config.intrusion_target)
    nominal_center = [target[axis] + offset[axis] for axis in range(3)]
    nominal_pose = nominal_center + [0.0, 0.0, 0.0, 1.0]
    nominal = [list(nominal_pose) for _ in range(frame_count)]
    truth = [list(nominal_pose) for _ in range(frame_count)]
    collision_frame = min(
        frame_count - 1, event_frame + config.intrusion_approach_frames
    )
    hold_until = min(
        frame_count - 1, collision_frame + config.intrusion_hold_frames - 1
    )
    return_until = min(
        frame_count - 1, hold_until + config.intrusion_approach_frames
    )
    approach_duration = max(1, collision_frame - event_frame)
    for frame in range(event_frame + 1, collision_frame + 1):
        fraction = _smoothstep((frame - event_frame) / approach_duration)
        truth[frame][:3] = [
            nominal_center[axis]
            + fraction * (target[axis] - nominal_center[axis])
            for axis in range(3)
        ]
    for frame in range(collision_frame, hold_until + 1):
        truth[frame][:3] = target
    return_duration = max(1, return_until - hold_until)
    for frame in range(hold_until + 1, return_until + 1):
        fraction = _smoothstep((frame - hold_until) / return_duration)
        truth[frame][:3] = [
            target[axis] + fraction * (nominal_center[axis] - target[axis])
            for axis in range(3)
        ]
    metadata = {
        "target": target,
        "nominal_center": nominal_center,
        "offset": offset,
        "offset_norm": offset_norm,
        "collision_frame": collision_frame,
        "hold_until_frame": hold_until,
        "return_until_frame": return_until,
        "reference_link": "link4",
        "reference_path_time_seconds": 0.6366385484641094,
    }
    return truth, nominal, metadata


def _temporal_gate_positions(
    frame_count: int,
    event_frame: int,
    config: ScenarioConfig,
    rng: random.Random,
) -> tuple[list[list[float]], dict]:
    angle = rng.uniform(-math.pi / 6.0, math.pi / 6.0)
    offset_norm = rng.uniform(
        config.temporal_gate_minimum_offset,
        config.temporal_gate_maximum_offset,
    )
    target = list(config.temporal_gate_target)
    open_center = [
        target[0],
        target[1] + offset_norm * math.cos(angle),
        target[2] + offset_norm * math.sin(angle),
    ]
    open_pose = open_center + [0.0, 0.0, 0.0, 1.0]
    positions = [list(open_pose) for _ in range(frame_count)]
    closed_from = min(
        frame_count - 1,
        event_frame + config.temporal_gate_approach_frames,
    )
    closed_until = min(
        frame_count - 1,
        closed_from + config.temporal_gate_hold_frames - 1,
    )
    open_after = min(
        frame_count - 1,
        closed_until + config.temporal_gate_departure_frames,
    )
    approach_duration = max(1, closed_from - event_frame)
    for frame in range(event_frame + 1, closed_from + 1):
        fraction = _smoothstep((frame - event_frame) / approach_duration)
        positions[frame][:3] = [
            open_center[axis] + fraction * (target[axis] - open_center[axis])
            for axis in range(3)
        ]
    for frame in range(closed_from, closed_until + 1):
        positions[frame][:3] = target
    departure_duration = max(1, open_after - closed_until)
    for frame in range(closed_until + 1, open_after + 1):
        fraction = _smoothstep((frame - closed_until) / departure_duration)
        positions[frame][:3] = [
            target[axis] + fraction * (open_center[axis] - target[axis])
            for axis in range(3)
        ]
    return positions, {
        "target": target,
        "open_center": open_center,
        "offset_norm": offset_norm,
        "closed_from_frame": closed_from,
        "closed_until_frame": closed_until,
        "open_after_frame": open_after,
        "reference_link": "link4",
        "reference_path_time_seconds": 0.6366385484641094,
    }


def build_scenario_truth(base_scene: dict, config: ScenarioConfig) -> dict:
    """Return a truth scene with one deterministic uncertainty event family."""

    config.validate()
    try:
        frame_count = int(base_scene["frame_count"])
        obstacles = base_scene["obstacles"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("scene must contain frame_count and obstacles") from error
    if frame_count <= 0 or not isinstance(obstacles, list):
        raise ValueError("scene frame_count must be positive and obstacles must be an array")

    scene = copy.deepcopy(base_scene)
    event_frame = _event_frame(frame_count, config.event_frame_ratio)
    dynamic_entries: list[tuple[int, dict, str]] = []
    used_identifiers: set[str] = set()
    for obstacle_index, obstacle in enumerate(scene["obstacles"]):
        if obstacle.get("type") not in {"dynamic_sphere", "dynamic_box"}:
            continue
        base_identifier = str(obstacle.get("name") or f"obstacle-{obstacle_index}")
        identifier = base_identifier
        if identifier in used_identifiers:
            identifier = f"{base_identifier}#{obstacle_index}"
        used_identifiers.add(identifier)
        dynamic_entries.append((obstacle_index, obstacle, identifier))

    obstacle_events: dict[str, dict] = {}
    affected_identifiers: set[str] = set()
    if config.kind in STANDARD_SCENARIO_KINDS and dynamic_entries:
        ordered_identifiers = sorted(
            (identifier for _, _, identifier in dynamic_entries),
            key=lambda identifier: (_obstacle_seed(config.seed, identifier), identifier),
        )
        affected_count = (
            len(ordered_identifiers)
            if config.kind == "complete_failure"
            else max(
                1,
                min(
                    len(ordered_identifiers),
                    int(
                        math.floor(
                            len(ordered_identifiers)
                            * config.affected_obstacle_fraction
                            + 0.5
                        )
                    ),
                ),
            )
        )
        affected_order = ordered_identifiers[:affected_count]
        affected_identifiers = set(affected_order)
        if config.kind in {"position_noise", "complete_failure"}:
            affected_frames = [event_frame] * affected_count
        else:
            affected_frames = _staggered_event_frames(
                frame_count,
                event_frame,
                affected_count,
                config.event_frame_jitter_fraction,
            )
        event_by_identifier = dict(zip(affected_order, affected_frames, strict=True))
        obstacle_events = {
            identifier: {
                "affected": identifier in affected_identifiers,
                "event_frame": event_by_identifier.get(identifier),
            }
            for _, _, identifier in dynamic_entries
        }

    metadata = {
        "schema_version": 1,
        "kind": config.kind,
        "seed": config.seed,
        "event_frame": event_frame,
        "config": asdict(config),
    }
    if config.kind in STANDARD_SCENARIO_KINDS:
        metadata["obstacle_events"] = obstacle_events
        metadata["dynamic_obstacle_count"] = len(dynamic_entries)
        metadata["affected_obstacle_count"] = len(affected_identifiers)

    if config.kind == "occlusion":
        metadata["occlusion_ranges"] = []
    if config.kind == "complete_failure":
        metadata["failure_frame"] = event_frame

    milestone_assigned = False
    temporal_gate_assigned = False

    identifier_by_index = {
        obstacle_index: identifier
        for obstacle_index, _, identifier in dynamic_entries
    }
    for obstacle_index, obstacle in enumerate(scene["obstacles"]):
        if obstacle.get("type") not in {"dynamic_sphere", "dynamic_box"}:
            continue
        positions = obstacle.get("positions")
        if not isinstance(positions, list) or len(positions) < frame_count:
            raise ValueError(
                f"dynamic obstacle {obstacle.get('name', '<unnamed>')} must cover every declared frame"
            )
        positions = positions[:frame_count]
        if config.kind == "milestone_intrusion" and not milestone_assigned:
            rng = random.Random(
                _obstacle_seed(config.seed, str(obstacle.get("name", "")))
            )
            truth_positions, nominal_positions, intrusion = (
                _milestone_intrusion_positions(
                    frame_count, event_frame, config, rng
                )
            )
            obstacle["radius"] = config.intrusion_sphere_radius
            obstacle["positions"] = truth_positions
            obstacle["_uncertainty_nominal_positions"] = nominal_positions
            metadata["intrusion"] = intrusion
            milestone_assigned = True
            continue
        if config.kind == "temporal_gate" and not temporal_gate_assigned:
            rng = random.Random(
                _obstacle_seed(config.seed, str(obstacle.get("name", "")))
            )
            gate_positions, gate = _temporal_gate_positions(
                frame_count, event_frame, config, rng
            )
            obstacle["radius"] = config.temporal_gate_sphere_radius
            obstacle["positions"] = gate_positions
            obstacle["_uncertainty_nominal_positions"] = [
                list(position) for position in gate_positions
            ]
            metadata["temporal_gate"] = gate
            temporal_gate_assigned = True
            continue
        obstacle["_uncertainty_nominal_positions"] = [
            list(position) for position in positions
        ]
        identifier = identifier_by_index[obstacle_index]
        affected = (
            identifier in affected_identifiers
            if config.kind in STANDARD_SCENARIO_KINDS
            else False
        )
        obstacle["_uncertainty_affected"] = affected
        obstacle_event_frame = obstacle_events.get(identifier, {}).get("event_frame")
        obstacle["_uncertainty_event_frame"] = obstacle_event_frame
        rng = random.Random(_obstacle_seed(config.seed, identifier))
        if not affected and config.kind in STANDARD_SCENARIO_KINDS:
            obstacle["positions"] = [list(position) for position in positions]
            if config.kind == "occlusion":
                obstacle["_uncertainty_occlusion_ranges"] = []
            continue
        if config.kind == "random_acceleration":
            obstacle["positions"] = _random_acceleration_positions(
                positions, int(obstacle_event_frame), config, rng
            )
        elif config.kind == "sudden_turn":
            obstacle["positions"] = _sudden_turn_positions(
                positions, int(obstacle_event_frame), config.turn_degrees
            )
        elif config.kind == "occlusion":
            duration = max(1, int(round(frame_count * config.occlusion_fraction)))
            last_hidden_frame = min(
                frame_count - 1,
                int(obstacle_event_frame) + duration - 1,
            )
            occlusion_range = [int(obstacle_event_frame), last_hidden_frame]
            obstacle["_uncertainty_occlusion_ranges"] = [occlusion_range]
            obstacle["positions"], offset = _occlusion_positions(
                positions,
                occlusion_range[0],
                occlusion_range[1],
                config,
                rng,
            )
            obstacle_events[identifier]["occlusion_range"] = occlusion_range
            obstacle_events[identifier]["reappearance_offset"] = offset
            metadata["occlusion_ranges"].append(occlusion_range)
        elif config.kind == "complete_failure":
            obstacle["positions"] = _unpredictable_positions(positions, event_frame, config, rng)
        else:
            obstacle["positions"] = [list(position) for position in positions]

    if config.kind == "milestone_intrusion" and not milestone_assigned:
        raise ValueError("milestone_intrusion requires at least one dynamic obstacle")
    if config.kind == "temporal_gate" and not temporal_gate_assigned:
        raise ValueError("temporal_gate requires at least one dynamic obstacle")

    scene["_uncertainty_scenario"] = metadata
    return scene


def generate_scenario(source_scene: Path, output_path: Path, config: ScenarioConfig) -> Path:
    source_scene = source_scene.resolve()
    base_scene = json.loads(source_scene.read_text(encoding="utf-8"))
    scenario = build_scenario_truth(base_scene, config)
    scenario["_uncertainty_scenario"]["source_scene"] = str(source_scene)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_scene", type=Path)
    parser.add_argument("output_scene", type=Path)
    parser.add_argument("--kind", choices=SCENARIO_KINDS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--event-frame-ratio", type=float, default=0.4)
    parser.add_argument("--acceleration-std", type=float, default=0.0006)
    parser.add_argument("--turn-degrees", type=float, default=90.0)
    parser.add_argument("--occlusion-fraction", type=float, default=0.2)
    parser.add_argument("--affected-obstacle-fraction", type=float, default=0.35)
    parser.add_argument("--event-frame-jitter-fraction", type=float, default=0.03)
    parser.add_argument("--occlusion-reappearance-offset-std", type=float, default=0.03)
    parser.add_argument(
        "--occlusion-reappearance-maximum-offset", type=float, default=0.10
    )
    parser.add_argument("--unpredictable-step-std", type=float, default=0.015)
    parser.add_argument("--intrusion-minimum-offset", type=float, default=0.16)
    parser.add_argument("--intrusion-maximum-offset", type=float, default=0.20)
    parser.add_argument("--intrusion-approach-frames", type=int, default=18)
    parser.add_argument("--intrusion-hold-frames", type=int, default=6)
    parser.add_argument("--intrusion-direction-spread-degrees", type=float, default=30.0)
    parser.add_argument("--intrusion-sphere-radius", type=float, default=0.075)
    parser.add_argument("--temporal-gate-minimum-offset", type=float, default=0.32)
    parser.add_argument("--temporal-gate-maximum-offset", type=float, default=0.40)
    parser.add_argument("--temporal-gate-approach-frames", type=int, default=12)
    parser.add_argument("--temporal-gate-hold-frames", type=int, default=60)
    parser.add_argument("--temporal-gate-departure-frames", type=int, default=12)
    parser.add_argument("--temporal-gate-sphere-radius", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = generate_scenario(
        args.source_scene,
        args.output_scene,
        ScenarioConfig(
            kind=args.kind,
            seed=args.seed,
            event_frame_ratio=args.event_frame_ratio,
            acceleration_std=args.acceleration_std,
            turn_degrees=args.turn_degrees,
            occlusion_fraction=args.occlusion_fraction,
            affected_obstacle_fraction=args.affected_obstacle_fraction,
            event_frame_jitter_fraction=args.event_frame_jitter_fraction,
            occlusion_reappearance_offset_std=args.occlusion_reappearance_offset_std,
            occlusion_reappearance_maximum_offset=(
                args.occlusion_reappearance_maximum_offset
            ),
            unpredictable_step_std=args.unpredictable_step_std,
            intrusion_minimum_offset=args.intrusion_minimum_offset,
            intrusion_maximum_offset=args.intrusion_maximum_offset,
            intrusion_approach_frames=args.intrusion_approach_frames,
            intrusion_hold_frames=args.intrusion_hold_frames,
            intrusion_direction_spread_degrees=args.intrusion_direction_spread_degrees,
            intrusion_sphere_radius=args.intrusion_sphere_radius,
            temporal_gate_minimum_offset=args.temporal_gate_minimum_offset,
            temporal_gate_maximum_offset=args.temporal_gate_maximum_offset,
            temporal_gate_approach_frames=args.temporal_gate_approach_frames,
            temporal_gate_hold_frames=args.temporal_gate_hold_frames,
            temporal_gate_departure_frames=args.temporal_gate_departure_frames,
            temporal_gate_sphere_radius=args.temporal_gate_sphere_radius,
        ),
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
