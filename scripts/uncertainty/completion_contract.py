#!/usr/bin/env python3
"""Frozen protocol and artifact-identity checks for phase-one completion."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


REQUIRED_METHOD_SCOPE = (
    "adaptive_tube_repair",
    "adaptive_tube_full_replan",
)
PROVENANCE_ROOTS = ("MSIRRT", "scripts", "tests")
PLANNER_CONTRACT_PATHS = (
    Path("MSIRRT"),
    Path("scripts/uncertainty/calibrate_tubes.py"),
    Path("scripts/uncertainty/completion_contract.py"),
    Path("scripts/uncertainty/conformal_tubes.py"),
    Path("scripts/uncertainty/generate_trial.py"),
    Path("scripts/uncertainty/run_adaptive_optimization.py"),
    Path("scripts/uncertainty/run_phase_one_completion.py"),
    Path("scripts/uncertainty/run_pilot.py"),
    Path("scripts/uncertainty/scenarios.py"),
    Path("docs/uncertainty_research/phase_one_completion_protocol_20260828.json"),
)
REQUIRED_IDENTITY_FIELDS = {
    "phase": ("phase", "phase"),
    "scenario": ("scenario", "scenario"),
    "policy": ("policy", "policy"),
    "seed": ("seed", "seed"),
    "trial_fingerprint": (
        "optimization_trial_fingerprint",
        "trial fingerprint",
    ),
    "source_state_digest": ("source_state_digest", "source state digest"),
    "planner_binary_sha256": (
        "planner_binary_sha256",
        "planner binary SHA-256",
    ),
    "selection_sha256": ("selection_sha256", "selection SHA-256"),
    "truth_scene_sha256": ("truth_scene_sha256", "truth scene SHA-256"),
    "completion_protocol_sha256": (
        "completion_protocol_sha256",
        "completion protocol SHA-256",
    ),
    "seed_registry_sha256": (
        "seed_registry_sha256",
        "seed registry SHA-256",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_state_digest(repo_root: Path) -> str:
    """Hash tracked patches and untracked source bytes under provenance roots."""
    root = repo_root.resolve()
    digest = hashlib.sha256()
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", *PROVENANCE_ROOTS],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    digest.update(b"tracked-diff\0")
    digest.update(diff)
    untracked_output = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            *PROVENANCE_ROOTS,
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    for relative in sorted(filter(None, untracked_output.splitlines())):
        path = root / relative
        digest.update(b"untracked\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def planner_contract_digest(
    repo_root: Path,
    *,
    paths: Iterable[Path] = PLANNER_CONTRACT_PATHS,
) -> str:
    """Hash only source/config bytes that can change a completion trial."""

    root = repo_root.resolve()
    files: list[Path] = []
    for relative in paths:
        candidate = root / relative
        if candidate.is_dir():
            files.extend(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and not path.name.endswith((".pyc", ".pyo"))
            )
        elif candidate.is_file():
            files.append(candidate)
        else:
            raise FileNotFoundError(f"planner contract path is missing: {candidate}")
    unique = sorted(set(files), key=lambda path: str(path.relative_to(root)))
    if not unique:
        raise ValueError("planner contract has no files")
    digest = hashlib.sha256()
    for path in unique:
        relative = str(path.relative_to(root))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _expand_seed_values(name: str, values: Any) -> tuple[int, ...]:
    if isinstance(values, Mapping):
        if set(values) != {"start", "stop"}:
            raise ValueError(f"seed range {name} must contain only start and stop")
        start = values["start"]
        stop = values["stop"]
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError(f"seed range {name} start must be an integer")
        if isinstance(stop, bool) or not isinstance(stop, int):
            raise ValueError(f"seed range {name} stop must be an integer")
        if stop < start:
            raise ValueError(f"seed range {name} stop must be at least start")
        return tuple(range(start, stop + 1))
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"seed registry entry {name} must be a sequence or range")
    expanded = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in expanded):
        raise ValueError(f"seed registry entry {name} must contain integers")
    return expanded


def validate_seed_registry(
    registry: Mapping[str, Iterable[int] | Mapping[str, int]],
) -> dict[str, tuple[int, ...]]:
    if not isinstance(registry, Mapping) or not registry:
        raise ValueError("seed registry must be a nonempty mapping")
    normalized: dict[str, tuple[int, ...]] = {}
    owners: dict[int, str] = {}
    for name, raw_values in registry.items():
        if not isinstance(name, str) or not name:
            raise ValueError("seed registry names must be nonempty strings")
        values = _expand_seed_values(name, raw_values)
        if not values:
            raise ValueError(f"seed registry entry {name} must be nonempty")
        if len(values) != len(set(values)):
            raise ValueError(f"seed overlap within {name}")
        for seed in values:
            previous = owners.get(seed)
            if previous is not None:
                raise ValueError(
                    f"seed overlap between {previous} and {name}: {seed}"
                )
            owners[seed] = name
        normalized[name] = values
    return normalized


def seed_registry_sha256(registry: Mapping[str, Iterable[int]]) -> str:
    normalized = {name: list(values) for name, values in sorted(registry.items())}
    encoded = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("completion protocol must be a JSON object")
    if protocol.get("schema_version") != 1:
        raise ValueError("completion protocol schema_version must be 1")
    if tuple(protocol.get("method_scope", ())) != REQUIRED_METHOD_SCOPE:
        raise ValueError(
            "completion protocol method_scope must contain only the two internal "
            "Adaptive-repair variants"
        )
    if protocol.get("deadline_ms") != 100.0:
        raise ValueError("completion protocol deadline_ms must be 100.0")
    if protocol.get("sudden_turn_budget_reuse_candidates") != [0.0, 0.5, 1.0]:
        raise ValueError(
            "completion protocol sudden_turn_budget_reuse_candidates must be "
            "[0.0, 0.5, 1.0]"
        )
    if protocol.get("conformal_alpha") != 0.05:
        raise ValueError("completion protocol conformal_alpha must be 0.05")
    protocol["seed_registry"] = validate_seed_registry(
        protocol.get("seed_registry", {})
    )
    return protocol


def _repository_root(manifest_path: Path, expected: Mapping[str, Any]) -> Path:
    configured = expected.get("repo_root")
    if configured is not None:
        return Path(configured).resolve()
    for parent in (manifest_path.parent, *manifest_path.parents):
        if (parent / ".git").exists():
            return parent.resolve()
    raise ValueError("repo root is required to resolve an /app result path")


def _resolve_result_path(
    reference: str, manifest_path: Path, expected: Mapping[str, Any]
) -> Path:
    candidate = Path(reference)
    if reference == "/app" or reference.startswith("/app/"):
        relative = Path(reference).relative_to("/app")
        return (_repository_root(manifest_path, expected) / relative).resolve()
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def validate_result_identity(
    run_manifest_path: Path, expected: Mapping[str, Any]
) -> Path:
    manifest_path = run_manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("run manifest must be a JSON object")
    missing = sorted(
        field for field in REQUIRED_IDENTITY_FIELDS if field not in expected
    )
    if "method" not in expected:
        missing.append("method")
    if missing:
        raise ValueError("expected identity is missing: " + ", ".join(missing))

    for expected_field, (manifest_field, label) in REQUIRED_IDENTITY_FIELDS.items():
        if manifest.get(manifest_field) != expected[expected_field]:
            raise ValueError(
                f"{label} mismatch: expected {expected[expected_field]!r}, "
                f"found {manifest.get(manifest_field)!r}"
            )

    method = expected["method"]
    if manifest.get("methods") != [method]:
        raise ValueError(
            f"method mismatch: expected [{method!r}], found {manifest.get('methods')!r}"
        )
    result_reference = manifest.get("results", {}).get(method)
    if not isinstance(result_reference, str) or not result_reference:
        raise ValueError(f"result path is missing for method {method}")
    result_path = _resolve_result_path(result_reference, manifest_path, expected)
    if not result_path.is_file():
        raise ValueError(f"result path does not exist: {result_path}")
    recorded_hash = manifest.get("result_sha256", {}).get(method)
    actual_hash = sha256_file(result_path)
    if recorded_hash != actual_hash:
        raise ValueError(
            f"result SHA-256 mismatch: recorded {recorded_hash!r}, actual {actual_hash}"
        )
    return result_path
