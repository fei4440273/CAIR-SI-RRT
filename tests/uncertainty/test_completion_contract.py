import hashlib
import json
from pathlib import Path

import pytest

from scripts.uncertainty.completion_contract import (
    load_protocol,
    planner_contract_digest,
    validate_result_identity,
    validate_seed_registry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_valid_result_fixture(tmp_path: Path) -> tuple[Path, dict, Path]:
    result_path = tmp_path / "adaptive_tube_repair.json"
    result_path.write_text('{"task_completed": true}\n', encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    identity = {
        "phase": "final",
        "scenario": "sudden_turn",
        "method": "adaptive_tube_repair",
        "policy": "optimized",
        "seed": 1101,
        "trial_fingerprint": "trial-1101",
        "source_state_digest": "1" * 64,
        "planner_binary_sha256": "2" * 64,
        "selection_sha256": "3" * 64,
        "truth_scene_sha256": "4" * 64,
        "completion_protocol_sha256": "5" * 64,
        "seed_registry_sha256": "6" * 64,
    }
    manifest_path.write_text(
        json.dumps(
            {
                "phase": identity["phase"],
                "scenario": identity["scenario"],
                "policy": identity["policy"],
                "seed": identity["seed"],
                "methods": [identity["method"]],
                "optimization_trial_fingerprint": identity["trial_fingerprint"],
                "source_state_digest": identity["source_state_digest"],
                "planner_binary_sha256": identity["planner_binary_sha256"],
                "selection_sha256": identity["selection_sha256"],
                "truth_scene_sha256": identity["truth_scene_sha256"],
                "completion_protocol_sha256": identity[
                    "completion_protocol_sha256"
                ],
                "seed_registry_sha256": identity["seed_registry_sha256"],
                "results": {identity["method"]: result_path.name},
                "result_sha256": {identity["method"]: _sha256(result_path)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path, identity, result_path


def test_completion_seed_registry_rejects_cross_split_overlap() -> None:
    with pytest.raises(ValueError, match="seed overlap"):
        validate_seed_registry({"sudden_dev": [1001], "sudden_final": [1001]})


def test_completion_seed_registry_expands_inclusive_ranges() -> None:
    assert validate_seed_registry(
        {"developer": {"start": 1001, "stop": 1003}, "final": [1101, 1102]}
    ) == {"developer": (1001, 1002, 1003), "final": (1101, 1102)}


def test_protocol_loader_normalizes_and_validates_registry(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method_scope": [
                    "adaptive_tube_repair",
                    "adaptive_tube_full_replan",
                ],
                "deadline_ms": 100.0,
                "seed_registry": {
                    "developer": {"start": 1001, "stop": 1002},
                    "final": [1101],
                },
                "sudden_turn_budget_reuse_candidates": [0.0, 0.5, 1.0],
                "conformal_alpha": 0.05,
            }
        ),
        encoding="utf-8",
    )
    protocol = load_protocol(path)
    assert protocol["seed_registry"] == {
        "developer": (1001, 1002),
        "final": (1101,),
    }


def test_result_identity_accepts_complete_matching_provenance(tmp_path: Path) -> None:
    manifest, expected, result_path = write_valid_result_fixture(tmp_path)
    assert validate_result_identity(manifest, expected) == result_path


def test_result_identity_rejects_changed_result_bytes(tmp_path: Path) -> None:
    manifest, expected, result_path = write_valid_result_fixture(tmp_path)
    result_path.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="result SHA-256"):
        validate_result_identity(manifest, expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("phase", "developer", "phase"),
        ("seed", 999, "seed"),
        ("planner_binary_sha256", "0" * 64, "planner binary SHA-256"),
    ),
)
def test_result_identity_rejects_wrong_phase_seed_and_binary(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    manifest, expected, _ = write_valid_result_fixture(tmp_path)
    altered = dict(expected)
    altered[field] = value
    with pytest.raises(ValueError, match=message):
        validate_result_identity(manifest, altered)


def test_result_identity_maps_container_result_path_to_repository(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    run_directory = repository / "runs" / "trial"
    run_directory.mkdir(parents=True)
    manifest, expected, result_path = write_valid_result_fixture(run_directory)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["results"]["adaptive_tube_repair"] = "/app/runs/trial/" + result_path.name
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    expected["repo_root"] = repository
    assert validate_result_identity(manifest, expected) == result_path


def test_planner_contract_digest_depends_only_on_declared_runtime_inputs(
    tmp_path: Path,
) -> None:
    (tmp_path / "MSIRRT").mkdir()
    (tmp_path / "MSIRRT" / "planner.cpp").write_text("first", encoding="utf-8")
    runtime = tmp_path / "scripts" / "uncertainty"
    runtime.mkdir(parents=True)
    (runtime / "run_pilot.py").write_text("runtime", encoding="utf-8")
    report = runtime / "generate_report.py"
    report.write_text("report-v1", encoding="utf-8")

    first = planner_contract_digest(
        tmp_path, paths=(Path("MSIRRT"), Path("scripts/uncertainty/run_pilot.py"))
    )
    report.write_text("report-v2", encoding="utf-8")
    assert planner_contract_digest(
        tmp_path, paths=(Path("MSIRRT"), Path("scripts/uncertainty/run_pilot.py"))
    ) == first

    (tmp_path / "MSIRRT" / "planner.cpp").write_text("second", encoding="utf-8")
    assert planner_contract_digest(
        tmp_path, paths=(Path("MSIRRT"), Path("scripts/uncertainty/run_pilot.py"))
    ) != first
