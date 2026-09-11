#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
smoke_root="${1:-/tmp/cair-si-rrt-smoke}"
planner="$repo_root/build/MSIRRT/MSIRRT_RepairSequence"

test -x "$planner" || {
    echo "Planner is missing; run ./scripts/build_and_test.sh first" >&2
    exit 2
}

mkdir -p "$smoke_root"
PYTHONPATH="$repo_root" python3 "$repo_root/scripts/uncertainty/scenarios.py" \
    "$repo_root/examples/one_sphere/scene_task.json" \
    "$smoke_root/truth.json" \
    --kind position_noise \
    --seed 4201 \
    --event-frame-ratio 0.15

export MSIRRT_ADAPTIVE_REPAIR_POLICY=optimized
export MSIRRT_CONDITIONAL_SAMPLING=1
export MSIRRT_REACHABLE_ELLIPSOID_SAMPLING=1
export MSIRRT_FEASIBILITY_AWARE_HORIZON=1
export MSIRRT_REUSE_STORED_PREVIOUS_INTERVALS=1
export MSIRRT_REUSE_FIRST_PREDICTION_SCENE=1
export MSIRRT_REUSE_PARSED_PREDICTION_METADATA=1
export MSIRRT_COOPERATIVE_DEADLINE_CHECKS=1
export MSIRRT_MAX_PLANNING_TIME=0.1

PYTHONPATH="$repo_root" python3 "$repo_root/scripts/uncertainty/run_pilot.py" \
    "$smoke_root/truth.json" \
    "$smoke_root/pilot" \
    --planner-binary "$planner" \
    --seed 4201 \
    --methods adaptive_tube_repair \
    --update-stride 15 \
    --initial-issue-frame 15 \
    --max-updates 1 \
    --rolling \
    --deadline-ms 100

PYTHONPATH="$repo_root" python3 - "$smoke_root/pilot/run_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
relative_result = manifest["results"]["adaptive_tube_repair"]
result_path = manifest_path.parent / relative_result
result = json.loads(result_path.read_text(encoding="utf-8"))
required = {"task_completed", "updates", "termination_reason"}
missing = sorted(required - result.keys())
if missing:
    raise SystemExit(f"smoke result is missing fields: {missing}")
if not isinstance(result["updates"], list) or not result["updates"]:
    raise SystemExit("smoke result has no update records")
print(
    json.dumps(
        {
            "result": str(result_path),
            "task_completed": result["task_completed"],
            "updates": len(result["updates"]),
            "termination_reason": result["termination_reason"],
        },
        ensure_ascii=False,
    )
)
PY
