#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -r /etc/profile.d/sirrt_dependencies.sh ]]; then
    set +u
    source /etc/profile.d/sirrt_dependencies.sh
    sirrt_use_release
    set -u
fi

max_jobs=32
detected_jobs="$(nproc)"
jobs="${CAIR_BUILD_JOBS:-$detected_jobs}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || {
    echo "CAIR_BUILD_JOBS must be a positive integer" >&2
    exit 2
}
((jobs <= max_jobs)) || jobs="$max_jobs"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build --parallel "$jobs"
ctest --test-dir build --output-on-failure

