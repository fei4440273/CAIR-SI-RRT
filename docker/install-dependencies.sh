#!/usr/bin/env bash
set -euo pipefail

proxy_host="${SIRRT_DOCKER_PROXY_HOST:-host.docker.internal}"

read_secret() {
  local name="$1"
  local path="/run/secrets/${name}"

  if [[ -f "$path" ]]; then
    cat "$path"
  fi
}

rewrite_proxy_value() {
  local value="$1"

  python3 - "$value" "$proxy_host" <<'PY'
import sys
from urllib.parse import urlsplit, urlunsplit

value, replacement = sys.argv[1:]
parsed = urlsplit(value)
if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
    sys.stdout.write(value)
    raise SystemExit

userinfo, separator, _ = parsed.netloc.rpartition("@")
prefix = f"{userinfo}{separator}" if separator else ""
replacement_host = f"[{replacement}]" if ":" in replacement else replacement
port = f":{parsed.port}" if parsed.port is not None else ""
sys.stdout.write(
    urlunsplit((parsed.scheme, f"{prefix}{replacement_host}{port}", parsed.path,
                parsed.query, parsed.fragment))
)
PY
}

bound_dependency_jobs() {
  local detected_jobs
  local jobs

  detected_jobs="$(nproc)"
  jobs="${SIRRT_JOBS:-$detected_jobs}"
  [[ "$jobs" =~ ^[1-9][0-9]*$ ]] || {
    printf 'SIRRT_JOBS must be a positive integer\n' >&2
    return 2
  }
  ((jobs <= 32)) || jobs=32
  export SIRRT_JOBS="$jobs"
}

verify_coal_qhull_linkage() {
  local dependencies_root="${DEPS_ROOT:-/opt/SIRRT_dependencies}"
  local variant
  local library
  local dynamic_section

  for variant in release debug; do
    library="${dependencies_root}/${variant}/lib/libcoal.so"
    [[ -f "$library" ]] || {
      printf 'Coal library not found: %s\n' "$library" >&2
      return 1
    }
    dynamic_section="$(readelf -d "$library")" || {
      printf 'Cannot inspect Coal library: %s\n' "$library" >&2
      return 1
    }
    grep -q 'Shared library: \[libqhullcpp' <<<"$dynamic_section" || {
      printf 'Coal does not link libqhullcpp: %s\n' "$library" >&2
      return 1
    }
    grep -q 'Shared library: \[libqhull_r' <<<"$dynamic_section" || {
      printf 'Coal does not link libqhull_r: %s\n' "$library" >&2
      return 1
    }
  done
}

main() {
  local http_proxy_value
  local https_proxy_value
  local no_proxy_value

  bound_dependency_jobs
  http_proxy_value="$(rewrite_proxy_value "$(read_secret sirrt_build_http_proxy)")"
  https_proxy_value="$(rewrite_proxy_value "$(read_secret sirrt_build_https_proxy)")"
  no_proxy_value="$(read_secret sirrt_build_no_proxy)"

  export HTTP_PROXY="$http_proxy_value"
  export HTTPS_PROXY="$https_proxy_value"
  export NO_PROXY="$no_proxy_value"
  export http_proxy="$http_proxy_value"
  export https_proxy="$https_proxy_value"
  export no_proxy="$no_proxy_value"

  bash /usr/local/src/sirrt/install.sh
  verify_coal_qhull_linkage
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
