#!/usr/bin/env bash
set -u

MAX_ATTEMPTS="${SIRRT_GIT_ATTEMPTS:-5}"
RETRY_DELAY_SECONDS="${SIRRT_GIT_RETRY_DELAY_SECONDS:-2}"
CONNECT_TIMEOUT_SECONDS="${SIRRT_CONNECT_TIMEOUT_SECONDS:-20}"
TRANSFER_TIMEOUT_SECONDS="${SIRRT_TRANSFER_TIMEOUT_SECONDS:-600}"
ARCHIVE_CACHE_ROOT="${SIRRT_ARCHIVE_CACHE_ROOT:-/var/cache/sirrt-archives}"

if [[ ! "$MAX_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SIRRT_GIT_ATTEMPTS must be a positive integer\n' >&2
  exit 2
fi

if [[ ! "$RETRY_DELAY_SECONDS" =~ ^[0-9]+$ ]]; then
  printf 'SIRRT_GIT_RETRY_DELAY_SECONDS must be a non-negative integer\n' >&2
  exit 2
fi

if [[ ! "$CONNECT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SIRRT_CONNECT_TIMEOUT_SECONDS must be a positive integer\n' >&2
  exit 2
fi

if [[ ! "$TRANSFER_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'SIRRT_TRANSFER_TIMEOUT_SECONDS must be a positive integer\n' >&2
  exit 2
fi

is_pinned_fetch() {
  [[ "$#" -eq 7 ]] \
    && [[ "$1" == "-C" ]] \
    && [[ "$3" == "fetch" ]] \
    && [[ "$4" == "--depth" ]] \
    && [[ "$5" == "1" ]] \
    && [[ "$6" == "origin" ]] \
    && [[ "$7" =~ ^[0-9a-f]{40}$ ]]
}

download_pinned_archive() {
  local dst="$2"
  local ref="$7"
  local remote
  local repo_path
  local cache_dir
  local archive
  local download
  local archive_url
  local expected_sha256=""
  local actual_sha256
  local -a curl_args

  remote="$(/usr/bin/git -C "$dst" remote get-url origin)" || return 1
  case "$remote" in
    https://github.com/*.git)
      repo_path="${remote#https://github.com/}"
      repo_path="${repo_path%.git}"
      ;;
    *)
      return 1
      ;;
  esac

  archive_url="https://codeload.github.com/${repo_path}/tar.gz/${ref}"
  case "${repo_path}@${ref}" in
    coal-library/coal@ff01445364a7b386e8e5c46ccd530ed9d9a8e8cf)
      expected_sha256="d730bf4ca2dffac51c37f12f3ab3e672d4aa1d0337b9c564939878b29954ca86"
      ;;
    jrl-umi3218/jrl-cmakemodules@e2df2bb1552848e7660d992cdc6102f70ff8c1dc)
      expected_sha256="0aa5569cfd8ce1c2f88a26f40c5d36c1c0dbb0af6da6262213a9d1f1c497ca36"
      ;;
    Tencent/rapidjson@24b5e7a8b27f42fa16b96fc70aade9106cf7102f)
      expected_sha256="2d2601a82d2d3b7e143a3c8d43ef616671391034bc46891a9816b79cf2d3e7a8"
      ;;
    jlblancoc/nanoflann@ba47cfcb127c3597d69196d87f5aa9ca8811b0a9)
      expected_sha256="f7db169b16d170f1f1d37606169e20467ac7ac23fb1d7d56f2a373dbe11947a9"
      ;;
    ompl/ompl@96b7b9c2c62f80570b5b93e03a95d56dea8410d9)
      archive_url="https://github.com/ompl/ompl/releases/download/2.0.0/ompl-2.0.0-Source.tar.gz"
      expected_sha256="7b6032ca69d0c69280ce3872737c1c6ff6105cfa38769befb57dd15610c90f75"
      ;;
    *)
      printf 'No checksum is pinned for %s@%s\n' "$repo_path" "$ref" >&2
      return 1
      ;;
  esac

  cache_dir="${ARCHIVE_CACHE_ROOT}/${repo_path}"
  archive="${cache_dir}/${ref}.tar.gz"
  download="${archive}.partial"
  mkdir -p "$cache_dir" || return 1

  if [[ -f "$archive" ]]; then
    actual_sha256="$(sha256sum "$archive" | cut -d ' ' -f 1)"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
      printf 'Checksum mismatch for cached %s@%s\n' "$repo_path" "$ref" >&2
      rm -f "$archive"
    elif ! tar -tzf "$archive" >/dev/null 2>&1; then
      printf 'Invalid cached archive for %s@%s\n' "$repo_path" "$ref" >&2
      rm -f "$archive"
    fi
  fi

  if [[ ! -f "$archive" ]]; then
    rm -f "$archive"
    printf 'Downloading pinned source archive %s@%s\n' "$repo_path" "$ref"

    curl_args=(-fsSL --http1.1 \
      --retry "$((MAX_ATTEMPTS - 1))" \
      --retry-delay "$RETRY_DELAY_SECONDS" \
      --retry-all-errors \
      --connect-timeout "$CONNECT_TIMEOUT_SECONDS" \
      --max-time "$TRANSFER_TIMEOUT_SECONDS")
    if ! curl "${curl_args[@]}" --continue-at - -o "$download" "$archive_url"; then
      rm -f "$download"
      printf 'Resumed download failed; retrying %s@%s from byte zero\n' \
        "$repo_path" "$ref" >&2
      if ! curl "${curl_args[@]}" -o "$download" "$archive_url"; then
        rm -f "$download"
        return 1
      fi
    fi

    actual_sha256="$(sha256sum "$download" | cut -d ' ' -f 1)"
    if [[ "$actual_sha256" != "$expected_sha256" ]]; then
      printf 'Checksum mismatch for %s@%s\n' "$repo_path" "$ref" >&2
      rm -f "$download"
      return 1
    fi

    if ! tar -tzf "$download" >/dev/null 2>&1; then
      rm -f "$download"
      return 1
    fi

    mv "$download" "$archive"
  else
    printf 'Using cached source archive %s@%s\n' "$repo_path" "$ref"
  fi

  if ! tar -xzf "$archive" --strip-components=1 -C "$dst"; then
    return 1
  fi

  printf '%s\n' "$ref" > "$dst/.git/sirrt-codeload-ref"
}

if is_pinned_fetch "$@"; then
  download_pinned_archive "$@"
  exit $?
fi

if [[ "$#" -eq 5 ]] \
  && [[ "$1" == "-C" ]] \
  && [[ "$3" == "checkout" ]] \
  && [[ "$4" == "--detach" ]] \
  && [[ "$5" == "FETCH_HEAD" ]] \
  && [[ -f "$2/.git/sirrt-codeload-ref" ]]; then
  exit 0
fi

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  /usr/bin/git -c http.version=HTTP/1.1 "$@"
  status=$?
  if ((status == 0)); then
    exit 0
  fi

  if ((attempt == MAX_ATTEMPTS)); then
    exit "$status"
  fi

  printf 'git command failed (attempt %d/%d); retrying...\n' \
    "$attempt" "$MAX_ATTEMPTS" >&2
  sleep "$((attempt * RETRY_DELAY_SECONDS))"
done
