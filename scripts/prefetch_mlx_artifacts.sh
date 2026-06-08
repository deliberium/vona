#!/usr/bin/env bash
set -euo pipefail

MLX_REF="${MLX_REF:-v0.25.1}"
MLX_CACHE_DIR="${MLX_CACHE_DIR:-resources/vendor/mlx}"
MLX_REPO_URL="${MLX_REPO_URL:-https://github.com/ml-explore/mlx.git}"
METAL_CPP_CACHE_DIR="${METAL_CPP_CACHE_DIR:-resources/vendor/metal-cpp}"
METAL_CPP_URL="${METAL_CPP_URL:-https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip}"
METAL_CPP_FORCE="${METAL_CPP_FORCE:-0}"
JSON_CACHE_DIR="${JSON_CACHE_DIR:-resources/vendor/nlohmann-json}"
JSON_REF="${JSON_REF:-v3.11.3}"
JSON_REPO_URL="${JSON_REPO_URL:-https://github.com/nlohmann/json.git}"
FMT_CACHE_DIR="${FMT_CACHE_DIR:-resources/vendor/fmt}"
FMT_REF="${FMT_REF:-12.1.0}"
FMT_REPO_URL="${FMT_REPO_URL:-https://github.com/fmtlib/fmt.git}"
GGUF_CACHE_DIR="${GGUF_CACHE_DIR:-resources/vendor/gguf-tools}"
GGUF_REF="${GGUF_REF:-main}"
GGUF_REPO_URL="${GGUF_REPO_URL:-https://github.com/antirez/gguf-tools.git}"

mkdir -p "$(dirname "$MLX_CACHE_DIR")"

if [ -d "$MLX_CACHE_DIR/.git" ]; then
  git -C "$MLX_CACHE_DIR" fetch --tags --prune origin
  git -C "$MLX_CACHE_DIR" checkout "$MLX_REF"
  git -C "$MLX_CACHE_DIR" submodule update --init --recursive
else
  git clone --recursive "$MLX_REPO_URL" "$MLX_CACHE_DIR"
  git -C "$MLX_CACHE_DIR" checkout "$MLX_REF"
  git -C "$MLX_CACHE_DIR" submodule update --init --recursive
fi

if [ "$METAL_CPP_FORCE" = "1" ]; then
  rm -rf "$METAL_CPP_CACHE_DIR"
fi

if [ ! -d "$METAL_CPP_CACHE_DIR/Metal" ] && [ ! -f "$METAL_CPP_CACHE_DIR/SingleHeader/Metal.hpp" ]; then
  tmp_zip="$(mktemp -t metal-cpp.XXXXXX.zip)"
  tmp_dir="$(mktemp -d -t metal-cpp.XXXXXX)"
  curl -L "$METAL_CPP_URL" -o "$tmp_zip"
  unzip -q "$tmp_zip" -d "$tmp_dir"
  rm -rf "$METAL_CPP_CACHE_DIR"
  mkdir -p "$METAL_CPP_CACHE_DIR"
  found_header_dir="$(find "$tmp_dir" -type d \( -name Metal -o -name SingleHeader \) -print -quit)"
  if [ -z "$found_header_dir" ]; then
    echo "Downloaded Metal C++ archive did not contain expected headers" >&2
    exit 1
  fi
  found_dir="$(dirname "$found_header_dir")"
  cp -R "$found_dir"/. "$METAL_CPP_CACHE_DIR"/
  rm -f "$tmp_zip"
  rm -rf "$tmp_dir"
fi

if [ -d "$JSON_CACHE_DIR/.git" ]; then
  git -C "$JSON_CACHE_DIR" fetch --tags --prune origin
  git -C "$JSON_CACHE_DIR" checkout "$JSON_REF"
else
  git clone "$JSON_REPO_URL" "$JSON_CACHE_DIR"
  git -C "$JSON_CACHE_DIR" checkout "$JSON_REF"
fi

if [ -d "$FMT_CACHE_DIR/.git" ]; then
  git -C "$FMT_CACHE_DIR" fetch --tags --prune origin
  git -C "$FMT_CACHE_DIR" checkout "$FMT_REF"
else
  git clone "$FMT_REPO_URL" "$FMT_CACHE_DIR"
  git -C "$FMT_CACHE_DIR" checkout "$FMT_REF"
fi

if [ -d "$GGUF_CACHE_DIR/.git" ]; then
  git -C "$GGUF_CACHE_DIR" fetch --tags --prune origin
  git -C "$GGUF_CACHE_DIR" checkout "$GGUF_REF"
else
  git clone "$GGUF_REPO_URL" "$GGUF_CACHE_DIR"
  git -C "$GGUF_CACHE_DIR" checkout "$GGUF_REF"
fi

cat <<EOF
Prefetched MLX source:
  path: $MLX_CACHE_DIR
  ref:  $(git -C "$MLX_CACHE_DIR" rev-parse --short HEAD)

Prefetched Metal C++ headers:
  path: $METAL_CPP_CACHE_DIR

Prefetched nlohmann/json:
  path: $JSON_CACHE_DIR
  ref:  $(git -C "$JSON_CACHE_DIR" rev-parse --short HEAD)

Prefetched fmt:
  path: $FMT_CACHE_DIR
  ref:  $(git -C "$FMT_CACHE_DIR" rev-parse --short HEAD)

Prefetched gguf-tools:
  path: $GGUF_CACHE_DIR
  ref:  $(git -C "$GGUF_CACHE_DIR" rev-parse --short HEAD)

Note: mlx-sys 0.2.0 currently fetches MLX through its bundled mlx-c CMake
project. Vona patches mlx-sys through Cargo [patch.crates-io], so native builds
can use this cache with:

  export MLX_SOURCE_DIR=$MLX_CACHE_DIR
  export METAL_CPP_SOURCE_DIR=$METAL_CPP_CACHE_DIR
  export JSON_SOURCE_DIR=$JSON_CACHE_DIR
  export FMT_SOURCE_DIR=$FMT_CACHE_DIR
  export GGUF_SOURCE_DIR=$GGUF_CACHE_DIR
EOF
