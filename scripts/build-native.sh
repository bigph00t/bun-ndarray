#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-debug}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE_DIR="$ROOT_DIR/native"

ZIG_BIN="${ZIG_BIN:-/tmp/zig-x86_64-linux-0.15.2/zig}"
if [[ ! -x "$ZIG_BIN" ]]; then
  ZIG_BIN="zig"
fi

build_args=()
if [[ "$MODE" == "release" ]]; then
  build_args+=("-Doptimize=ReleaseFast")
fi

(
  cd "$NATIVE_DIR"
  ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
  ZIG_LOCAL_CACHE_DIR=.zig-cache \
  "$ZIG_BIN" build "${build_args[@]}"
)
