#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NATIVE_DIR="$ROOT_DIR/native"

ZIG_BIN="${ZIG_BIN:-/tmp/zig-x86_64-linux-0.15.2/zig}"
if [[ ! -x "$ZIG_BIN" ]]; then
  ZIG_BIN="zig"
fi

TARGETS=(
  "x86_64-linux-gnu"
  "aarch64-linux-gnu"
  "x86_64-macos"
  "aarch64-macos"
  "x86_64-windows-gnu"
)

for target in "${TARGETS[@]}"; do
  echo "==> Building $target"
  out_prefix="$NATIVE_DIR/zig-out-matrix/$target"
  mkdir -p "$out_prefix"
  (
    cd "$NATIVE_DIR"
    ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
    ZIG_LOCAL_CACHE_DIR=.zig-cache \
    "$ZIG_BIN" build -Doptimize=ReleaseFast -Dtarget="$target" --prefix "$out_prefix"
  )
done

echo "All matrix builds completed. Artifacts are under native/zig-out-matrix/."
