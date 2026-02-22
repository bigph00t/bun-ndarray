#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="$ROOT_DIR/native/zig-out-matrix"
PREBUILDS_DIR="$ROOT_DIR/prebuilds"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "Copied: $src -> $dst"
  else
    echo "Skipped missing artifact: $src" >&2
  fi
}

copy_if_exists "$ARTIFACT_DIR/x86_64-linux-gnu/lib/libndarray.so.0.1.0" "$PREBUILDS_DIR/linux-x64/libndarray.so"
copy_if_exists "$ARTIFACT_DIR/aarch64-linux-gnu/lib/libndarray.so.0.1.0" "$PREBUILDS_DIR/linux-arm64/libndarray.so"
copy_if_exists "$ARTIFACT_DIR/x86_64-macos/lib/libndarray.0.1.0.dylib" "$PREBUILDS_DIR/darwin-x64/libndarray.dylib"
copy_if_exists "$ARTIFACT_DIR/aarch64-macos/lib/libndarray.0.1.0.dylib" "$PREBUILDS_DIR/darwin-arm64/libndarray.dylib"
copy_if_exists "$ARTIFACT_DIR/x86_64-windows-gnu/bin/ndarray.dll" "$PREBUILDS_DIR/win32-x64/ndarray.dll"

echo "Prebuild staging complete."
