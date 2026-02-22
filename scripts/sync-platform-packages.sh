#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREBUILDS="$ROOT_DIR/prebuilds"
PLATFORMS="$ROOT_DIR/platform-packages"

copy_or_warn() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "Synced: $src -> $dst"
  else
    echo "Missing source artifact: $src" >&2
  fi
}

copy_or_warn "$PREBUILDS/linux-x64/libndarray.so" "$PLATFORMS/bun-ndarray-linux-x64-gnu/libndarray.so"
copy_or_warn "$PREBUILDS/linux-arm64/libndarray.so" "$PLATFORMS/bun-ndarray-linux-arm64-gnu/libndarray.so"
copy_or_warn "$PREBUILDS/darwin-x64/libndarray.dylib" "$PLATFORMS/bun-ndarray-darwin-x64/libndarray.dylib"
copy_or_warn "$PREBUILDS/darwin-arm64/libndarray.dylib" "$PLATFORMS/bun-ndarray-darwin-arm64/libndarray.dylib"
copy_or_warn "$PREBUILDS/win32-x64/ndarray.dll" "$PLATFORMS/bun-ndarray-windows-x64-gnu/ndarray.dll"

echo "Platform package sync complete."
