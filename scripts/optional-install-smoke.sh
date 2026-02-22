#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
PACK_DIR="$TMP_DIR/packs"
CONSUMER_DIR="$TMP_DIR/consumer"
export NPM_CONFIG_CACHE="$TMP_DIR/npm-cache"
export BUN_TMPDIR="$TMP_DIR/bun-tmp"
export BUN_INSTALL="$TMP_DIR/bun-install"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$PACK_DIR" "$CONSUMER_DIR" "$BUN_TMPDIR" "$BUN_INSTALL"

(
  cd "$ROOT_DIR"
  npm pack --silent --pack-destination "$PACK_DIR" >/dev/null
)

for pkg in "$ROOT_DIR"/platform-packages/*; do
  (
    cd "$pkg"
    npm pack --silent --pack-destination "$PACK_DIR" >/dev/null
  )
done

platform_key="$(node -p "process.platform + '-' + process.arch")"
case "$platform_key" in
  linux-x64) platform_pkg="bun-ndarray-linux-x64-gnu" ;;
  linux-arm64) platform_pkg="bun-ndarray-linux-arm64-gnu" ;;
  darwin-x64) platform_pkg="bun-ndarray-darwin-x64" ;;
  darwin-arm64) platform_pkg="bun-ndarray-darwin-arm64" ;;
  win32-x64) platform_pkg="bun-ndarray-windows-x64-gnu" ;;
  *)
    echo "[optional-install-smoke] No platform package mapping for $platform_key; skipping."
    exit 0
    ;;
esac

root_tgzs=( "$PACK_DIR"/bun-ndarray-*.tgz )
platform_tgzs=( "$PACK_DIR"/"$platform_pkg"-*.tgz )

if [[ ! -f "${root_tgzs[0]}" ]]; then
  echo "[optional-install-smoke] Missing packed root tarball in $PACK_DIR" >&2
  exit 1
fi
if [[ ! -f "${platform_tgzs[0]}" ]]; then
  echo "[optional-install-smoke] Missing packed platform tarball for $platform_pkg in $PACK_DIR" >&2
  exit 1
fi

root_tgz="${root_tgzs[0]}"
platform_tgz="${platform_tgzs[0]}"

cat > "$CONSUMER_DIR/package.json" <<JSON
{
  "name": "bun-ndarray-optional-install-smoke",
  "private": true,
  "type": "module",
  "dependencies": {
    "bun-ndarray": "file:${root_tgz}",
    "${platform_pkg}": "file:${platform_tgz}"
  }
}
JSON

(
  cd "$CONSUMER_DIR"
  bun install --offline
  EXPECTED_PLATFORM_PKG="$platform_pkg" bun -e '
    import { NDArray } from "bun-ndarray";
    import { nativeLibraryPath } from "bun-ndarray/src/ffi";

    const expected = process.env.EXPECTED_PLATFORM_PKG;
    if (!expected) {
      throw new Error("EXPECTED_PLATFORM_PKG is not set");
    }

    using arr = NDArray.ones([4]);
    const values = Array.from(arr.toFloat64Array({ copy: true }));
    if (values.join(",") !== "1,1,1,1") {
      throw new Error(`unexpected values: ${values.join(",")}`);
    }

    if (!nativeLibraryPath.includes(expected)) {
      throw new Error(
        `expected optional platform package path containing "${expected}", got "${nativeLibraryPath}"`,
      );
    }

    console.log(`[optional-install-smoke] Loaded ${nativeLibraryPath}`);
  '
)
