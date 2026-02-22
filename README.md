# bun-ndarray

Zig-native `ndarray` core for Bun, implemented as a Phase 1 production scaffold for the Loaf roadmap.

This repository is a working baseline: native ABI + TypeScript wrapper + test harness + cross-target packaging pipeline. It is intentionally not a full NumPy replacement yet.

## What Is Implemented

- Native C ABI (`native/include/ndarray.h`) and Bun FFI bridge
- Explicit NDArray lifetime model with `dispose()` and `[Symbol.dispose]`
- DTypes: `f64`, `f32`, `i32`
- Core array ops: create, reshape, transpose, slice views, elementwise ops, comparisons, reductions, `matmul`
- Cross-target artifact build/staging for:
1. Linux x64 GNU
2. Linux arm64 GNU
3. macOS x64
4. macOS arm64
5. Windows x64 GNU
- Optional dependency platform-package model (esbuild-style split packages)
- Artifact metadata manifests with checksums for release traceability
- CI, nightly, and release workflows with smoke gates

## Requirements

- Bun `>=1.3.4`
- Zig `0.15.2` (required for local native builds)
- Python 3.11 + NumPy (only for differential tests)

## Install and Runtime Model

Root package:
- `bun-ndarray`

Optional platform packages:
1. `bun-ndarray-linux-x64-gnu`
2. `bun-ndarray-linux-arm64-gnu`
3. `bun-ndarray-darwin-x64`
4. `bun-ndarray-darwin-arm64`
5. `bun-ndarray-windows-x64-gnu`

Runtime native library resolution (`src/ffi.ts`) checks, in order:
1. Local dev artifact in `native/zig-out/`
2. Installed optional platform package
3. Checked-in `prebuilds/` fallback

## Memory and Safety Contract

Native memory is not managed by JS garbage collection.

- Call `dispose()` explicitly, or use `using` with `[Symbol.dispose]`
- `FinalizationRegistry` is best-effort fallback only
- Zero-copy typed-array exports are invalid after disposal of the owning NDArray
- Use `toTypedArray({ copy: true })` when lifetime isolation is required

## Quick Start

```bash
bun install
bun run build:native:release
```

```ts
import { NDArray } from "bun-ndarray";

using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]), [2, 2]);
using b = NDArray.ones([2, 2]);
using out = a
  .add(b)
  .transpose([1, 0])
  .slice([{ step: -1 }, null], { squeeze: false })
  .contiguous();

console.log(out.shape); // [2, 2]
console.log(Array.from(out.toFloat64Array({ copy: true })));
console.log(out.sum());
```

## API Surface (Current Scaffold)

Creation:
- `NDArray.zeros(shape, dtype?)`
- `NDArray.ones(shape, dtype?)`
- `NDArray.fromTyped(data, shape?, dtype?)`
- `NDArray.fromTypedArray(float64, shape?)`

Metadata:
- `dtype`, `shape`, `strides`, `size`, `length`, `byteLength`, `isContiguous`

Transforms:
- `clone()`
- `contiguous()`
- `reshape(shape)`
- `transpose(perm?)`
- `slice(specs?, options?)`

Slice notes:
- `SliceSpec` supports span specs (`{ start, stop, step }`), numeric index specs, and `null` full-axis specs.
- `slice(..., { squeeze: true })` drops numeric-indexed axes and can produce scalar-shaped outputs.

Ops:
- Arithmetic: `add`, `sub`, `mul`, `div`
- Compare/select: `eq`, `lt`, `gt`, `where`
- Reductions: `sumAll`, `sumAxis`, `sum`
- Linear algebra: `matmul`

Export:
- `toTypedArray({ copy?: boolean })`
- `toFloat64Array({ copy?: boolean })`

## Development Commands

```bash
# Host debug native build
bun run build:native

# Host release native build
bun run build:native:release

# Full unit/integration test suite
bun test --bail

# Differential checks vs NumPy (optional)
bun run test:numpy

# Optional dependency install/runtime smoke
bun run test:optional-install-smoke

# Basic benchmark smoke
bun run bench:smoke
```

Cross-target artifact staging:

```bash
bun run build:native:matrix
bun run build:packages
```

Outputs:
- Target binaries under `prebuilds/`
- Mirrored platform package payloads under `platform-packages/*`
- Root metadata manifest at `prebuilds/manifest.json`
- Per-platform metadata at `platform-packages/*/artifact-metadata.json`

## Artifact Metadata

`scripts/write-artifact-metadata.ts` emits checksummed metadata for each artifact:

- package/version identity
- ABI version
- minimum Bun version
- Zig version
- git SHA
- platform key and target triple
- artifact filename, SHA-256, and byte size

Validation:
- `test/artifact-metadata.test.ts`
- `scripts/optional-install-smoke.sh` cross-checks root manifest against installed platform metadata

## CI, Nightly, and Release

CI (`.github/workflows/ci.yml`) runs:
1. full build + test pass
2. cross-OS FFI smoke
3. memory/registry stress tests
4. NumPy differential checks
5. optional install/package smoke

Nightly (`.github/workflows/nightly.yml`) runs:
1. NumPy differential checks
2. benchmark smoke

Release (`.github/workflows/release.yml`) runs:
1. full tests
2. cross-target build/package staging
3. optional install smoke
4. artifact uploads (`prebuilds`, `platform-packages`, ABI header)

## Planning and Completion Status

Planning/concept basis for this implementation:
- `CODEX-TAKEOFF.md`
- `PLAN.md`
- `LOAF.md`

Note: those planning files are workspace-level project documents and are not packaged with this repository.

Current status tracking:
- `IMPLEMENTATION_COVERAGE.md` maps implemented scaffold coverage against plan sections
- `RISK_REGISTER.md` tracks active technical/release risks and next mitigation gates

Interpretation:
- Phase 1 scaffold is working and test-backed
- Some roadmap items are intentionally left for hardening/refinement passes

## Known Gaps (Intentional for This Pass)

- Job API remains stubbed (`ND_E_NOT_IMPLEMENTED`)
- `toArrayBuffer` deallocator callback path needs deeper cross-OS soak hardening
- DType optimization parity is incomplete (`f64` most optimized; `f32` partial fast paths; `i32` scalar baseline in generic paths)
- Advanced indexing beyond current slice model (fancy/index-array semantics) is pending
- Platform package npm publish execution is scaffolded, but not yet completed from this repository
