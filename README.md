# bun-ndarray

Zig-native `ndarray` scaffold for Bun, designed as a professional jump-off point for full-plan refinement.

## Current Scope

This repository now includes a working scaffold for the broader Phase 1/Section 21 API surface:

- Explicit lifetime model: `dispose()` + `[Symbol.dispose]` (`using` support)
- Handle-based ABI with stale-handle detection
- Creation APIs (`alloc`, `from_host_copy`) for `f32`, `f64`, `i32`
- Metadata APIs (`dtype`, `shape`, `strides`, `elem_count`, `byte_len`, `is_contiguous`)
- Transform APIs (`clone`, `make_contiguous`, `reshape`, `transpose`, `slice`)
- Ops (`add`, `sub`, `mul`, `div`, `eq`, `lt`, `gt`, `where`, `sum_all`, `sum_axis`, `matmul`)
- Export bridge (`nd_array_export_bytes`) for Bun `toArrayBuffer`
- Async matmul job API stubs (returns `ND_E_NOT_IMPLEMENTED`)
- Legacy low-level symbols retained for regression/perf tests (`nd_add_into`, raw SIMD hooks)
- Differential test scaffold (`test/numpy-differential.test.ts`) that auto-runs when NumPy exists

## Safety Contract

- Native memory is **not** garbage collected.
- You **must** call `dispose()` (or use `using`).
- `FinalizationRegistry` is best-effort only.
- Zero-copy typed views become invalid after parent dispose.

## Build and Test

```bash
bun run build:native:release
bun test
bun run test:numpy  # optional; skips when numpy is unavailable
```

## Cross-Target Scaffold

```bash
bun run build:native:matrix
bun run build:packages
bun run sync:platform-packages
```

Artifacts are staged under `native/zig-out-matrix/` and copied into `prebuilds/`.
Platform package artifacts are mirrored under `platform-packages/` for optional-dependency publishing workflows.

## Quick Example

```ts
import { NDArray } from "bun-ndarray";

using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]), [2, 2]);
using b = NDArray.ones([2, 2]);
using c = a.add(b).transpose().slice([{ step: -1 }, {}]).contiguous();

console.log(c.shape);           // [2, 2]
console.log(c.toFloat64Array());
console.log(c.sum());
```

## Deliberate Scaffold Gaps

These are present as explicit next-step work, not silent omissions:

- Job API is stubbed (`ND_E_NOT_IMPLEMENTED`)
- `toArrayBuffer` deallocator callback path is wired in scaffold form and still needs production hardening in CI across platforms
- Full dtype kernel parity is incomplete (`f64` is most optimized; `f32` has contiguous fast paths; `i32` remains baseline scalar)
- Slicing supports empty outputs and negative-step defaults; richer slicing DSL parity still needs refinement
- Packaging is scaffolded with optional dependency package templates but not yet published as split npm artifacts
- CI workflows are scaffolded under `.github/workflows/` but not yet battle-tested in remote runners
