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
```

Artifacts are staged under `native/zig-out-matrix/` and copied into `prebuilds/`.

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
- `toArrayBuffer` deallocator callback path is not yet wired (export includes placeholders)
- Full dtype kernel parity is incomplete (`f64` is most optimized; `f32`/`i32` are baseline scalar paths)
- Empty-slice semantics and richer slicing DSL behavior still need full parity work
- Packaging is scaffolded but not published as split optional-dependency platform packages
- CI workflows are scaffolded under `.github/workflows/` but not yet battle-tested in remote runners
