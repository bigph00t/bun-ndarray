# Implementation Coverage (Scaffold Pass)

Date: 2026-02-22

This file maps current implementation status back to `CODEX-TAKEOFF.md`, `PLAN.md`, and `LOAF.md`.

## CODEX-TAKEOFF.md

Status: Complete and superseded by broader scaffold.

- Core ABI + wrapper + tests + build: complete
- Memory/lifetime constraints from spikes: enforced
- SIMD hooks + stress tests: complete

## PLAN.md (Phase 1)

Legend:
- COMPLETE: implemented and tested in this repository
- SCAFFOLDED: implemented as a working baseline or explicit stub for future hardening
- PENDING: not yet implemented in this pass

### Sections 1-13 (Phase 1 execution)

1. Scope & success criteria: SCAFFOLDED
2. Repository structure: SCAFFOLDED
3. Zig core library:
- Lifecycle, metadata, create/copy: COMPLETE
- Elementwise add/sub/mul/div: COMPLETE (broadcast-aware for f64)
- Elementwise eq/lt/gt/where: COMPLETE (broadcast-aware)
- Sum reduction: COMPLETE
- Axis reduction (`sum_axis`): COMPLETE
- Matmul baseline: COMPLETE (2D f64)
- Broadcasting engine: COMPLETE for current op surface
- Slicing views: COMPLETE baseline (views, empty outputs, negative-step defaults, optional squeeze for indexed axes)
4. TypeScript wrapper:
- NDArray lifecycle + core ops + transform APIs: COMPLETE
- Broader dtype surface (f32/i32/f64) with explicit math limits: COMPLETE (f32 contiguous fast path + i32 baseline)
5. FFI bridge design: COMPLETE (status + out-params + symbol table parity)
6. Memory management: COMPLETE for explicit ownership; deallocator callback path is SCAFFOLDED (with dedicated export-bridge tests + copy-path soak guard)
7. SIMD strategy: SCAFFOLDED (raw SIMD hooks + contiguous fast path)
8. Error handling across FFI: COMPLETE
9. Build + cross compilation: COMPLETE (matrix script + artifact staging)
10. Testing strategy: SCAFFOLDED (unit + stress + ABI + fuzz + wrapper coverage + numpy differential in CI)
11. Benchmarking strategy: SCAFFOLDED (`bench/basic.bench.ts`)
12. Distribution & packaging: SCAFFOLDED (prebuild staging + postinstall detection + optional-dependency package templates + CI/release workflow scaffolds + install smoke gate)
13. Step-by-step implementation order: COMPLETE through practical scaffold equivalents

### Sections 14-16

14. Open questions: SCAFFOLDED via implementation defaults; still needs final product decisions
15. Risk register: COMPLETE baseline (`RISK_REGISTER.md`)
16. References: N/A (documentation section)

### Sections 17-22 (Codex append sections)

17-20 research constraints: COMPLETE adoption where code-relevant
21 exact ABI packet: SCAFFOLDED with broad parity and explicit stubs where unresolved
22 spike updates: COMPLETE adoption (manual dispose, alignment expectations, deallocator caveat)

## LOAF.md (Project Vision)

Status: Phase 1-focused scaffold only.

- Phase 1 (`bun-ndarray` core): SCAFFOLDED/WORKING
- Phase 2+ (full op breadth, distribution ecosystem, `bun-frame`, MiniLoaf, full Loaf platform): PENDING

## Explicit Remaining Work Before "Full Plan Complete"

1. Final production-hardening for the `toArrayBuffer` deallocator callback path (leak/soak guards and stricter CI gates)
2. Full dtype optimization parity (SIMD/fast paths for f32/i32, not just scalar baselines)
3. Advanced slicing/indexing parity (dimension squeeze/index arrays) beyond the current baseline
4. Publishing platform split packages to npm (optional dependency install flow is now smoke-validated in CI/release workflows)
5. Battle-hardening and validating release automation in remote runners
