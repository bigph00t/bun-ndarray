# Risk Register

Date: 2026-02-22
Scope: `bun-ndarray` Phase 1 scaffold in this repository.

| ID | Risk | Likelihood | Impact | Current Mitigation | Next Gate |
|---|---|---|---|---|---|
| R1 | Bun `toArrayBuffer` deallocator callback behavior differs by platform/runtime updates | Medium | High | Wrapper has explicit fallback and `nd_export_release_ctx` path; CI matrix runs dedicated export-bridge tests | Add stricter leak/soak gates for zero-copy callback path on macOS/Windows runners |
| R2 | SIMD parity remains uneven across dtypes (`f32`/`i32` rely on scalar baselines) | High | Medium | `f64` fast paths are in place; dtype tests exist for correctness | Add perf thresholds for `f32`/`i32` hot ops and implement vectorized kernels |
| R3 | API drift between C header and TS symbol map | Low | High | `test/abi-contract.test.ts` enforces symbol parity | Keep ABI contract test required on PR CI |
| R4 | Handle lifecycle bugs under heavy churn (stale handles, double release) | Medium | High | Registry fuzz + lifecycle tests + stale-handle checks are running | Extend fuzz corpus and run stress job in nightly |
| R5 | Cross-target artifacts may regress silently in packaging flow | Medium | Medium | Matrix build script + package sync script + staged prebuilds + optional-install smoke script in CI/release workflows + artifact metadata manifests/checksums | Add remote workflow assertions for platform package publish metadata before npm release |
| R6 | Differential correctness against NumPy may drift as op surface grows | Medium | High | Optional local differential + nightly NumPy differential workflow | Promote differential to required job for release tags |
