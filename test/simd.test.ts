import { ptr } from "bun:ffi";
import { describe, expect, test } from "bun:test";
import { native } from "../src/ffi";
import { NDArray } from "../src/index";

function averageNs(fn: () => void, iters: number): number {
  const start = Bun.nanoseconds();
  for (let i = 0; i < iters; i++) {
    fn();
  }
  return (Bun.nanoseconds() - start) / iters;
}

describe("simd", () => {
  test("reports SIMD width", () => {
    expect(NDArray.simdWidthF64()).toBeGreaterThanOrEqual(1);
  });

  test("raw native SIMD add stays below latency threshold at N=10K", () => {
    const n = 10_000;
    const warmup = 50;
    const runs = 400;

    const a = new Float64Array(n);
    const b = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      a[i] = (i % 13) * 1.25;
      b[i] = (i % 17) * 0.75;
    }

    const jsOut = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      jsOut[i] = a[i] + b[i];
    }

    const outNative = new Float64Array(n);
    let sink = 0;

    const nativeAdd = () => {
      const status = Number(
        native.nd_simd_add_f64_raw(ptr(a), ptr(b), ptr(outNative), BigInt(n)),
      );
      if (status !== 0) {
        throw new Error(`nd_simd_add_f64_raw failed with status ${status}`);
      }
      sink += outNative[0] ?? 0;
    };

    averageNs(nativeAdd, warmup);
    const nativeNs = averageNs(nativeAdd, runs);
    const nativeMs = nativeNs / 1e6;

    // Correctness guard: compare several points across the output.
    for (let i = 0; i < n; i += 997) {
      expect(outNative[i]).toBeCloseTo(jsOut[i], 12);
    }
    expect(Number.isFinite(sink)).toBe(true);
    // Regression gate from spike baseline envelope (0.002ms baseline, 0.01ms strict gate).
    // We keep a wider bound here for CI/sandbox jitter while still catching major regressions.
    expect(nativeMs).toBeLessThan(0.1);
  });
});
