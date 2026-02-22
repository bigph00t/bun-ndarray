import { ptr } from "bun:ffi";
import { describe, expect, test } from "bun:test";
import { native } from "../src/ffi";
import { NDArray } from "../src/index";

describe("reduce", () => {
  test("sum of small array", () => {
    using arr = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4, 5]));
    expect(arr.sum()).toBe(15);
  });

  test("sum of generated array", () => {
    const data = new Float64Array(1000);
    data.fill(0.5);

    using arr = NDArray.fromTypedArray(data);
    expect(arr.sum()).toBeCloseTo(500, 10);
  });

  test("raw native SIMD sum stays within latency regression threshold at 1M elements", () => {
    const n = 1_000_000;
    const data = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = i * 0.001;
    }

    const jsSum = () => {
      let s = 0;
      for (let i = 0; i < n; i++) {
        s += data[i];
      }
      return s;
    };

    const out = new Float64Array(1);
    const nativeSum = () => {
      const status = Number(native.nd_simd_sum_f64_raw(ptr(data), BigInt(n), ptr(out)));
      if (status !== 0) {
        throw new Error(`nd_simd_sum_f64_raw failed with status ${status}`);
      }
      return out[0] ?? 0;
    };
    const averageMs = (fn: () => number, reps: number): number => {
      let out = 0;
      const start = Bun.nanoseconds();
      for (let i = 0; i < reps; i++) {
        out += fn();
      }
      const elapsed = Number(Bun.nanoseconds() - start) / 1e6;
      expect(Number.isFinite(out)).toBe(true);
      return elapsed / reps;
    };

    // Warmup
    jsSum();
    nativeSum();

    const jsMs = averageMs(jsSum, 20);
    const nativeMs = averageMs(nativeSum, 20);
    expect(nativeSum()).toBeCloseTo(jsSum(), 6);
    expect(jsMs).toBeGreaterThan(0);
    // Keep this as a regression gate without assuming isolated benchmark conditions.
    expect(nativeMs).toBeLessThan(3.0);
  });
});
