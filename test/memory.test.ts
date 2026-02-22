import { ptr } from "bun:ffi";
import { describe, expect, test } from "bun:test";
import { native } from "../src/ffi";

describe("memory", () => {
  test("stress create/dispose does not grow RSS beyond threshold", () => {
    const loops = 10_000;
    const before = process.memoryUsage().rss;
    const shape = new BigInt64Array([32n]);

    for (let i = 0; i < loops; i++) {
      const out = new BigUint64Array(1);
      const allocStatus = Number(native.nd_array_alloc(4, ptr(shape), 1, 0, ptr(out)));
      expect(allocStatus).toBe(0);

      const releaseStatus = Number(native.nd_array_release(out[0]));
      expect(releaseStatus).toBe(0);
    }

    Bun.gc(true);
    const after = process.memoryUsage().rss;
    const delta = after - before;

    // Spike baseline was ~1.25MB for 10K alloc/free cycles.
    expect(delta).toBeLessThan(10 * 1024 * 1024);
  });
});
