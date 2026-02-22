import { ptr } from "bun:ffi";
import { describe, expect, test } from "bun:test";
import { native } from "../src/ffi";

function randInt(maxExclusive: number): number {
  return (Math.random() * maxExclusive) | 0;
}

function alloc1d(len: number): bigint {
  const shape = new BigInt64Array([BigInt(len)]);
  const out = new BigUint64Array(1);
  const s = Number(native.nd_array_alloc(4, ptr(shape), 1, 0, ptr(out)));
  expect(s).toBe(0);
  return out[0];
}

describe("registry fuzz", () => {
  test("random retain/release/lookup sequences preserve stale-handle safety", () => {
    const live = new Map<bigint, number>(); // handle -> logical refcount
    const stale: bigint[] = [];
    const outLen = new BigUint64Array(1);

    const steps = 20_000;
    for (let i = 0; i < steps; i++) {
      const r = Math.random();

      if (r < 0.35 || live.size === 0) {
        if (live.size < 256) {
          const h = alloc1d(1 + randInt(64));
          live.set(h, 1);
        }
        continue;
      }

      const keys = [...live.keys()];
      const h = keys[randInt(keys.length)]!;

      if (r < 0.60) {
        const s = Number(native.nd_array_retain(h));
        expect(s).toBe(0);
        live.set(h, (live.get(h) ?? 0) + 1);
        continue;
      }

      if (r < 0.90) {
        const s = Number(native.nd_array_release(h));
        expect(s).toBe(0);
        const rc = (live.get(h) ?? 0) - 1;
        if (rc <= 0) {
          live.delete(h);
          stale.push(h);
        } else {
          live.set(h, rc);
        }
        continue;
      }

      // Metadata probe on live handle should always succeed.
      const s = Number(native.nd_array_len(h, ptr(outLen)));
      expect(s).toBe(0);
      expect(Number(outLen[0])).toBeGreaterThan(0);
    }

    // Cleanup live handles fully.
    for (const [h, rc] of live) {
      for (let i = 0; i < rc; i++) {
        const s = Number(native.nd_array_release(h));
        expect(s).toBe(0);
      }
      stale.push(h);
    }

    // Sample stale handles and assert they stay stale.
    const sampleCount = Math.min(200, stale.length);
    for (let i = 0; i < sampleCount; i++) {
      const h = stale[randInt(stale.length)]!;
      const s1 = Number(native.nd_array_len(h, ptr(outLen)));
      expect(s1).toBe(6);
      const s2 = Number(native.nd_array_release(h));
      expect(s2).toBe(6);
    }
  });
});
