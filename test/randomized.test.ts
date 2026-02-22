import { describe, expect, test } from "bun:test";
import { NDArray } from "../src/index";

describe("randomized correctness", () => {
  test("add and sum agree with JS across random 1D cases", () => {
    const cases = 200;

    for (let c = 0; c < cases; c++) {
      const n = 1 + (Math.random() * 1024) | 0;
      const a = new Float64Array(n);
      const b = new Float64Array(n);

      for (let i = 0; i < n; i++) {
        a[i] = (Math.random() - 0.5) * 1e3;
        b[i] = (Math.random() - 0.5) * 1e3;
      }

      using na = NDArray.fromTypedArray(a);
      using nb = NDArray.fromTypedArray(b);
      using nc = na.add(nb);

      const out = nc.toFloat64Array({ copy: true });
      let jsSum = 0;
      for (let i = 0; i < n; i++) {
        const expected = a[i] + b[i];
        jsSum += expected;
        expect(out[i]).toBeCloseTo(expected, 10);
      }

      expect(nc.sum()).toBeCloseTo(jsSum, 8);
    }
  });
});

