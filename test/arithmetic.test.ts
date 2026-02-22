import { describe, expect, test } from "bun:test";
import { NDArray, NativeError } from "../src/index";

describe("arithmetic", () => {
  test("element-wise add", () => {
    using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]));
    using b = NDArray.fromTypedArray(new Float64Array([10, 20, 30, 40]));
    using c = a.add(b);

    expect(Array.from(c.toFloat64Array())).toEqual([11, 22, 33, 44]);
  });

  test("add shape mismatch throws native error", () => {
    using a = NDArray.zeros([3]);
    using b = NDArray.zeros([4]);

    expect(() => a.add(b)).toThrow(NativeError);
  });
});
