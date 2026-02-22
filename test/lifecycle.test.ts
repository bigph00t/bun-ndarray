import { ptr } from "bun:ffi";
import { describe, expect, test } from "bun:test";
import { NDArray } from "../src/index";
import { native } from "../src/ffi";

describe("lifecycle", () => {
  test("dispose is idempotent in wrapper", () => {
    const arr = NDArray.zeros([8]);
    arr.dispose();
    arr.dispose();
    expect(arr.disposed).toBe(true);
  });

  test("use after dispose throws", () => {
    const arr = NDArray.zeros([4]);
    arr.dispose();
    expect(() => arr.toFloat64Array()).toThrow("disposed");
  });

  test("native stale handle detection after release", () => {
    const shape = new BigInt64Array([4n]);
    const out = new BigUint64Array(1);

    const s1 = Number(native.nd_array_alloc(4, ptr(shape), 1, 0, ptr(out)));
    expect(s1).toBe(0);

    const handle = out[0];
    const s2 = Number(native.nd_array_release(handle));
    expect(s2).toBe(0);

    const s3 = Number(native.nd_array_release(handle));
    expect(s3).toBe(6);
  });

  test("Symbol.dispose is implemented", () => {
    const arr = NDArray.zeros([2]);
    arr[Symbol.dispose]();
    expect(arr.disposed).toBe(true);
  });
});
