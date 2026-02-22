import { describe, expect, test } from "bun:test";
import { NDArray } from "../src/index";

describe("creation", () => {
  test("zeros creates f64 ndarray", () => {
    using arr = NDArray.zeros([2, 3]);

    expect(arr.dtype).toBe("f64");
    expect(arr.shape).toEqual([2, 3]);
    expect(arr.length).toBe(6);
    expect(arr.strides).toEqual([24, 8]);
    expect(Array.from(arr.toFloat64Array())).toEqual([0, 0, 0, 0, 0, 0]);
  });

  test("ones creates f64 ndarray", () => {
    using arr = NDArray.ones([4]);
    expect(Array.from(arr.toFloat64Array())).toEqual([1, 1, 1, 1]);
  });

  test("fromTypedArray copies data and supports shape override", () => {
    const src = new Float64Array([1, 2, 3, 4, 5, 6]);
    using arr = NDArray.fromTypedArray(src, [2, 3]);

    expect(arr.shape).toEqual([2, 3]);
    expect(arr.length).toBe(6);
    expect(Array.from(arr.toFloat64Array())).toEqual([1, 2, 3, 4, 5, 6]);
  });

  test("fromTypedArray rejects shape/data length mismatch", () => {
    const src = new Float64Array([1, 2, 3, 4]);
    expect(() => NDArray.fromTypedArray(src, [3, 3])).toThrow("shape product");
  });

  test("zeros supports empty dimensions and empty exports", () => {
    using arr = NDArray.zeros([2, 0, 3]);
    expect(arr.shape).toEqual([2, 0, 3]);
    expect(arr.length).toBe(0);
    expect(arr.byteLength).toBe(0);
    expect(Array.from(arr.toFloat64Array())).toEqual([]);
  });

  test("zeros rejects negative dimensions", () => {
    expect(() => NDArray.zeros([2, -1])).toThrow("invalid dimension");
  });

  test("fromTyped supports empty typed arrays", () => {
    using arr = NDArray.fromTypedArray(new Float64Array(0), [0]);
    expect(arr.shape).toEqual([0]);
    expect(arr.length).toBe(0);
    expect(Array.from(arr.toFloat64Array())).toEqual([]);
  });

  test("scalar shape [] is supported", () => {
    using arr = NDArray.zeros([]);
    expect(arr.shape).toEqual([]);
    expect(arr.length).toBe(1);
    expect(arr.toFloat64Array()[0]).toBe(0);
  });

  test("toFloat64Array copy mode isolates from native writes", () => {
    using arr = NDArray.fromTypedArray(new Float64Array([1, 2, 3]));
    const copied = arr.toFloat64Array({ copy: true });
    const view = arr.toFloat64Array();

    view[0] = 999;
    expect(copied[0]).toBe(1);
    expect(view[0]).toBe(999);
  });
});
