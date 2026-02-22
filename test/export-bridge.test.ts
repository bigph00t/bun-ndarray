import { describe, expect, test } from "bun:test";
import { NDArray } from "../src/index";

describe("export bridge", () => {
  test("zero-copy export on contiguous arrays stays writable", () => {
    using arr = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]));
    const view = arr.toFloat64Array();
    view[1] = 99;
    expect(Array.from(arr.toFloat64Array({ copy: true }))).toEqual([1, 99, 3, 4]);
  });

  test("copy export does not alias source storage", () => {
    using arr = NDArray.fromTypedArray(new Float64Array([5, 6, 7]));
    const copied = arr.toFloat64Array({ copy: true });
    arr.toFloat64Array()[0] = 42;
    expect(Array.from(copied)).toEqual([5, 6, 7]);
  });

  test("copy export densifies non-contiguous views", () => {
    using arr = NDArray.fromTypedArray(new Float64Array([
      1, 2, 3,
      4, 5, 6,
    ]), [2, 3]);
    using t = arr.transpose([1, 0]);
    expect(t.isContiguous).toBe(false);
    expect(Array.from(t.toFloat64Array({ copy: true }))).toEqual([1, 4, 2, 5, 3, 6]);
  });

  test("empty view export is stable for copy and zero-copy modes", () => {
    using arr = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]));
    using empty = arr.slice([{ start: 2, stop: 2 }]);

    expect(empty.length).toBe(0);
    expect(Array.from(empty.toFloat64Array())).toEqual([]);
    expect(Array.from(empty.toFloat64Array({ copy: true }))).toEqual([]);
  });
});

