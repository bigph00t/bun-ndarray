import { ptr } from "bun:ffi";
import { describe, expect, test } from "bun:test";
import { native } from "../src/ffi";
import { NDArray, NativeError } from "../src/index";

describe("scaffold API", () => {
  test("abi/build version are exposed", () => {
    expect(NDArray.abiVersion()).toBe(1);
    expect(NDArray.buildVersion().length).toBeGreaterThan(0);
  });

  test("clone creates independent data copy", () => {
    using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]), [2, 2]);
    using b = a.clone();

    a.toFloat64Array()[0] = 99;
    expect(Array.from(b.toFloat64Array())).toEqual([1, 2, 3, 4]);
  });

  test("reshape creates a view over the same storage", () => {
    using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]), [2, 2]);
    using r = a.reshape([4]);

    expect(r.shape).toEqual([4]);

    const rView = r.toFloat64Array();
    rView[1] = 222;
    expect(a.toFloat64Array()[1]).toBe(222);
  });

  test("transpose returns non-contiguous view and contiguous() densifies", () => {
    using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]), [2, 2]);
    using t = a.transpose([1, 0]);

    expect(t.shape).toEqual([2, 2]);
    expect(t.isContiguous).toBe(false);
    expect(() => t.toFloat64Array()).toThrow(NativeError);
    expect(Array.from(t.toFloat64Array({ copy: true }))).toEqual([1, 3, 2, 4]);

    using dense = t.contiguous();
    expect(dense.isContiguous).toBe(true);
    expect(Array.from(dense.toFloat64Array())).toEqual([1, 3, 2, 4]);
  });

  test("zero-length non-contiguous views export safely", () => {
    using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]), [2, 2]);
    using t = a.transpose([1, 0]);
    using empty = t.slice([{ start: 0, stop: 0 }, {}]);

    expect(empty.length).toBe(0);
    expect(empty.isContiguous).toBe(false);
    expect(Array.from(empty.toFloat64Array({ copy: true }))).toEqual([]);
  });

  test("broadcasted elementwise ops work", () => {
    using a = NDArray.fromTypedArray(new Float64Array([1, 2]), [2, 1]);
    using b = NDArray.fromTypedArray(new Float64Array([10, 20, 30]), [1, 3]);
    using ones = NDArray.ones([2, 3]);
    using twos = NDArray.fromTypedArray(new Float64Array([2, 2, 2, 2, 2, 2]), [2, 3]);

    using add = a.add(b);
    using sub = add.sub(ones);
    using mul = add.mul(twos);
    using div = mul.div(twos);

    expect(add.shape).toEqual([2, 3]);
    expect(Array.from(add.toFloat64Array())).toEqual([11, 21, 31, 12, 22, 32]);
    expect(Array.from(sub.toFloat64Array())).toEqual([10, 20, 30, 11, 21, 31]);
    expect(Array.from(mul.toFloat64Array())).toEqual([22, 42, 62, 24, 44, 64]);
    expect(Array.from(div.toFloat64Array())).toEqual([11, 21, 31, 12, 22, 32]);
  });

  test("matmul computes baseline 2D product", () => {
    using a = NDArray.fromTypedArray(new Float64Array([
      1, 2, 3,
      4, 5, 6,
    ]), [2, 3]);

    using b = NDArray.fromTypedArray(new Float64Array([
      7, 8,
      9, 10,
      11, 12,
    ]), [3, 2]);

    using out = a.matmul(b);
    expect(out.shape).toEqual([2, 2]);
    expect(Array.from(out.toFloat64Array())).toEqual([58, 64, 139, 154]);
  });

  test("f32/i32 dtypes round-trip through the scaffold", () => {
    using f32 = NDArray.fromTyped(new Float32Array([1.5, 2.5, 3.5]));
    using i32 = NDArray.fromTyped(new Int32Array([1, 2, 3]));

    expect(f32.dtype).toBe("f32");
    expect(i32.dtype).toBe("i32");

    const f32View = f32.toTypedArray();
    const i32View = i32.toTypedArray();

    expect(f32View instanceof Float32Array).toBe(true);
    expect(i32View instanceof Int32Array).toBe(true);
    expect(Array.from(f32View as Float32Array)).toEqual([1.5, 2.5, 3.5]);
    expect(Array.from(i32View as Int32Array)).toEqual([1, 2, 3]);
  });

  test("f32 math paths are operational", () => {
    using a = NDArray.fromTyped(new Float32Array([1, 2, 3]));
    using b = NDArray.fromTyped(new Float32Array([4, 5, 6]));
    using out = a.add(b).mul(b).div(b).sub(a);
    const view = out.toTypedArray();
    expect(view instanceof Float32Array).toBe(true);
    expect(Array.from(view as Float32Array)).toEqual([4, 5, 6]);
  });

  test("i32 math paths are operational and division by zero errors", () => {
    using a = NDArray.fromTyped(new Int32Array([8, 12, 16]));
    using b = NDArray.fromTyped(new Int32Array([2, 3, 4]));
    using out = a.div(b);
    expect(Array.from(out.toTypedArray() as Int32Array)).toEqual([4, 4, 4]);

    using z = NDArray.fromTyped(new Int32Array([1, 0, 1]));
    expect(() => a.div(z)).toThrow(NativeError);
  });

  test("slice creates view semantics and supports copy extraction", () => {
    using a = NDArray.fromTypedArray(new Float64Array([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
    ]), [3, 3]);

    using s = a.slice([
      { start: 0, stop: 3, step: 2 },
      { start: 1, stop: 3 },
    ]);

    expect(s.shape).toEqual([2, 2]);
    expect(s.isContiguous).toBe(false);
    expect(Array.from(s.toFloat64Array({ copy: true }))).toEqual([2, 3, 8, 9]);
    expect(s.sum()).toBeCloseTo(22, 12);
  });

  test("slice supports negative-step defaults and empty outputs", () => {
    using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]));
    using reversed = a.slice([{ step: -1 }]);
    using empty = a.slice([{ step: -1, stop: -1 }]);
    using negRange = a.slice([{ start: -10, stop: 10, step: 2 }]);
    using reverseWindow = a.slice([{ start: 3, stop: 0, step: -1 }]);
    using reverseEmpty = a.slice([{ start: 0, stop: 0, step: -1 }]);

    expect(reversed.shape).toEqual([4]);
    expect(Array.from(reversed.toFloat64Array({ copy: true }))).toEqual([4, 3, 2, 1]);

    expect(empty.shape).toEqual([0]);
    expect(empty.length).toBe(0);
    expect(Array.from(empty.toFloat64Array())).toEqual([]);
    expect(empty.sum()).toBe(0);

    expect(Array.from(negRange.toFloat64Array({ copy: true }))).toEqual([1, 3]);
    expect(Array.from(reverseWindow.toFloat64Array({ copy: true }))).toEqual([4, 3, 2]);
    expect(reverseEmpty.length).toBe(0);
    expect(Array.from(reverseEmpty.toFloat64Array())).toEqual([]);
  });

  test("compare + where produce mask-select outputs", () => {
    using a = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]));
    using b = NDArray.fromTypedArray(new Float64Array([2, 2, 2, 2]));
    using mask = a.gt(b);
    using out = mask.where(a, b);

    expect(mask.dtype).toBe("i32");
    expect(Array.from(mask.toTypedArray() as Int32Array)).toEqual([0, 0, 1, 1]);
    expect(Array.from(out.toFloat64Array())).toEqual([2, 2, 3, 4]);
  });

  test("sumAxis reduces along requested axis", () => {
    using a = NDArray.fromTypedArray(new Float64Array([
      1, 2, 3,
      4, 5, 6,
    ]), [2, 3]);

    using s0 = a.sumAxis(0);
    using s1 = a.sumAxis(1);
    using s1neg = a.sumAxis(-1);
    expect(s0.shape).toEqual([3]);
    expect(s1.shape).toEqual([2]);
    expect(Array.from(s0.toFloat64Array())).toEqual([5, 7, 9]);
    expect(Array.from(s1.toFloat64Array())).toEqual([6, 15]);
    expect(Array.from(s1neg.toFloat64Array())).toEqual([6, 15]);
  });

  test("job API is scaffolded as ND_E_NOT_IMPLEMENTED", () => {
    const outJob = new BigUint64Array(1);
    const outState = new Uint32Array(1);
    const outStatus = new Int32Array(1);
    const outHandle = new BigUint64Array(1);

    expect(Number(native.nd_job_submit_matmul(1n, 2n, ptr(outJob)))).toBe(9);
    expect(Number(native.nd_job_poll(1n, ptr(outState), ptr(outStatus)))).toBe(9);
    expect(Number(native.nd_job_take_result(1n, ptr(outHandle)))).toBe(9);
    expect(Number(native.nd_job_cancel(1n))).toBe(9);
  });
});
