import { ptr, toArrayBuffer } from "bun:ffi";
import { describe, expect, test } from "bun:test";
import { native } from "../src/ffi";

function alloc1d(len: number): bigint {
  const shape = new BigInt64Array([BigInt(len)]);
  const out = new BigUint64Array(1);
  const status = Number(native.nd_array_alloc(4, ptr(shape), 1, 0, ptr(out)));
  expect(status).toBe(0);
  return out[0];
}

function exportBytes(handle: bigint): BigUint64Array {
  const out4 = new BigUint64Array(4);
  const status = Number(native.nd_array_export_bytes(handle, ptr(out4)));
  expect(status).toBe(0);
  return out4;
}

function readF64(handle: bigint, expectedLen: number): Float64Array {
  const out4 = exportBytes(handle);
  const p = Number(out4[0]);
  const bytes = Number(out4[1]);
  const ctx = out4[3];
  expect(p).toBeGreaterThan(0);
  expect(bytes).toBe(expectedLen * Float64Array.BYTES_PER_ELEMENT);
  const copied = new Float64Array(new Float64Array(toArrayBuffer(p, 0, bytes)));
  if (ctx > 0n) {
    expect(Number(native.nd_export_release_ctx(ctx))).toBe(0);
  }
  return copied;
}

function readI32(handle: bigint, expectedLen: number): Int32Array {
  const out4 = exportBytes(handle);
  const p = Number(out4[0]);
  const bytes = Number(out4[1]);
  const ctx = out4[3];
  expect(p).toBeGreaterThan(0);
  expect(bytes).toBe(expectedLen * Int32Array.BYTES_PER_ELEMENT);
  const copied = new Int32Array(new Int32Array(toArrayBuffer(p, 0, bytes)));
  if (ctx > 0n) {
    expect(Number(native.nd_export_release_ctx(ctx))).toBe(0);
  }
  return copied;
}

describe("direct ABI", () => {
  test("ABI version is pinned", () => {
    expect(Number(native.nd_abi_version())).toBe(1);
  });

  test("invalid dtype returns ND_E_INVALID_DTYPE with message", () => {
    const shape = new BigInt64Array([4n]);
    const out = new BigUint64Array(1);

    const status = Number(native.nd_array_alloc(999, ptr(shape), 1, 0, ptr(out)));
    expect(status).toBe(2);
    expect(Number(native.nd_last_error_code())).toBe(2);

    const msgBuf = new Uint8Array(256);
    const msgLen = new BigUint64Array(1);
    const msgStatus = Number(native.nd_last_error_message(ptr(msgBuf), BigInt(msgBuf.length), ptr(msgLen)));
    expect(msgStatus).toBe(0);

    const msg = new TextDecoder().decode(msgBuf.subarray(0, Number(msgLen[0]))).replace(/\0+$/, "");
    expect(msg.length).toBeGreaterThan(0);
  });

  test("invalid shape (zero dimension) returns ND_E_INVALID_SHAPE", () => {
    const shape = new BigInt64Array([0n]);
    const out = new BigUint64Array(1);
    const status = Number(native.nd_array_alloc(4, ptr(shape), 1, 0, ptr(out)));
    expect(status).toBe(3);
  });

  test("shape_copy validates output capacity", () => {
    const shape = new BigInt64Array([2n, 3n, 4n]);
    const out = new BigUint64Array(1);
    const s1 = Number(native.nd_array_alloc(4, ptr(shape), 3, 0, ptr(out)));
    expect(s1).toBe(0);

    try {
      const tooSmall = new BigInt64Array(2);
      const s2 = Number(native.nd_array_shape_copy(out[0], ptr(tooSmall), 2));
      expect(s2).toBe(1);
    } finally {
      const s3 = Number(native.nd_array_release(out[0]));
      expect(s3).toBe(0);
    }
  });

  test("handle generation changes after reuse and stale handle is rejected", () => {
    const h1 = alloc1d(4);
    expect(Number(native.nd_array_release(h1))).toBe(0);

    const h2 = alloc1d(4);
    expect(h2).not.toBe(h1);

    const outLen = new BigUint64Array(1);
    const staleStatus = Number(native.nd_array_len(h1, ptr(outLen)));
    expect(staleStatus).toBe(6);
    const stalePtr = Number(native.nd_array_data_ptr(h1));
    expect(stalePtr).toBe(0);
    expect(Number(native.nd_last_error_code())).toBe(6);

    const validStatus = Number(native.nd_array_len(h2, ptr(outLen)));
    expect(validStatus).toBe(0);
    expect(Number(outLen[0])).toBe(4);

    expect(Number(native.nd_array_release(h2))).toBe(0);
  });

  test("retain/release balance keeps object alive until final release", () => {
    const h = alloc1d(8);
    expect(Number(native.nd_array_retain(h))).toBe(0);

    const outLen = new BigUint64Array(1);
    expect(Number(native.nd_array_len(h, ptr(outLen)))).toBe(0);
    expect(Number(outLen[0])).toBe(8);

    expect(Number(native.nd_array_release(h))).toBe(0);
    expect(Number(native.nd_array_len(h, ptr(outLen)))).toBe(0);
    expect(Number(native.nd_array_release(h))).toBe(0);
    expect(Number(native.nd_array_release(h))).toBe(6);
  });

  test("sum_all produces scalar handle (ndim=0, len=1)", () => {
    const data = new Float64Array([1, 2, 3, 4, 5]);
    const shape = new BigInt64Array([5n]);

    const arr = new BigUint64Array(1);
    expect(Number(native.nd_array_from_host_copy(ptr(data), 4, ptr(shape), 0, 1, 0, ptr(arr)))).toBe(0);

    const sumHandle = new BigUint64Array(1);
    expect(Number(native.nd_sum_all(arr[0], ptr(sumHandle)))).toBe(0);

    try {
      const outNdim = new Uint8Array(1);
      expect(Number(native.nd_array_ndim(sumHandle[0], ptr(outNdim)))).toBe(0);
      expect(outNdim[0]).toBe(0);

      const outLen = new BigUint64Array(1);
      expect(Number(native.nd_array_len(sumHandle[0], ptr(outLen)))).toBe(0);
      expect(Number(outLen[0])).toBe(1);

      const dataPtr = Number(native.nd_array_data_ptr(sumHandle[0]));
      expect(dataPtr).toBeGreaterThan(0);
      const view = new Float64Array(toArrayBuffer(dataPtr, 0, 8));
      expect(view[0]).toBeCloseTo(15, 12);
    } finally {
      expect(Number(native.nd_array_release(sumHandle[0]))).toBe(0);
      expect(Number(native.nd_array_release(arr[0]))).toBe(0);
    }
  });

  test("raw SIMD add returns errors for null pointers and computes correct output", () => {
    const n = 256;
    const a = new Float64Array(n);
    const b = new Float64Array(n);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      a[i] = i * 1.5;
      b[i] = i * 2.5;
    }

    const nullStatus = Number(native.nd_simd_add_f64_raw(0, ptr(b), ptr(out), BigInt(n)));
    expect(nullStatus).toBe(1);
    expect(Number(native.nd_last_error_code())).toBe(1);

    const status = Number(native.nd_simd_add_f64_raw(ptr(a), ptr(b), ptr(out), BigInt(n)));
    expect(status).toBe(0);

    for (let i = 0; i < n; i++) {
      expect(out[i]).toBeCloseTo(a[i] + b[i], 12);
    }
  });

  test("last_error_message cap behavior is bounded and reports length", () => {
    const shape = new BigInt64Array([4n]);
    const out = new BigUint64Array(1);
    const status = Number(native.nd_array_alloc(999, ptr(shape), 1, 0, ptr(out)));
    expect(status).toBe(2);

    const lenOut = new BigUint64Array(1);
    const zeroCapBuf = new Uint8Array(1);
    expect(Number(native.nd_last_error_message(ptr(zeroCapBuf), 0n, ptr(lenOut)))).toBe(0);
    expect(Number(lenOut[0])).toBeGreaterThan(0);

    const tiny = new Uint8Array(4);
    const lenOut2 = new BigUint64Array(1);
    expect(Number(native.nd_last_error_message(ptr(tiny), 4n, ptr(lenOut2)))).toBe(0);
    expect(Number(lenOut2[0])).toBeGreaterThan(0);
    expect(tiny.length).toBe(4);
  });

  test("error code clears after successful operation", () => {
    const shape = new BigInt64Array([4n]);
    const badOut = new BigUint64Array(1);
    expect(Number(native.nd_array_alloc(999, ptr(shape), 1, 0, ptr(badOut)))).toBe(2);
    expect(Number(native.nd_last_error_code())).toBe(2);

    const goodOut = new BigUint64Array(1);
    expect(Number(native.nd_array_alloc(4, ptr(shape), 1, 0, ptr(goodOut)))).toBe(0);
    expect(Number(native.nd_last_error_code())).toBe(0);
    expect(Number(native.nd_array_release(goodOut[0]))).toBe(0);
  });

  test("add_into supports output aliasing with lhs", () => {
    const a = new Float64Array([1, 2, 3, 4]);
    const b = new Float64Array([10, 20, 30, 40]);
    const shape = new BigInt64Array([4n]);
    const ha = new BigUint64Array(1);
    const hb = new BigUint64Array(1);

    expect(Number(native.nd_array_from_host_copy(ptr(a), 4, ptr(shape), 0, 1, 0, ptr(ha)))).toBe(0);
    expect(Number(native.nd_array_from_host_copy(ptr(b), 4, ptr(shape), 0, 1, 0, ptr(hb)))).toBe(0);

    try {
      expect(Number(native.nd_add_into(ha[0], hb[0], ha[0]))).toBe(0);

      const p = Number(native.nd_array_data_ptr(ha[0]));
      const view = new Float64Array(toArrayBuffer(p, 0, 4 * 8));
      expect(Array.from(view)).toEqual([11, 22, 33, 44]);
    } finally {
      expect(Number(native.nd_array_release(ha[0]))).toBe(0);
      expect(Number(native.nd_array_release(hb[0]))).toBe(0);
    }
  });

  test("build_version_cstr and metadata APIs return usable values", () => {
    expect(String(native.nd_build_version_cstr()).length).toBeGreaterThan(0);

    const h = alloc1d(5);
    try {
      const elems = new BigUint64Array(1);
      const bytes = new BigUint64Array(1);
      const contiguous = new Uint32Array(1);
      const dtype = new Uint32Array(1);

      expect(Number(native.nd_array_elem_count(h, ptr(elems)))).toBe(0);
      expect(Number(native.nd_array_byte_len(h, ptr(bytes)))).toBe(0);
      expect(Number(native.nd_array_is_contiguous(h, ptr(contiguous)))).toBe(0);
      expect(Number(native.nd_array_dtype(h, ptr(dtype)))).toBe(0);

      expect(Number(elems[0])).toBe(5);
      expect(Number(bytes[0])).toBe(40);
      expect(contiguous[0]).toBe(1);
      expect(dtype[0]).toBe(4);
    } finally {
      expect(Number(native.nd_array_release(h))).toBe(0);
    }
  });

  test("reshape/transpose/make_contiguous and export_bytes integrate correctly", () => {
    const data = new Float64Array([1, 2, 3, 4]);
    const shape2x2 = new BigInt64Array([2n, 2n]);
    const base = new BigUint64Array(1);
    expect(Number(native.nd_array_from_host_copy(ptr(data), 4, ptr(shape2x2), 0, 2, 0, ptr(base)))).toBe(0);

    const reshaped = new BigUint64Array(1);
    const shape4 = new BigInt64Array([4n]);
    expect(Number(native.nd_array_reshape(base[0], ptr(shape4), 1, ptr(reshaped)))).toBe(0);

    const transposed = new BigUint64Array(1);
    const perm = new BigInt64Array([1n, 0n]);
    expect(Number(native.nd_array_transpose(base[0], ptr(perm), 2, ptr(transposed)))).toBe(0);

    const isContig = new Uint32Array(1);
    expect(Number(native.nd_array_is_contiguous(transposed[0], ptr(isContig)))).toBe(0);
    expect(isContig[0]).toBe(0);
    const out4 = new BigUint64Array(4);
    expect(Number(native.nd_array_export_bytes(transposed[0], ptr(out4)))).toBe(8);

    const dense = new BigUint64Array(1);
    expect(Number(native.nd_array_make_contiguous(transposed[0], ptr(dense)))).toBe(0);
    expect(Number(native.nd_array_is_contiguous(dense[0], ptr(isContig)))).toBe(0);
    expect(isContig[0]).toBe(1);

    const denseView = readF64(dense[0], 4);
    expect(Array.from(denseView)).toEqual([1, 3, 2, 4]);

    const reshapeView = readF64(reshaped[0], 4);
    expect(Array.from(reshapeView)).toEqual([1, 2, 3, 4]);

    expect(Number(native.nd_array_release(dense[0]))).toBe(0);
    expect(Number(native.nd_array_release(transposed[0]))).toBe(0);
    expect(Number(native.nd_array_release(reshaped[0]))).toBe(0);
    expect(Number(native.nd_array_release(base[0]))).toBe(0);
  });

  test("sub/mul/div/matmul produce expected outputs via ABI", () => {
    const shape1d = new BigInt64Array([4n]);
    const aData = new Float64Array([10, 20, 30, 40]);
    const bData = new Float64Array([1, 2, 3, 4]);
    const ha = new BigUint64Array(1);
    const hb = new BigUint64Array(1);
    expect(Number(native.nd_array_from_host_copy(ptr(aData), 4, ptr(shape1d), 0, 1, 0, ptr(ha)))).toBe(0);
    expect(Number(native.nd_array_from_host_copy(ptr(bData), 4, ptr(shape1d), 0, 1, 0, ptr(hb)))).toBe(0);

    const hSub = new BigUint64Array(1);
    const hMul = new BigUint64Array(1);
    const hDiv = new BigUint64Array(1);
    expect(Number(native.nd_sub(ha[0], hb[0], ptr(hSub)))).toBe(0);
    expect(Number(native.nd_mul(ha[0], hb[0], ptr(hMul)))).toBe(0);
    expect(Number(native.nd_div(ha[0], hb[0], ptr(hDiv)))).toBe(0);

    expect(Array.from(readF64(hSub[0], 4))).toEqual([9, 18, 27, 36]);
    expect(Array.from(readF64(hMul[0], 4))).toEqual([10, 40, 90, 160]);
    expect(Array.from(readF64(hDiv[0], 4))).toEqual([10, 10, 10, 10]);

    const mShapeA = new BigInt64Array([2n, 3n]);
    const mShapeB = new BigInt64Array([3n, 2n]);
    const mA = new BigUint64Array(1);
    const mB = new BigUint64Array(1);
    expect(Number(native.nd_array_from_host_copy(ptr(new Float64Array([1, 2, 3, 4, 5, 6])), 4, ptr(mShapeA), 0, 2, 0, ptr(mA)))).toBe(0);
    expect(Number(native.nd_array_from_host_copy(ptr(new Float64Array([7, 8, 9, 10, 11, 12])), 4, ptr(mShapeB), 0, 2, 0, ptr(mB)))).toBe(0);

    const mOut = new BigUint64Array(1);
    expect(Number(native.nd_matmul(mA[0], mB[0], ptr(mOut)))).toBe(0);
    expect(Array.from(readF64(mOut[0], 4))).toEqual([58, 64, 139, 154]);

    expect(Number(native.nd_array_release(mOut[0]))).toBe(0);
    expect(Number(native.nd_array_release(mB[0]))).toBe(0);
    expect(Number(native.nd_array_release(mA[0]))).toBe(0);
    expect(Number(native.nd_array_release(hDiv[0]))).toBe(0);
    expect(Number(native.nd_array_release(hMul[0]))).toBe(0);
    expect(Number(native.nd_array_release(hSub[0]))).toBe(0);
    expect(Number(native.nd_array_release(hb[0]))).toBe(0);
    expect(Number(native.nd_array_release(ha[0]))).toBe(0);
  });

  test("slice/compare/where/sum_axis operate through the C ABI", () => {
    const baseData = new Float64Array([
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
    ]);
    const shape = new BigInt64Array([3n, 3n]);
    const hBase = new BigUint64Array(1);
    expect(Number(native.nd_array_from_host_copy(ptr(baseData), 4, ptr(shape), 0, 2, 0, ptr(hBase)))).toBe(0);

    const starts = new BigInt64Array([0n, 1n]);
    const stops = new BigInt64Array([3n, 3n]);
    const steps = new BigInt64Array([2n, 1n]);
    const hSlice = new BigUint64Array(1);
    expect(Number(native.nd_array_slice(hBase[0], ptr(starts), ptr(stops), ptr(steps), 2, ptr(hSlice)))).toBe(0);

    const isContig = new Uint32Array(1);
    expect(Number(native.nd_array_is_contiguous(hSlice[0], ptr(isContig)))).toBe(0);
    expect(isContig[0]).toBe(0);
    const out4 = new BigUint64Array(4);
    expect(Number(native.nd_array_export_bytes(hSlice[0], ptr(out4)))).toBe(8);

    const hDense = new BigUint64Array(1);
    expect(Number(native.nd_array_make_contiguous(hSlice[0], ptr(hDense)))).toBe(0);
    expect(Array.from(readF64(hDense[0], 4))).toEqual([2, 3, 8, 9]);

    const hCmpRhs = new BigUint64Array(1);
    const cmpData = new Float64Array([2, 2, 2, 2]);
    const cmpShape = new BigInt64Array([4n]);
    expect(Number(native.nd_array_from_host_copy(ptr(cmpData), 4, ptr(cmpShape), 0, 1, 0, ptr(hCmpRhs)))).toBe(0);

    const hCmpLhs = new BigUint64Array(1);
    const lhsData = new Float64Array([1, 2, 3, 4]);
    expect(Number(native.nd_array_from_host_copy(ptr(lhsData), 4, ptr(cmpShape), 0, 1, 0, ptr(hCmpLhs)))).toBe(0);

    const hMask = new BigUint64Array(1);
    expect(Number(native.nd_gt(hCmpLhs[0], hCmpRhs[0], ptr(hMask)))).toBe(0);
    expect(Array.from(readI32(hMask[0], 4))).toEqual([0, 0, 1, 1]);

    const hWhere = new BigUint64Array(1);
    expect(Number(native.nd_where(hMask[0], hCmpLhs[0], hCmpRhs[0], ptr(hWhere)))).toBe(0);
    expect(Array.from(readF64(hWhere[0], 4))).toEqual([2, 2, 3, 4]);

    const hAxis = new BigUint64Array(1);
    expect(Number(native.nd_sum_axis(hBase[0], 1, ptr(hAxis)))).toBe(0);
    expect(Array.from(readF64(hAxis[0], 3))).toEqual([6, 15, 24]);

    expect(Number(native.nd_array_release(hAxis[0]))).toBe(0);
    expect(Number(native.nd_array_release(hWhere[0]))).toBe(0);
    expect(Number(native.nd_array_release(hMask[0]))).toBe(0);
    expect(Number(native.nd_array_release(hCmpLhs[0]))).toBe(0);
    expect(Number(native.nd_array_release(hCmpRhs[0]))).toBe(0);
    expect(Number(native.nd_array_release(hDense[0]))).toBe(0);
    expect(Number(native.nd_array_release(hSlice[0]))).toBe(0);
    expect(Number(native.nd_array_release(hBase[0]))).toBe(0);
  });
});
