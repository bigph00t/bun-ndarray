import { ptr, toArrayBuffer } from "bun:ffi";
import { native } from "./ffi";
import { checkStatus } from "./errors";
import {
  ND_BOOL_TRUE,
  ND_DTYPE_F32,
  ND_DTYPE_F64,
  ND_DTYPE_I32,
} from "./symbols";

export type DType = "f32" | "f64" | "i32";
export type Typed = Float32Array | Float64Array | Int32Array;
export type SliceSpec = {
  start?: number;
  stop?: number;
  step?: number;
};

const finalizer = new FinalizationRegistry<number>((handle) => {
  // Best effort only; explicit dispose is still required.
  native.nd_array_release(BigInt(handle));
});

function product(shape: readonly number[]): number {
  let out = 1;
  for (const dim of shape) {
    out *= dim;
  }
  return out;
}

function assertShape(shape: readonly number[]): void {
  if (shape.length > 8) {
    throw new Error("ndarray scaffold supports up to 8 dimensions");
  }
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim <= 0) {
      throw new Error(`invalid dimension: ${dim}`);
    }
  }
}

function ptrOrNull(view: ArrayBufferView): number {
  return view.byteLength === 0 ? 0 : ptr(view);
}

function outHandle(): BigUint64Array {
  return new BigUint64Array(1);
}

function toHandle(value: bigint): number {
  const handle = Number(value);
  if (!Number.isSafeInteger(handle) || handle <= 0) {
    throw new Error(`native returned invalid handle: ${value.toString()}`);
  }
  return handle;
}

function shapeToI64(shape: readonly number[]): BigInt64Array {
  const out = new BigInt64Array(shape.length);
  for (let i = 0; i < shape.length; i++) {
    out[i] = BigInt(shape[i]);
  }
  return out;
}

function permToI64(perm: readonly number[]): BigInt64Array {
  const out = new BigInt64Array(perm.length);
  for (let i = 0; i < perm.length; i++) {
    out[i] = BigInt(perm[i]);
  }
  return out;
}

function withOutHandle(symbol: string, invoker: (out: BigUint64Array) => number): number {
  const out = outHandle();
  checkStatus(symbol, Number(invoker(out)));
  return toHandle(out[0]);
}

function dtypeToCode(dtype: DType): number {
  switch (dtype) {
    case "f32":
      return ND_DTYPE_F32;
    case "f64":
      return ND_DTYPE_F64;
    case "i32":
      return ND_DTYPE_I32;
  }
}

function codeToDType(code: number): DType {
  switch (code) {
    case ND_DTYPE_F32:
      return "f32";
    case ND_DTYPE_F64:
      return "f64";
    case ND_DTYPE_I32:
      return "i32";
    default:
      throw new Error(`unknown dtype code: ${code}`);
  }
}

function inferDTypeFromTyped(data: Typed): DType {
  if (data instanceof Float32Array) {
    return "f32";
  }
  if (data instanceof Float64Array) {
    return "f64";
  }
  if (data instanceof Int32Array) {
    return "i32";
  }
  throw new Error("unsupported TypedArray");
}

function typedArrayCtor(dtype: DType): { new (buffer: ArrayBuffer): Typed; BYTES_PER_ELEMENT: number } {
  switch (dtype) {
    case "f32":
      return Float32Array as unknown as { new (buffer: ArrayBuffer): Typed; BYTES_PER_ELEMENT: number };
    case "f64":
      return Float64Array as unknown as { new (buffer: ArrayBuffer): Typed; BYTES_PER_ELEMENT: number };
    case "i32":
      return Int32Array as unknown as { new (buffer: ArrayBuffer): Typed; BYTES_PER_ELEMENT: number };
  }
}

function asFloat64(data: Typed): Float64Array {
  if (!(data instanceof Float64Array)) {
    throw new Error("operation currently supports f64 arrays only");
  }
  return data;
}

export class NDArray {
  #handle: number;
  #disposed = false;

  private constructor(handle: number) {
    this.#handle = handle;
    finalizer.register(this, handle, this);
  }

  static zeros(shape: readonly number[], dtype: DType = "f64"): NDArray {
    assertShape(shape);
    const shapeBuf = shapeToI64(shape);
    const shapePtr = ptrOrNull(shapeBuf);

    const handle = withOutHandle("nd_array_alloc", (out) =>
      Number(native.nd_array_alloc(dtypeToCode(dtype), shapePtr, shape.length, 0, ptr(out))),
    );

    return new NDArray(handle);
  }

  static ones(shape: readonly number[], dtype: DType = "f64"): NDArray {
    const arr = NDArray.zeros(shape, dtype);
    arr.toTypedArray().fill(1);
    return arr;
  }

  static fromTyped(data: Typed, shape?: readonly number[], dtype?: DType): NDArray {
    const inferred = inferDTypeFromTyped(data);
    const resolvedDType = dtype ?? inferred;

    if (resolvedDType !== inferred) {
      throw new Error(`dtype ${resolvedDType} does not match source typed array ${inferred}`);
    }

    const resolvedShape = shape ? [...shape] : [data.length];
    assertShape(resolvedShape);

    if (product(resolvedShape) !== data.length) {
      throw new Error("shape product must match typed array length");
    }

    const shapeBuf = shapeToI64(resolvedShape);
    const shapePtr = ptrOrNull(shapeBuf);

    const handle = withOutHandle("nd_array_from_host_copy", (out) =>
      Number(
        native.nd_array_from_host_copy(
          ptr(data),
          dtypeToCode(resolvedDType),
          shapePtr,
          0,
          resolvedShape.length,
          0,
          ptr(out),
        ),
      ),
    );

    return new NDArray(handle);
  }

  static fromTypedArray(data: Float64Array, shape?: readonly number[]): NDArray {
    return NDArray.fromTyped(data, shape, "f64");
  }

  static simdWidthF64(): number {
    return Number(native.nd_simd_width_f64());
  }

  static abiVersion(): number {
    return Number(native.nd_abi_version());
  }

  static buildVersion(): string {
    return String(native.nd_build_version_cstr());
  }

  get disposed(): boolean {
    return this.#disposed;
  }

  get dtype(): DType {
    this.#assertAlive();

    const out = new Uint32Array(1);
    checkStatus("nd_array_dtype", Number(native.nd_array_dtype(BigInt(this.#handle), ptr(out))));
    return codeToDType(out[0] ?? -1);
  }

  get shape(): number[] {
    this.#assertAlive();

    const ndimOut = new Uint8Array(1);
    checkStatus("nd_array_ndim", Number(native.nd_array_ndim(BigInt(this.#handle), ptr(ndimOut))));

    const ndim = ndimOut[0];
    if (ndim === 0) {
      return [];
    }

    const out = new BigInt64Array(ndim);
    checkStatus("nd_array_shape_copy", Number(native.nd_array_shape_copy(BigInt(this.#handle), ptr(out), ndim)));
    return Array.from(out, Number);
  }

  get strides(): number[] {
    this.#assertAlive();

    const ndimOut = new Uint8Array(1);
    checkStatus("nd_array_ndim", Number(native.nd_array_ndim(BigInt(this.#handle), ptr(ndimOut))));

    const ndim = ndimOut[0];
    if (ndim === 0) {
      return [];
    }

    const out = new BigInt64Array(ndim);
    checkStatus(
      "nd_array_strides_copy",
      Number(native.nd_array_strides_copy(BigInt(this.#handle), ptr(out), ndim)),
    );
    return Array.from(out, Number);
  }

  get size(): number {
    this.#assertAlive();

    const out = new BigUint64Array(1);
    checkStatus(
      "nd_array_elem_count",
      Number(native.nd_array_elem_count(BigInt(this.#handle), ptr(out))),
    );
    return Number(out[0]);
  }

  get length(): number {
    return this.size;
  }

  get byteLength(): number {
    this.#assertAlive();

    const out = new BigUint64Array(1);
    checkStatus("nd_array_byte_len", Number(native.nd_array_byte_len(BigInt(this.#handle), ptr(out))));
    return Number(out[0]);
  }

  get isContiguous(): boolean {
    this.#assertAlive();

    const out = new Uint32Array(1);
    checkStatus(
      "nd_array_is_contiguous",
      Number(native.nd_array_is_contiguous(BigInt(this.#handle), ptr(out))),
    );
    return out[0] === ND_BOOL_TRUE;
  }

  #assertAlive(): void {
    if (this.#disposed) {
      throw new Error("NDArray is disposed");
    }
  }

  #asBigIntHandle(): bigint {
    return BigInt(this.#handle);
  }

  clone(): NDArray {
    this.#assertAlive();
    const handle = withOutHandle("nd_array_clone", (out) =>
      Number(native.nd_array_clone(this.#asBigIntHandle(), ptr(out))),
    );
    return new NDArray(handle);
  }

  contiguous(): NDArray {
    this.#assertAlive();
    const handle = withOutHandle("nd_array_make_contiguous", (out) =>
      Number(native.nd_array_make_contiguous(this.#asBigIntHandle(), ptr(out))),
    );
    return new NDArray(handle);
  }

  reshape(shape: readonly number[]): NDArray {
    this.#assertAlive();
    assertShape(shape);

    const shapeBuf = shapeToI64(shape);
    const shapePtr = ptrOrNull(shapeBuf);
    const handle = withOutHandle("nd_array_reshape", (out) =>
      Number(native.nd_array_reshape(this.#asBigIntHandle(), shapePtr, shape.length, ptr(out))),
    );

    return new NDArray(handle);
  }

  transpose(perm?: readonly number[]): NDArray {
    this.#assertAlive();

    const ndim = this.shape.length;
    if (perm && perm.length !== ndim) {
      throw new Error(`transpose permutation must have ${ndim} entries`);
    }

    const permBuf = perm ? permToI64(perm) : null;
    const permPtr = permBuf ? ptrOrNull(permBuf) : 0;
    const ndimArg = perm ? perm.length : ndim;

    const handle = withOutHandle("nd_array_transpose", (out) =>
      Number(native.nd_array_transpose(this.#asBigIntHandle(), permPtr, ndimArg, ptr(out))),
    );

    return new NDArray(handle);
  }

  slice(specs: readonly SliceSpec[]): NDArray {
    this.#assertAlive();

    const shape = this.shape;
    const ndim = shape.length;
    if (specs.length > ndim) {
      throw new Error(`slice expects at most ${ndim} dimensions`);
    }

    const starts = new BigInt64Array(ndim);
    const stops = new BigInt64Array(ndim);
    const steps = new BigInt64Array(ndim);

    for (let i = 0; i < ndim; i++) {
      const spec = specs[i] ?? {};
      const step = spec.step ?? 1;
      if (!Number.isInteger(step) || step === 0) {
        throw new Error(`invalid slice step at dim ${i}: ${step}`);
      }
      const defaultStart = step > 0 ? 0 : shape[i] - 1;
      const defaultStop = step > 0 ? shape[i] : -1;

      starts[i] = BigInt(spec.start ?? defaultStart);
      stops[i] = BigInt(spec.stop ?? defaultStop);
      steps[i] = BigInt(step);
    }

    const handle = withOutHandle("nd_array_slice", (out) =>
      Number(
        native.nd_array_slice(
          this.#asBigIntHandle(),
          ptrOrNull(starts),
          ptrOrNull(stops),
          ptrOrNull(steps),
          ndim,
          ptr(out),
        ),
      ),
    );

    return new NDArray(handle);
  }

  #binary(rhs: NDArray, symbol: "nd_add" | "nd_sub" | "nd_mul" | "nd_div"): NDArray {
    this.#assertAlive();
    rhs.#assertAlive();

    const handle = withOutHandle(symbol, (out) => {
      if (symbol === "nd_add") {
        return Number(native.nd_add(this.#asBigIntHandle(), rhs.#asBigIntHandle(), ptr(out)));
      }
      if (symbol === "nd_sub") {
        return Number(native.nd_sub(this.#asBigIntHandle(), rhs.#asBigIntHandle(), ptr(out)));
      }
      if (symbol === "nd_mul") {
        return Number(native.nd_mul(this.#asBigIntHandle(), rhs.#asBigIntHandle(), ptr(out)));
      }
      return Number(native.nd_div(this.#asBigIntHandle(), rhs.#asBigIntHandle(), ptr(out)));
    });

    return new NDArray(handle);
  }

  add(rhs: NDArray): NDArray {
    return this.#binary(rhs, "nd_add");
  }

  sub(rhs: NDArray): NDArray {
    return this.#binary(rhs, "nd_sub");
  }

  mul(rhs: NDArray): NDArray {
    return this.#binary(rhs, "nd_mul");
  }

  div(rhs: NDArray): NDArray {
    return this.#binary(rhs, "nd_div");
  }

  #compare(rhs: NDArray, symbol: "nd_eq" | "nd_lt" | "nd_gt"): NDArray {
    this.#assertAlive();
    rhs.#assertAlive();

    const handle = withOutHandle(symbol, (out) => {
      if (symbol === "nd_eq") {
        return Number(native.nd_eq(this.#asBigIntHandle(), rhs.#asBigIntHandle(), ptr(out)));
      }
      if (symbol === "nd_lt") {
        return Number(native.nd_lt(this.#asBigIntHandle(), rhs.#asBigIntHandle(), ptr(out)));
      }
      return Number(native.nd_gt(this.#asBigIntHandle(), rhs.#asBigIntHandle(), ptr(out)));
    });

    return new NDArray(handle);
  }

  eq(rhs: NDArray): NDArray {
    return this.#compare(rhs, "nd_eq");
  }

  lt(rhs: NDArray): NDArray {
    return this.#compare(rhs, "nd_lt");
  }

  gt(rhs: NDArray): NDArray {
    return this.#compare(rhs, "nd_gt");
  }

  where(whenTrue: NDArray, whenFalse: NDArray): NDArray {
    this.#assertAlive();
    whenTrue.#assertAlive();
    whenFalse.#assertAlive();

    const handle = withOutHandle("nd_where", (out) =>
      Number(
        native.nd_where(
          this.#asBigIntHandle(),
          whenTrue.#asBigIntHandle(),
          whenFalse.#asBigIntHandle(),
          ptr(out),
        ),
      ),
    );
    return new NDArray(handle);
  }

  addInto(rhs: NDArray, out: NDArray): void {
    this.#assertAlive();
    rhs.#assertAlive();
    out.#assertAlive();

    checkStatus(
      "nd_add_into",
      Number(native.nd_add_into(this.#asBigIntHandle(), rhs.#asBigIntHandle(), out.#asBigIntHandle())),
    );
  }

  sumAll(): NDArray {
    this.#assertAlive();

    const handle = withOutHandle("nd_sum_all", (out) =>
      Number(native.nd_sum_all(this.#asBigIntHandle(), ptr(out))),
    );
    return new NDArray(handle);
  }

  sumAxis(axis: number): NDArray {
    this.#assertAlive();
    if (!Number.isInteger(axis)) {
      throw new Error("axis must be an integer");
    }

    const handle = withOutHandle("nd_sum_axis", (out) =>
      Number(native.nd_sum_axis(this.#asBigIntHandle(), axis, ptr(out))),
    );
    return new NDArray(handle);
  }

  sum(): number {
    using scalar = this.sumAll();
    const view = asFloat64(scalar.toTypedArray());
    return view[0] ?? 0;
  }

  matmul(rhs: NDArray): NDArray {
    this.#assertAlive();
    rhs.#assertAlive();

    const handle = withOutHandle("nd_matmul", (out) =>
      Number(native.nd_matmul(this.#asBigIntHandle(), rhs.#asBigIntHandle(), ptr(out))),
    );
    return new NDArray(handle);
  }

  toTypedArray(options?: { copy?: boolean }): Typed {
    this.#assertAlive();

    const out4 = new BigUint64Array(4);
    const status = Number(native.nd_array_export_bytes(this.#asBigIntHandle(), ptr(out4)));
    if (status !== 0) {
      if (options?.copy) {
        using dense = this.contiguous();
        return dense.toTypedArray({ copy: true });
      }
      checkStatus("nd_array_export_bytes", status);
    }

    const dataPtr = Number(out4[0]);
    const byteLen = Number(out4[1]);
    if (!Number.isFinite(dataPtr) || dataPtr <= 0) {
      throw new Error("nd_array_export_bytes returned invalid data pointer");
    }

    const ab = toArrayBuffer(dataPtr, 0, byteLen);
    const ctor = typedArrayCtor(this.dtype);
    const view = new ctor(ab);

    if (options?.copy) {
      return view.slice();
    }
    return view;
  }

  toFloat64Array(options?: { copy?: boolean }): Float64Array {
    return asFloat64(this.toTypedArray(options));
  }

  dispose(): void {
    if (this.#disposed) {
      return;
    }

    this.#disposed = true;
    finalizer.unregister(this);

    checkStatus("nd_array_release", Number(native.nd_array_release(this.#asBigIntHandle())));
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}
