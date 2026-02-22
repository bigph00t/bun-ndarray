import { FFIType } from "bun:ffi";

export const symbolDefs = {
  nd_abi_version: { args: [], returns: FFIType.u32 },
  nd_build_version_cstr: { args: [], returns: FFIType.cstring },
  nd_last_error_code: { args: [], returns: FFIType.i32 },
  nd_last_error_message: { args: [FFIType.ptr, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },

  nd_array_alloc: {
    args: [FFIType.u32, FFIType.ptr, FFIType.u8, FFIType.u32, FFIType.ptr],
    returns: FFIType.i32,
  },
  nd_array_from_host_copy: {
    args: [FFIType.ptr, FFIType.u32, FFIType.ptr, FFIType.ptr, FFIType.u8, FFIType.u32, FFIType.ptr],
    returns: FFIType.i32,
  },

  nd_array_retain: { args: [FFIType.u64], returns: FFIType.i32 },
  nd_array_release: { args: [FFIType.u64], returns: FFIType.i32 },
  nd_array_clone: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_array_make_contiguous: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },

  nd_array_dtype: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_array_ndim: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_array_shape_copy: { args: [FFIType.u64, FFIType.ptr, FFIType.u8], returns: FFIType.i32 },
  nd_array_strides_copy: { args: [FFIType.u64, FFIType.ptr, FFIType.u8], returns: FFIType.i32 },
  nd_array_elem_count: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_array_byte_len: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_array_is_contiguous: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },

  nd_array_reshape: { args: [FFIType.u64, FFIType.ptr, FFIType.u8, FFIType.ptr], returns: FFIType.i32 },
  nd_array_transpose: { args: [FFIType.u64, FFIType.ptr, FFIType.u8, FFIType.ptr], returns: FFIType.i32 },
  nd_array_slice: {
    args: [FFIType.u64, FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.u8, FFIType.ptr],
    returns: FFIType.i32,
  },
  nd_array_export_bytes: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },

  nd_add: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_sub: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_mul: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_div: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_eq: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_lt: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_gt: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_where: { args: [FFIType.u64, FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_sum_all: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_sum_axis: { args: [FFIType.u64, FFIType.i32, FFIType.ptr], returns: FFIType.i32 },
  nd_matmul: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },

  nd_job_submit_matmul: { args: [FFIType.u64, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_job_poll: { args: [FFIType.u64, FFIType.ptr, FFIType.ptr], returns: FFIType.i32 },
  nd_job_take_result: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  nd_job_cancel: { args: [FFIType.u64], returns: FFIType.i32 },

  // Legacy symbols retained for low-level regression/perf tests.
  nd_add_into: { args: [FFIType.u64, FFIType.u64, FFIType.u64], returns: FFIType.i32 },
  nd_array_data_ptr: { args: [FFIType.u64], returns: FFIType.ptr },
  nd_array_len: { args: [FFIType.u64, FFIType.ptr], returns: FFIType.i32 },

  nd_simd_width_f64: { args: [], returns: FFIType.u64 },
  nd_simd_add_f64_raw: { args: [FFIType.ptr, FFIType.ptr, FFIType.ptr, FFIType.u64], returns: FFIType.i32 },
  nd_simd_sum_f64_raw: { args: [FFIType.ptr, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
} as const;

export type NativeSymbols = {
  [K in keyof typeof symbolDefs]: (...args: any[]) => any;
};

export const ND_DTYPE_F32 = 1;
export const ND_DTYPE_I32 = 3;
export const ND_DTYPE_F64 = 4;

export const ND_BOOL_FALSE = 0;
export const ND_BOOL_TRUE = 1;
