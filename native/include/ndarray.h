#ifndef NDARRAY_H
#define NDARRAY_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
  #define ND_API __declspec(dllexport)
#else
  #define ND_API __attribute__((visibility("default")))
#endif

typedef int32_t  nd_status_t;
typedef uint64_t nd_handle_t;
typedef uint32_t nd_dtype_t;
typedef uint32_t nd_flags_t;
typedef uint32_t nd_bool_t;

enum {
  ND_OK = 0,
  ND_E_INVALID_ARG = 1,
  ND_E_INVALID_DTYPE = 2,
  ND_E_INVALID_SHAPE = 3,
  ND_E_INVALID_STRIDES = 4,
  ND_E_INVALID_ALIGNMENT = 5,
  ND_E_STALE_HANDLE = 6,
  ND_E_OOM = 7,
  ND_E_NOT_CONTIGUOUS = 8,
  ND_E_NOT_IMPLEMENTED = 9,
  ND_E_INTERNAL = 255
};

enum {
  ND_DTYPE_F32 = 1,
  ND_DTYPE_I32 = 3,
  ND_DTYPE_F64 = 4
};

enum {
  ND_BOOL_FALSE = 0,
  ND_BOOL_TRUE = 1
};

enum {
  ND_FLAG_READONLY = 1u << 0
};

ND_API uint32_t nd_abi_version(void);
ND_API const char* nd_build_version_cstr(void);
ND_API nd_status_t nd_last_error_code(void);
ND_API nd_status_t nd_last_error_message(uint8_t* out_utf8, uint64_t cap, uint64_t* out_len);

ND_API nd_status_t nd_array_alloc(
  nd_dtype_t dtype,
  const int64_t* shape,
  uint8_t ndim,
  nd_flags_t flags,
  nd_handle_t* out_handle
);

ND_API nd_status_t nd_array_from_host_copy(
  const uint8_t* data,
  nd_dtype_t dtype,
  const int64_t* shape,
  const int64_t* strides_or_null,
  uint8_t ndim,
  nd_flags_t flags,
  nd_handle_t* out_handle
);

ND_API nd_status_t nd_array_retain(nd_handle_t h);
ND_API nd_status_t nd_array_release(nd_handle_t h);
ND_API nd_status_t nd_array_clone(nd_handle_t h, nd_handle_t* out_handle);
ND_API nd_status_t nd_array_make_contiguous(nd_handle_t h, nd_handle_t* out_handle);

ND_API nd_status_t nd_array_ndim(nd_handle_t h, uint8_t* out_ndim);
ND_API nd_status_t nd_array_shape_copy(nd_handle_t h, int64_t* out_shape, uint8_t cap);
ND_API nd_status_t nd_array_strides_copy(nd_handle_t h, int64_t* out_strides, uint8_t cap);
ND_API nd_status_t nd_array_dtype(nd_handle_t h, nd_dtype_t* out_dtype);
ND_API nd_status_t nd_array_elem_count(nd_handle_t h, uint64_t* out_len);
ND_API nd_status_t nd_array_byte_len(nd_handle_t h, uint64_t* out_len);
ND_API nd_status_t nd_array_is_contiguous(nd_handle_t h, nd_bool_t* out_bool);

ND_API nd_status_t nd_array_reshape(
  nd_handle_t h,
  const int64_t* shape,
  uint8_t ndim,
  nd_handle_t* out_handle
);

ND_API nd_status_t nd_array_transpose(
  nd_handle_t h,
  const int64_t* perm_or_null,
  uint8_t ndim,
  nd_handle_t* out_handle
);

ND_API nd_status_t nd_array_slice(
  nd_handle_t h,
  const int64_t* starts_or_null,
  const int64_t* stops_or_null,
  const int64_t* steps_or_null,
  uint8_t ndim,
  nd_handle_t* out_handle
);

// out4: [data_ptr, byte_len, deallocator_fn_ptr, deallocator_ctx]
ND_API nd_status_t nd_array_export_bytes(nd_handle_t h, uint64_t* out4);

ND_API nd_status_t nd_add(nd_handle_t a, nd_handle_t b, nd_handle_t* out_handle);
ND_API nd_status_t nd_sub(nd_handle_t a, nd_handle_t b, nd_handle_t* out_handle);
ND_API nd_status_t nd_mul(nd_handle_t a, nd_handle_t b, nd_handle_t* out_handle);
ND_API nd_status_t nd_div(nd_handle_t a, nd_handle_t b, nd_handle_t* out_handle);
ND_API nd_status_t nd_eq(nd_handle_t a, nd_handle_t b, nd_handle_t* out_handle);
ND_API nd_status_t nd_lt(nd_handle_t a, nd_handle_t b, nd_handle_t* out_handle);
ND_API nd_status_t nd_gt(nd_handle_t a, nd_handle_t b, nd_handle_t* out_handle);
ND_API nd_status_t nd_where(nd_handle_t cond, nd_handle_t x, nd_handle_t y, nd_handle_t* out_handle);
ND_API nd_status_t nd_sum_all(nd_handle_t a, nd_handle_t* out_handle);
ND_API nd_status_t nd_sum_axis(nd_handle_t a, int32_t axis, nd_handle_t* out_handle);
ND_API nd_status_t nd_matmul(nd_handle_t a, nd_handle_t b, nd_handle_t* out_handle);

ND_API nd_status_t nd_job_submit_matmul(nd_handle_t a, nd_handle_t b, uint64_t* out_job_id);
ND_API nd_status_t nd_job_poll(uint64_t job_id, uint32_t* out_state, nd_status_t* out_result_status);
ND_API nd_status_t nd_job_take_result(uint64_t job_id, nd_handle_t* out_handle);
ND_API nd_status_t nd_job_cancel(uint64_t job_id);

// Legacy low-level test hooks kept for benchmark/regression scaffolding.
ND_API nd_status_t nd_add_into(nd_handle_t a, nd_handle_t b, nd_handle_t out_handle);
ND_API uint8_t* nd_array_data_ptr(nd_handle_t h);
ND_API nd_status_t nd_array_len(nd_handle_t h, uint64_t* out_len);

ND_API uint64_t nd_simd_width_f64(void);
ND_API nd_status_t nd_simd_add_f64_raw(const double* a, const double* b, double* out, uint64_t len);
ND_API nd_status_t nd_simd_sum_f64_raw(const double* data, uint64_t len, double* out_sum);

#endif
