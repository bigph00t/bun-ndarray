const std = @import("std");
const abi_mod = @import("abi.zig");
const dtype_mod = @import("dtype.zig");
const error_mod = @import("error.zig");
const header_mod = @import("header.zig");
const iter_mod = @import("iter.zig");
const jobs_mod = @import("jobs.zig");
const registry = @import("registry.zig");
const elem_kernel = @import("kernels/add.zig");
const reduce_kernel = @import("kernels/reduce.zig");
const matmul_kernel = @import("kernels/matmul.zig");

const DType = dtype_mod.DType;
const NdStatus = error_mod.NdStatus;
const ArrayHeader = header_mod.ArrayHeader;

const allocator = std.heap.page_allocator;

fn setErr(status: NdStatus, msg: []const u8) i32 {
    _ = error_mod.set(status, msg);
    return @intFromEnum(status);
}

fn clearErr() void {
    error_mod.clear();
}

fn mapError(err: anyerror) i32 {
    return switch (err) {
        error.OutOfMemory => setErr(.oom, "out of memory"),
        error.InvalidArg => setErr(.invalid_arg, @errorName(err)),
        error.InvalidShape, error.ShapeOverflow, error.ShapeMismatch => setErr(.invalid_shape, @errorName(err)),
        error.InvalidStrides, error.InvalidOffset => setErr(.invalid_strides, @errorName(err)),
        error.InvalidDType => setErr(.invalid_dtype, @errorName(err)),
        error.InvalidHandle, error.StaleHandle, error.HeaderAlreadyReleased, error.BlockAlreadyFreed => setErr(.stale_handle, @errorName(err)),
        else => setErr(.internal, @errorName(err)),
    };
}

fn parseShape(ndim: u8, shape_ptr: ?[*]const i64, shape_buf: *[header_mod.MAX_DIMS]i64) ![]const i64 {
    if (ndim > header_mod.MAX_DIMS) return error.InvalidShape;
    if (ndim == 0) return shape_buf[0..0];

    const src = shape_ptr orelse return error.InvalidShape;
    var i: usize = 0;
    while (i < ndim) : (i += 1) {
        shape_buf[i] = src[i];
    }
    return shape_buf[0..ndim];
}

fn parseOptionalI64Slice(ndim: u8, ptr_in: ?[*]const i64, buf: *[header_mod.MAX_DIMS]i64) !?[]const i64 {
    if (ptr_in == null) return null;
    if (ndim > header_mod.MAX_DIMS) return error.InvalidShape;

    const src = ptr_in.?;
    var i: usize = 0;
    while (i < ndim) : (i += 1) {
        buf[i] = src[i];
    }
    return buf[0..ndim];
}

fn parsePermutation(ndim: u8, perm_ptr: ?[*]const i64, perm_buf: *[header_mod.MAX_DIMS]i64) ![]const i64 {
    if (ndim > header_mod.MAX_DIMS) return error.InvalidShape;
    if (ndim == 0) return perm_buf[0..0];

    var seen: [header_mod.MAX_DIMS]bool = [_]bool{false} ** header_mod.MAX_DIMS;

    if (perm_ptr) |perm| {
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            const value = perm[i];
            if (value < 0 or value >= ndim) return error.InvalidShape;
            const idx: usize = @intCast(value);
            if (seen[idx]) return error.InvalidShape;
            seen[idx] = true;
            perm_buf[i] = value;
        }
    } else {
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            perm_buf[i] = @as(i64, @intCast(ndim - 1 - i));
        }
    }

    return perm_buf[0..ndim];
}

fn linearByteOffset(flat: u64, shape: []const i64, strides: []const i64) !i64 {
    return iter_mod.linearByteOffset(flat, shape, strides);
}

fn copyHostStridedToContiguous(dst: *ArrayHeader, src_ptr: [*]const u8, src_shape: []const i64, src_strides: []const i64) !void {
    if (src_shape.len != src_strides.len) return error.InvalidShape;
    if (src_shape.len != dst.shape.len) return error.InvalidShape;

    const elem_size = dst.dtype.byteSize();
    const dst_bytes = dst.block.ptr[0..dst.block.byte_len];

    var flat: u64 = 0;
    while (flat < dst.len) : (flat += 1) {
        const src_off_i64 = try linearByteOffset(flat, src_shape, src_strides);
        if (src_off_i64 < 0) return error.InvalidStrides;

        const src_off: usize = @intCast(src_off_i64);
        const dst_off_u64 = std.math.mul(u64, flat, elem_size) catch return error.ShapeOverflow;
        if (dst_off_u64 > std.math.maxInt(usize)) return error.ShapeOverflow;
        const dst_off: usize = @intCast(dst_off_u64);

        @memcpy(dst_bytes[dst_off..][0..elem_size], src_ptr[src_off..][0..elem_size]);
    }
}

fn copyArrayToContiguous(src: *const ArrayHeader, flags: u32) !*ArrayHeader {
    return src.cloneContiguous(allocator, flags);
}

const Broadcast = struct {
    shape: []const i64,
    a_strides: []const i64,
    b_strides: []const i64,
};

fn computeBroadcast(
    a: *const ArrayHeader,
    b: *const ArrayHeader,
    shape_buf: *[header_mod.MAX_DIMS]i64,
    a_strides_buf: *[header_mod.MAX_DIMS]i64,
    b_strides_buf: *[header_mod.MAX_DIMS]i64,
) !Broadcast {
    const a_ndim = a.shape.len;
    const b_ndim = b.shape.len;
    const out_ndim = @max(a_ndim, b_ndim);
    if (out_ndim > header_mod.MAX_DIMS) return error.InvalidShape;

    var d: usize = 0;
    while (d < out_ndim) : (d += 1) {
        const a_has_dim = d + a_ndim >= out_ndim;
        const b_has_dim = d + b_ndim >= out_ndim;

        const a_idx = if (a_has_dim) d + a_ndim - out_ndim else 0;
        const b_idx = if (b_has_dim) d + b_ndim - out_ndim else 0;

        const a_dim = if (a_has_dim) a.shape[a_idx] else 1;
        const b_dim = if (b_has_dim) b.shape[b_idx] else 1;

        if (a_dim != b_dim and a_dim != 1 and b_dim != 1) return error.ShapeMismatch;

        shape_buf[d] = if (a_dim > b_dim) a_dim else b_dim;
        a_strides_buf[d] = if (!a_has_dim or a_dim == 1) 0 else a.strides[a_idx];
        b_strides_buf[d] = if (!b_has_dim or b_dim == 1) 0 else b.strides[b_idx];
    }

    return .{
        .shape = shape_buf[0..out_ndim],
        .a_strides = a_strides_buf[0..out_ndim],
        .b_strides = b_strides_buf[0..out_ndim],
    };
}

fn shapeEquals(header: *const ArrayHeader, shape: []const i64) bool {
    if (header.shape.len != shape.len) return false;
    var i: usize = 0;
    while (i < shape.len) : (i += 1) {
        if (header.shape[i] != shape[i]) return false;
    }
    return true;
}

fn applyBinaryF64(op: elem_kernel.BinaryOp, a: f64, b: f64) f64 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
    };
}

fn applyBinaryF32(op: elem_kernel.BinaryOp, a: f32, b: f32) f32 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
    };
}

fn applyBinaryI32(op: elem_kernel.BinaryOp, a: i32, b: i32) !i32 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => blk: {
            if (b == 0) return error.InvalidArg;
            break :blk @divTrunc(a, b);
        },
    };
}

fn applyBinaryGeneric(op: elem_kernel.BinaryOp, a: *const ArrayHeader, b: *const ArrayHeader, out: *ArrayHeader) !void {
    if (a.dtype != b.dtype or a.dtype != out.dtype) return error.InvalidDType;

    var shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var a_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var b_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const bc = try computeBroadcast(a, b, &shape_buf, &a_strides_buf, &b_strides_buf);

    if (!shapeEquals(out, bc.shape)) return error.ShapeMismatch;

    if (a.dtype == .f64 and a.hasSameShape(b) and a.hasSameShape(out) and a.isContiguous() and b.isContiguous() and out.isContiguous()) {
        const pa = try a.asConstF64();
        const pb = try b.asConstF64();
        const po = try out.asMutF64();
        const len: usize = @intCast(out.len);
        elem_kernel.binaryRawF64(op, pa, pb, po, len);
        return;
    }

    var flat: u64 = 0;
    while (flat < out.len) : (flat += 1) {
        const a_off = try linearByteOffset(flat, bc.shape, bc.a_strides);
        const b_off = try linearByteOffset(flat, bc.shape, bc.b_strides);

        switch (a.dtype) {
            .f64 => {
                const av = try a.readF64AtByteOffset(a_off);
                const bv = try b.readF64AtByteOffset(b_off);
                try out.writeF64AtByteOffset(try out.linearOffset(flat), applyBinaryF64(op, av, bv));
            },
            .f32 => {
                const av = try a.readF32AtByteOffset(a_off);
                const bv = try b.readF32AtByteOffset(b_off);
                try out.writeF32AtByteOffset(try out.linearOffset(flat), applyBinaryF32(op, av, bv));
            },
            .i32 => {
                const av = try a.readI32AtByteOffset(a_off);
                const bv = try b.readI32AtByteOffset(b_off);
                try out.writeI32AtByteOffset(try out.linearOffset(flat), try applyBinaryI32(op, av, bv));
            },
        }
    }
}

fn createBroadcastOutput(a: *const ArrayHeader, b: *const ArrayHeader) !*ArrayHeader {
    if (a.dtype != b.dtype) return error.InvalidDType;

    var shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var a_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var b_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const bc = try computeBroadcast(a, b, &shape_buf, &a_strides_buf, &b_strides_buf);

    return ArrayHeader.allocateZeros(allocator, a.dtype, bc.shape, 0);
}

fn createBroadcastOutputWithDType(a: *const ArrayHeader, b: *const ArrayHeader, dtype: DType) !*ArrayHeader {
    var shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var a_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var b_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const bc = try computeBroadcast(a, b, &shape_buf, &a_strides_buf, &b_strides_buf);
    return ArrayHeader.allocateZeros(allocator, dtype, bc.shape, 0);
}

fn computeBroadcastAgainstShape(
    src_shape: []const i64,
    src_strides: []const i64,
    out_shape: []const i64,
    out_strides_buf: *[header_mod.MAX_DIMS]i64,
) ![]const i64 {
    if (src_shape.len != src_strides.len) return error.InvalidShape;
    if (out_shape.len > header_mod.MAX_DIMS) return error.InvalidShape;
    if (src_shape.len > out_shape.len) return error.ShapeMismatch;

    const out_ndim = out_shape.len;
    const src_ndim = src_shape.len;
    var i: usize = 0;
    while (i < out_ndim) : (i += 1) {
        const has_src = i + src_ndim >= out_ndim;
        if (!has_src) {
            out_strides_buf[i] = 0;
            continue;
        }

        const src_idx = i + src_ndim - out_ndim;
        const src_dim = src_shape[src_idx];
        const out_dim = out_shape[i];
        if (src_dim == out_dim) {
            out_strides_buf[i] = src_strides[src_idx];
        } else if (src_dim == 1) {
            out_strides_buf[i] = 0;
        } else {
            return error.ShapeMismatch;
        }
    }

    return out_strides_buf[0..out_ndim];
}

const CompareOp = enum {
    eq,
    lt,
    gt,
};

fn compareAt(op: CompareOp, dtype: DType, a: *const ArrayHeader, b: *const ArrayHeader, a_off: i64, b_off: i64) !i32 {
    return switch (dtype) {
        .f64 => blk: {
            const av = try a.readF64AtByteOffset(a_off);
            const bv = try b.readF64AtByteOffset(b_off);
            break :blk switch (op) {
                .eq => if (av == bv) 1 else 0,
                .lt => if (av < bv) 1 else 0,
                .gt => if (av > bv) 1 else 0,
            };
        },
        .f32 => blk: {
            const av = try a.readF32AtByteOffset(a_off);
            const bv = try b.readF32AtByteOffset(b_off);
            break :blk switch (op) {
                .eq => if (av == bv) 1 else 0,
                .lt => if (av < bv) 1 else 0,
                .gt => if (av > bv) 1 else 0,
            };
        },
        .i32 => blk: {
            const av = try a.readI32AtByteOffset(a_off);
            const bv = try b.readI32AtByteOffset(b_off);
            break :blk switch (op) {
                .eq => if (av == bv) 1 else 0,
                .lt => if (av < bv) 1 else 0,
                .gt => if (av > bv) 1 else 0,
            };
        },
    };
}

fn sumF64AnyLayout(a: *const ArrayHeader) !f64 {
    var acc: f64 = 0;

    if (a.dtype == .f64 and a.isContiguous()) {
        return reduce_kernel.sumAllF64(a);
    }

    var flat: u64 = 0;
    while (flat < a.len) : (flat += 1) {
        const off = try a.linearOffset(flat);
        acc += switch (a.dtype) {
            .f64 => try a.readF64AtByteOffset(off),
            .f32 => @as(f64, try a.readF32AtByteOffset(off)),
            .i32 => @as(f64, @floatFromInt(try a.readI32AtByteOffset(off))),
        };
    }
    return acc;
}

fn registerHeader(header: *ArrayHeader, out_handle: *u64) i32 {
    const h = registry.register(header) catch |err| {
        header.destroy();
        return switch (err) {
            error.RegistryFull => setErr(.oom, "handle registry full"),
            else => mapError(err),
        };
    };

    out_handle.* = h;
    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_abi_version() u32 {
    return abi_mod.ABI_VERSION;
}

export fn nd_build_version_cstr() [*:0]const u8 {
    return abi_mod.BUILD_VERSION;
}

export fn nd_last_error_code() i32 {
    return error_mod.code();
}

export fn nd_last_error_message(out_utf8: ?[*]u8, cap: u64, out_len: ?*u64) i32 {
    const status = error_mod.writeMessage(out_utf8, cap, out_len);
    return @intFromEnum(status);
}

export fn nd_array_alloc(dtype: u32, shape_ptr: ?[*]const i64, ndim: u8, flags: u32, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const parsed_dtype = DType.fromAbi(dtype) catch return setErr(.invalid_dtype, "unsupported dtype");

    var shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const shape = parseShape(ndim, shape_ptr, &shape_buf) catch |err| {
        return mapError(err);
    };

    const header = ArrayHeader.allocateZeros(allocator, parsed_dtype, shape, flags) catch |err| {
        return mapError(err);
    };

    return registerHeader(header, out);
}

export fn nd_array_from_host_copy(
    data: ?[*]const u8,
    dtype: u32,
    shape_ptr: ?[*]const i64,
    strides_or_null: ?[*]const i64,
    ndim: u8,
    flags: u32,
    out_handle: ?*u64,
) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const src = data orelse return setErr(.invalid_arg, "data is null");
    const parsed_dtype = DType.fromAbi(dtype) catch return setErr(.invalid_dtype, "unsupported dtype");

    var shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const shape = parseShape(ndim, shape_ptr, &shape_buf) catch |err| {
        return mapError(err);
    };

    const header = ArrayHeader.allocateZeros(allocator, parsed_dtype, shape, flags) catch |err| {
        return mapError(err);
    };
    errdefer header.destroy();

    var strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const maybe_strides = parseOptionalI64Slice(ndim, strides_or_null, &strides_buf) catch |err| {
        return mapError(err);
    };

    if (maybe_strides) |src_strides| {
        copyHostStridedToContiguous(header, src, shape, src_strides) catch |err| {
            return mapError(err);
        };
    } else {
        const dst = header.block.ptr[0..header.block.byte_len];
        @memcpy(dst, src[0..header.block.byte_len]);
    }

    return registerHeader(header, out);
}

export fn nd_array_retain(handle: u64) i32 {
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    header.retainHandle() catch |err| {
        return mapError(err);
    };

    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_array_release(handle: u64) i32 {
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    const zero = header.releaseHandle() catch |err| {
        return mapError(err);
    };

    if (zero) {
        registry.unregister(handle) catch |err| {
            return mapError(err);
        };
        header.destroy();
    }

    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_array_clone(handle: u64, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    const cloned = copyArrayToContiguous(header, header.flags) catch |err| {
        return mapError(err);
    };

    return registerHeader(cloned, out);
}

export fn nd_array_make_contiguous(handle: u64, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    const contiguous = copyArrayToContiguous(header, header.flags) catch |err| {
        return mapError(err);
    };

    return registerHeader(contiguous, out);
}

export fn nd_array_ndim(handle: u64, out_ndim: ?*u8) i32 {
    const out = out_ndim orelse return setErr(.invalid_arg, "out_ndim is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    out.* = header.ndim;
    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_array_shape_copy(handle: u64, out_shape: ?[*]i64, cap: u8) i32 {
    const out = out_shape orelse return setErr(.invalid_arg, "out_shape is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    if (cap < header.ndim) return setErr(.invalid_arg, "shape buffer too small");

    var i: usize = 0;
    while (i < header.shape.len) : (i += 1) {
        out[i] = header.shape[i];
    }

    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_array_strides_copy(handle: u64, out_strides: ?[*]i64, cap: u8) i32 {
    const out = out_strides orelse return setErr(.invalid_arg, "out_strides is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    if (cap < header.ndim) return setErr(.invalid_arg, "strides buffer too small");

    var i: usize = 0;
    while (i < header.strides.len) : (i += 1) {
        out[i] = header.strides[i];
    }

    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_array_dtype(handle: u64, out_dtype: ?*u32) i32 {
    const out = out_dtype orelse return setErr(.invalid_arg, "out_dtype is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    out.* = @intFromEnum(header.dtype);
    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_array_elem_count(handle: u64, out_len: ?*u64) i32 {
    const out = out_len orelse return setErr(.invalid_arg, "out_len is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    out.* = header.len;
    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_array_byte_len(handle: u64, out_bytes: ?*u64) i32 {
    const out = out_bytes orelse return setErr(.invalid_arg, "out_bytes is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    out.* = header.byteLen();
    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_array_is_contiguous(handle: u64, out_bool: ?*u32) i32 {
    const out = out_bool orelse return setErr(.invalid_arg, "out_bool is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    out.* = if (header.isContiguous()) @intFromEnum(abi_mod.NdBool.true_) else @intFromEnum(abi_mod.NdBool.false_);
    clearErr();
    return @intFromEnum(NdStatus.ok);
}

// Legacy alias retained for backwards compatibility with earlier POC tests.
export fn nd_array_len(handle: u64, out_len: ?*u64) i32 {
    return nd_array_elem_count(handle, out_len);
}

export fn nd_array_data_ptr(handle: u64) ?[*]u8 {
    const header = registry.lookup(handle) catch |err| {
        _ = mapError(err);
        return null;
    };

    const p = header.dataPtr() catch |err| {
        _ = mapError(err);
        return null;
    };

    clearErr();
    return p;
}

export fn nd_array_reshape(handle: u64, shape_ptr: ?[*]const i64, ndim: u8, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    var new_shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const new_shape = parseShape(ndim, shape_ptr, &new_shape_buf) catch |err| {
        return mapError(err);
    };

    const new_len = ArrayHeader.computeLen(new_shape) catch |err| {
        return mapError(err);
    };

    if (new_len != header.len) return setErr(.invalid_shape, "reshape element count mismatch");
    if (!header.isContiguous()) return setErr(.not_contiguous, "reshape requires contiguous source");

    var new_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const new_strides = new_strides_buf[0..new_shape.len];
    ArrayHeader.computeContiguousStrides(header.dtype, new_shape, new_strides) catch |err| {
        return mapError(err);
    };

    const view = ArrayHeader.createView(allocator, header, new_shape, new_strides, header.offset_bytes, header.flags) catch |err| {
        return mapError(err);
    };

    return registerHeader(view, out);
}

export fn nd_array_transpose(handle: u64, perm_or_null: ?[*]const i64, ndim: u8, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    if (ndim != header.ndim) return setErr(.invalid_shape, "transpose ndim mismatch");

    var perm_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const perm = parsePermutation(ndim, perm_or_null, &perm_buf) catch |err| {
        return mapError(err);
    };

    var shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var strides_buf: [header_mod.MAX_DIMS]i64 = undefined;

    var i: usize = 0;
    while (i < perm.len) : (i += 1) {
        const src_dim: usize = @intCast(perm[i]);
        shape_buf[i] = header.shape[src_dim];
        strides_buf[i] = header.strides[src_dim];
    }

    const view = ArrayHeader.createView(allocator, header, shape_buf[0..perm.len], strides_buf[0..perm.len], header.offset_bytes, header.flags) catch |err| {
        return mapError(err);
    };

    return registerHeader(view, out);
}

export fn nd_array_slice(
    handle: u64,
    starts_or_null: ?[*]const i64,
    stops_or_null: ?[*]const i64,
    steps_or_null: ?[*]const i64,
    ndim: u8,
    out_handle: ?*u64,
) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    if (ndim != header.ndim) return setErr(.invalid_shape, "slice ndim mismatch");

    var starts_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var stops_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var steps_buf: [header_mod.MAX_DIMS]i64 = undefined;

    const starts = parseOptionalI64Slice(ndim, starts_or_null, &starts_buf) catch |err| {
        return mapError(err);
    };
    const stops = parseOptionalI64Slice(ndim, stops_or_null, &stops_buf) catch |err| {
        return mapError(err);
    };
    const steps = parseOptionalI64Slice(ndim, steps_or_null, &steps_buf) catch |err| {
        return mapError(err);
    };

    var out_shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var out_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var offset_delta: i64 = 0;

    var i: usize = 0;
    while (i < header.shape.len) : (i += 1) {
        const dim = header.shape[i];
        const step = if (steps) |s| s[i] else 1;
        if (step == 0) return setErr(.invalid_arg, "slice step cannot be zero");

        var start = if (starts) |s| s[i] else if (step > 0) 0 else dim - 1;
        var stop = if (stops) |s| s[i] else if (step > 0) dim else -1;

        if (start < 0) start += dim;
        if (stop < 0) stop += dim;

        if (step > 0) {
            if (start < 0) start = 0;
            if (start > dim) start = dim;
            if (stop < 0) stop = 0;
            if (stop > dim) stop = dim;

            const span = stop - start;
            if (span <= 0) return setErr(.invalid_shape, "empty slices are not supported in scaffold");
            out_shape_buf[i] = @divTrunc(span - 1, step) + 1;
        } else {
            if (start < -1) start = -1;
            if (start >= dim) start = dim - 1;
            if (stop < -1) stop = -1;
            if (stop >= dim) stop = dim - 1;

            const span = start - stop;
            if (span <= 0) return setErr(.invalid_shape, "empty slices are not supported in scaffold");
            out_shape_buf[i] = @divTrunc(span - 1, -step) + 1;
        }

        out_strides_buf[i] = std.math.mul(i64, header.strides[i], step) catch {
            return setErr(.invalid_strides, "slice stride overflow");
        };
        offset_delta = std.math.add(i64, offset_delta, std.math.mul(i64, header.strides[i], start) catch return setErr(.invalid_strides, "slice offset overflow")) catch {
            return setErr(.invalid_strides, "slice offset overflow");
        };
    }

    const view_offset = std.math.add(i64, header.offset_bytes, offset_delta) catch {
        return setErr(.invalid_strides, "slice base offset overflow");
    };
    const view = ArrayHeader.createView(
        allocator,
        header,
        out_shape_buf[0..header.shape.len],
        out_strides_buf[0..header.shape.len],
        view_offset,
        header.flags,
    ) catch |err| {
        return mapError(err);
    };

    return registerHeader(view, out);
}

export fn nd_array_export_bytes(handle: u64, out4: ?[*]u64) i32 {
    const out = out4 orelse return setErr(.invalid_arg, "out4 is null");
    const header = registry.lookup(handle) catch |err| {
        return mapError(err);
    };

    if (!header.isContiguous()) return setErr(.not_contiguous, "array is not contiguous");

    const p = header.dataPtr() catch |err| {
        return mapError(err);
    };

    out[0] = @intFromPtr(p);
    out[1] = header.byteLen();
    out[2] = 0;
    out[3] = 0;

    clearErr();
    return @intFromEnum(NdStatus.ok);
}

fn binaryEntry(op: elem_kernel.BinaryOp, a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");

    const a = registry.lookup(a_handle) catch |err| {
        return mapError(err);
    };
    const b = registry.lookup(b_handle) catch |err| {
        return mapError(err);
    };

    const out_arr = createBroadcastOutput(a, b) catch |err| {
        return mapError(err);
    };

    applyBinaryGeneric(op, a, b, out_arr) catch |err| {
        out_arr.destroy();
        return mapError(err);
    };

    return registerHeader(out_arr, out);
}

export fn nd_add(a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    return binaryEntry(.add, a_handle, b_handle, out_handle);
}

export fn nd_sub(a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    return binaryEntry(.sub, a_handle, b_handle, out_handle);
}

export fn nd_mul(a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    return binaryEntry(.mul, a_handle, b_handle, out_handle);
}

export fn nd_div(a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    return binaryEntry(.div, a_handle, b_handle, out_handle);
}

fn compareEntry(op: CompareOp, a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const a = registry.lookup(a_handle) catch |err| {
        return mapError(err);
    };
    const b = registry.lookup(b_handle) catch |err| {
        return mapError(err);
    };

    if (a.dtype != b.dtype) return setErr(.invalid_dtype, "compare requires matching dtypes");

    const out_arr = createBroadcastOutputWithDType(a, b, .i32) catch |err| {
        return mapError(err);
    };

    var shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var a_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var b_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const bc = computeBroadcast(a, b, &shape_buf, &a_strides_buf, &b_strides_buf) catch |err| {
        out_arr.destroy();
        return mapError(err);
    };

    var flat: u64 = 0;
    while (flat < out_arr.len) : (flat += 1) {
        const a_off = linearByteOffset(flat, bc.shape, bc.a_strides) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };
        const b_off = linearByteOffset(flat, bc.shape, bc.b_strides) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };

        const cmp = compareAt(op, a.dtype, a, b, a_off, b_off) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };
        out_arr.writeI32AtByteOffset(out_arr.linearOffset(flat) catch |err| {
            out_arr.destroy();
            return mapError(err);
        }, cmp) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };
    }

    return registerHeader(out_arr, out);
}

export fn nd_eq(a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    return compareEntry(.eq, a_handle, b_handle, out_handle);
}

export fn nd_lt(a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    return compareEntry(.lt, a_handle, b_handle, out_handle);
}

export fn nd_gt(a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    return compareEntry(.gt, a_handle, b_handle, out_handle);
}

export fn nd_where(cond_handle: u64, x_handle: u64, y_handle: u64, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const cond = registry.lookup(cond_handle) catch |err| {
        return mapError(err);
    };
    const x = registry.lookup(x_handle) catch |err| {
        return mapError(err);
    };
    const y = registry.lookup(y_handle) catch |err| {
        return mapError(err);
    };

    if (x.dtype != y.dtype) return setErr(.invalid_dtype, "where requires matching x/y dtypes");

    const out_arr = createBroadcastOutputWithDType(x, y, x.dtype) catch |err| {
        return mapError(err);
    };

    var xy_shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var x_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var y_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const bc_xy = computeBroadcast(x, y, &xy_shape_buf, &x_strides_buf, &y_strides_buf) catch |err| {
        out_arr.destroy();
        return mapError(err);
    };

    var cond_strides_buf: [header_mod.MAX_DIMS]i64 = undefined;
    const cond_bc_strides = computeBroadcastAgainstShape(cond.shape, cond.strides, bc_xy.shape, &cond_strides_buf) catch |err| {
        out_arr.destroy();
        return mapError(err);
    };

    var flat: u64 = 0;
    while (flat < out_arr.len) : (flat += 1) {
        const cond_off = linearByteOffset(flat, bc_xy.shape, cond_bc_strides) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };
        const cond_true = switch (cond.dtype) {
            .f64 => (cond.readF64AtByteOffset(cond_off) catch |err| {
                out_arr.destroy();
                return mapError(err);
            }) != 0,
            .f32 => (cond.readF32AtByteOffset(cond_off) catch |err| {
                out_arr.destroy();
                return mapError(err);
            }) != 0,
            .i32 => (cond.readI32AtByteOffset(cond_off) catch |err| {
                out_arr.destroy();
                return mapError(err);
            }) != 0,
        };

        const x_off = linearByteOffset(flat, bc_xy.shape, bc_xy.a_strides) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };
        const y_off = linearByteOffset(flat, bc_xy.shape, bc_xy.b_strides) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };
        const out_off = out_arr.linearOffset(flat) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };

        switch (x.dtype) {
            .f64 => {
                const xv = x.readF64AtByteOffset(x_off) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
                const yv = y.readF64AtByteOffset(y_off) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
                out_arr.writeF64AtByteOffset(out_off, if (cond_true) xv else yv) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
            },
            .f32 => {
                const xv = x.readF32AtByteOffset(x_off) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
                const yv = y.readF32AtByteOffset(y_off) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
                out_arr.writeF32AtByteOffset(out_off, if (cond_true) xv else yv) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
            },
            .i32 => {
                const xv = x.readI32AtByteOffset(x_off) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
                const yv = y.readI32AtByteOffset(y_off) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
                out_arr.writeI32AtByteOffset(out_off, if (cond_true) xv else yv) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
            },
        }
    }

    return registerHeader(out_arr, out);
}

// Legacy helper retained from earlier POC; requires out to be preallocated broadcast result shape.
export fn nd_add_into(a_handle: u64, b_handle: u64, out_handle: u64) i32 {
    const a = registry.lookup(a_handle) catch |err| {
        return mapError(err);
    };
    const b = registry.lookup(b_handle) catch |err| {
        return mapError(err);
    };
    const out = registry.lookup(out_handle) catch |err| {
        return mapError(err);
    };

    applyBinaryGeneric(.add, a, b, out) catch |err| {
        return mapError(err);
    };

    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_sum_all(a_handle: u64, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const a = registry.lookup(a_handle) catch |err| {
        return mapError(err);
    };

    const sum = sumF64AnyLayout(a) catch |err| {
        return mapError(err);
    };

    const scalar = ArrayHeader.allocateZeros(allocator, .f64, &[_]i64{}, 0) catch |err| {
        return mapError(err);
    };

    const data = scalar.asMutF64() catch |err| {
        scalar.destroy();
        return mapError(err);
    };
    data[0] = sum;

    return registerHeader(scalar, out);
}

export fn nd_sum_axis(a_handle: u64, axis: i32, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const a = registry.lookup(a_handle) catch |err| {
        return mapError(err);
    };

    if (a.ndim == 0) return setErr(.invalid_arg, "sum_axis requires ndim >= 1");

    var axis_norm = axis;
    if (axis_norm < 0) axis_norm += @as(i32, a.ndim);
    if (axis_norm < 0 or axis_norm >= a.ndim) return setErr(.invalid_arg, "axis out of range");
    const axis_idx: usize = @intCast(axis_norm);

    var out_shape_buf: [header_mod.MAX_DIMS]i64 = undefined;
    var out_i: usize = 0;
    var d: usize = 0;
    while (d < a.shape.len) : (d += 1) {
        if (d == axis_idx) continue;
        out_shape_buf[out_i] = a.shape[d];
        out_i += 1;
    }
    const out_shape = out_shape_buf[0..out_i];

    const out_arr = ArrayHeader.allocateZeros(allocator, a.dtype, out_shape, 0) catch |err| {
        return mapError(err);
    };

    const axis_len = a.shape[axis_idx];
    var flat: u64 = 0;
    while (flat < out_arr.len) : (flat += 1) {
        var rem = flat;
        var base_off: i64 = 0;
        var out_pos = out_shape.len;
        var dim_rev: usize = a.shape.len;
        while (dim_rev > 0) {
            dim_rev -= 1;
            if (dim_rev == axis_idx) continue;
            out_pos -= 1;
            const dim = out_shape[out_pos];
            const dim_u64: u64 = @intCast(dim);
            const idx_u64 = rem % dim_u64;
            rem /= dim_u64;
            const idx_i64: i64 = @intCast(idx_u64);
            const delta = std.math.mul(i64, idx_i64, a.strides[dim_rev]) catch {
                out_arr.destroy();
                return setErr(.invalid_shape, "sum_axis offset overflow");
            };
            base_off = std.math.add(i64, base_off, delta) catch {
                out_arr.destroy();
                return setErr(.invalid_shape, "sum_axis offset overflow");
            };
        }

        const out_off = out_arr.linearOffset(flat) catch |err| {
            out_arr.destroy();
            return mapError(err);
        };

        switch (a.dtype) {
            .f64 => {
                var acc: f64 = 0;
                var k: i64 = 0;
                while (k < axis_len) : (k += 1) {
                    const delta = std.math.mul(i64, k, a.strides[axis_idx]) catch {
                        out_arr.destroy();
                        return setErr(.invalid_shape, "sum_axis stride overflow");
                    };
                    const off = std.math.add(i64, base_off, delta) catch {
                        out_arr.destroy();
                        return setErr(.invalid_shape, "sum_axis offset overflow");
                    };
                    acc += a.readF64AtByteOffset(off) catch |err| {
                        out_arr.destroy();
                        return mapError(err);
                    };
                }
                out_arr.writeF64AtByteOffset(out_off, acc) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
            },
            .f32 => {
                var acc: f64 = 0;
                var k: i64 = 0;
                while (k < axis_len) : (k += 1) {
                    const delta = std.math.mul(i64, k, a.strides[axis_idx]) catch {
                        out_arr.destroy();
                        return setErr(.invalid_shape, "sum_axis stride overflow");
                    };
                    const off = std.math.add(i64, base_off, delta) catch {
                        out_arr.destroy();
                        return setErr(.invalid_shape, "sum_axis offset overflow");
                    };
                    acc += @as(f64, a.readF32AtByteOffset(off) catch |err| {
                        out_arr.destroy();
                        return mapError(err);
                    });
                }
                out_arr.writeF32AtByteOffset(out_off, @floatCast(acc)) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
            },
            .i32 => {
                var acc: i64 = 0;
                var k: i64 = 0;
                while (k < axis_len) : (k += 1) {
                    const delta = std.math.mul(i64, k, a.strides[axis_idx]) catch {
                        out_arr.destroy();
                        return setErr(.invalid_shape, "sum_axis stride overflow");
                    };
                    const off = std.math.add(i64, base_off, delta) catch {
                        out_arr.destroy();
                        return setErr(.invalid_shape, "sum_axis offset overflow");
                    };
                    acc += @as(i64, a.readI32AtByteOffset(off) catch |err| {
                        out_arr.destroy();
                        return mapError(err);
                    });
                }
                const narrowed = std.math.cast(i32, acc) orelse {
                    out_arr.destroy();
                    return setErr(.invalid_shape, "sum_axis i32 overflow");
                };
                out_arr.writeI32AtByteOffset(out_off, narrowed) catch |err| {
                    out_arr.destroy();
                    return mapError(err);
                };
            },
        }
    }

    return registerHeader(out_arr, out);
}

export fn nd_matmul(a_handle: u64, b_handle: u64, out_handle: ?*u64) i32 {
    const out = out_handle orelse return setErr(.invalid_arg, "out_handle is null");
    const a = registry.lookup(a_handle) catch |err| {
        return mapError(err);
    };
    const b = registry.lookup(b_handle) catch |err| {
        return mapError(err);
    };

    if (a.dtype != .f64 or b.dtype != .f64) return setErr(.invalid_dtype, "matmul currently supports f64 only");
    if (a.ndim != 2 or b.ndim != 2) return setErr(.invalid_shape, "matmul requires 2D inputs");
    if (a.shape[1] != b.shape[0]) return setErr(.invalid_shape, "matmul inner dimensions mismatch");

    const out_shape = [_]i64{ a.shape[0], b.shape[1] };
    const out_arr = ArrayHeader.allocateZeros(allocator, .f64, &out_shape, 0) catch |err| {
        return mapError(err);
    };

    matmul_kernel.matmulF64(a, b, out_arr) catch |err| {
        out_arr.destroy();
        return mapError(err);
    };

    return registerHeader(out_arr, out);
}

export fn nd_job_submit_matmul(a: u64, b: u64, out_job_id: ?*u64) i32 {
    _ = jobs_mod.submitMatmul(a, b) catch {};
    if (out_job_id) |job_id| {
        job_id.* = 0;
    }
    return setErr(.not_implemented, "job API is scaffold-only in this build");
}

export fn nd_job_poll(job_id: u64, out_state: ?*u32, out_result_status: ?*i32) i32 {
    if (out_state) |state| {
        state.* = @intFromEnum(jobs_mod.JobState.queued);
    }
    if (out_result_status) |status| {
        status.* = @intFromEnum(NdStatus.not_implemented);
    }
    _ = job_id;
    return setErr(.not_implemented, "job API is scaffold-only in this build");
}

export fn nd_job_take_result(job_id: u64, out_handle: ?*u64) i32 {
    _ = job_id;
    _ = out_handle;
    return setErr(.not_implemented, "job API is scaffold-only in this build");
}

export fn nd_job_cancel(job_id: u64) i32 {
    _ = job_id;
    return setErr(.not_implemented, "job API is scaffold-only in this build");
}

export fn nd_simd_width_f64() u64 {
    return std.simd.suggestVectorLength(f64) orelse 1;
}

export fn nd_simd_add_f64_raw(a_ptr: ?[*]const f64, b_ptr: ?[*]const f64, out_ptr: ?[*]f64, len: u64) i32 {
    const a = a_ptr orelse return setErr(.invalid_arg, "a_ptr is null");
    const b = b_ptr orelse return setErr(.invalid_arg, "b_ptr is null");
    const out = out_ptr orelse return setErr(.invalid_arg, "out_ptr is null");

    if (len > std.math.maxInt(usize)) return setErr(.invalid_arg, "len is too large");

    elem_kernel.binaryRawF64(.add, a, b, out, @intCast(len));
    clearErr();
    return @intFromEnum(NdStatus.ok);
}

export fn nd_simd_sum_f64_raw(data_ptr: ?[*]const f64, len: u64, out_sum: ?*f64) i32 {
    const data = data_ptr orelse return setErr(.invalid_arg, "data_ptr is null");
    const out = out_sum orelse return setErr(.invalid_arg, "out_sum is null");

    if (len > std.math.maxInt(usize)) return setErr(.invalid_arg, "len is too large");

    out.* = reduce_kernel.sumRawF64(data, @intCast(len));
    clearErr();
    return @intFromEnum(NdStatus.ok);
}
