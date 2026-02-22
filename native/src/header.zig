const std = @import("std");
const dtype_mod = @import("dtype.zig");
const block_mod = @import("block.zig");

const Allocator = std.mem.Allocator;
const DType = dtype_mod.DType;
const DataBlock = block_mod.DataBlock;
const AtomicU32 = std.atomic.Value(u32);

pub const MAX_DIMS: usize = 8;

pub const ArrayHeader = struct {
    block: *DataBlock,
    dtype: DType,
    ndim: u8,
    shape: []i64,
    strides: []i64, // in bytes
    offset_bytes: i64,
    flags: u32,
    len: u64,
    handle_refs: AtomicU32,
    allocator: Allocator,

    pub fn computeLen(shape: []const i64) !u64 {
        if (shape.len == 0) return 1;

        var len: u64 = 1;
        var has_zero = false;
        for (shape) |dim| {
            if (dim < 0) return error.InvalidShape;
            if (dim == 0) {
                has_zero = true;
                continue;
            }
            if (!has_zero) {
                len = std.math.mul(u64, len, @as(u64, @intCast(dim))) catch return error.ShapeOverflow;
            }
        }
        return if (has_zero) 0 else len;
    }

    pub fn computeContiguousStrides(dtype: DType, shape: []const i64, out_strides: []i64) !void {
        if (shape.len != out_strides.len) return error.InvalidShape;
        var stride: i64 = @intCast(dtype.byteSize());
        var i: usize = shape.len;
        while (i > 0) {
            i -= 1;
            out_strides[i] = stride;
            stride = std.math.mul(i64, stride, shape[i]) catch return error.ShapeOverflow;
        }
    }

    pub fn allocateZeros(allocator: Allocator, dtype: DType, shape: []const i64, flags: u32) !*ArrayHeader {
        if (shape.len > MAX_DIMS) return error.InvalidShape;

        const len = try computeLen(shape);

        const byte_len_u64 = std.math.mul(u64, len, dtype.byteSize()) catch return error.ShapeOverflow;
        if (byte_len_u64 > std.math.maxInt(usize)) return error.ShapeOverflow;
        const byte_len: usize = @intCast(byte_len_u64);

        const shape_copy = try allocator.alloc(i64, shape.len);
        errdefer allocator.free(shape_copy);
        @memcpy(shape_copy, shape);

        const strides = try allocator.alloc(i64, shape.len);
        errdefer allocator.free(strides);

        try computeContiguousStrides(dtype, shape_copy, strides);

        const block = try DataBlock.create(allocator, byte_len);
        errdefer {
            _ = block.release() catch {};
        }

        const header = try allocator.create(ArrayHeader);
        header.* = .{
            .block = block,
            .dtype = dtype,
            .ndim = @intCast(shape.len),
            .shape = shape_copy,
            .strides = strides,
            .offset_bytes = 0,
            .flags = flags,
            .len = len,
            .handle_refs = AtomicU32.init(1),
            .allocator = allocator,
        };
        return header;
    }

    pub fn fromHostCopy(allocator: Allocator, src: [*]const u8, dtype: DType, shape: []const i64, flags: u32) !*ArrayHeader {
        const header = try allocateZeros(allocator, dtype, shape, flags);
        const dst = header.block.ptr[0..header.block.byte_len];
        const src_slice = src[0..header.block.byte_len];
        @memcpy(dst, src_slice);
        return header;
    }

    pub fn createView(allocator: Allocator, base: *ArrayHeader, shape: []const i64, strides: []const i64, offset_bytes: i64, flags: u32) !*ArrayHeader {
        if (shape.len > MAX_DIMS or shape.len != strides.len) return error.InvalidShape;
        const len = try computeLen(shape);
        const required_bytes = if (len == 0) 0 else base.dtype.byteSize();
        _ = try base.resolveRange(offset_bytes, required_bytes);

        const shape_copy = try allocator.alloc(i64, shape.len);
        errdefer allocator.free(shape_copy);
        @memcpy(shape_copy, shape);

        const strides_copy = try allocator.alloc(i64, strides.len);
        errdefer allocator.free(strides_copy);
        @memcpy(strides_copy, strides);

        try base.block.retain();
        errdefer {
            _ = base.block.release() catch {};
        }

        const header = try allocator.create(ArrayHeader);
        header.* = .{
            .block = base.block,
            .dtype = base.dtype,
            .ndim = @intCast(shape.len),
            .shape = shape_copy,
            .strides = strides_copy,
            .offset_bytes = offset_bytes,
            .flags = flags,
            .len = len,
            .handle_refs = AtomicU32.init(1),
            .allocator = allocator,
        };
        return header;
    }

    pub fn cloneContiguous(self: *const ArrayHeader, allocator: Allocator, flags: u32) !*ArrayHeader {
        const out = try allocateZeros(allocator, self.dtype, self.shape, flags);
        const dst = out.block.ptr[0..out.block.byte_len];

        if (self.isContiguous()) {
            const src_off = try self.resolveRange(0, out.block.byte_len);
            const src = self.block.ptr[src_off..][0..out.block.byte_len];
            @memcpy(dst, src);
            return out;
        }

        var flat: u64 = 0;
        const elem_size = self.dtype.byteSize();
        while (flat < self.len) : (flat += 1) {
            const src_logical = try self.linearOffset(flat);
            const src_off = try self.resolveRange(src_logical, elem_size);
            const dst_off_u64 = std.math.mul(u64, flat, elem_size) catch return error.ShapeOverflow;
            const dst_off: usize = @intCast(dst_off_u64);
            @memcpy(dst[dst_off..][0..elem_size], self.block.ptr[src_off..][0..elem_size]);
        }

        return out;
    }

    pub fn byteLen(self: *const ArrayHeader) usize {
        const len_usize: usize = @intCast(self.len);
        return len_usize * self.dtype.byteSize();
    }

    pub fn isContiguous(self: *const ArrayHeader) bool {
        var expected: i64 = @intCast(self.dtype.byteSize());
        var i: usize = self.shape.len;
        while (i > 0) {
            i -= 1;
            if (self.strides[i] != expected) return false;
            expected = std.math.mul(i64, expected, self.shape[i]) catch return false;
        }
        return true;
    }

    pub fn retainHandle(self: *ArrayHeader) !void {
        const prev = self.handle_refs.fetchAdd(1, .monotonic);
        if (prev == 0) {
            _ = self.handle_refs.fetchSub(1, .monotonic);
            return error.HeaderAlreadyReleased;
        }
    }

    pub fn releaseHandle(self: *ArrayHeader) !bool {
        const current = self.handle_refs.load(.monotonic);
        if (current == 0) {
            return error.HeaderAlreadyReleased;
        }

        const prev = self.handle_refs.fetchSub(1, .release);
        if (prev == 0) {
            _ = self.handle_refs.fetchAdd(1, .monotonic);
            return error.HeaderAlreadyReleased;
        }

        if (prev == 1) {
            _ = self.handle_refs.load(.acquire);
            return true;
        }

        return false;
    }

    pub fn destroy(self: *ArrayHeader) void {
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
        _ = self.block.release() catch {};
        self.allocator.destroy(self);
    }

    pub fn asConstF64(self: *const ArrayHeader) ![*]const f64 {
        if (self.dtype != .f64) return error.InvalidDType;
        if (!self.isContiguous()) return error.InvalidStrides;
        const required_bytes: usize = if (self.byteLen() == 0) 0 else 1;
        const off = try self.resolveRange(0, required_bytes);
        return @alignCast(@ptrCast(self.block.ptr + off));
    }

    pub fn asMutF64(self: *ArrayHeader) ![*]f64 {
        if (self.dtype != .f64) return error.InvalidDType;
        if (!self.isContiguous()) return error.InvalidStrides;
        const required_bytes: usize = if (self.byteLen() == 0) 0 else 1;
        const off = try self.resolveRange(0, required_bytes);
        return @alignCast(@ptrCast(self.block.ptr + off));
    }

    pub fn dataPtr(self: *const ArrayHeader) ![*]u8 {
        if (!self.isContiguous()) return error.InvalidStrides;
        const required_bytes: usize = if (self.byteLen() == 0) 0 else 1;
        const off = try self.resolveRange(0, required_bytes);
        return self.block.ptr + off;
    }

    pub fn linearOffset(self: *const ArrayHeader, flat: u64) !i64 {
        var rem = flat;
        var offset: i64 = 0;

        var d: usize = self.shape.len;
        while (d > 0) {
            d -= 1;
            const dim = self.shape[d];
            if (dim <= 0) return error.InvalidShape;
            const dim_u64: u64 = @intCast(dim);
            const idx_u64 = rem % dim_u64;
            rem /= dim_u64;
            const idx_i64: i64 = @intCast(idx_u64);
            const delta = std.math.mul(i64, idx_i64, self.strides[d]) catch return error.ShapeOverflow;
            offset = std.math.add(i64, offset, delta) catch return error.ShapeOverflow;
        }

        return offset;
    }

    pub fn resolveRange(self: *const ArrayHeader, logical_byte_offset: i64, byte_count: usize) !usize {
        const start = std.math.add(i64, self.offset_bytes, logical_byte_offset) catch return error.InvalidOffset;
        if (start < 0) return error.InvalidOffset;

        const start_u: usize = @intCast(start);
        const end = std.math.add(usize, start_u, byte_count) catch return error.InvalidOffset;
        if (end > self.block.byte_len) return error.InvalidOffset;
        return start_u;
    }

    pub fn readF64AtByteOffset(self: *const ArrayHeader, byte_offset: i64) !f64 {
        if (self.dtype != .f64) return error.InvalidDType;
        const off = try self.resolveRange(byte_offset, @sizeOf(f64));
        const p = self.block.ptr + off;
        return @as(*const f64, @alignCast(@ptrCast(p))).*;
    }

    pub fn writeF64AtByteOffset(self: *ArrayHeader, byte_offset: i64, value: f64) !void {
        if (self.dtype != .f64) return error.InvalidDType;
        const off = try self.resolveRange(byte_offset, @sizeOf(f64));
        const p = self.block.ptr + off;
        @as(*f64, @alignCast(@ptrCast(p))).* = value;
    }

    pub fn readF32AtByteOffset(self: *const ArrayHeader, byte_offset: i64) !f32 {
        if (self.dtype != .f32) return error.InvalidDType;
        const off = try self.resolveRange(byte_offset, @sizeOf(f32));
        const p = self.block.ptr + off;
        return @as(*const f32, @alignCast(@ptrCast(p))).*;
    }

    pub fn writeF32AtByteOffset(self: *ArrayHeader, byte_offset: i64, value: f32) !void {
        if (self.dtype != .f32) return error.InvalidDType;
        const off = try self.resolveRange(byte_offset, @sizeOf(f32));
        const p = self.block.ptr + off;
        @as(*f32, @alignCast(@ptrCast(p))).* = value;
    }

    pub fn readI32AtByteOffset(self: *const ArrayHeader, byte_offset: i64) !i32 {
        if (self.dtype != .i32) return error.InvalidDType;
        const off = try self.resolveRange(byte_offset, @sizeOf(i32));
        const p = self.block.ptr + off;
        return @as(*const i32, @alignCast(@ptrCast(p))).*;
    }

    pub fn writeI32AtByteOffset(self: *ArrayHeader, byte_offset: i64, value: i32) !void {
        if (self.dtype != .i32) return error.InvalidDType;
        const off = try self.resolveRange(byte_offset, @sizeOf(i32));
        const p = self.block.ptr + off;
        @as(*i32, @alignCast(@ptrCast(p))).* = value;
    }

    pub fn hasSameShape(self: *const ArrayHeader, other: *const ArrayHeader) bool {
        if (self.ndim != other.ndim) return false;
        var i: usize = 0;
        while (i < self.shape.len) : (i += 1) {
            if (self.shape[i] != other.shape[i]) return false;
        }
        return true;
    }
};
