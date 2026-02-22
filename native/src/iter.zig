const std = @import("std");

pub fn linearByteOffset(flat: u64, shape: []const i64, strides: []const i64) !i64 {
    if (shape.len != strides.len) return error.InvalidShape;

    var rem = flat;
    var offset: i64 = 0;
    var d: usize = shape.len;

    while (d > 0) {
        d -= 1;

        const dim = shape[d];
        if (dim <= 0) return error.InvalidShape;

        const dim_u64: u64 = @intCast(dim);
        const idx_u64 = rem % dim_u64;
        rem /= dim_u64;

        const idx_i64: i64 = @intCast(idx_u64);
        const delta = std.math.mul(i64, idx_i64, strides[d]) catch return error.ShapeOverflow;
        offset = std.math.add(i64, offset, delta) catch return error.ShapeOverflow;
    }

    return offset;
}
