const std = @import("std");
const header_mod = @import("../header.zig");

const ArrayHeader = header_mod.ArrayHeader;

fn elemOffset(arr: *const ArrayHeader, row: usize, col: usize) i64 {
    return @as(i64, @intCast(row)) * arr.strides[0] + @as(i64, @intCast(col)) * arr.strides[1];
}

pub fn matmulF64(a: *const ArrayHeader, b: *const ArrayHeader, out: *ArrayHeader) !void {
    if (a.dtype != .f64 or b.dtype != .f64 or out.dtype != .f64) return error.InvalidDType;
    if (a.ndim != 2 or b.ndim != 2 or out.ndim != 2) return error.InvalidShape;

    const m: usize = @intCast(a.shape[0]);
    const k_a: usize = @intCast(a.shape[1]);
    const k_b: usize = @intCast(b.shape[0]);
    const n: usize = @intCast(b.shape[1]);

    if (k_a != k_b) return error.ShapeMismatch;
    if (out.shape[0] != a.shape[0] or out.shape[1] != b.shape[1]) return error.ShapeMismatch;

    var i: usize = 0;
    while (i < m) : (i += 1) {
        var j: usize = 0;
        while (j < n) : (j += 1) {
            var acc: f64 = 0;
            var k: usize = 0;
            while (k < k_a) : (k += 1) {
                const a_off = elemOffset(a, i, k);
                const b_off = elemOffset(b, k, j);
                const av = try a.readF64AtByteOffset(a_off);
                const bv = try b.readF64AtByteOffset(b_off);
                acc += av * bv;
            }
            const out_off = elemOffset(out, i, j);
            try out.writeF64AtByteOffset(out_off, acc);
        }
    }
}
