const std = @import("std");
const header_mod = @import("../header.zig");

const ArrayHeader = header_mod.ArrayHeader;

pub fn sumRawF64(data: [*]const f64, len: usize) f64 {
    const vec_len = std.simd.suggestVectorLength(f64) orelse 1;
    var i: usize = 0;
    var result: f64 = 0;

    if (vec_len > 1) {
        const V = @Vector(vec_len, f64);
        var acc: V = @splat(0.0);
        while (i + vec_len <= len) : (i += vec_len) {
            const v: V = data[i..][0..vec_len].*;
            acc += v;
        }
        result = @reduce(.Add, acc);
    }

    while (i < len) : (i += 1) {
        result += data[i];
    }

    return result;
}

pub fn sumAllF64(a: *const ArrayHeader) !f64 {
    if (a.dtype != .f64) return error.InvalidDType;

    const data = try a.asConstF64();
    const len: usize = @intCast(a.len);
    return sumRawF64(data, len);
}
