const std = @import("std");
const header_mod = @import("../header.zig");

const ArrayHeader = header_mod.ArrayHeader;

pub const BinaryOp = enum {
    add,
    sub,
    mul,
    div,
};

fn applyOp(op: BinaryOp, a: f64, b: f64) f64 {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
    };
}

pub fn binaryRawF64(op: BinaryOp, pa: [*]const f64, pb: [*]const f64, po: [*]f64, len: usize) void {
    const vec_len = std.simd.suggestVectorLength(f64) orelse 1;
    var i: usize = 0;

    if (vec_len > 1) {
        const V = @Vector(vec_len, f64);
        while (i + vec_len <= len) : (i += vec_len) {
            const va: V = pa[i..][0..vec_len].*;
            const vb: V = pb[i..][0..vec_len].*;
            const vr: V = switch (op) {
                .add => va + vb,
                .sub => va - vb,
                .mul => va * vb,
                .div => va / vb,
            };
            po[i..][0..vec_len].* = vr;
        }
    }

    while (i < len) : (i += 1) {
        po[i] = applyOp(op, pa[i], pb[i]);
    }
}

pub fn addRawF64(pa: [*]const f64, pb: [*]const f64, po: [*]f64, len: usize) void {
    binaryRawF64(.add, pa, pb, po, len);
}

pub fn binaryF64(op: BinaryOp, a: *const ArrayHeader, b: *const ArrayHeader, out: *ArrayHeader) !void {
    if (a.dtype != .f64 or b.dtype != .f64 or out.dtype != .f64) return error.InvalidDType;
    if (!a.hasSameShape(b) or !a.hasSameShape(out)) return error.ShapeMismatch;

    const len: usize = @intCast(a.len);
    const pa = try a.asConstF64();
    const pb = try b.asConstF64();
    const po = try out.asMutF64();
    binaryRawF64(op, pa, pb, po, len);
}

pub fn addF64(a: *const ArrayHeader, b: *const ArrayHeader, out: *ArrayHeader) !void {
    try binaryF64(.add, a, b, out);
}

pub fn subF64(a: *const ArrayHeader, b: *const ArrayHeader, out: *ArrayHeader) !void {
    try binaryF64(.sub, a, b, out);
}

pub fn mulF64(a: *const ArrayHeader, b: *const ArrayHeader, out: *ArrayHeader) !void {
    try binaryF64(.mul, a, b, out);
}

pub fn divF64(a: *const ArrayHeader, b: *const ArrayHeader, out: *ArrayHeader) !void {
    try binaryF64(.div, a, b, out);
}
