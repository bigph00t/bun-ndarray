const std = @import("std");

pub const NdStatus = enum(i32) {
    ok = 0,
    invalid_arg = 1,
    invalid_dtype = 2,
    invalid_shape = 3,
    invalid_strides = 4,
    invalid_alignment = 5,
    stale_handle = 6,
    oom = 7,
    not_contiguous = 8,
    not_implemented = 9,
    internal = 255,
};

threadlocal var last_error_code: NdStatus = .ok;
threadlocal var last_error_buf: [256]u8 = [_]u8{0} ** 256;

pub fn clear() void {
    last_error_code = .ok;
    last_error_buf[0] = 0;
}

pub fn set(status: NdStatus, message: []const u8) NdStatus {
    last_error_code = status;
    const n = @min(message.len, last_error_buf.len - 1);
    @memcpy(last_error_buf[0..n], message[0..n]);
    last_error_buf[n] = 0;
    return status;
}

pub fn code() i32 {
    return @intFromEnum(last_error_code);
}

pub fn writeMessage(out_utf8: ?[*]u8, cap: u64, out_len: ?*u64) NdStatus {
    const msg_len = std.mem.indexOfScalar(u8, &last_error_buf, 0) orelse last_error_buf.len;
    if (out_len) |n| {
        n.* = msg_len;
    }

    if (cap == 0) {
        return .ok;
    }

    const out = out_utf8 orelse return set(.invalid_arg, "nd_last_error_message: out_utf8 is null");
    const max_copy = @as(usize, @intCast(cap - 1));
    const copy_len = @min(msg_len, max_copy);
    if (copy_len > 0) {
        @memcpy(out[0..copy_len], last_error_buf[0..copy_len]);
    }
    out[copy_len] = 0;
    return .ok;
}
