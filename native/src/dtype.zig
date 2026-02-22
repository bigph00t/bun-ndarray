const std = @import("std");

pub const DType = enum(u32) {
    f32 = 1,
    i32 = 3,
    f64 = 4,

    pub fn fromAbi(code: u32) !DType {
        return switch (code) {
            @intFromEnum(DType.f32) => .f32,
            @intFromEnum(DType.i32) => .i32,
            @intFromEnum(DType.f64) => .f64,
            else => error.InvalidDType,
        };
    }

    pub fn byteSize(self: DType) usize {
        return switch (self) {
            .f32 => @sizeOf(f32),
            .f64 => @sizeOf(f64),
            .i32 => @sizeOf(i32),
        };
    }

    pub fn alignment(self: DType) std.mem.Alignment {
        return switch (self) {
            .f32 => @alignOf(f32),
            .f64 => @alignOf(f64),
            .i32 => @alignOf(i32),
        };
    }

    pub fn name(self: DType) []const u8 {
        return switch (self) {
            .f32 => "f32",
            .f64 => "f64",
            .i32 => "i32",
        };
    }
};
