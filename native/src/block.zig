const std = @import("std");

const Allocator = std.mem.Allocator;
const AtomicU32 = std.atomic.Value(u32);

pub const ALIGN_BYTES: usize = 64;
pub const ALIGNMENT: std.mem.Alignment = @enumFromInt(6); // 2^6 = 64

pub const DataBlock = struct {
    ptr: [*]u8,
    byte_len: usize,
    refcount: AtomicU32,
    allocator: Allocator,

    pub fn create(allocator: Allocator, byte_len: usize) !*DataBlock {
        const raw = try allocator.alignedAlloc(u8, ALIGNMENT, byte_len);
        @memset(raw, 0);

        const block = try allocator.create(DataBlock);
        block.* = .{
            .ptr = raw.ptr,
            .byte_len = byte_len,
            .refcount = AtomicU32.init(1),
            .allocator = allocator,
        };
        return block;
    }

    pub fn retain(self: *DataBlock) !void {
        const prev = self.refcount.fetchAdd(1, .monotonic);
        if (prev == 0) {
            _ = self.refcount.fetchSub(1, .monotonic);
            return error.BlockAlreadyFreed;
        }
    }

    pub fn release(self: *DataBlock) !bool {
        const current = self.refcount.load(.monotonic);
        if (current == 0) {
            return error.BlockAlreadyFreed;
        }

        const prev = self.refcount.fetchSub(1, .release);
        if (prev == 0) {
            _ = self.refcount.fetchAdd(1, .monotonic);
            return error.BlockAlreadyFreed;
        }

        if (prev == 1) {
            _ = self.refcount.load(.acquire);
            const raw: [*]align(ALIGN_BYTES) u8 = @alignCast(@ptrCast(self.ptr));
            self.allocator.free(raw[0..self.byte_len]);
            self.allocator.destroy(self);
            return true;
        }

        return false;
    }
};
