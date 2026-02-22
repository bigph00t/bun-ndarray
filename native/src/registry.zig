const std = @import("std");
const header_mod = @import("header.zig");

const ArrayHeader = header_mod.ArrayHeader;

pub const MAX_SLOTS: usize = 4096;
const MAX_GEN: u32 = (1 << 21) - 1;
const MAX_SAFE_HANDLE: u64 = 9_007_199_254_740_991;

const Slot = struct {
    generation: u32 = 1,
    header: ?*ArrayHeader = null,
    in_use: bool = false,
};

var slots: [MAX_SLOTS]Slot = [_]Slot{.{}} ** MAX_SLOTS;
var mutex: std.Thread.Mutex = .{};

fn encodeHandle(index: u32, generation: u32) !u64 {
    const handle = (@as(u64, generation) << 32) | @as(u64, index);
    if (handle > MAX_SAFE_HANDLE) return error.HandleOverflow;
    return handle;
}

fn decodeHandle(handle: u64) struct { index: u32, generation: u32 } {
    return .{
        .index = @intCast(handle & 0xffff_ffff),
        .generation = @intCast((handle >> 32) & MAX_GEN),
    };
}

pub fn register(header: *ArrayHeader) !u64 {
    mutex.lock();
    defer mutex.unlock();

    var i: usize = 0;
    while (i < MAX_SLOTS) : (i += 1) {
        if (!slots[i].in_use) {
            if (slots[i].generation == 0) {
                slots[i].generation = 1;
            }
            slots[i].in_use = true;
            slots[i].header = header;
            return encodeHandle(@intCast(i), slots[i].generation);
        }
    }

    return error.RegistryFull;
}

pub fn lookup(handle: u64) !*ArrayHeader {
    const decoded = decodeHandle(handle);
    if (decoded.index >= MAX_SLOTS) return error.InvalidHandle;

    mutex.lock();
    defer mutex.unlock();

    const slot = &slots[decoded.index];
    if (!slot.in_use or slot.header == null) return error.StaleHandle;
    if (slot.generation != decoded.generation) return error.StaleHandle;

    return slot.header.?;
}

pub fn unregister(handle: u64) !void {
    const decoded = decodeHandle(handle);
    if (decoded.index >= MAX_SLOTS) return error.InvalidHandle;

    mutex.lock();
    defer mutex.unlock();

    const slot = &slots[decoded.index];
    if (!slot.in_use or slot.header == null) return error.StaleHandle;
    if (slot.generation != decoded.generation) return error.StaleHandle;

    slot.in_use = false;
    slot.header = null;
    slot.generation = (slot.generation + 1) & MAX_GEN;
    if (slot.generation == 0) {
        slot.generation = 1;
    }
}
