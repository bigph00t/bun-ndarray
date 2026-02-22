pub const ABI_VERSION: u32 = 1;
pub const BUILD_VERSION: [*:0]const u8 = "bun-ndarray/0.2.0-scaffold";

pub const NdBool = enum(u32) {
    false_ = 0,
    true_ = 1,
};

pub const NdFlags = struct {
    pub const readonly: u32 = 1 << 0;
};
