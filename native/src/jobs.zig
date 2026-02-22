pub const JobState = enum(u32) {
    queued = 0,
    running = 1,
    done = 2,
    failed = 3,
    cancelled = 4,
};

pub const JobError = error{NotImplemented};

pub fn submitMatmul(_: u64, _: u64) JobError!u64 {
    return JobError.NotImplemented;
}

pub fn poll(_: u64) JobError!JobState {
    return JobError.NotImplemented;
}

pub fn takeResult(_: u64) JobError!u64 {
    return JobError.NotImplemented;
}

pub fn cancel(_: u64) JobError!void {
    return JobError.NotImplemented;
}
