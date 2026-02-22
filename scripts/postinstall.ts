const platformKey = `${process.platform}-${process.arch}`;

const mapping: Record<string, string> = {
  "linux-x64": "prebuilds/linux-x64/libndarray.so",
  "linux-arm64": "prebuilds/linux-arm64/libndarray.so",
  "darwin-x64": "prebuilds/darwin-x64/libndarray.dylib",
  "darwin-arm64": "prebuilds/darwin-arm64/libndarray.dylib",
  "win32-x64": "prebuilds/win32-x64/ndarray.dll",
};

const candidate = mapping[platformKey];
if (!candidate) {
  console.warn(`[bun-ndarray] No prebuild mapping for ${platformKey}.`);
  process.exit(0);
}

const exists = await Bun.file(new URL(`../${candidate}`, import.meta.url)).exists();
if (!exists) {
  console.warn(
    `[bun-ndarray] Missing prebuild for ${platformKey} at ${candidate}. Run: bun run build:native:matrix && bun run build:packages`,
  );
} else {
  console.log(`[bun-ndarray] Prebuild detected for ${platformKey}: ${candidate}`);
}
