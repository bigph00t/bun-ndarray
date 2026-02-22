import { dlopen, suffix } from "bun:ffi";
import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { symbolDefs, type NativeSymbols } from "./symbols";

function resolveLibraryPath(): string {
  const req = createRequire(import.meta.url);
  const platformKey = `${process.platform}-${process.arch}`;
  const base = join(import.meta.dir, "..", "native", "zig-out");
  const inLib = join(base, "lib", `libndarray.${suffix}`);
  const inBin = join(base, "bin", suffix === "dll" ? `ndarray.${suffix}` : `libndarray.${suffix}`);

  if (existsSync(inLib)) {
    return inLib;
  }
  if (existsSync(inBin)) {
    return inBin;
  }

  const optionalPackageCandidates: Record<string, { pkg: string; file: string }> = {
    "linux-x64": { pkg: "bun-ndarray-linux-x64-gnu", file: "libndarray.so" },
    "linux-arm64": { pkg: "bun-ndarray-linux-arm64-gnu", file: "libndarray.so" },
    "darwin-x64": { pkg: "bun-ndarray-darwin-x64", file: "libndarray.dylib" },
    "darwin-arm64": { pkg: "bun-ndarray-darwin-arm64", file: "libndarray.dylib" },
    "win32-x64": { pkg: "bun-ndarray-windows-x64-gnu", file: "ndarray.dll" },
  };
  const optional = optionalPackageCandidates[platformKey];
  if (optional) {
    try {
      const pkgJson = req.resolve(`${optional.pkg}/package.json`);
      const packageRoot = join(pkgJson, "..");
      const candidate = join(packageRoot, optional.file);
      if (existsSync(candidate)) {
        return candidate;
      }
    } catch {
      // Ignore optional dependency resolution errors and continue fallback chain.
    }
  }

  const prebuildCandidates: Record<string, string> = {
    "linux-x64": join(import.meta.dir, "..", "prebuilds", "linux-x64", "libndarray.so"),
    "linux-arm64": join(import.meta.dir, "..", "prebuilds", "linux-arm64", "libndarray.so"),
    "darwin-x64": join(import.meta.dir, "..", "prebuilds", "darwin-x64", "libndarray.dylib"),
    "darwin-arm64": join(import.meta.dir, "..", "prebuilds", "darwin-arm64", "libndarray.dylib"),
    "win32-x64": join(import.meta.dir, "..", "prebuilds", "win32-x64", "ndarray.dll"),
  };
  const prebuild = prebuildCandidates[platformKey];
  if (prebuild && existsSync(prebuild)) {
    return prebuild;
  }

  throw new Error(
    `Native library not found. Tried:\n- ${prebuild ?? "(no prebuild for this platform)"}\n- ${inLib}\n- ${inBin}\nRun: bun run build:native:release`,
  );
}

const libPath = resolveLibraryPath();
const opened = dlopen(libPath, symbolDefs);

export const native: NativeSymbols = opened.symbols as NativeSymbols;
export const closeNative = opened.close;
export const nativeLibraryPath = libPath;
