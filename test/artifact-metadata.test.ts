import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { native } from "../src/ffi";

type Manifest = {
  packageName: string;
  version: string;
  abiVersion: number;
  artifacts: Array<{
    packageName: string;
    platformKey: string;
    targetTriple: string;
    file: string;
    sha256: string;
    sizeBytes: number;
  }>;
};

describe("artifact metadata", () => {
  test("prebuild manifest exists and matches ABI expectations", () => {
    const manifestPath = join(import.meta.dir, "..", "prebuilds", "manifest.json");
    expect(existsSync(manifestPath)).toBe(true);

    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Manifest;
    expect(manifest.packageName).toBe("bun-ndarray");
    expect(manifest.version).toBe("0.1.0");
    expect(manifest.abiVersion).toBe(Number(native.nd_abi_version()));
    expect(manifest.artifacts.length).toBe(5);

    for (const artifact of manifest.artifacts) {
      expect(artifact.packageName.length).toBeGreaterThan(0);
      expect(artifact.platformKey.length).toBeGreaterThan(0);
      expect(artifact.targetTriple.length).toBeGreaterThan(0);
      expect(artifact.file.length).toBeGreaterThan(0);
      expect(artifact.sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(artifact.sizeBytes).toBeGreaterThan(0);
    }
  });

  test("platform package metadata files align with manifest", () => {
    const manifestPath = join(import.meta.dir, "..", "prebuilds", "manifest.json");
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as Manifest;

    for (const artifact of manifest.artifacts) {
      const metadataPath = join(
        import.meta.dir,
        "..",
        "platform-packages",
        artifact.packageName,
        "artifact-metadata.json",
      );
      expect(existsSync(metadataPath)).toBe(true);
      const metadata = JSON.parse(readFileSync(metadataPath, "utf8")) as {
        packageName: string;
        platformKey: string;
        targetTriple: string;
        file: string;
        sha256: string;
        sizeBytes: number;
        abiVersion: number;
      };
      expect(metadata.packageName).toBe(artifact.packageName);
      expect(metadata.platformKey).toBe(artifact.platformKey);
      expect(metadata.targetTriple).toBe(artifact.targetTriple);
      expect(metadata.file).toBe(artifact.file);
      expect(metadata.sha256).toBe(artifact.sha256);
      expect(metadata.sizeBytes).toBe(artifact.sizeBytes);
      expect(metadata.abiVersion).toBe(manifest.abiVersion);
    }
  });
});

