import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

type ArtifactDef = {
  packageName: string;
  platformKey: string;
  targetTriple: string;
  prebuildRelPath: string;
  packageDirRelPath: string;
  packageFile: string;
};

const rootDir = join(import.meta.dir, "..");
const packageJson = JSON.parse(readFileSync(join(rootDir, "package.json"), "utf8")) as {
  name: string;
  version: string;
  engines?: { bun?: string };
};

const gitSha = (() => {
  const out = Bun.spawnSync({
    cmd: ["git", "rev-parse", "HEAD"],
    cwd: rootDir,
    stdout: "pipe",
    stderr: "ignore",
  });
  return out.exitCode === 0 ? out.stdout.toString().trim() : "unknown";
})();

const generatedAt = new Date().toISOString();
const abiVersion = 1;
const bunMinVersion = packageJson.engines?.bun ?? "unknown";
const zigVersion = "0.15.2";

const artifacts: ArtifactDef[] = [
  {
    packageName: "bun-ndarray-linux-x64-gnu",
    platformKey: "linux-x64",
    targetTriple: "x86_64-linux-gnu",
    prebuildRelPath: "prebuilds/linux-x64/libndarray.so",
    packageDirRelPath: "platform-packages/bun-ndarray-linux-x64-gnu",
    packageFile: "libndarray.so",
  },
  {
    packageName: "bun-ndarray-linux-arm64-gnu",
    platformKey: "linux-arm64",
    targetTriple: "aarch64-linux-gnu",
    prebuildRelPath: "prebuilds/linux-arm64/libndarray.so",
    packageDirRelPath: "platform-packages/bun-ndarray-linux-arm64-gnu",
    packageFile: "libndarray.so",
  },
  {
    packageName: "bun-ndarray-darwin-x64",
    platformKey: "darwin-x64",
    targetTriple: "x86_64-macos",
    prebuildRelPath: "prebuilds/darwin-x64/libndarray.dylib",
    packageDirRelPath: "platform-packages/bun-ndarray-darwin-x64",
    packageFile: "libndarray.dylib",
  },
  {
    packageName: "bun-ndarray-darwin-arm64",
    platformKey: "darwin-arm64",
    targetTriple: "aarch64-macos",
    prebuildRelPath: "prebuilds/darwin-arm64/libndarray.dylib",
    packageDirRelPath: "platform-packages/bun-ndarray-darwin-arm64",
    packageFile: "libndarray.dylib",
  },
  {
    packageName: "bun-ndarray-windows-x64-gnu",
    platformKey: "win32-x64",
    targetTriple: "x86_64-windows-gnu",
    prebuildRelPath: "prebuilds/win32-x64/ndarray.dll",
    packageDirRelPath: "platform-packages/bun-ndarray-windows-x64-gnu",
    packageFile: "ndarray.dll",
  },
];

const manifestArtifacts: Array<{
  packageName: string;
  platformKey: string;
  targetTriple: string;
  file: string;
  sha256: string;
  sizeBytes: number;
}> = [];

for (const artifact of artifacts) {
  const prebuildAbsPath = join(rootDir, artifact.prebuildRelPath);
  if (!existsSync(prebuildAbsPath)) {
    throw new Error(`Missing artifact: ${artifact.prebuildRelPath}`);
  }

  const bytes = readFileSync(prebuildAbsPath);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const sizeBytes = bytes.byteLength;

  manifestArtifacts.push({
    packageName: artifact.packageName,
    platformKey: artifact.platformKey,
    targetTriple: artifact.targetTriple,
    file: artifact.packageFile,
    sha256,
    sizeBytes,
  });

  const platformMetadata = {
    packageName: artifact.packageName,
    version: packageJson.version,
    abiVersion,
    bunMinVersion,
    zigVersion,
    gitSha,
    generatedAt,
    platformKey: artifact.platformKey,
    targetTriple: artifact.targetTriple,
    file: artifact.packageFile,
    sha256,
    sizeBytes,
  };

  const platformMetadataPath = join(rootDir, artifact.packageDirRelPath, "artifact-metadata.json");
  writeFileSync(platformMetadataPath, `${JSON.stringify(platformMetadata, null, 2)}\n`);
}

const manifest = {
  packageName: packageJson.name,
  version: packageJson.version,
  abiVersion,
  bunMinVersion,
  zigVersion,
  gitSha,
  generatedAt,
  artifacts: manifestArtifacts,
};

const manifestPath = join(rootDir, "prebuilds", "manifest.json");
writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);

console.log(`[artifact-metadata] wrote ${manifestPath}`);
for (const artifact of manifestArtifacts) {
  console.log(
    `[artifact-metadata] ${artifact.packageName} ${artifact.file} sha256=${artifact.sha256.slice(0, 12)}... size=${artifact.sizeBytes}`,
  );
}

