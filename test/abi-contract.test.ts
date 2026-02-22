import { describe, expect, test } from "bun:test";
import { native } from "../src/ffi";
import { symbolDefs } from "../src/symbols";

function extractHeaderSymbols(header: string): string[] {
  const re = /^ND_API\s+[^(]+\s+(nd_[a-z0-9_]+)\s*\(/gm;
  const out: string[] = [];
  for (const m of header.matchAll(re)) {
    out.push(m[1]!);
  }
  return out;
}

describe("ABI contract", () => {
  test("header and TS symbol table match exactly", async () => {
    const headerPath = new URL("../native/include/ndarray.h", import.meta.url);
    const text = await Bun.file(headerPath).text();

    const headerSyms = extractHeaderSymbols(text).sort();
    const tsSyms = Object.keys(symbolDefs).sort();

    expect(tsSyms).toEqual(headerSyms);
  });

  test("loaded native symbols include all declared TS symbols", () => {
    for (const key of Object.keys(symbolDefs)) {
      expect(typeof (native as any)[key]).toBe("function");
    }
  });
});

