import { ptr } from "bun:ffi";
import { native } from "./ffi";

const textDecoder = new TextDecoder();

const STATUS_LABELS = new Map<number, string>([
  [0, "ND_OK"],
  [1, "ND_E_INVALID_ARG"],
  [2, "ND_E_INVALID_DTYPE"],
  [3, "ND_E_INVALID_SHAPE"],
  [4, "ND_E_INVALID_STRIDES"],
  [5, "ND_E_INVALID_ALIGNMENT"],
  [6, "ND_E_STALE_HANDLE"],
  [7, "ND_E_OOM"],
  [8, "ND_E_NOT_CONTIGUOUS"],
  [9, "ND_E_NOT_IMPLEMENTED"],
  [255, "ND_E_INTERNAL"],
]);

export class NativeError extends Error {
  readonly code: number;
  readonly codeLabel: string;
  readonly symbol: string;

  constructor(symbol: string, code: number, message: string) {
    const label = STATUS_LABELS.get(code) ?? `ND_STATUS_${code}`;
    super(`${symbol} failed with ${label}: ${message}`);
    this.name = "NativeError";
    this.code = code;
    this.codeLabel = label;
    this.symbol = symbol;
  }
}

export function getLastError(): { code: number; message: string } {
  const code = Number(native.nd_last_error_code());

  const buf = new Uint8Array(256);
  const outLen = new BigUint64Array(1);
  native.nd_last_error_message(ptr(buf), BigInt(buf.length), ptr(outLen));

  const len = Number(outLen[0]);
  const safeLen = Number.isFinite(len) ? Math.min(len, buf.length) : 0;
  const message = textDecoder.decode(buf.subarray(0, safeLen)).replace(/\0+$/, "");

  return {
    code,
    message: message || "native operation failed",
  };
}

export function checkStatus(symbol: string, status: number): void {
  if (status === 0) {
    return;
  }

  const last = getLastError();
  const code = last.code || status;
  throw new NativeError(symbol, code, last.message);
}
