import { NDArray } from "../src/index";

function timeMs(fn: () => void): number {
  const start = Bun.nanoseconds();
  fn();
  return Number(Bun.nanoseconds() - start) / 1e6;
}

function benchAdd(n: number): void {
  const a = new Float64Array(n);
  const b = new Float64Array(n);
  a.fill(1);
  b.fill(2);

  const js = timeMs(() => {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      out[i] = a[i] + b[i];
    }
  });

  using na = NDArray.fromTypedArray(a);
  using nb = NDArray.fromTypedArray(b);

  const native = timeMs(() => {
    using out = na.add(nb);
    out.toFloat64Array();
  });

  const speedup = js / native;
  console.log(`add n=${n}: js=${js.toFixed(4)}ms native=${native.toFixed(4)}ms speedup=${speedup.toFixed(2)}x`);
}

function benchSum(n: number): void {
  const data = new Float64Array(n);
  data.fill(1.25);

  const js = timeMs(() => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += data[i];
    }
    if (sum === 0) {
      throw new Error("impossible");
    }
  });

  using arr = NDArray.fromTypedArray(data);
  const native = timeMs(() => {
    const sum = arr.sum();
    if (sum === 0) {
      throw new Error("impossible");
    }
  });

  const speedup = js / native;
  console.log(`sum n=${n}: js=${js.toFixed(4)}ms native=${native.toFixed(4)}ms speedup=${speedup.toFixed(2)}x`);
}

console.log(`SIMD width f64: ${NDArray.simdWidthF64()}`);
for (const n of [1_000, 10_000, 100_000, 1_000_000]) {
  benchAdd(n);
}
for (const n of [1_000, 10_000, 100_000, 1_000_000]) {
  benchSum(n);
}
