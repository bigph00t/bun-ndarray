import { describe, expect, test } from "bun:test";
import { NDArray } from "../src/index";

const PY = `import numpy`;
const numpyProbe = Bun.spawnSync(["python3", "-c", PY]);
const HAS_NUMPY = numpyProbe.exitCode === 0;

function runNumpy(payload: unknown): any {
  const proc = Bun.spawnSync(["python3", "scripts/numpy-diff.py"], {
    stdin: JSON.stringify(payload),
    stdout: "pipe",
    stderr: "pipe",
  });

  const outText = new TextDecoder().decode(proc.stdout);
  if (proc.exitCode !== 0) {
    throw new Error(`numpy-diff.py failed: ${new TextDecoder().decode(proc.stderr)}`);
  }
  return JSON.parse(outText);
}

const maybeTest = HAS_NUMPY ? test : test.skip;

describe("numpy differential", () => {
  maybeTest("add/sum/sum_axis/matmul/where agree with numpy", () => {
    const aData = [1, 2, 3, 4, 5, 6];
    const bData = [10, 20, 30, 40, 50, 60];

    using a = NDArray.fromTypedArray(new Float64Array(aData), [2, 3]);
    using b = NDArray.fromTypedArray(new Float64Array(bData), [2, 3]);

    using add = a.add(b);
    const addPy = runNumpy({
      op: "add",
      a: { dtype: "f64", data: aData, shape: [2, 3] },
      b: { dtype: "f64", data: bData, shape: [2, 3] },
    });
    expect(addPy.ok).toBe(true);
    expect(add.shape).toEqual(addPy.shape);
    expect(Array.from(add.toFloat64Array())).toEqual(addPy.data);

    const sumPy = runNumpy({
      op: "sum",
      a: { dtype: "f64", data: aData, shape: [2, 3] },
    });
    expect(sumPy.ok).toBe(true);
    expect(a.sum()).toBeCloseTo(sumPy.value, 12);

    using sAxis = a.sumAxis(1);
    const sumAxisPy = runNumpy({
      op: "sum_axis",
      axis: 1,
      a: { dtype: "f64", data: aData, shape: [2, 3] },
    });
    expect(sumAxisPy.ok).toBe(true);
    expect(sAxis.shape).toEqual(sumAxisPy.shape);
    expect(Array.from(sAxis.toFloat64Array())).toEqual(sumAxisPy.data);

    using mA = NDArray.fromTypedArray(new Float64Array([
      1, 2, 3,
      4, 5, 6,
    ]), [2, 3]);
    using mB = NDArray.fromTypedArray(new Float64Array([
      7, 8,
      9, 10,
      11, 12,
    ]), [3, 2]);
    using mOut = mA.matmul(mB);

    const mmPy = runNumpy({
      op: "matmul",
      a: { dtype: "f64", data: [1, 2, 3, 4, 5, 6], shape: [2, 3] },
      b: { dtype: "f64", data: [7, 8, 9, 10, 11, 12], shape: [3, 2] },
    });
    expect(mmPy.ok).toBe(true);
    expect(mOut.shape).toEqual(mmPy.shape);
    expect(Array.from(mOut.toFloat64Array())).toEqual(mmPy.data);

    using cond = NDArray.fromTyped(new Int32Array([0, 1, 0, 1]), [2, 2]);
    using x = NDArray.fromTypedArray(new Float64Array([100, 200, 300, 400]), [2, 2]);
    using y = NDArray.fromTypedArray(new Float64Array([1, 2, 3, 4]), [2, 2]);
    using w = cond.where(x, y);

    const wherePy = runNumpy({
      op: "where",
      cond: { dtype: "i32", data: [0, 1, 0, 1], shape: [2, 2] },
      x: { dtype: "f64", data: [100, 200, 300, 400], shape: [2, 2] },
      y: { dtype: "f64", data: [1, 2, 3, 4], shape: [2, 2] },
    });
    expect(wherePy.ok).toBe(true);
    expect(w.shape).toEqual(wherePy.shape);
    expect(Array.from(w.toFloat64Array())).toEqual(wherePy.data);
  });
});
