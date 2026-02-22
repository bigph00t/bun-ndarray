#!/usr/bin/env python3
import json
import sys

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    print(json.dumps({"ok": False, "error": f"numpy-unavailable: {e}"}))
    sys.exit(0)


def as_array(payload):
    dtype = payload.get("dtype", "f64")
    if dtype == "f32":
        dt = np.float32
    elif dtype == "f64":
        dt = np.float64
    elif dtype == "i32":
        dt = np.int32
    else:
        raise ValueError(f"unsupported dtype: {dtype}")

    data = np.array(payload["data"], dtype=dt)
    shape = tuple(payload["shape"])
    return data.reshape(shape)


def main():
    req = json.loads(sys.stdin.read())
    op = req["op"]

    if op == "add":
        a = as_array(req["a"])
        b = as_array(req["b"])
        out = a + b
        print(json.dumps({"ok": True, "data": out.reshape(-1).tolist(), "shape": list(out.shape)}))
        return

    if op == "sum":
        a = as_array(req["a"])
        out = a.sum()
        print(json.dumps({"ok": True, "value": float(out)}))
        return

    if op == "sum_axis":
        a = as_array(req["a"])
        axis = int(req["axis"])
        out = a.sum(axis=axis)
        print(json.dumps({"ok": True, "data": np.array(out).reshape(-1).tolist(), "shape": list(np.array(out).shape)}))
        return

    if op == "matmul":
        a = as_array(req["a"])
        b = as_array(req["b"])
        out = a @ b
        print(json.dumps({"ok": True, "data": out.reshape(-1).tolist(), "shape": list(out.shape)}))
        return

    if op == "where":
        cond = as_array(req["cond"]) != 0
        x = as_array(req["x"])
        y = as_array(req["y"])
        out = np.where(cond, x, y)
        print(json.dumps({"ok": True, "data": out.reshape(-1).tolist(), "shape": list(out.shape)}))
        return

    print(json.dumps({"ok": False, "error": f"unsupported op: {op}"}))


if __name__ == "__main__":
    main()
