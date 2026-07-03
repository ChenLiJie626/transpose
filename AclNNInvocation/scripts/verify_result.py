#!/usr/bin/env python3
import argparse
import os

import numpy as np


ROWS = 136
INNER = 8
COLS = 256


def reference(x: np.ndarray, batch: int) -> np.ndarray:
    return x.reshape(batch, ROWS, INNER, COLS).transpose(1, 3, 0, 2).reshape(ROWS, COLS, batch * INNER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--input-dir", default=os.path.join(os.path.dirname(__file__), "..", "input"))
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    args = parser.parse_args()

    input_shape = (args.batch * ROWS, INNER, COLS)
    output_shape = (ROWS, COLS, INNER * args.batch)
    a_r = np.fromfile(os.path.join(args.input_dir, "input_ar.bin"), dtype=np.float32).reshape(input_shape)
    a_i = np.fromfile(os.path.join(args.input_dir, "input_ai.bin"), dtype=np.float32).reshape(input_shape)
    c_r = np.fromfile(os.path.join(args.output_dir, "output_cr.bin"), dtype=np.float32).reshape(output_shape)
    c_i = np.fromfile(os.path.join(args.output_dir, "output_ci.bin"), dtype=np.float32).reshape(output_shape)

    diff_r = np.max(np.abs(c_r - reference(a_r, args.batch)))
    diff_i = np.max(np.abs(c_i - reference(a_i, args.batch)))
    print(f"max_diff_r={diff_r} max_diff_i={diff_i}")
    if diff_r != 0.0 or diff_i != 0.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
