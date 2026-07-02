#!/usr/bin/env python3
import argparse
import os

import numpy as np


ROWS = 136
INNER = 8
COLS = 256


def fmt(values, precision=3):
    return np.array2string(
        np.asarray(values),
        precision=precision,
        separator=", ",
        suppress_small=False,
        max_line_width=180,
    )


def read_float_bin(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise ValueError(f"{path}: expected {expected} float32 values, got {data.size}")
    return data.reshape(shape)


def print_linear(name, array, count):
    flat = array.reshape(-1)
    print(f"{name} first {count} linear float32 values:")
    print(fmt(flat[:count]))
    print()


def print_mapping(a_r, a_i, c_r, c_i, coords):
    batch = a_r.shape[0] // ROWS
    print("coordinate mapping checks:")
    print("  c[row, col, inner * batch + batch_idx] == a[batch_idx * 136 + row, inner, col]")
    for batch_idx, row, inner, col in coords:
        in_row = batch_idx * ROWS + row
        out_last = inner * batch + batch_idx
        ar = a_r[in_row, inner, col]
        ai = a_i[in_row, inner, col]
        cr = c_r[row, col, out_last]
        ci = c_i[row, col, out_last]
        print(
            f"  b={batch_idx:2d} row={row:3d} inner={inner} col={col:3d} "
            f"| a_r[{in_row},{inner},{col}]={ar:10.3f} -> c_r[{row},{col},{out_last}]={cr:10.3f} "
            f"| a_i={ai:10.3f} -> c_i={ci:10.3f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--input-dir", default=os.path.join(os.path.dirname(__file__), "..", "input"))
    parser.add_argument("--output-dir", default=os.path.join(os.path.dirname(__file__), "..", "output"))
    parser.add_argument("--linear-count", type=int, default=32)
    args = parser.parse_args()

    input_shape = (args.batch * ROWS, INNER, COLS)
    output_shape = (ROWS, COLS, INNER * args.batch)
    a_r = read_float_bin(os.path.join(args.input_dir, "input_ar.bin"), input_shape)
    a_i = read_float_bin(os.path.join(args.input_dir, "input_ai.bin"), input_shape)
    c_r = read_float_bin(os.path.join(args.output_dir, "output_cr.bin"), output_shape)
    c_i = read_float_bin(os.path.join(args.output_dir, "output_ci.bin"), output_shape)

    print(f"a_r shape={a_r.shape}, a_i shape={a_i.shape}")
    print(f"c_r shape={c_r.shape}, c_i shape={c_i.shape}")
    print()

    print_linear("input_ar.bin", a_r, args.linear_count)
    print_linear("input_ai.bin", a_i, args.linear_count)
    print_linear("output_cr.bin", c_r, args.linear_count)
    print_linear("output_ci.bin", c_i, args.linear_count)

    print("a_r[0, :, 0:8]  # batch 0, row 0, all inner, first 8 cols")
    print(fmt(a_r[0, :, 0:8]))
    print()
    if args.batch > 1:
        print("a_r[136, :, 0:8]  # batch 1, row 0, all inner, first 8 cols")
        print(fmt(a_r[136, :, 0:8]))
        print()

    print("c_r[0, 0, :]  # row 0, col 0, all inner*batch lanes")
    print(fmt(c_r[0, 0, :]))
    print()
    print("c_i[0, 0, :]  # row 0, col 0, all inner*batch lanes")
    print(fmt(c_i[0, 0, :]))
    print()
    print("c_r[0, 0:8, 0:8]  # row 0, first 8 cols, inner 0 across first batches")
    print(fmt(c_r[0, 0:8, 0:8]))
    print()

    coords = [
        (0, 0, 0, 0),
        (0, 0, 7, 0),
        (0, 0, 3, 5),
        (0, 135, 7, 255),
        (args.batch // 2, 12, 3, 128),
        (args.batch - 1, 0, 0, 0),
        (args.batch - 1, 135, 7, 255),
    ]
    print_mapping(a_r, a_i, c_r, c_i, coords)

    ref_r = a_r.reshape(args.batch, ROWS, INNER, COLS).transpose(1, 3, 2, 0).reshape(output_shape)
    ref_i = a_i.reshape(args.batch, ROWS, INNER, COLS).transpose(1, 3, 2, 0).reshape(output_shape)
    print(f"max_diff_r={np.max(np.abs(c_r - ref_r))}")
    print(f"max_diff_i={np.max(np.abs(c_i - ref_i))}")


if __name__ == "__main__":
    main()
