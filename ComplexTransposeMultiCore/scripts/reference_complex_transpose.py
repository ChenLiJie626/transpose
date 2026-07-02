#!/usr/bin/env python3
import argparse
import numpy as np


def complex_transpose(a_r: np.ndarray, a_i: np.ndarray):
    if a_r.shape != a_i.shape:
        raise ValueError(f"a_r and a_i shapes differ: {a_r.shape} vs {a_i.shape}")
    if a_r.ndim != 3 or a_r.shape[0] % 136 != 0 or a_r.shape[1:] != (8, 256):
        raise ValueError(f"expected input shape [batch * 136, 8, 256], got {a_r.shape}")

    batch = a_r.shape[0] // 136
    c_r = a_r.reshape(batch, 136, 8, 256).transpose(1, 3, 2, 0).reshape(136, 256, 8 * batch)
    c_i = a_i.reshape(batch, 136, 8, 256).transpose(1, 3, 2, 0).reshape(136, 256, 8 * batch)
    return c_r, c_i


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    shape = (args.batch * 136, 8, 256)
    a_r = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    a_i = -a_r
    c_r, c_i = complex_transpose(a_r, a_i)
    print("a_r", a_r.shape, "c_r", c_r.shape)
    print("check", c_r[0, 0, :8].tolist(), c_i[0, 0, :8].tolist())


if __name__ == "__main__":
    main()
