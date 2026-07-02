#!/usr/bin/env python3
import argparse
import os

import numpy as np


ROWS = 136
INNER = 8
COLS = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "..", "input"))
    args = parser.parse_args()

    shape = (args.batch * ROWS, INNER, COLS)
    total = int(np.prod(shape))
    centered = (np.arange(total, dtype=np.int64) % 4096) - 2048
    a_r = (centered.astype(np.float32) * np.float32(0.125)).reshape(shape)
    a_i = (-a_r.reshape(-1) + (np.arange(total, dtype=np.int64) % 7).astype(np.float32) * np.float32(0.25)).reshape(shape)

    os.makedirs(args.out_dir, exist_ok=True)
    a_r.tofile(os.path.join(args.out_dir, "input_ar.bin"))
    a_i.tofile(os.path.join(args.out_dir, "input_ai.bin"))


if __name__ == "__main__":
    main()
