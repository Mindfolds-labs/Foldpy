from __future__ import annotations

import time
import numpy as np

import foldpy as fp


def main() -> None:
    img = np.random.rand(256, 256, 3).astype(np.float32)
    grid = fp.ImageGrid(img, layout="HWC")

    t0 = time.perf_counter()
    for _ in range(200):
        _ = fp.patchify(grid, (16, 16))
    t1 = time.perf_counter()

    print("patchify 200x:", round(t1 - t0, 6), "s")


if __name__ == "__main__":
    main()