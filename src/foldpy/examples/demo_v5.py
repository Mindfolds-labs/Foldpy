from __future__ import annotations

import numpy as np

import foldpy as fp


def main() -> None:
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[12:52, 16:48, 0] = 1.0
    img[20:44, 22:42, 1] = 0.7
    img[26:38, 28:36, 2] = 0.5

    grid = fp.ImageGrid(img, layout="HWC")

    edges = fp.edge_map(grid)
    region_info = fp.mask_region(grid, threshold=0.1)
    bbox = region_info["bbox"]

    zmap = fp.lift_approx(grid, smooth_passes=4, mask_threshold=0.1, use_depth_hint=True)
    mesh = fp.surface_mesh_hint(zmap)
    proj = fp.projective_score(grid)

    pts1 = np.array([[10, 10], [20, 15], [30, 25]], dtype=np.float32)
    pts2 = np.array([[9, 10], [18, 15], [27, 25]], dtype=np.float32)
    points3d = fp.triangulate_points(pts1, pts2, baseline=1.0, focal=1.0)

    print("Edges shape:", edges.shape)
    print("BBox:", bbox)
    print("ZMap range:", float(zmap.min()), float(zmap.max()))
    print("Mesh vertices:", mesh["vertices"].shape)
    print("Mesh faces:", mesh["faces"].shape)
    print("Projective score:", proj)
    print("Triangulated points:\n", points3d)


if __name__ == "__main__":
    main()