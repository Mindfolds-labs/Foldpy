from __future__ import annotations

import numpy as np


def triangulate_points(
    pts1: np.ndarray,
    pts2: np.ndarray,
    baseline: float = 1.0,
    focal: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Triangulação estéreo simplificada.

    pts1, pts2: arrays [N,2] com coordenadas correspondentes.
    Usa disparidade em x:
        z = f * B / d

    Retorna pontos [N,3] = (x, y, z)
    """
    pts1 = np.asarray(pts1, dtype=np.float32)
    pts2 = np.asarray(pts2, dtype=np.float32)

    if pts1.shape != pts2.shape or pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("pts1 and pts2 must have shape [N,2]")

    disparity = pts1[:, 0] - pts2[:, 0]
    z = (focal * baseline) / (np.abs(disparity) + eps)

    x = pts1[:, 0] * z / focal
    y = pts1[:, 1] * z / focal

    return np.stack([x, y, z], axis=1).astype(np.float32)


def surface_mesh_hint(zmap: np.ndarray) -> dict[str, np.ndarray]:
    """
    Gera malha aproximada a partir de height-field 2D.
    Retorna vertices e faces triangulares.
    """
    zmap = np.asarray(zmap, dtype=np.float32)
    if zmap.ndim != 2:
        raise ValueError("zmap must be 2D")

    h, w = zmap.shape
    yy, xx = np.mgrid[0:h, 0:w]

    vertices = np.stack(
        [
            xx.reshape(-1),
            yy.reshape(-1),
            zmap.reshape(-1),
        ],
        axis=1,
    ).astype(np.float32)

    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            i0 = y * w + x
            i1 = y * w + (x + 1)
            i2 = (y + 1) * w + x
            i3 = (y + 1) * w + (x + 1)

            faces.append((i0, i1, i2))
            faces.append((i1, i3, i2))

    return {
        "vertices": vertices,
        "faces": np.asarray(faces, dtype=np.int32),
    }