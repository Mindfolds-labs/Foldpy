from __future__ import annotations

import numpy as np

from .image import ImageGrid


def _center_crop(arr, target_h, target_w):

    h, w = arr.shape[:2]

    y0 = max(0, (h - target_h) // 2)
    x0 = max(0, (w - target_w) // 2)

    return arr[y0:y0 + target_h, x0:x0 + target_w]


def multi_view_align(images: list[ImageGrid]) -> dict:
    """
    Alinha múltiplas imagens por recorte central.

    Retorna tensor:
        [N, H, W, C]
    """

    if len(images) == 0:
        raise ValueError("images list empty")

    heights = [img.height for img in images]
    widths = [img.width for img in images]

    target_h = min(heights)
    target_w = min(widths)

    aligned = []

    for img in images:

        arr = img.as_hwc()

        cropped = _center_crop(arr, target_h, target_w)

        aligned.append(cropped)

    aligned = np.asarray(aligned, dtype=np.float32)

    return {
        "aligned": aligned,
        "views": len(images),
        "height": target_h,
        "width": target_w,
        "channels": aligned.shape[-1] if aligned.ndim == 4 else 1,
    }


def multi_view_stack(images: list[ImageGrid]) -> np.ndarray:
    """
    Empilha imagens alinhadas.

    Output:
        tensor [N, H, W, C]
    """

    aligned = multi_view_align(images)

    return aligned["aligned"]


def multi_view_grayscale(images: list[ImageGrid]) -> np.ndarray:
    """
    Converte múltiplas imagens para grayscale stack.

    Output:
        [N, H, W]
    """

    stack = multi_view_stack(images)

    if stack.ndim == 4:

        gray = stack.mean(axis=3)

        return gray

    return stack


def detect_points_simple(image: ImageGrid, threshold=0.2):
    """
    Detector simples baseado em intensidade.
    """

    gray = image.grayscale().data.astype(np.float32)

    mask = gray > threshold

    ys, xs = np.where(mask)

    pts = np.stack([xs, ys], axis=1)

    return pts.astype(np.float32)


def match_points_simple(pts1, pts2, max_dist=3.0):
    """
    Match de pontos simples por distância.

    Retorna pares correspondentes.
    """

    matches1 = []
    matches2 = []

    for p1 in pts1:

        d = np.linalg.norm(pts2 - p1, axis=1)

        idx = np.argmin(d)

        if d[idx] < max_dist:

            matches1.append(p1)
            matches2.append(pts2[idx])

    if len(matches1) == 0:
        return None, None

    return np.array(matches1), np.array(matches2)


def stereo_correspondence(img1: ImageGrid, img2: ImageGrid):
    """
    Pipeline simples:

    detectar pontos
    casar pontos
    retornar correspondências
    """

    pts1 = detect_points_simple(img1)

    pts2 = detect_points_simple(img2)

    p1, p2 = match_points_simple(pts1, pts2)

    return p1, p2


def multi_view_triangulate(img1: ImageGrid, img2: ImageGrid, triangulate_fn):
    """
    Pipeline estéreo completo:

    detect
    match
    triangulate
    """

    p1, p2 = stereo_correspondence(img1, img2)

    if p1 is None:
        return None

    pts3d = triangulate_fn(p1, p2)

    return pts3d