from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np

from .image import ImageGrid


def _parse_color(value: str | None) -> tuple[int, int, int]:
    if value is None:
        return (255, 255, 255)

    value = value.strip().lower()
    named = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }
    if value in named:
        return named[value]
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    return (255, 255, 255)


def from_svg(path: str) -> dict:
    """
    Parser SVG básico.
    Suporta rect, circle, line.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    width = root.attrib.get("width", "256")
    height = root.attrib.get("height", "256")

    width_num = float(re.findall(r"[\d.]+", width)[0])
    height_num = float(re.findall(r"[\d.]+", height)[0])

    shapes = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag == "rect":
            shapes.append({
                "type": "rect",
                "x": float(elem.attrib.get("x", 0)),
                "y": float(elem.attrib.get("y", 0)),
                "width": float(elem.attrib.get("width", 0)),
                "height": float(elem.attrib.get("height", 0)),
                "fill": _parse_color(elem.attrib.get("fill")),
            })

        elif tag == "circle":
            shapes.append({
                "type": "circle",
                "cx": float(elem.attrib.get("cx", 0)),
                "cy": float(elem.attrib.get("cy", 0)),
                "r": float(elem.attrib.get("r", 0)),
                "fill": _parse_color(elem.attrib.get("fill")),
            })

        elif tag == "line":
            shapes.append({
                "type": "line",
                "x1": float(elem.attrib.get("x1", 0)),
                "y1": float(elem.attrib.get("y1", 0)),
                "x2": float(elem.attrib.get("x2", 0)),
                "y2": float(elem.attrib.get("y2", 0)),
                "stroke": _parse_color(elem.attrib.get("stroke")),
            })

    return {
        "width": int(round(width_num)),
        "height": int(round(height_num)),
        "shapes": shapes,
    }


def _draw_rect(canvas: np.ndarray, shape: dict) -> None:
    x = int(round(shape["x"]))
    y = int(round(shape["y"]))
    w = int(round(shape["width"]))
    h = int(round(shape["height"]))
    color = np.asarray(shape["fill"], dtype=np.uint8)

    y0 = max(0, y)
    x0 = max(0, x)
    y1 = min(canvas.shape[0], y + h)
    x1 = min(canvas.shape[1], x + w)

    if y0 < y1 and x0 < x1:
        canvas[y0:y1, x0:x1] = color


def _draw_circle(canvas: np.ndarray, shape: dict) -> None:
    cx = float(shape["cx"])
    cy = float(shape["cy"])
    r = float(shape["r"])
    color = np.asarray(shape["fill"], dtype=np.uint8)

    yy, xx = np.mgrid[0:canvas.shape[0], 0:canvas.shape[1]]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    canvas[mask] = color


def _draw_line(canvas: np.ndarray, shape: dict) -> None:
    x1 = float(shape["x1"])
    y1 = float(shape["y1"])
    x2 = float(shape["x2"])
    y2 = float(shape["y2"])
    color = np.asarray(shape["stroke"], dtype=np.uint8)

    steps = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
    xs = np.linspace(x1, x2, steps)
    ys = np.linspace(y1, y2, steps)

    for x, y in zip(xs, ys):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= yi < canvas.shape[0] and 0 <= xi < canvas.shape[1]:
            canvas[yi, xi] = color


def svg_to_raster(scene: dict, shape: tuple[int, int] | None = None) -> ImageGrid:
    h = scene["height"]
    w = scene["width"]

    if shape is not None:
        h, w = shape

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for item in scene["shapes"]:
        if item["type"] == "rect":
            _draw_rect(canvas, item)
        elif item["type"] == "circle":
            _draw_circle(canvas, item)
        elif item["type"] == "line":
            _draw_line(canvas, item)

    return ImageGrid(canvas, domain="visual", layout="HWC")