"""SVG parsing and raster bridge."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import numpy as np

from .image import ImageGrid


def _parse_color(value: str | None) -> tuple[int, int, int]:
    if value is None:
        return (255, 255, 255)
    value = value.strip().lower()
    if value.startswith("#") and len(value) == 7:
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    mapping = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    return mapping.get(value, (255, 255, 255))


def from_svg(path: str) -> dict:
    """Parse basic SVG shapes into a scene dictionary."""
    root = ET.parse(path).getroot()
    width = int(float(re.findall(r"[\d.]+", root.attrib.get("width", "256"))[0]))
    height = int(float(re.findall(r"[\d.]+", root.attrib.get("height", "256"))[0]))
    shapes: list[dict] = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "rect":
            shapes.append({"type": "rect", "x": float(elem.attrib.get("x", 0)), "y": float(elem.attrib.get("y", 0)), "width": float(elem.attrib.get("width", 0)), "height": float(elem.attrib.get("height", 0)), "fill": _parse_color(elem.attrib.get("fill"))})
        elif tag == "circle":
            shapes.append({"type": "circle", "cx": float(elem.attrib.get("cx", 0)), "cy": float(elem.attrib.get("cy", 0)), "r": float(elem.attrib.get("r", 0)), "fill": _parse_color(elem.attrib.get("fill"))})
    return {"width": width, "height": height, "shapes": shapes}


def svg_to_raster(scene: dict, shape: tuple[int, int] | None = None) -> ImageGrid:
    """Rasterize scene dict into ImageGrid."""
    h, w = shape if shape is not None else (scene["height"], scene["width"])
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for obj in scene["shapes"]:
        if obj["type"] == "rect":
            x, y = int(obj["x"]), int(obj["y"])
            canvas[y:y + int(obj["height"]), x:x + int(obj["width"])] = np.asarray(obj["fill"], dtype=np.uint8)
        elif obj["type"] == "circle":
            yy, xx = np.mgrid[0:h, 0:w]
            mask = (xx - obj["cx"]) ** 2 + (yy - obj["cy"]) ** 2 <= obj["r"] ** 2
            canvas[mask] = np.asarray(obj["fill"], dtype=np.uint8)
    return ImageGrid(canvas, layout="HWC")
