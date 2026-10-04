"""Explicit crop/resize geometry, recorded for nuisance audits."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ._types import BBox, ImageInput, Metadata

if TYPE_CHECKING:
    from .embedder import EmbedderConfig


def read_image(source: ImageInput) -> Image.Image:
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return image.convert("RGB")
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8 or source.ndim not in (2, 3):
            raise ValueError("Image arrays must be uint8 HxW or HxWxC.")
        if source.ndim == 3:
            if source.shape[2] == 1:
                source = source[:, :, 0]
            elif source.shape[2] not in (3, 4):
                raise ValueError("Image arrays must have 1, 3 (RGB), or 4 (RGBA) channels.")
        if min(source.shape[:2]) < 1:
            raise ValueError("Image must have nonzero width and height.")
        return Image.fromarray(source).convert("RGB")
    raise TypeError("Image must be a local path, PIL image, or uint8 NumPy array.")


def prepare_image(
    source: ImageInput,
    bbox: BBox | None,
    config: EmbedderConfig,
) -> tuple[Image.Image, Metadata]:
    image = read_image(source)
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError("Image must have nonzero width and height.")
    if bbox is None:
        x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)
    else:
        coordinates = np.asarray(bbox, dtype=float)
        if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
            raise ValueError("bbox must be four finite xyxy coordinates in pixels.")
        x1, y1, x2, y2 = coordinates.tolist()
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("bbox must have positive area and lie within the image.")
    box_width, box_height = x2 - x1, y2 - y1
    # Context is added on each side as a fraction of the original bbox size.
    cx, cy = config.context_fraction * box_width, config.context_fraction * box_height
    bounds = (
        max(0, math.floor(x1 - cx)),
        max(0, math.floor(y1 - cy)),
        min(width, math.ceil(x2 + cx)),
        min(height, math.ceil(y2 + cy)),
    )
    crop = image.crop(bounds)
    cw, ch = crop.size
    size = config.image_size
    if config.resize_mode == "none":
        if crop.size != (size, size):
            raise ValueError(f"resize_mode='none' requires crops of exactly {size}x{size}.")
        output = crop
        rw, rh, left, top = size, size, 0, 0
    elif config.resize_mode == "stretch":
        output = crop.resize((size, size), Image.Resampling.BICUBIC)
        rw, rh, left, top = size, size, 0, 0
    else:
        scale = size / max(cw, ch)
        rw, rh = max(1, round(cw * scale)), max(1, round(ch * scale))
        left, top = (size - rw) // 2, (size - rh) // 2
        output = Image.new("RGB", (size, size), config.pad_color)
        output.paste(crop.resize((rw, rh), Image.Resampling.BICUBIC), (left, top))
    metadata: Metadata = {
        "image_width": width,
        "image_height": height,
        "bbox_width": box_width,
        "bbox_height": box_height,
        "bbox_area_fraction": box_width * box_height / (width * height),
        "aspect_ratio": box_width / box_height,
        "crop_x1": bounds[0],
        "crop_y1": bounds[1],
        "crop_x2": bounds[2],
        "crop_y2": bounds[3],
        "crop_width": cw,
        "crop_height": ch,
        "resize_scale_x": rw / cw,
        "resize_scale_y": rh / ch,
        "padding_left": left,
        "padding_top": top,
        "padding_right": size - rw - left,
        "padding_bottom": size - rh - top,
        "padding_fraction": 1 - rw * rh / (size * size),
    }
    return output, metadata
