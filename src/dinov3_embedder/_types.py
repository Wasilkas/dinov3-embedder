"""Shared input and output types for image embedding."""

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray
from PIL import Image

ImageInput = str | Path | Image.Image | NDArray[np.uint8]
BBox = ArrayLike
Metadata = dict[str, int | float]
ScalarLabel = str | int | float
