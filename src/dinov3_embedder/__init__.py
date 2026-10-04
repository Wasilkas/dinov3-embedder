"""Frozen DINOv3 features with aligned object IDs and annotations."""

from .embedder import DINOv3Embedder, EmbedderConfig
from .result import EmbeddingResult
from .samples import ImageSample, load_coco, load_csv

__all__ = [
    "DINOv3Embedder",
    "EmbedderConfig",
    "ImageSample",
    "EmbeddingResult",
    "load_coco",
    "load_csv",
]
__version__ = "0.1.0"
