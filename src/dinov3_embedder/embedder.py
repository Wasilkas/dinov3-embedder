"""DINOv3 ViT batch inference: CLS and/or patch-token pooling."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ._types import BBox, ImageInput, Metadata, ScalarLabel
from .preprocessing import prepare_image, read_image
from .result import EmbeddingResult
from .samples import ImageSample

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class EmbedderConfig:
    model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    revision: str | None = None
    device: str = "auto"
    dtype: str = "float32"
    batch_size: int = 16
    image_size: int = 512
    resize_mode: str = "letterbox"
    context_fraction: float = 0.0
    pad_color: tuple[int, int, int] = (0, 0, 0)
    pooling: str = "cls"
    normalize: bool = True
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.model_id or not isinstance(self.model_id, str):
            raise ValueError("model_id must be a Hugging Face model ID or local directory.")
        for name in ("batch_size", "image_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.resize_mode not in ("letterbox", "stretch", "none"):
            raise ValueError("resize_mode must be letterbox, stretch, or none.")
        if self.pooling not in ("cls", "patch_mean", "cls_patch_mean"):
            raise ValueError("pooling must be cls, patch_mean, or cls_patch_mean.")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32, float16, or bfloat16.")
        if not np.isfinite(self.context_fraction) or self.context_fraction < 0:
            raise ValueError("context_fraction must be finite and nonnegative.")
        if len(self.pad_color) != 3 or any(
            isinstance(c, bool) or not isinstance(c, Integral) or not 0 <= c <= 255
            for c in self.pad_color
        ):
            raise ValueError("pad_color must contain three integers in [0, 255].")
        if not isinstance(self.normalize, bool) or not isinstance(self.local_files_only, bool):
            raise ValueError("normalize and local_files_only must be boolean.")


class DINOv3Embedder:
    def __init__(
        self,
        config: EmbedderConfig | None = None,
        *,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Load a frozen HF DINOv3 ViT model and its image processor.

        token defaults to Hugging Face's configured authentication. It is never
        included in exported config. local_files_only prevents model downloads.
        """
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.config = config or EmbedderConfig()
        config = self.config
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is not available. Use device='cpu' or install CUDA-enabled PyTorch."
            )
        if self.device.type == "cpu" and config.dtype != "float32":
            raise ValueError("CPU inference uses dtype='float32'.")
        self.dtype = getattr(torch, config.dtype)
        loading: dict[str, Any] = {
            "revision": config.revision,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": config.local_files_only,
            "trust_remote_code": False,
        }
        self.processor = AutoImageProcessor.from_pretrained(config.model_id, **loading)
        self.model = AutoModel.from_pretrained(config.model_id, dtype=self.dtype, **loading)
        model_config = self.model.config
        if model_config.model_type != "dinov3_vit":
            raise ValueError(
                "This embedder supports DINOv3 ViT checkpoints (model_type='dinov3_vit')."
            )
        patch = model_config.patch_size
        self.patch_size: tuple[int, ...] = (
            (int(patch), int(patch)) if isinstance(patch, Integral) else tuple(patch)
        )
        if len(self.patch_size) != 2 or any(config.image_size % size for size in self.patch_size):
            raise ValueError(f"image_size must be divisible by model patch size {self.patch_size}.")
        self.num_register_tokens = model_config.num_register_tokens
        self.hidden_size = model_config.hidden_size
        self.embedding_dim = self.hidden_size * (2 if config.pooling == "cls_patch_mean" else 1)
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.model.requires_grad_(False)

    def encode(
        self,
        images: Iterable[ImageInput],
        *,
        bboxes: Iterable[BBox | None] | None = None,
    ) -> NDArray[np.float32]:
        """Return float32 (N,D) NumPy features, preserving iterable order.

        images is an iterable of paths, PIL images, or uint8 NumPy images.
        Optional bboxes is an equally sized iterable of pixel xyxy or None.
        """
        if isinstance(images, (str, Path, Image.Image)) or (
            isinstance(images, np.ndarray) and images.ndim < 4
        ):
            raise ValueError("encode expects an iterable of images; wrap a single image in a list.")

        def samples() -> Iterator[ImageSample]:
            if bboxes is None:
                for i, image in enumerate(images):
                    yield ImageSample(str(i), image)
            else:
                sentinel = object()
                for i, (candidate_image, candidate_bbox) in enumerate(
                    itertools.zip_longest(images, bboxes, fillvalue=sentinel)
                ):
                    if candidate_image is sentinel or candidate_bbox is sentinel:
                        raise ValueError("images and bboxes must have the same length.")
                    yield ImageSample(
                        str(i),
                        cast(ImageInput, candidate_image),
                        bbox=cast(BBox | None, candidate_bbox),
                    )

        return self.embed_samples(samples()).embeddings

    def encode_bboxes(self, image: ImageInput, bboxes: Iterable[BBox]) -> NDArray[np.float32]:
        """Encode multiple boxes of one image; decode the source only once."""
        if bboxes is None:
            raise ValueError("bboxes must be an iterable of xyxy boxes.")
        source = read_image(image)
        return self.embed_samples(
            ImageSample(str(i), source, bbox=box) for i, box in enumerate(bboxes)
        ).embeddings

    def embed_samples(self, samples: Iterable[ImageSample]) -> EmbeddingResult:
        """Encode ImageSample objects and retain annotations/geometry for audit.

        Pixel memory is bounded by batch_size; output features remain in RAM.
        Labels/groups must each be present for every sample or absent for all.
        An invalid object raises an error rather than silently skipping a row.
        """
        iterator = iter(samples)
        features: list[NDArray[np.float32]] = []
        ids: list[str] = []
        labels: list[ScalarLabel | None] = []
        groups: list[ScalarLabel | None] = []
        geometry: list[Metadata] = []
        seen = set()
        while batch := list(itertools.islice(iterator, self.config.batch_size)):
            for sample in batch:
                if not isinstance(sample, ImageSample):
                    raise TypeError("embed_samples expects ImageSample objects.")
                if sample.sample_id is None or not str(sample.sample_id):
                    raise ValueError("Every object needs a nonempty sample_id.")
                sample_id = str(sample.sample_id)
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id: {sample_id}.")
                seen.add(sample_id)
                ids.append(sample_id)
                labels.append(sample.label)
                groups.append(sample.group)
            vectors, rows = self._encode_batch(batch)
            features.append(vectors)
            geometry.extend(rows)
        output = (
            np.concatenate(features)
            if features
            else np.empty((0, self.embedding_dim), dtype=np.float32)
        )
        return EmbeddingResult(
            embeddings=output,
            sample_ids=np.asarray(ids, dtype=str),
            labels=self._optional_array(labels, "labels"),
            groups=self._optional_array(groups, "groups"),
            metadata=geometry,
            info={**self.info(), "n_samples": len(ids)},
        )

    @staticmethod
    def _optional_array(values: Sequence[ScalarLabel | None], name: str) -> NDArray[np.str_] | None:
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"{name} must be present for every object or absent for all.")
        if any(
            isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in values
        ):
            raise ValueError(f"{name} must not contain nonfinite values.")
        return np.asarray(values, dtype=str)

    def _encode_batch(
        self, samples: Sequence[ImageSample]
    ) -> tuple[NDArray[np.float32], list[Metadata]]:
        import torch

        images: list[Image.Image] = []
        rows: list[Metadata] = []
        decoded: dict[str, Image.Image] = {}
        for sample in samples:
            source = sample.image
            if isinstance(source, (str, Path)):
                key = str(source)
                if key not in decoded:
                    decoded[key] = read_image(source)
                source = decoded[key]
            image, metadata = prepare_image(source, sample.bbox, self.config)
            images.append(image)
            rows.append(metadata)
        # Shape is controlled above; preserve model-specific rescale/normalization.
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixels = inputs["pixel_values"]
        expected = (len(samples), 3, self.config.image_size, self.config.image_size)
        if tuple(pixels.shape) != expected:
            raise RuntimeError(
                f"Processor changed the image geometry: {tuple(pixels.shape)} != {expected}."
            )
        pixels = pixels.to(device=self.device, dtype=self.dtype)
        with torch.inference_mode():
            output = self.model(pixel_values=pixels, return_dict=True)
            vectors = self._pool(output.last_hidden_state).float()
            if not torch.isfinite(vectors).all():
                raise RuntimeError("DINOv3 produced nonfinite embeddings.")
            if self.config.normalize:
                norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
                if (norms == 0).any():
                    raise RuntimeError("Cannot normalize zero embeddings.")
                vectors = vectors / norms
        array: NDArray[np.float32] = vectors.cpu().numpy().copy()
        return array, rows

    def _pool(self, tokens: Tensor) -> Tensor:
        import torch

        ph, pw = self.patch_size
        patch_count = (self.config.image_size // ph) * (self.config.image_size // pw)
        prefix = 1 + self.num_register_tokens
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (prefix + patch_count, self.hidden_size):
            raise RuntimeError(
                "Unexpected DINOv3 token layout; check model and processor compatibility."
            )
        cls = tokens[:, 0]
        if self.config.pooling == "cls":
            return cls
        patches = tokens[:, prefix:].mean(dim=1)  # CLS and register tokens excluded
        return patches if self.config.pooling == "patch_mean" else torch.cat((cls, patches), dim=1)

    def info(self) -> dict[str, Any]:
        """Record the extraction recipe and library versions, excluding credentials."""
        import torch
        import transformers

        return {
            "config": asdict(self.config),
            "resolved_device": str(self.device),
            "resolved_model_revision": getattr(self.model.config, "_commit_hash", None),
            "embedding_dim": self.embedding_dim,
            "patch_size": list(self.patch_size),
            "num_register_tokens": self.num_register_tokens,
            "image_processor": {
                "class": type(self.processor).__name__,
                "image_mean": self.processor.image_mean,
                "image_std": self.processor.image_std,
                "rescale_factor": self.processor.rescale_factor,
                "resize": False,
                "center_crop": False,
            },
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
            },
            "pooling_patch_reference": "all patches, including padded image regions",
        }
