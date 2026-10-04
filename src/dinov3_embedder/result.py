"""NumPy artifacts compatible with embeddings-quality."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from numpy.typing import NDArray

from ._types import Metadata


class _ExportArrays(TypedDict, total=False):
    embeddings: NDArray[np.float32]
    sample_ids: NDArray[np.str_]
    labels: NDArray[np.str_]
    groups: NDArray[np.str_]


@dataclass
class EmbeddingResult:
    embeddings: NDArray[np.float32]
    sample_ids: NDArray[np.str_]
    labels: NDArray[np.str_] | None
    groups: NDArray[np.str_] | None
    metadata: list[Metadata]
    info: dict[str, Any]

    def save(self, path: str | Path) -> Path:
        """Write .npz plus .metadata.csv and .config.json siblings.

        Existing files at these paths are overwritten. Labels are omitted for
        unlabeled datasets; such NPZs need labels added before quality auditing.
        """
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Output path must end with .npz.")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: _ExportArrays = {
            "embeddings": self.embeddings,
            "sample_ids": self.sample_ids,
        }
        if self.labels is not None:
            arrays["labels"] = self.labels
        if self.groups is not None:
            arrays["groups"] = self.groups
        np.savez_compressed(path, **arrays)
        with path.with_suffix(".metadata.csv").open("w", encoding="utf-8", newline="") as file:
            columns = ["sample_id", *self.metadata[0].keys()] if self.metadata else ["sample_id"]
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for sample_id, row in zip(self.sample_ids, self.metadata):
                writer.writerow({"sample_id": sample_id, **row})
        path.with_suffix(".config.json").write_text(
            json.dumps(self.info, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return path
