"""Single object samples, CSV xyxy and COCO xywh annotation readers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._types import BBox, ImageInput, ScalarLabel


@dataclass(frozen=True)
class ImageSample:
    sample_id: str | int
    image: ImageInput
    label: ScalarLabel | None = None
    group: ScalarLabel | None = None
    bbox: BBox | None = None


def _root(manifest: str | Path, image_root: str | Path | None) -> Path:
    return Path(image_root) if image_root is not None else Path(manifest).resolve().parent


def load_csv(path: str | Path, *, image_root: str | Path | None = None) -> list[ImageSample]:
    """Read object rows; relative image paths resolve against image_root or CSV parent.

    Required: sample_id, image_path. Optional: label, group,
    x1/y1/x2/y2 (all four, in absolute pixels). Default group is image path.
    """
    root = _root(path, image_root)
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        if not {"sample_id", "image_path"} <= columns:
            raise ValueError("CSV needs sample_id and image_path columns.")
        bbox_columns = {"x1", "y1", "x2", "y2"}
        if columns & bbox_columns and not bbox_columns <= columns:
            raise ValueError("CSV needs all four bbox columns: x1,y1,x2,y2.")
        samples = []
        for row_number, row in enumerate(reader, 2):
            if not row["sample_id"] or not row["image_path"]:
                raise ValueError(f"CSV row {row_number}: empty sample_id/image_path.")
            image = (root / row["image_path"]).resolve()
            bbox = None
            if bbox_columns <= columns:
                coordinates = [row[key] for key in ("x1", "y1", "x2", "y2")]
                if any(coordinates):
                    if not all(coordinates):
                        raise ValueError(f"CSV row {row_number}: partial bbox.")
                    x1, y1, x2, y2 = (float(v) for v in coordinates)
                    bbox = (x1, y1, x2, y2)
            samples.append(
                ImageSample(
                    sample_id=row["sample_id"],
                    image=image,
                    label=row.get("label") or None,
                    group=row.get("group") or str(image),
                    bbox=bbox,
                )
            )
    _validate_samples(samples)
    return samples


def load_coco(path: str | Path, *, image_root: str | Path | None = None) -> list[ImageSample]:
    """Read COCO detection annotations; every bbox becomes one sample.

    COCO [x,y,width,height] is converted to pixel xyxy. annotation.id is the
    sample ID, image.id the group, category.name the label. iscrowd entries
    are included. Annotation order is preserved.
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("COCO document must be a JSON object.")
    root = _root(path, image_root)
    images = _index(document.get("images", []), "images")
    categories = _index(document.get("categories", []), "categories")
    annotations = document.get("annotations", [])
    if not isinstance(annotations, list):
        raise ValueError("COCO annotations must be a list.")
    samples = []
    for annotation in annotations:
        try:
            image = images[annotation["image_id"]]
            category = categories[annotation["category_id"]]
            x, y, w, h = (float(v) for v in annotation["bbox"])
            samples.append(
                ImageSample(
                    sample_id=str(annotation["id"]),
                    image=(root / image["file_name"]).resolve(),
                    label=str(category["name"]),
                    group=str(image["id"]),
                    bbox=(x, y, x + w, y + h),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid COCO annotation, image reference, category, or bbox."
            ) from exc
    _validate_samples(samples)
    return samples


def _index(rows: list[dict[str, Any]], name: str) -> dict[Any, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"COCO {name} must be a list.")
    try:
        indexed = {row["id"]: row for row in rows}
    except (KeyError, TypeError) as exc:
        raise ValueError(f"COCO {name} entries need IDs.") from exc
    if len(indexed) != len(rows):
        raise ValueError(f"COCO {name} IDs must be unique.")
    return indexed


def _validate_samples(samples: list[ImageSample]) -> None:
    if not samples:
        raise ValueError("Manifest contains no objects.")
    ids = [sample.sample_id for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("sample_id must be unique per object.")
