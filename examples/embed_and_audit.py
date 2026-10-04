"""Source images + CSV/COCO objects -> DINOv3 vectors -> original-space audit."""

import argparse
from pathlib import Path

from embeddings_quality import AuditConfig, evaluate_embeddings

from dinov3_embedder import DINOv3Embedder, EmbedderConfig, load_coco, load_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment"))
    parser.add_argument("--model", default=EmbedderConfig.model_id)
    parser.add_argument("--pooling", choices=["cls", "patch_mean", "cls_patch_mean"], default="cls")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    loader = load_coco if args.manifest.suffix.lower() == ".json" else load_csv
    samples = loader(args.manifest, image_root=args.image_root)
    if any(sample.label is None for sample in samples):
        parser.error("Quality auditing requires a label for every object.")
    embedder = DINOv3Embedder(
        EmbedderConfig(
            model_id=args.model,
            pooling=args.pooling,
            batch_size=args.batch_size,
            image_size=args.image_size,
            local_files_only=args.local_files_only,
        )
    )
    result = embedder.embed_samples(samples)
    result.save(args.output / "dinov3.npz")
    metadata = {
        key: [row[key] for row in result.metadata]
        for key in ("bbox_area_fraction", "aspect_ratio", "padding_fraction")
    }
    report = evaluate_embeddings(
        result.embeddings,
        result.labels,
        sample_ids=result.sample_ids,
        groups=result.groups,
        metadata=metadata,
        config=AuditConfig(random_state=42),
    )
    report.save(args.output / "quality", plots=True)
    print(f"Saved features and quality report to {args.output}")


if __name__ == "__main__":
    main()
