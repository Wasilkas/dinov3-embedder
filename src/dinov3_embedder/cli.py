"""Extract image/bbox embeddings to an audit-compatible NPZ."""

import argparse
from pathlib import Path

from .embedder import DINOv3Embedder, EmbedderConfig
from .samples import load_coco, load_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract frozen DINOv3 ViT embeddings from CSV or COCO objects."
    )
    parser.add_argument("manifest", help="CSV with image paths/bbox or COCO JSON")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--image-root", help="Base directory for relative image paths")
    parser.add_argument("--format", choices=["csv", "coco"])
    parser.add_argument(
        "--model", default=EmbedderConfig.model_id, help="HF ID or local HF model directory"
    )
    parser.add_argument("--revision", help="HF revision/commit for reproducible extraction")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--resize-mode", choices=["letterbox", "stretch", "none"], default="letterbox"
    )
    parser.add_argument("--context-fraction", type=float, default=0)
    parser.add_argument("--pooling", choices=["cls", "patch_mean", "cls_patch_mean"], default="cls")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    try:
        if Path(args.output).suffix.lower() != ".npz":
            raise ValueError("--output must end with .npz.")
        config = EmbedderConfig(
            model_id=args.model,
            revision=args.revision,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            image_size=args.image_size,
            resize_mode=args.resize_mode,
            context_fraction=args.context_fraction,
            pooling=args.pooling,
            normalize=not args.no_normalize,
            local_files_only=args.local_files_only,
        )
        format_name = args.format or (
            "coco" if Path(args.manifest).suffix.lower() == ".json" else "csv"
        )
        samples = (load_coco if format_name == "coco" else load_csv)(
            args.manifest, image_root=args.image_root
        )
        embedder = DINOv3Embedder(config, cache_dir=args.cache_dir)
        result = embedder.embed_samples(samples)
        result.save(args.output)
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))
    print(
        f"Saved {len(result.embeddings)} embeddings of dimension {result.embeddings.shape[1]} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
