import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from dinov3_embedder import DINOv3Embedder, EmbedderConfig, ImageSample
from dinov3_embedder.cli import main


def test_real_dinov3_inference_order_batching_and_frozen_weights(embedder):
    rng = np.random.default_rng(2)
    images = [rng.integers(0, 256, (40, 64, 3), dtype=np.uint8) for _ in range(5)]
    snapshots = [image.copy() for image in images]
    batch = embedder.encode(iter(images))
    individually = np.concatenate([embedder.encode([image]) for image in images])
    assert batch.shape == (5, 32) and batch.dtype == np.float32
    np.testing.assert_allclose(batch, individually, atol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-6)
    for image, snapshot in zip(images, snapshots):
        np.testing.assert_array_equal(image, snapshot)
    assert not embedder.model.training
    assert not any(parameter.requires_grad for parameter in embedder.model.parameters())
    assert not any(parameter.grad is not None for parameter in embedder.model.parameters())


@pytest.mark.parametrize(
    "pooling,expected_dim", [("cls", 32), ("patch_mean", 32), ("cls_patch_mean", 64)]
)
def test_pooling_excludes_register_tokens(embedder, pooling, expected_dim):
    embedder.config = replace(embedder.config, pooling=pooling)
    tokens = torch.ones(1, 7, 32)
    tokens[:, 1:3] = 1000  # registers must never enter the patch average
    tokens[:, 3:] = 3
    output = embedder._pool(tokens)
    assert output.shape == (1, expected_dim)
    if pooling == "cls":
        assert torch.all(output == 1)
    elif pooling == "patch_mean":
        assert torch.all(output == 3)
    else:
        assert torch.all(output[:, :32] == 1) and torch.all(output[:, 32:] == 3)
    with pytest.raises(RuntimeError, match="layout"):
        embedder._pool(tokens[:, :-1])


def test_annotated_objects_save_and_quality_integration(tmp_path, embedder):
    samples = [
        ImageSample(
            f"point-{i}",
            Image.new("RGB", (80, 60), (100, 50 * (i % 2), 80)),
            label="scratch" if i % 2 else "crack",
            group=f"image-{i // 2}",
            bbox=(10, 20, 50, 40),
        )
        for i in range(8)
    ]
    result = embedder.embed_samples(iter(samples))
    path = result.save(tmp_path / "dino.npz")
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["sample_ids"], [s.sample_id for s in samples])
        np.testing.assert_array_equal(data["groups"], [s.group for s in samples])
        assert data["embeddings"].shape == (8, 32)
        assert data["labels"].dtype.kind == "U"
    assert (tmp_path / "dino.metadata.csv").read_text().startswith("sample_id,")
    info = json.loads((tmp_path / "dino.config.json").read_text())
    assert info["config"]["image_size"] == 32 and info["num_register_tokens"] == 2
    quality = pytest.importorskip("embeddings_quality")
    audit = quality.evaluate_embeddings(
        result.embeddings,
        result.labels,
        sample_ids=result.sample_ids,
        groups=result.groups,
        config=quality.AuditConfig(ks=(1,), cv_folds=2),
    )
    assert audit.summary["n_features"] == 32
    assert audit.summary["linear_probe"]["status"] == "ok"


def test_bbox_convenience_matches_pre_cropped_source(embedder):
    rng = np.random.default_rng(8)
    image = rng.integers(0, 256, (60, 80, 3), dtype=np.uint8)
    actual = embedder.encode_bboxes(image, [(0, 0, 40, 20), (20, 20, 80, 60)])
    expected = embedder.encode([image[0:20, 0:40], image[20:60, 20:80]])
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_empty_and_unlabeled_results(tmp_path, embedder):
    assert embedder.encode([]).shape == (0, 32)
    result = embedder.embed_samples([ImageSample("1", Image.new("L", (10, 10)))])
    assert result.labels is None and result.groups is None
    result.save(tmp_path / "unlabeled.npz")
    with np.load(tmp_path / "unlabeled.npz", allow_pickle=False) as data:
        assert "labels" not in data and "groups" not in data


def test_wrong_lengths_partial_labels_and_duplicates(embedder):
    image = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError, match="same length"):
        embedder.encode([image, image], bboxes=[None])
    with pytest.raises(ValueError, match="single image"):
        embedder.encode(image)
    with pytest.raises(ValueError, match="Duplicate"):
        embedder.embed_samples([ImageSample("x", image), ImageSample("x", image)])
    with pytest.raises(ValueError, match="present for every"):
        embedder.embed_samples([ImageSample("x", image, label="scratch"), ImageSample("y", image)])


def test_model_patch_size_validation(local_model):
    with pytest.raises(ValueError, match="divisible"):
        DINOv3Embedder(
            EmbedderConfig(model_id=str(local_model), image_size=33, local_files_only=True)
        )
    with pytest.raises(ValueError, match="CPU"):
        DINOv3Embedder(EmbedderConfig(model_id=str(local_model), device="cpu", dtype="float16"))


def test_cli_full_local_pipeline(tmp_path, local_model, capsys):
    Image.new("RGB", (64, 48), (100, 20, 10)).save(tmp_path / "metal.png")
    manifest = tmp_path / "objects.csv"
    manifest.write_text(
        "sample_id,image_path,label,x1,y1,x2,y2\n"
        "q1,metal.png,scratch,0,0,32,32\nq2,metal.png,crack,32,0,64,32\n",
    )
    output = tmp_path / "cli.npz"
    assert (
        main(
            [
                str(manifest),
                "--output",
                str(output),
                "--model",
                str(local_model),
                "--local-files-only",
                "--image-size",
                "32",
                "--pooling",
                "cls_patch_mean",
            ]
        )
        == 0
    )
    assert "2 embeddings of dimension 64" in capsys.readouterr().out
    with np.load(output, allow_pickle=False) as data:
        assert data["embeddings"].shape == (2, 64)
        assert data["groups"][0] == data["groups"][1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"image_size": 3.5},
        {"context_fraction": -0.1},
        {"context_fraction": float("nan")},
        {"pooling": "mean_all"},
        {"resize_mode": "crop"},
        {"dtype": "int8"},
        {"normalize": "yes"},
        {"pad_color": (0, 300, 0)},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        EmbedderConfig(**kwargs)
