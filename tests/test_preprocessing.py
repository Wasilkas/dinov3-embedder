import numpy as np
import pytest
from PIL import Image

from dinov3_embedder import EmbedderConfig
from dinov3_embedder.preprocessing import prepare_image, read_image


def test_bbox_selects_object_and_letterbox_geometry():
    pixels = np.zeros((80, 160, 3), dtype=np.uint8)
    pixels[20:40, 40:120] = (240, 10, 20)
    image, metadata = prepare_image(pixels, (40, 20, 120, 40), EmbedderConfig(image_size=32))
    assert image.size == (32, 32)
    result = np.asarray(image)
    np.testing.assert_array_equal(result[12:20], np.broadcast_to([240, 10, 20], (8, 32, 3)))
    assert not result[:12].any() and not result[20:].any()
    assert metadata["padding_fraction"] == 0.75
    assert metadata["bbox_area_fraction"] == 0.125
    assert metadata["resize_scale_x"] == metadata["resize_scale_y"] == 0.4


def test_context_is_clipped_and_float_bbox_covers_pixels():
    image, metadata = prepare_image(
        Image.new("RGB", (100, 60)),
        (0.2, 2.3, 20.4, 22.1),
        EmbedderConfig(image_size=32, context_fraction=0.5),
    )
    assert image.size == (32, 32)
    assert metadata["crop_x1"] == metadata["crop_y1"] == 0
    assert metadata["crop_x2"] == 31 and metadata["crop_y2"] == 32


def test_stretch_and_none():
    pixels = np.zeros((16, 32), dtype=np.uint8)
    image, metadata = prepare_image(
        pixels, None, EmbedderConfig(image_size=32, resize_mode="stretch")
    )
    assert image.mode == "RGB" and metadata["padding_fraction"] == 0
    assert metadata["resize_scale_x"] == 1 and metadata["resize_scale_y"] == 2
    with pytest.raises(ValueError, match="exactly"):
        prepare_image(pixels, None, EmbedderConfig(image_size=32, resize_mode="none"))
    square = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    output, _ = prepare_image(square, None, EmbedderConfig(image_size=32, resize_mode="none"))
    np.testing.assert_array_equal(output, square)


@pytest.mark.parametrize(
    "bbox",
    [
        (10, 0, 5, 2),
        (-1, 0, 2, 2),
        (0, 0, 40, 20),
        (0, 0, 0, 1),
        (0, 0, float("nan"), 2),
        (1, 2, 3),
    ],
)
def test_bad_boxes(bbox):
    with pytest.raises(ValueError, match="bbox"):
        prepare_image(Image.new("RGB", (32, 32)), bbox, EmbedderConfig(image_size=32))


@pytest.mark.parametrize(
    "source",
    [
        np.ones((10, 10, 3)),
        np.zeros((3, 10, 10), dtype=np.uint8),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_ambiguous_or_empty_arrays(source):
    with pytest.raises(ValueError):
        read_image(source)


def test_rgb_grayscale_rgba_paths_and_no_input_mutation(tmp_path):
    for channels in (1, 3, 4):
        source = np.ones((8, 12, channels), dtype=np.uint8) * 30
        snapshot = source.copy()
        image = read_image(source)
        assert image.mode == "RGB" and image.size == (12, 8)
        np.testing.assert_array_equal(source, snapshot)
    file = tmp_path / "image.png"
    Image.new("L", (8, 12), 123).save(file)
    assert read_image(file).mode == "RGB"
