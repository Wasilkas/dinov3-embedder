import json

import pytest

from dinov3_embedder import load_coco, load_csv


def test_csv_preserves_ids_order_and_groups(tmp_path):
    file = tmp_path / "objects.csv"
    file.write_text(
        "sample_id,image_path,label,group,x1,y1,x2,y2\n"
        "q-5,image.png,scratch,coil-3,0,2,10,20\n"
        "q-2,image.png,crack,,,,,\n",
        encoding="utf-8",
    )
    samples = load_csv(file)
    assert [s.sample_id for s in samples] == ["q-5", "q-2"]
    assert samples[0].bbox == (0, 2, 10, 20)
    assert samples[1].bbox is None
    assert samples[0].group == "coil-3"
    assert samples[1].group == str(tmp_path / "image.png")


@pytest.mark.parametrize(
    "text,match",
    [
        ("id,image_path\na,x\n", "sample_id"),
        ("sample_id,image_path,x1\na,x,0\n", "all four"),
        ("sample_id,image_path,x1,y1,x2,y2\na,x,0,0,,1\n", "partial"),
        ("sample_id,image_path\na,x\na,y\n", "unique"),
        ("sample_id,image_path\n,x\n", "empty"),
        ("sample_id,image_path\n", "no objects"),
    ],
)
def test_bad_csv(tmp_path, text, match):
    path = tmp_path / "objects.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_csv(path)


def test_coco_xywh_conversion_and_image_groups(tmp_path):
    document = {
        "images": [{"id": 11, "file_name": "metal.png"}],
        "categories": [{"id": 4, "name": "scratch"}],
        "annotations": [
            {"id": 101, "image_id": 11, "category_id": 4, "bbox": [10, 20, 30, 5]},
            {"id": 102, "image_id": 11, "category_id": 4, "bbox": [0, 0, 2, 3]},
        ],
    }
    file = tmp_path / "coco.json"
    file.write_text(json.dumps(document))
    samples = load_coco(file, image_root=tmp_path / "images")
    assert [s.sample_id for s in samples] == ["101", "102"]
    assert samples[0].bbox == (10, 20, 40, 25)
    assert samples[0].group == samples[1].group == "11"
    assert samples[0].image == tmp_path / "images" / "metal.png"
    document["annotations"][0]["category_id"] = 100
    file.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Invalid COCO"):
        load_coco(file)
