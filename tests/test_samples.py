import json

import pytest

from dinov3_embedder import load_coco, load_csv

CSV_COLUMNS = ["image_path", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]
CSV_HEADER = ",".join(CSV_COLUMNS) + "\n"


def test_csv_required_columns_only(tmp_path):
    file = tmp_path / "objects.csv"
    file.write_text(
        CSV_HEADER + "image.png,scratch,0,2.5,10,20\nimage.png,crack,1,3,12,21\n",
        encoding="utf-8-sig",
    )
    samples = load_csv(file, image_root=tmp_path / "images")
    assert [s.sample_id for s in samples] == ["1", "2"]
    assert [s.label for s in samples] == ["scratch", "crack"]
    assert samples[0].bbox == (0, 2.5, 10, 20)
    assert samples[1].bbox == (1, 3, 12, 21)
    assert samples[0].image == tmp_path / "images" / "image.png"
    assert samples[0].group == samples[1].group == str(samples[0].image)


def test_csv_preserves_ids_order_and_groups(tmp_path):
    file = tmp_path / "objects.csv"
    file.write_text(
        "sample_id,group," + CSV_HEADER + "q-5,coil-3,image.png,scratch,0,2,10,20\n"
        "q-2,,image.png,crack,1,3,12,21\n",
        encoding="utf-8",
    )
    samples = load_csv(file)
    assert [s.sample_id for s in samples] == ["q-5", "q-2"]
    assert samples[0].bbox == (0, 2, 10, 20)
    assert samples[1].bbox == (1, 3, 12, 21)
    assert [s.label for s in samples] == ["scratch", "crack"]
    assert samples[0].group == "coil-3"
    assert samples[1].group == str(tmp_path / "image.png")


@pytest.mark.parametrize(
    "text,match",
    [
        ("sample_id,image_path,label,x1,y1,x2,y2\na,x,scratch,0,0,1,1\n", "instance_label"),
        (CSV_HEADER + "x,scratch,0,0,,1\n", "row 2: empty bbox_x_br"),
        (CSV_HEADER + "x,scratch,,,,\n", "empty bbox_x_tl"),
        (CSV_HEADER + "x,scratch,0,0,1\n", "empty bbox_y_br"),
        (CSV_HEADER + "x,scratch,0,0,invalid,1\n", "row 2: bbox coordinates"),
        ("sample_id," + CSV_HEADER + "a,x,scratch,0,0,1,1\na,y,crack,0,0,1,1\n", "unique"),
        ("sample_id," + CSV_HEADER + ",x,scratch,0,0,1,1\n", "empty sample_id"),
        (CSV_HEADER, "no objects"),
    ],
)
def test_bad_csv(tmp_path, text, match):
    path = tmp_path / "objects.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=match):
        load_csv(path)


@pytest.mark.parametrize("column", CSV_COLUMNS)
def test_csv_rejects_missing_required_column(tmp_path, column):
    path = tmp_path / "objects.csv"
    path.write_text(",".join(key for key in CSV_COLUMNS if key != column) + "\n")
    with pytest.raises(ValueError, match=column):
        load_csv(path)


@pytest.mark.parametrize("column", CSV_COLUMNS)
@pytest.mark.parametrize("empty", ["", "   "])
def test_csv_rejects_empty_required_value(tmp_path, column, empty):
    values = ["image.png", "scratch", "0", "0", "10", "20"]
    values[CSV_COLUMNS.index(column)] = empty
    path = tmp_path / "objects.csv"
    path.write_text(CSV_HEADER + ",".join(values) + "\n")
    with pytest.raises(ValueError, match=f"row 2: empty {column}"):
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
