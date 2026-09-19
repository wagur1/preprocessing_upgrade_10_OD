"""Unit tests for the Open Images V6 -> COCO probe converter."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("probe_openimages", ROOT / "ops" / "probe_openimages.py")
oid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oid)


def _write_csv(path, rows, header):
    lines = [",".join(header)] + [",".join(str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_build_mid_to_coco_matches_aliases_and_case(tmp_path):
    csv = _write_csv(tmp_path / "desc.csv", [
        ("/m/01g317", "Person"),
        ("/m/0k4j", "Car"),
        ("/m/02tw4j", "Sofa"),
        ("/m/0h8mzrc", "mobile phone"),
        ("/m/01c648", "Table"),
        ("/m/0174n1x", "Goose"),          # no COCO match
    ], ["LabelName", "DisplayName"])
    m = oid.build_mid_to_coco(csv)
    assert m["/m/01g317"] == 1      # person
    assert m["/m/0k4j"] == 3        # car
    assert m["/m/02tw4j"] == 58     # sofa -> couch
    assert m["/m/0h8mzrc"] == 68    # mobile phone -> cell phone
    assert m["/m/01c648"] == 61     # table -> dining table
    assert "/m/0174n1x" not in m


def test_sample_ids_filters_and_is_deterministic(tmp_path):
    mid2coco = {"/m/01g317": 1, "/m/0k4j": 3}
    rows = []
    # img A: 3 mapped boxes + 1 IsGroupOf row; img B: 2 mapped; img C: 1 mapped (too few)
    rows += [("aaaaaaaaaaaaaaaa", "/m/01g317", 0), ("aaaaaaaaaaaaaaaa", "/m/0k4j", 0),
             ("aaaaaaaaaaaaaaaa", "/m/01g317", 0), ("aaaaaaaaaaaaaaaa", "/m/0k4j", 1),
             ("bbbbbbbbbbbbbbbb", "/m/01g317", 0), ("bbbbbbbbbbbbbbbb", "/m/0k4j", 0),
             ("cccccccccccccccc", "/m/01g317", 0)]
    header = ["ImageID", "LabelName", "IsGroupOf"]
    csv = _write_csv(tmp_path / "bbox.csv", rows, header)
    ids, _ = oid.sample_ids(csv, mid2coco, 2, min_boxes=2, seed=0)
    assert set(ids) == {"aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"}
    again, _ = oid.sample_ids(csv, mid2coco, 2, min_boxes=2, seed=0)
    assert ids == again
    with pytest.raises(ValueError):
        oid.sample_ids(csv, mid2coco, 3, min_boxes=2, seed=0)


def test_to_coco_json_normalizes_boxes_and_restricts_categories(tmp_path):
    import zlib
    mid2coco = {"/m/01g317": 1, "/m/0k4j": 3, "/m/0bt9lr": 41}  # person, car, dog... 41=bottle
    meta = {zlib.crc32(b"aaaaaaaaaaaaaaaa"): (1000, 500, "aaaaaaaaaaaaaaaa.jpg"),
            zlib.crc32(b"cccccccccccccccc"): (200, 200, "cccccccccccccccc.jpg")}
    rows = [
        # normalized box fully inside
        ("aaaaaaaaaaaaaaaa", "/m/01g317", 0.0, 0.0, 1.0, 1.0, 0),
        # box overflowing the right/bottom edge must clip
        ("aaaaaaaaaaaaaaaa", "/m/0k4j", 0.9, 0.9, 1.2, 1.4, 0),
        # IsGroupOf row is dropped
        ("aaaaaaaaaaaaaaaa", "/m/0bt9lr", 0.1, 0.1, 0.2, 0.2, 1),
        # unmapped label is dropped
        ("cccccccccccccccc", "/m/0174n1x", 0.1, 0.1, 0.2, 0.2, 0),
    ]
    header = ["ImageID", "LabelName", "XMin", "YMin", "XMax", "YMax", "IsGroupOf"]
    csv = _write_csv(tmp_path / "bbox.csv", rows, header)
    coco = oid.to_coco_json(csv, mid2coco, meta)
    assert {im["id"] for im in coco["images"]} == set(meta)
    assert [im["file_name"] for im in coco["images"]] == \
        ["aaaaaaaaaaaaaaaa.jpg", "cccccccccccccccc.jpg"]
    person = next(a for a in coco["annotations"] if a["category_id"] == 1)
    assert person["bbox"] == [0.0, 0.0, 1000.0, 500.0]
    car = next(a for a in coco["annotations"] if a["category_id"] == 3)
    assert car["bbox"][0] == 900.0 and car["bbox"][2] == 100.0   # clipped at the edge
    assert car["bbox"][1] == 450.0 and car["bbox"][3] == 50.0
    assert all(a["category_id"] in {1, 3} for a in coco["annotations"])
    assert {c["name"] for c in coco["categories"]} == {"person", "car"}
