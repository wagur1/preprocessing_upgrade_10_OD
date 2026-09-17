"""Regression tests using cached COCO xywh predictions, without inference."""
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import detection_bootstrap as db

ANN = {"categories": [{"id": 1, "name": "thing"}]}
BOX = [10.0, 20.0, 100.0, 50.0]
GT = {i: [{"id": i, "image_id": i, "category_id": 1, "iscrowd": 0,
           "area": 5000.0, "bbox": BOX}] for i in (7, 8)}
PREDS = [{"image_id": i, "category_id": 1, "bbox": BOX, "score": 0.9}
         for i in (7, 8)]


def records():
    # Actual AP varies from 1 at QP30 to 0 at QP40, not merely confidence.
    return {arm: {qp: {i: (rate * factor, [PREDS[i - 7]] if qp == 30 else [])
                      for i in (7, 8)}
                  for qp, rate in ((30, 0.2), (40, 0.1))}
            for arm, factor in (("anchor", 1), ("prep", 0.5), ("sandwich", 1.5))}


def test_perfect_predictions_and_duplicates_preserve_xywh_and_inputs():
    before = deepcopy((PREDS, GT, ANN))
    for sample in ([7, 8], [7, 7, 8]):
        assert db.coco_map(PREDS, GT, sample, ANN) == pytest.approx((1, 1))
    assert (PREDS, GT, ANN) == before


def test_unique_image_and_annotation_ids_per_occurrence():
    dataset, detections = db._resampled_coco(PREDS, GT, [7, 7, 8], ANN)
    assert [im["id"] for im in dataset["images"]] == [1, 2, 3]
    for rows in (dataset["annotations"], detections):
        assert len({row["id"] for row in rows}) == len(rows) == 3
        assert [row["image_id"] for row in rows] == [1, 2, 3]
        assert all(row["bbox"] == BOX for row in rows)


def test_duplicate_samples_change_ap_and_rate_not_deduplicated():
    # Only image 7 is detected. Repeating it increases recall from 1/2 to 2/3.
    ordinary = db.coco_map(PREDS[:1], GT, [7, 8], ANN)[0]
    weighted = db.coco_map(PREDS[:1], GT, [7, 7, 8], ANN)[0]
    assert ordinary == pytest.approx(51 / 101)
    assert weighted == pytest.approx(67 / 101)
    points = {30: {7: (1.0, PREDS[:1]), 8: (4.0, [])}}
    curve = db.detection_curve(points, [30], GT, [7, 7, 8], ANN)
    assert curve["rate"] == [2.0]
    assert curve["mAP"] == pytest.approx([weighted])


def test_empty_predictions_and_undefined_ground_truth():
    assert db.coco_map([], GT, [7, 8], ANN) == (0, 0)
    assert np.isnan(db.coco_map([], {7: []}, [7], ANN)[0])
    crowd = deepcopy(GT)
    for annotations in crowd.values():
        annotations[0]["iscrowd"] = 1
    assert np.isnan(db.coco_map(PREDS, crowd, [7, 8], ANN)[0])


def test_separate_arm_draws_and_confidence_intervals():
    summary = db.paired_detection_bootstrap(records(), [30, 40], GT, [7, 8], ANN,
                                             n_draws=8, seed=17)
    assert set(summary) == {"prep", "sandwich"}
    for arm, expected in (("prep", -50), ("sandwich", 50)):
        ci = summary[arm]
        assert ci["draws"] == pytest.approx([expected] * 8)
        assert (ci["lo"], ci["hi"]) == pytest.approx((expected, expected))
        assert ci["n_draws"] == ci["n_requested"] == 8
        assert ci["n_invalid"] == 0
        assert ci["seed"] == 17


def test_exact_n_paired_samples_across_all_arms_qps_and_seed(monkeypatch):
    observed = []
    original = db.coco_map

    def capture(predictions, gt, ids, meta):
        observed.append(list(ids))
        return original(predictions, gt, ids, meta)

    monkeypatch.setattr(db, "coco_map", capture)
    cached = records()
    # Vary rate savings by image so seed changes genuinely change BD draws.
    for qp in (30, 40):
        rate, preds = cached["prep"][qp][8]
        cached["prep"][qp][8] = (rate * 1.5, preds)
    first = db.paired_detection_bootstrap(cached, [30, 40], GT, [7, 8], ANN, 8, 17)
    rng = random.Random(17)
    expected = [[rng.choice([7, 8]) for _ in range(2)] for _ in range(8)]
    assert observed == [sample for sample in expected for _ in range(6)]
    assert any(len(set(sample)) < 2 for sample in expected)
    assert first == db.paired_detection_bootstrap(cached, [30, 40], GT, [7, 8], ANN, 8, 17)
    other = db.paired_detection_bootstrap(cached, [30, 40], GT, [7, 8], ANN, 8, 18)
    assert first["prep"]["draws"] != other["prep"]["draws"]


def test_invalid_draws_counted_per_arm_without_losing_valid_arm():
    cached = records()
    # A flat sandwich AP curve is undefined for BD-rate, but prep stays valid.
    for i in (7, 8):
        rate, _ = cached["sandwich"][30][i]
        cached["sandwich"][30][i] = (rate, [])
    summary = db.paired_detection_bootstrap(cached, [30, 40], GT, [7, 8], ANN, 5, 0)
    assert summary["prep"]["n_draws"] == 5
    assert summary["prep"]["n_invalid"] == 0
    assert summary["sandwich"]["draws"] == [None] * 5
    assert summary["sandwich"]["n_invalid"] == 5
    assert summary["sandwich"]["n_draws"] == 0
    assert summary["sandwich"]["lo"] is summary["sandwich"]["hi"] is None


@pytest.mark.parametrize("ids", [[], [7, 7]])
def test_reject_invalid_source_population(ids):
    with pytest.raises(ValueError):
        db.paired_detection_bootstrap(records(), [30, 40], GT, ids, ANN, 1)


def test_reject_unpaired_records():
    cached = records()
    del cached["prep"][30][7]
    with pytest.raises(ValueError, match="every sampled image"):
        db.paired_detection_bootstrap(cached, [30, 40], GT, [7, 8], ANN, 1)
