import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.retrieval_eval import evaluate as eval_retrieval, load_test_set
from eval.confidence_flag_eval import evaluate as eval_confidence
from eval.segmentation_eval import iou


def test_retrieval_test_set_loads():
    rows = load_test_set()
    assert len(rows) == 90
    assert set(rows[0].keys()) == {"food_id", "correct_name", "query_label", "difficulty"}


def test_retrieval_eval_produces_sane_accuracy():
    results = eval_retrieval()
    assert 0.0 <= results["top_k_accuracy"][1] <= 1.0
    assert results["top_k_accuracy"][3] >= results["top_k_accuracy"][1]  # top-3 can't be worse than top-1


def test_confidence_flag_eval_produces_sane_metrics():
    r = eval_confidence(threshold=70.0)
    assert r["n"] == 90
    assert r["tp"] + r["fp"] + r["tn"] + r["fn"] == 90
    assert 0.0 <= r["precision"] <= 1.0
    assert 0.0 <= r["recall"] <= 1.0


def test_confidence_flag_higher_threshold_flags_more():
    low = eval_confidence(threshold=50.0)
    high = eval_confidence(threshold=90.0)
    flagged_low = low["tp"] + low["fp"]
    flagged_high = high["tp"] + high["fp"]
    assert flagged_high >= flagged_low  # a looser (higher) threshold should never flag fewer items


def test_iou_identical_masks_is_one():
    import numpy as np
    m = np.zeros((5, 5), dtype=bool)
    m[1:4, 1:4] = True
    assert iou(m, m) == 1.0


def test_iou_disjoint_masks_is_zero():
    import numpy as np
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[7:10, 7:10] = True
    assert iou(a, b) == 0.0


def test_iou_partial_overlap_matches_known_value():
    import numpy as np
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True  # 16 px
    b = np.zeros((10, 10), dtype=bool)
    b[3:7, 3:7] = True  # 16 px, intersection 9px, union 23px
    assert abs(iou(a, b) - 9 / 23) < 1e-9
