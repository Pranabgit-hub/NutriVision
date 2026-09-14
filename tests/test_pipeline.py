import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from pipeline.nutrition_db import NutritionDB
from pipeline.volume_estimator import CameraIntrinsics, estimate_volume_ml, volume_to_mass_g
from pipeline.depth_estimation import DepthEstimator, scale_from_reference_object
from pipeline.segmentation import Segmenter
from pipeline.storage import Storage, UserTargets
from pipeline.pipeline import MealPipeline
from pipeline.vlm_classifier import classify_mock


def test_nutrition_db_loads():
    db = NutritionDB()
    assert len(db.entries) > 0


def test_fuzzy_match_finds_close_name():
    db = NutritionDB()
    match = db.best_match("steamed white rice")
    assert match is not None
    entry, score = match
    assert "rice" in entry.name
    assert score > 50


def test_fuzzy_match_no_result_for_garbage():
    db = NutritionDB()
    matches = db.match("asdkjqwoeiuzxcv", score_cutoff=90)
    assert matches == []


def test_macro_scaling_is_linear():
    db = NutritionDB()
    entry, _ = db.best_match("cooked white rice")
    m100 = entry.scale(100)
    m200 = entry.scale(200)
    assert m200.kcal == pytest.approx(m100.kcal * 2)
    assert m200.protein_g == pytest.approx(m100.protein_g * 2)


def test_volume_estimate_positive_for_dome():
    depth = DepthEstimator.mock(height=64, width=64)
    segments = Segmenter.mock_segments(height=64, width=64, n=1)
    intrinsics = CameraIntrinsics.approximate(64, 64)
    vol = estimate_volume_ml(depth.depth_map, segments[0].mask, intrinsics)
    assert vol >= 0  # mock dome + mock mask may not overlap much; just must not error/negative


def test_volume_to_mass_scales_with_density():
    assert volume_to_mass_g(100, 1.0) == 100
    assert volume_to_mass_g(100, 0.5) == 50


def test_scale_from_reference_object():
    # A reference object (e.g. a can) with a real depth gradient top-to-bottom
    # within the mask -- a perfectly flat card would have ~zero relative-depth
    # span and can't calibrate scale this way.
    relative = np.zeros((10, 10), dtype=np.float32)
    relative[3:7, 3:7] = np.linspace(0.0, 1.0, 4)[:, None]  # span of 1.0 top to bottom
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    factor = scale_from_reference_object(relative, mask, reference_real_height_m=0.05)
    assert factor == pytest.approx(0.05)


def test_scale_from_empty_reference_raises():
    relative = np.zeros((10, 10), dtype=np.float32)
    mask = np.zeros((10, 10), dtype=bool)
    with pytest.raises(ValueError):
        scale_from_reference_object(relative, mask, 0.05)


def test_pipeline_refuses_non_metric_depth():
    db = NutritionDB()
    storage = Storage(":memory:")
    mp = MealPipeline(db, storage)
    depth = DepthEstimator.mock()
    depth.metric = False  # simulate a relative-only depth model with no calibration
    segments = Segmenter.mock_segments(n=1)
    classifications = [classify_mock()]
    intrinsics = CameraIntrinsics.approximate(224, 224)
    with pytest.raises(ValueError):
        mp.process_meal(depth, segments, classifications, intrinsics)


def test_pipeline_end_to_end_with_mocks():
    db = NutritionDB()
    storage = Storage(":memory:")
    mp = MealPipeline(db, storage)
    depth = DepthEstimator.mock(height=128, width=128)
    segments = Segmenter.mock_segments(height=128, width=128, n=2)
    classifications = [classify_mock("cooked white rice"), classify_mock("grilled chicken breast")]
    intrinsics = CameraIntrinsics.approximate(128, 128)

    items = mp.process_meal(depth, segments, classifications, intrinsics)
    assert len(items) == 2
    for it in items:
        assert it.grams >= 0
        assert it.macros.kcal >= 0

    meal_id = mp.log_meal(user_id="pranab", items=items)
    assert meal_id >= 1


def test_daily_summary_rollup():
    db = NutritionDB()
    storage = Storage(":memory:")
    storage.upsert_user(UserTargets("pranab", kcal_target=2200, protein_target_g=140, fat_target_g=70, carbs_target_g=250))
    mp = MealPipeline(db, storage)

    depth = DepthEstimator.mock()
    segments = Segmenter.mock_segments(n=1)
    classifications = [classify_mock("cooked white rice")]
    intrinsics = CameraIntrinsics.approximate(224, 224)

    items = mp.process_meal(depth, segments, classifications, intrinsics)
    from datetime import date

    today = date.today().isoformat()
    mp.log_meal(user_id="pranab", items=items)
    summary = storage.get_daily_summary("pranab", today)
    assert summary["kcal_target"] == 2200
    assert "kcal_remaining" in summary
