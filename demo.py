"""
Runnable end-to-end demo using mock depth/segmentation/VLM (no model-hub
network access or GPU required). Real usage: swap the `.mock()` /
`classify_mock()` calls for the real `.load()` / `classify_segment()` paths
once you're running somewhere with model-hub access and an ANTHROPIC_API_KEY.

Run: python demo.py
"""
from pipeline.depth_estimation import DepthEstimator
from pipeline.segmentation import Segmenter
from pipeline.vlm_classifier import classify_mock
from pipeline.nutrition_db import NutritionDB
from pipeline.storage import Storage, UserTargets
from pipeline.pipeline import MealPipeline
from pipeline.volume_estimator import CameraIntrinsics


def main():
    user_id = "pranab"
    db = NutritionDB()
    storage = Storage("demo_nutrivision.db")
    storage.upsert_user(
        UserTargets(user_id, kcal_target=2400, protein_target_g=150, fat_target_g=80, carbs_target_g=280)
    )
    mp = MealPipeline(db, storage)

    # ---- "captured photo" stage (mocked: no real depth/seg/VLM weights here) ----
    depth = DepthEstimator.mock(height=256, width=256)
    segments = Segmenter.mock_segments(height=256, width=256, n=3)
    labels = ["cooked white rice", "grilled chicken breast", "mixed vegetable curry"]
    classifications = [classify_mock(l) for l in labels]
    intrinsics = CameraIntrinsics.approximate(256, 256)

    items = mp.process_meal(depth, segments, classifications, intrinsics)

    print("Detected items for this meal:\n")
    for it in items:
        flag = "  [NEEDS USER CONFIRMATION]" if it.needs_confirmation else ""
        print(f"  - {it.label}: {it.grams:.0f}g -> {it.macros.kcal:.0f} kcal, "
              f"{it.macros.protein_g:.1f}g protein{flag}")

    meal_id = mp.log_meal(user_id=user_id, items=items)
    print(f"\nLogged as meal #{meal_id}.")

    from datetime import date
    summary = storage.get_daily_summary(user_id, date.today().isoformat())
    print("\nDaily progress:")
    print(f"  {summary['kcal']:.0f} / {summary['kcal_target']:.0f} kcal "
          f"({summary['kcal_remaining']:.0f} remaining)")
    print(f"  {summary['protein_g']:.0f} / {summary['protein_target_g']:.0f} g protein "
          f"({summary['protein_remaining_g']:.0f} remaining)")

    print("\n--- Note ---")
    print("Volumes/masses above come from a SYNTHETIC mock depth map and mock")
    print("segmentation masks (no real image was processed) -- this demo proves")
    print("the pipeline wiring, retrieval, and storage are correct end-to-end.")
    print("Plug in real depth_estimation.DepthEstimator.load() + segmentation.Segmenter.load()")
    print("+ vlm_classifier.classify_segment() where you have model-hub/API access")
    print("to get real numbers from a real photo.")


if __name__ == "__main__":
    main()
