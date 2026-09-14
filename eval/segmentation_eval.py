"""
Benchmark 3: segmentation quality via mean IoU against human-annotated
ground-truth masks.

The IoU computation itself (`iou`, `evaluate`) is real and tested below on
synthetic masks purely to prove the math is correct -- that is NOT a
segmentation quality benchmark, since synthetic masks aren't a real model's
predictions. To get a real number you need:

  1. ~50 real meal photos run through the REAL segmenter (Segmenter.load(),
     not .mock_segments() -- needs model-hub network access this sandbox
     lacks).
  2. Human-annotated ground-truth masks for the same photos (e.g. via CVAT,
     Labelbox, or even manual polygon annotation in an image editor,
     exported as binary PNG masks).
  3. Point `evaluate()` at matched (predicted_mask, ground_truth_mask) pairs.

Usage once you have real masks:
    python -m eval.segmentation_eval --pred-dir <preds/> --gt-dir <gt/>
Masks are matched by filename stem.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def load_mask_png(path: Path) -> np.ndarray:
    from PIL import Image

    arr = np.array(Image.open(path).convert("L"))
    return arr > 127  # treat >50% grey as foreground


def evaluate(pred_dir: Path, gt_dir: Path) -> dict:
    pred_files = {p.stem: p for p in pred_dir.glob("*.png")}
    gt_files = {p.stem: p for p in gt_dir.glob("*.png")}
    common = sorted(set(pred_files) & set(gt_files))

    missing_pred = sorted(set(gt_files) - set(pred_files))
    missing_gt = sorted(set(pred_files) - set(gt_files))
    if missing_pred:
        print(f"Warning: {len(missing_pred)} ground-truth masks have no matching prediction, skipped.")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} predictions have no matching ground truth, skipped.")

    scores = []
    per_image = {}
    for stem in common:
        pred_mask = load_mask_png(pred_files[stem])
        gt_mask = load_mask_png(gt_files[stem])
        score = iou(pred_mask, gt_mask)
        scores.append(score)
        per_image[stem] = score

    return {
        "n": len(scores),
        "mean_iou": sum(scores) / len(scores) if scores else float("nan"),
        "min_iou": min(scores) if scores else None,
        "max_iou": max(scores) if scores else None,
        "per_image": per_image,
    }


def _smoke_test():
    """Proves the IoU math is correct -- NOT a segmentation benchmark result."""
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True  # 4x4 = 16 px
    b = np.zeros((10, 10), dtype=bool)
    b[3:7, 3:7] = True  # 4x4 = 16 px, offset by 1 in each dim -> 9px intersection, 23px union
    score = iou(a, b)
    expected = 9 / 23
    assert abs(score - expected) < 1e-9, f"IoU math check failed: got {score}, expected {expected}"
    print(f"Smoke test passed: IoU({4}x4 box, 1px-offset 4x4 box) = {score:.4f} (expected {expected:.4f})")
    print("This confirms the IoU computation is correct -- it is NOT a segmentation quality result.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path)
    parser.add_argument("--gt-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true", help="Verify the IoU math only, no real data needed")
    args = parser.parse_args()

    if args.smoke_test or not (args.pred_dir and args.gt_dir):
        _smoke_test()
        if not (args.pred_dir and args.gt_dir):
            print("\nNo --pred-dir/--gt-dir given, so no benchmark was run. See module docstring.")
        return

    results = evaluate(args.pred_dir, args.gt_dir)
    print(f"Segmentation IoU over {results['n']} matched image pairs")
    print(f"  mean IoU: {results['mean_iou']:.3f}")
    if results["n"]:
        print(f"  min/max:  {results['min_iou']:.3f} / {results['max_iou']:.3f}")


if __name__ == "__main__":
    main()
