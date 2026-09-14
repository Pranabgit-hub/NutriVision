"""
Benchmark 5: does the low-confidence flag (`needs_confirmation` in
pipeline.py, driven by NutritionDB match score < threshold) actually predict
when the retrieval got the wrong answer?

This reuses the SAME 90 labeled (query, correct_food_id) pairs as
retrieval_eval.py, and the SAME NutritionDB.best_match() call the real
pipeline uses -- there's no synthetic/injected label noise here. Ground
truth "wrong" = the retrieved top-1 food_id != the labeled correct food_id.
Predicted "flagged" = match score < MealPipeline's confirmation_threshold.

This isolates one specific design decision (is 70.0 a good threshold?) with
a real, reproducible number, separate from the harder-to-source end-to-end
photo accuracy (see nutrition5k_eval.py) and segmentation quality
(segmentation_eval.py), neither of which have real data available yet.

Usage: python -m eval.confidence_flag_eval [--threshold 70]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.nutrition_db import NutritionDB
from pipeline.pipeline import MealPipeline
from eval.retrieval_eval import load_test_set


def evaluate(threshold: float = 70.0) -> dict:
    db = NutritionDB()
    rows = load_test_set()

    tp = fp = tn = fn = 0
    examples = {"fp": [], "fn": []}  # the two error types worth inspecting

    for row in rows:
        query = row["query_label"]
        correct_id = row["food_id"]

        match = db.best_match(query)
        if match is None:
            top1_id, score = None, 0.0
        else:
            top1_id, score = match[0].food_id, match[1]

        actually_wrong = top1_id != correct_id
        flagged = score < threshold

        if flagged and actually_wrong:
            tp += 1
        elif flagged and not actually_wrong:
            fp += 1
            examples["fp"].append((query, row["correct_name"], score))
        elif not flagged and actually_wrong:
            fn += 1
            examples["fn"].append((query, row["correct_name"], score))
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else float("nan")

    return {
        "threshold": threshold,
        "n": len(rows),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "examples": examples,
    }


def sweep_thresholds(thresholds=(50, 60, 65, 70, 75, 80, 85, 90)) -> list[dict]:
    return [evaluate(t) for t in thresholds]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=70.0)
    parser.add_argument("--sweep", action="store_true", help="Report precision/recall across a range of thresholds")
    args = parser.parse_args()

    if args.sweep:
        print(f"{'threshold':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'flagged':>8}")
        for r in sweep_thresholds():
            flagged = r["tp"] + r["fp"]
            print(f"{r['threshold']:>10.0f} {r['precision']*100:>9.1f}% {r['recall']*100:>9.1f}% {r['f1']*100:>9.1f}% {flagged:>8d}")
        return

    r = evaluate(args.threshold)
    print(f"Confidence-flag benchmark (threshold={r['threshold']}, n={r['n']})\n")
    print(f"  precision: {r['precision']*100:.1f}%  (of flagged items, % that were actually wrong)")
    print(f"  recall:    {r['recall']*100:.1f}%  (of actually-wrong items, % that got flagged)")
    print(f"  f1:        {r['f1']*100:.1f}%")
    print(f"  confusion: TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")

    if r["examples"]["fn"]:
        print(f"\nFalse negatives -- wrong AND not flagged (the dangerous case, silently wrong):")
        for q, correct, score in r["examples"]["fn"]:
            print(f"  '{q}' (expected '{correct}') scored {score:.1f}, above threshold")


if __name__ == "__main__":
    main()
