"""
Benchmark 4: Ingredient-label -> nutrition-DB retrieval accuracy.

This is the one benchmark in this eval/ folder that needs no external
dataset or images -- it tests NutritionDB.match() (rapidfuzz retrieval)
against eval/data/retrieval_test_set.csv, a hand-labeled set of 90
(free-text label, correct food_id) pairs modeled on realistic VLM output
style (paraphrases, verbose descriptions, regional names, typos/uncertainty
hedges). Numbers from this script are real -- there is no mocking here.

Usage: python -m eval.retrieval_eval
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.nutrition_db import NutritionDB

TEST_SET_PATH = Path(__file__).resolve().parent / "data" / "retrieval_test_set.csv"


def load_test_set(path: Path = TEST_SET_PATH) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluate(top_k_report: tuple[int, ...] = (1, 3)) -> dict:
    db = NutritionDB()
    rows = load_test_set()

    hits = {k: 0 for k in top_k_report}
    by_difficulty = defaultdict(lambda: {k: [0, 0] for k in top_k_report})  # [hits, total]
    misses = []

    max_k = max(top_k_report)
    for row in rows:
        query = row["query_label"]
        correct_id = row["food_id"]
        difficulty = row["difficulty"]

        candidates = db.match(query, top_k=max_k, score_cutoff=0.0)  # no cutoff -- we're measuring ranking quality itself
        candidate_ids = [c[0].food_id for c in candidates]

        for k in top_k_report:
            is_hit = correct_id in candidate_ids[:k]
            by_difficulty[difficulty][k][1] += 1
            if is_hit:
                hits[k] += 1
                by_difficulty[difficulty][k][0] += 1

        if correct_id not in candidate_ids[:1]:
            top1_name = candidates[0][0].name if candidates else "(no match)"
            misses.append((query, row["correct_name"], top1_name))

    n = len(rows)
    results = {
        "n": n,
        "top_k_accuracy": {k: hits[k] / n for k in top_k_report},
        "by_difficulty": {
            diff: {k: (v[0] / v[1] if v[1] else None) for k, v in kmap.items()}
            for diff, kmap in by_difficulty.items()
        },
        "misses_top1": misses,
    }
    return results


def main():
    results = evaluate()
    print(f"Retrieval accuracy over {results['n']} labeled (label -> food_id) pairs\n")
    for k, acc in results["top_k_accuracy"].items():
        print(f"  top-{k} accuracy: {acc*100:.1f}%")

    print("\nBy difficulty bucket:")
    for diff in ("easy", "medium", "hard"):
        if diff in results["by_difficulty"]:
            row = results["by_difficulty"][diff]
            parts = ", ".join(f"top-{k}: {v*100:.1f}%" for k, v in row.items() if v is not None)
            print(f"  {diff:8s} {parts}")

    if results["misses_top1"]:
        print(f"\nTop-1 misses ({len(results['misses_top1'])}):")
        for query, correct, got in results["misses_top1"]:
            print(f"  '{query}' -> expected '{correct}', got '{got}'")


if __name__ == "__main__":
    main()
