"""
Benchmark 2: end-to-end macro accuracy against Nutrition5k.

Nutrition5k (Google, 2021) is the standard public benchmark for this task:
real dishes with RGB(-D) images plus scale-measured ground-truth mass,
calories, and macros. This script computes MAPE (mean absolute percentage
error) between pipeline predictions and Nutrition5k ground truth.

IMPORTANT -- this cannot produce a real number in this environment. The
dataset (~1.5M images, hosted on Google Cloud Storage) is not reachable
from this sandbox's network allowlist, and running it for real also needs
the depth/segmentation/VLM stages to use REAL weights (see
depth_estimation.py / segmentation.py), not the mocks. Treat this file as
the harness to run once you have both of those -- not a source of resume
numbers today.

To get real numbers:
  1. Download Nutrition5k: https://github.com/google-research-datasets/Nutrition5k
     (metadata CSVs + `gsutil -m cp -r gs://nutrition5k_dataset .`)
  2. Point NUTRITION5K_METADATA_CSV / NUTRITION5K_IMAGE_DIR below at your copy.
  3. Wire depth_estimation.DepthEstimator.load() and segmentation.Segmenter.load()
     to real weights (needs model-hub network access this sandbox lacks).
  4. Run: python -m eval.nutrition5k_eval --metadata <path> --images <dir>

The `data/nutrition5k_sample_SYNTHETIC.csv` file alongside this script is
NOT real Nutrition5k data -- it's 4 fabricated rows with the right schema,
included only so you can smoke-test that the MAPE computation itself is
correct before pointing this at the real dataset. Do not report numbers
from it as a benchmark result.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SYNTHETIC_SAMPLE = Path(__file__).resolve().parent / "data" / "nutrition5k_sample_SYNTHETIC.csv"

# Nutrition5k's own metadata schema (subset of columns relevant here):
# dish_id, total_calories, total_mass, total_fat, total_carb, total_protein
METRICS = ["mass", "calories", "protein", "fat", "carb"]


def mape(predicted: list[float], actual: list[float]) -> float:
    errors = []
    for p, a in zip(predicted, actual):
        if a == 0:
            continue  # undefined; skip rather than divide by zero
        errors.append(abs(p - a) / a)
    return sum(errors) / len(errors) * 100 if errors else float("nan")


def run_pipeline_on_image(image_path: Path):
    """Placeholder for the real per-image pipeline call.

    A real implementation loads the image, runs Segmenter + DepthEstimator +
    vlm_classifier (all REAL, not mocked) via pipeline.MealPipeline, and
    returns predicted (mass_g, kcal, protein_g, fat_g, carb_g) totals summed
    across detected items. Left unimplemented here because it requires the
    real model weights this sandbox doesn't have access to -- wire it up
    once depth_estimation/segmentation have real .load() paths available.
    """
    raise NotImplementedError(
        "Wire this to the real (non-mock) pipeline once depth/segmentation/VLM "
        "weights are available -- see module docstring."
    )


def evaluate(metadata_csv: Path, image_dir: Path, allow_synthetic: bool = False) -> dict:
    if metadata_csv == SYNTHETIC_SAMPLE and not allow_synthetic:
        raise SystemExit(
            "Refusing to run against the synthetic sample without --allow-synthetic. "
            "Numbers from synthetic data are not a real benchmark result -- see docstring."
        )

    predictions = {m: [] for m in METRICS}
    actuals = {m: [] for m in METRICS}

    with open(metadata_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path = image_dir / row["image_path"]
            pred = run_pipeline_on_image(image_path)  # raises until real weights are wired up
            predictions["mass"].append(pred["mass_g"])
            predictions["calories"].append(pred["kcal"])
            predictions["protein"].append(pred["protein_g"])
            predictions["fat"].append(pred["fat_g"])
            predictions["carb"].append(pred["carb_g"])
            actuals["mass"].append(float(row["total_mass"]))
            actuals["calories"].append(float(row["total_calories"]))
            actuals["protein"].append(float(row["total_protein"]))
            actuals["fat"].append(float(row["total_fat"]))
            actuals["carb"].append(float(row["total_carb"]))

    return {m: mape(predictions[m], actuals[m]) for m in METRICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=SYNTHETIC_SAMPLE)
    parser.add_argument("--images", type=Path, default=Path("."))
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()

    results = evaluate(args.metadata, args.images, allow_synthetic=args.allow_synthetic)
    for metric, value in results.items():
        print(f"  {metric:>10s} MAPE: {value:.1f}%")


if __name__ == "__main__":
    main()
