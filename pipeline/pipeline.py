"""
End-to-end orchestration: image -> per-item macros -> stored meal.

This wires the modules together and enforces the one hard rule the rest of
the codebase assumes: we never compute a volume from a depth map that hasn't
been marked metric (see depth_estimation.DepthResult.metric). Skipping that
check is exactly how you'd silently ship volume estimates that are off by an
arbitrary factor -- see the README section on scale ambiguity.
"""
from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np

from .depth_estimation import DepthResult
from .nutrition_db import NutritionDB, MacroResult
from .segmentation import Segment
from .storage import Storage
from .volume_estimator import CameraIntrinsics, estimate_volume_ml, volume_to_mass_g
from .vlm_classifier import ClassificationResult


@dataclasses.dataclass
class DetectedItem:
    label: str
    match_score: float
    grams: float
    macros: MacroResult
    needs_confirmation: bool


class MealPipeline:
    def __init__(self, db: NutritionDB, storage: Storage, confirmation_threshold: float = 70.0):
        self.db = db
        self.storage = storage
        self.confirmation_threshold = confirmation_threshold

    def process_meal(
        self,
        depth: DepthResult,
        segments: list[Segment],
        classifications: list[ClassificationResult],
        intrinsics: CameraIntrinsics,
    ) -> list[DetectedItem]:
        if not depth.metric:
            raise ValueError(
                "Refusing to estimate volume from non-metric (relative) depth. "
                "Calibrate with depth_estimation.scale_from_reference_object() first, "
                "or use a metric-depth model."
            )
        if len(segments) != len(classifications):
            raise ValueError("segments and classifications must be the same length and aligned")

        items = []
        for seg, cls in zip(segments, classifications):
            match = self.db.best_match(cls.label)
            if match is None:
                # No DB entry even loosely matches -- surface to user rather than guessing.
                items.append(
                    DetectedItem(
                        label=cls.label, match_score=0.0, grams=0.0,
                        macros=MacroResult(cls.label, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                        needs_confirmation=True,
                    )
                )
                continue
            entry, score = match
            volume_ml = estimate_volume_ml(depth.depth_map, seg.mask, intrinsics)
            grams = volume_to_mass_g(volume_ml, entry.density_g_per_ml)
            macros = entry.scale(grams)
            items.append(
                DetectedItem(
                    label=entry.name,
                    match_score=score,
                    grams=grams,
                    macros=macros,
                    needs_confirmation=(score < self.confirmation_threshold) or (cls.confidence_note != "confident"),
                )
            )
        return items

    def log_meal(self, user_id: str, items: list[DetectedItem], user_corrected: bool = False) -> int:
        macro_list = [it.macros for it in items]
        total = macro_list[0]
        for m in macro_list[1:]:
            total = total + m
        return self.storage.log_meal(
            user_id=user_id,
            logged_at_iso=dt.datetime.now().isoformat(),
            items=macro_list,
            totals=total,
            user_corrected=user_corrected,
        )
