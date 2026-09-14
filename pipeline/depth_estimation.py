"""
Monocular depth estimation.

IMPORTANT LIMITATION (see README): plain relative depth from a single RGB
image has no metric scale. This module supports two modes:

  - "metric": use a metric-depth model (e.g. Depth Pro, ZoeDepth) that
    predicts real-world distances directly, OR a relative-depth model plus an
    explicit scale reference (see `scale_from_reference_object`).
  - "relative": relative depth only (e.g. Depth Anything V2 base checkpoint)
    -- volumes derived from this WILL be wrong by an unknown constant factor
    unless a scale reference is supplied. The pipeline refuses to silently
    proceed in this case; see `pipeline.py`.

Model weights are not bundled here (this environment has no model-hub
network access). `load_model()` will attempt a real `transformers` load and
raise a clear error if unavailable; `DepthEstimator.mock()` gives you a
synthetic depth map so the rest of the pipeline (segmentation -> volume ->
mass -> macros -> storage -> chatbot) is runnable and testable end-to-end
without GPU/model-hub access.
"""
from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass
class DepthResult:
    depth_map: np.ndarray  # HxW, meters if metric=True, else unitless relative depth
    metric: bool
    scale_factor: float | None = None  # set if a reference object calibrated relative->metric


class DepthEstimator:
    def __init__(self, backend: str = "depth_anything_v2", metric: bool = False):
        self.backend = backend
        self.metric = metric
        self._model = None

    def load(self):
        """Attempt to load real weights via `transformers`.

        Raises RuntimeError with a clear message if the model hub isn't
        reachable -- callers should catch this and fall back to `.mock()`
        during development, but must NOT ship that fallback to production.
        """
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as e:
            raise RuntimeError(
                "transformers not installed. `pip install transformers torch pillow`."
            ) from e
        try:
            self._model = hf_pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
            )
        except Exception as e:  # network / hub access, auth, etc.
            raise RuntimeError(
                "Could not download depth model weights (no model-hub network "
                "access in this environment). Run this module where huggingface.co "
                "is reachable, or point `model=` at a locally cached checkpoint."
            ) from e
        return self

    def estimate(self, image: np.ndarray) -> DepthResult:
        if self._model is None:
            raise RuntimeError("Call .load() first, or use DepthEstimator.mock() for testing.")
        from PIL import Image

        out = self._model(Image.fromarray(image))
        depth = np.array(out["depth"], dtype=np.float32)
        return DepthResult(depth_map=depth, metric=self.metric)

    @staticmethod
    def mock(height: int = 224, width: int = 224, seed: int = 0, camera_to_plate_m: float = 0.35) -> DepthResult:
        """Synthetic depth map: a dome shape (like food piled on a plate),
        for exercising the rest of the pipeline without real weights.

        Depth convention (matches real depth models): depth = distance from
        camera to surface. Food piled higher is CLOSER to the camera, so its
        depth value is SMALLER than the surrounding plate, not larger --
        `camera_to_plate_m - bump_height`, not `bump_height` alone.
        """
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:height, 0:width]
        cy, cx = height / 2, width / 2
        r = np.sqrt(((yy - cy) / (height / 2)) ** 2 + ((xx - cx) / (width / 2)) ** 2)
        dome = np.clip(1.0 - r, 0, 1) ** 1.5  # 0 at edges, 1 at center
        bump_height_m = dome * 0.04  # ~4cm peak pile height
        depth = camera_to_plate_m - bump_height_m + rng.normal(0, 0.0005, size=(height, width))
        depth = np.clip(depth, 0.01, None)
        return DepthResult(depth_map=depth.astype(np.float32), metric=True, scale_factor=1.0)


def scale_from_reference_object(
    relative_depth: np.ndarray,
    reference_mask: np.ndarray,
    reference_real_height_m: float,
) -> float:
    """Compute a relative->metric scale factor from a known-size reference
    object in frame (e.g. a card, coin, or standard plate rim of known
    diameter). This is the calibration step that makes single-image volume
    estimation trustworthy -- see README for why this can't be skipped.

    `reference_mask` is a boolean HxW mask over the reference object.
    Returns a multiplier: metric_depth = relative_depth * scale_factor.
    """
    if reference_mask.sum() == 0:
        raise ValueError("Empty reference mask -- no scale reference detected in frame.")
    ref_relative_span = (
        relative_depth[reference_mask].max() - relative_depth[reference_mask].min()
    )
    if ref_relative_span <= 0:
        raise ValueError("Reference object has no depth variation to calibrate against.")
    return reference_real_height_m / ref_relative_span
