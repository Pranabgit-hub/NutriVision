"""
Per-food instance segmentation.

Real backend: SAM2 (Segment Anything 2), prompted with either automatic
mask generation + VLM filtering, or points/boxes from the VLM's own
grounding output. Weights aren't bundled here for the same model-hub-access
reason as depth_estimation.py. `SegmentationResult.mock()` produces
plausible-looking blob masks so the pipeline is testable end-to-end.
"""
from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass
class Segment:
    mask: np.ndarray  # boolean HxW
    label_hint: str | None = None  # optional prompt/box source label


class Segmenter:
    def __init__(self, backend: str = "sam2"):
        self.backend = backend
        self._model = None

    def load(self):
        raise RuntimeError(
            "SAM2 weights are not available in this environment (no model-hub "
            "network access). Install `sam2` + checkpoint where you have hub "
            "access, or use Segmenter.mock_segments() during development."
        )

    def segment(self, image: np.ndarray) -> list[Segment]:
        if self._model is None:
            raise RuntimeError("Call .load() first, or use mock_segments() for testing.")
        raise NotImplementedError  # real SAM2 call goes here

    @staticmethod
    def mock_segments(height: int = 224, width: int = 224, n: int = 2, seed: int = 0) -> list[Segment]:
        """N angular wedge masks over the central region, standing in for real
        instance masks. Deliberately overlaps the center of the frame (where
        `DepthEstimator.mock()`'s synthetic food dome lives) so the demo
        pipeline produces non-trivial, differentiated volumes per item rather
        than segments that miss the "food" entirely.
        """
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:height, 0:width]
        cy, cx = height / 2, width / 2
        angle = np.arctan2(yy - cy, xx - cx)  # -pi..pi
        radius = np.sqrt(((yy - cy) / (height / 2)) ** 2 + ((xx - cx) / (width / 2)) ** 2)
        within_plate = radius <= 0.9

        segments = []
        wedge_width = 2 * np.pi / n
        # small random offset per wedge so items aren't perfectly equal-sized
        for i in range(n):
            lo = -np.pi + i * wedge_width
            hi = lo + wedge_width * rng.uniform(0.7, 1.0)
            mask = within_plate & (angle >= lo) & (angle < hi)
            segments.append(Segment(mask=mask, label_hint=f"mock_item_{i}"))
        return segments
