"""
Depth map + segmentation mask + camera intrinsics -> volume (mL).

This is the one piece of the "3D reconstruction" stage that's pure math, no
model weights required, so it's fully implemented and tested (see
tests/test_volume_estimator.py).

Method: back-project each masked pixel to a 3D point using the pinhole
camera model, take the per-pixel height above a fitted base-plane (the
plate/table surface, estimated from the depth just outside the food mask),
and integrate height * pixel_footprint_area over the mask. This is the same
approach used in the food-volume-estimation literature (e.g. Graikos et al.
2020's point-cloud-to-volume step): reconstruct a heightmap relative to a
support surface rather than trusting absolute depth values, which cancels
out a lot of scale error as long as the *relative* depth between food and
plate is accurate -- absolute metric scale still needs the calibration step
in depth_estimation.scale_from_reference_object.
"""
from __future__ import annotations

import dataclasses
import numpy as np


@dataclasses.dataclass
class CameraIntrinsics:
    focal_length_px: float
    principal_point: tuple[float, float]  # (cx, cy)

    @staticmethod
    def approximate(image_width: int, image_height: int, horizontal_fov_deg: float = 65.0):
        """Rough intrinsics from image size + assumed FOV, for phones without
        calibration data. Real EXIF-derived intrinsics are far more accurate --
        prefer those when available.
        """
        fov_rad = np.deg2rad(horizontal_fov_deg)
        focal_length_px = (image_width / 2) / np.tan(fov_rad / 2)
        return CameraIntrinsics(
            focal_length_px=focal_length_px,
            principal_point=(image_width / 2, image_height / 2),
        )


def estimate_base_plane_depth(depth_map: np.ndarray, mask: np.ndarray, ring_px: int = 8) -> float:
    """Estimate the support-surface (plate/table) depth as the median depth
    in a ring immediately surrounding the food mask.
    """
    from scipy.ndimage import binary_dilation

    dilated = binary_dilation(mask, iterations=ring_px)
    ring = dilated & ~mask
    if ring.sum() == 0:
        # mask fills the frame; fall back to the mask's own shallowest depth
        return float(np.percentile(depth_map[mask], 5))
    return float(np.median(depth_map[ring]))


def estimate_volume_ml(
    depth_map: np.ndarray,
    mask: np.ndarray,
    intrinsics: CameraIntrinsics,
    base_plane_depth_m: float | None = None,
) -> float:
    """Integrate height-above-plane * pixel footprint area over the mask.

    depth_map: metric depth in meters (see depth_estimation.py for why this
        must be metric, not raw relative depth).
    mask: boolean HxW, the food instance.
    Returns volume in milliliters.
    """
    if mask.sum() == 0:
        return 0.0

    if base_plane_depth_m is None:
        base_plane_depth_m = estimate_base_plane_depth(depth_map, mask)

    height_above_plane = np.clip(base_plane_depth_m - depth_map, 0, None)  # camera looks down at food
    # If depth increases with distance from camera (typical convention), food
    # sticking up toward the camera has SMALLER depth than the plate behind it.
    # height = plane_depth - food_depth, clipped at 0.

    ys, xs = np.nonzero(mask)
    depths = depth_map[ys, xs]
    heights = height_above_plane[ys, xs]

    # Pixel footprint area at each point's depth (pinhole model): a pixel at
    # depth z subtends real-world area (z / f)^2 per pixel.
    f = intrinsics.focal_length_px
    pixel_areas_m2 = (depths / f) ** 2

    volume_m3 = float(np.sum(heights * pixel_areas_m2))
    volume_ml = volume_m3 * 1e6  # 1 m^3 = 1e6 mL
    return max(volume_ml, 0.0)


def volume_to_mass_g(volume_ml: float, density_g_per_ml: float) -> float:
    return volume_ml * density_g_per_ml
