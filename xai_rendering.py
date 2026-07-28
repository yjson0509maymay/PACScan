"""Substantia-nigra-restricted 3D CAM post-processing and MRI rendering.

The classifier still receives the same 56x56x56 tensor. This module changes only XAI
visualization: CAM is intersected with a substantia-nigra (SN) ROI and then the
strongest bilateral focal responses inside that ROI are rendered over the Min-Max-
normalized, pre-resize volume.

A supplied anatomical SN mask is preferred. When no mask is available, a conservative
bilateral midbrain ROI is estimated from the visible anatomy. The fallback is explicitly
reported as an estimated ROI and must not be interpreted as an anatomical segmentation.
"""
from __future__ import annotations

import base64
import io
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_opening,
    generate_binary_structure,
    label,
    maximum_filter,
)


def _display_window_mri(plane: np.ndarray) -> np.ndarray:
    plane = np.nan_to_num(np.asarray(plane, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane, dtype=np.float32)
    nonzero = finite[np.abs(finite) > 1e-8]
    sample = nonzero if nonzero.size >= 128 else finite
    lo, hi = np.percentile(sample, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(plane, dtype=np.float32)
    return np.power(np.clip((plane - lo) / (hi - lo), 0.0, 1.0), 0.92).astype(np.float32)


def _interpolate_palette(gray: np.ndarray, stops_c: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0.0, 1.0)
    stops_t = np.linspace(0.0, 1.0, num=len(stops_c), dtype=np.float32)
    return np.stack(
        [np.interp(gray, stops_t, stops_c[:, channel]) for channel in range(3)],
        axis=-1,
    )


def _class_palette(gray: np.ndarray, class_label: str | None = None) -> np.ndarray:
    """Return a class-specific RGB palette.

    The hue family reflects the predicted class, while brightness/saturation still reflect
    CAM strength *within* that class. This avoids interpreting red as "more severe" for
    Prodromal cases.
    """
    label = (class_label or 'PD').strip()
    if label == 'Control':
        # Blue -> cyan only
        stops_c = np.array([[30, 60, 220], [0, 140, 230], [0, 200, 200], [110, 240, 255]], dtype=np.float32)
    elif label == 'Prodromal':
        # Yellow -> orange only (no red)
        stops_c = np.array([[255, 232, 120], [255, 214, 64], [255, 176, 38], [255, 140, 0]], dtype=np.float32)
    else:  # PD and fallback
        # Orange -> red
        stops_c = np.array([[255, 190, 70], [255, 145, 20], [240, 90, 25], [220, 30, 30]], dtype=np.float32)
    return _interpolate_palette(gray, stops_c)


def _visible_anatomy_mask(volume: np.ndarray) -> np.ndarray:
    data = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    positive = data[data > 1e-8]
    if positive.size == 0:
        return np.zeros_like(data, dtype=bool)
    floor = max(float(np.percentile(positive, 1.0)), 0.008)
    mask = data > floor
    structure = generate_binary_structure(3, 1)
    mask = binary_closing(mask, structure=structure, iterations=1)
    return mask


def _bounding_box(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        lo = np.zeros(3, dtype=np.float32)
        hi = np.asarray(mask.shape, dtype=np.float32) - 1.0
    else:
        lo = coords.min(axis=0).astype(np.float32)
        hi = coords.max(axis=0).astype(np.float32)
    return lo, hi


def build_substantia_nigra_roi(
    volume: np.ndarray,
    voxel_spacing: Iterable[float] = (1.0, 1.0, 1.0),
    *,
    supplied_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    data = np.asarray(volume, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"3D volume이 필요합니다: shape={data.shape}")

    anatomy = _visible_anatomy_mask(data)
    structure = generate_binary_structure(3, 1)

    if supplied_mask is not None and np.shape(supplied_mask) == data.shape:
        roi = np.asarray(supplied_mask, dtype=np.float32) > 0.5
        roi = binary_opening(roi, structure=structure, iterations=1)
        roi = binary_closing(roi, structure=structure, iterations=1)
        if int(roi.sum()) > 0:
            return roi.astype(bool), {
                "roi_source": "anatomical_mask",
                "roi_label": "해부학적 흑질 마스크",
                "roi_voxels": int(roi.sum()),
                "estimated": False,
            }

    lo, hi = _bounding_box(anatomy)
    extent = np.maximum(hi - lo, 1.0)
    spacing = np.asarray(tuple(voxel_spacing)[:3], dtype=np.float32)
    spacing = np.maximum(spacing, 1e-3)

    center_y = lo[1] + 0.47 * extent[1]
    center_z = lo[2] + 0.30 * extent[2]
    center_x = lo[0] + 0.50 * extent[0]
    side_offset = max(3.5 / spacing[0], 0.045 * extent[0])

    radius_x = max(3.8 / spacing[0], 0.036 * extent[0])
    radius_y = max(5.4 / spacing[1], 0.042 * extent[1])
    radius_z = max(4.6 / spacing[2], 0.060 * extent[2])

    gx, gy, gz = np.ogrid[0:data.shape[0], 0:data.shape[1], 0:data.shape[2]]
    gx = gx.astype(np.float32, copy=False)
    gy = gy.astype(np.float32, copy=False)
    gz = gz.astype(np.float32, copy=False)
    roi = np.zeros(data.shape, dtype=bool)
    for sign in (-1.0, 1.0):
        cx = center_x + sign * side_offset
        ellipsoid = (
            ((gx - cx) / max(radius_x, 1.0)) ** 2
            + ((gy - center_y) / max(radius_y, 1.0)) ** 2
            + ((gz - center_z) / max(radius_z, 1.0)) ** 2
            <= 1.0
        )
        roi |= ellipsoid

    anatomy_support = binary_dilation(anatomy, structure=structure, iterations=2)
    constrained = roi & anatomy_support
    if int(constrained.sum()) >= 4:
        roi = constrained

    return roi.astype(bool), {
        "roi_source": "estimated_midbrain",
        "roi_label": "추정 양측 흑질 ROI",
        "roi_voxels": int(roi.sum()),
        "estimated": True,
        "center_fraction_xyz": [0.50, 0.47, 0.30],
    }


def _split_bilateral_roi(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(roi)
    if coords.size == 0:
        empty = np.zeros_like(roi, dtype=bool)
        return empty, empty
    xmin, xmax = int(coords[:, 0].min()), int(coords[:, 0].max())
    mid = (xmin + xmax) / 2.0
    gx = np.arange(roi.shape[0], dtype=np.float32)[:, None, None]
    left = roi & (gx < mid)
    right = roi & (gx >= mid)
    if int(left.sum()) == 0 or int(right.sum()) == 0:
        mid = roi.shape[0] / 2.0
        left = roi & (gx < mid)
        right = roi & (gx >= mid)
    return left.astype(bool), right.astype(bool)


def _select_side_hotspot(
    norm: np.ndarray,
    side_roi: np.ndarray,
    *,
    global_peak: float,
    side_name: str,
    target_side_coverage: float = 0.24,
    min_peak_ratio: float = 0.04,
) -> tuple[np.ndarray, dict]:
    empty = {
        "side": side_name,
        "component_count": 0,
        "coverage_percent": 0.0,
        "peak_ratio": 0.0,
        "peak_value": 0.0,
        "threshold": 0.0,
        "displayed": False,
        "voxel_count": 0,
    }
    if int(side_roi.sum()) == 0:
        return np.zeros_like(norm, dtype=np.float32), empty

    side_values = norm[side_roi]
    side_values = side_values[side_values > 1e-8]
    if side_values.size == 0:
        return np.zeros_like(norm, dtype=np.float32), empty

    side_peak = float(side_values.max())
    peak_ratio = side_peak / max(global_peak, 1e-8)
    if peak_ratio < min_peak_ratio:
        meta = {**empty, "peak_ratio": round(float(peak_ratio), 4), "peak_value": round(float(side_peak), 4)}
        return np.zeros_like(norm, dtype=np.float32), meta

    threshold = max(0.06, side_peak * 0.34)
    chosen_percentile = 75.0
    mask = np.zeros_like(side_roi, dtype=bool)
    side_count = max(int(side_roi.sum()), 1)
    for percentile in (70.0, 75.0, 80.0, 85.0, 90.0, 92.5, 95.0):
        candidate_threshold = max(float(np.percentile(side_values, percentile)), side_peak * 0.34, 0.06)
        candidate_mask = side_roi & (norm >= candidate_threshold)
        coverage = float(candidate_mask.sum()) / side_count
        threshold = candidate_threshold
        chosen_percentile = percentile
        mask = candidate_mask
        if int(candidate_mask.sum()) >= 2 and coverage <= target_side_coverage:
            break

    structure = generate_binary_structure(3, 2)
    mask = binary_closing(mask, structure=structure, iterations=1) & side_roi
    components, count = label(mask, structure=structure)
    best_mask = np.zeros_like(mask, dtype=bool)
    best_score = -1.0
    best_size = 0
    for component_id in range(1, count + 1):
        component_mask = components == component_id
        size = int(component_mask.sum())
        if size == 0:
            continue
        values = norm[component_mask]
        score = float(values.max()) * 0.8 + float(values.mean()) * 0.2
        if score > best_score:
            best_score = score
            best_size = size
            best_mask = component_mask

    if int(best_mask.sum()) == 0:
        best_mask = mask
        best_size = int(mask.sum())

    display_mask = binary_dilation(best_mask, structure=generate_binary_structure(3, 1), iterations=1) & side_roi
    display_mask = binary_closing(display_mask, structure=generate_binary_structure(3, 1), iterations=1) & side_roi

    # Keep the display compact but not dot-like.
    min_voxels = 5
    max_voxels = max(min_voxels, int(np.ceil(side_count * target_side_coverage)))
    keep_indices = np.flatnonzero(display_mask)
    if keep_indices.size > max_voxels:
        score_field = maximum_filter(norm * best_mask.astype(np.float32), size=3).reshape(-1)
        values = score_field[keep_indices]
        top_order = np.argpartition(values, -max_voxels)[-max_voxels:]
        capped = np.zeros_like(display_mask, dtype=bool).reshape(-1)
        capped[keep_indices[top_order]] = True
        display_mask = capped.reshape(display_mask.shape)
    elif 0 < keep_indices.size < min_voxels:
        score_field = maximum_filter(norm * best_mask.astype(np.float32), size=3)
        expanded = binary_dilation(display_mask, structure=generate_binary_structure(3, 1), iterations=2) & side_roi
        candidate_indices = np.flatnonzero(expanded)
        if candidate_indices.size:
            values = score_field.reshape(-1)[candidate_indices]
            k = min(min_voxels, candidate_indices.size)
            top_order = np.argpartition(values, -k)[-k:]
            boosted = np.zeros_like(display_mask, dtype=bool).reshape(-1)
            boosted[candidate_indices[top_order]] = True
            display_mask = boosted.reshape(display_mask.shape)

    side_signal = maximum_filter(norm * best_mask.astype(np.float32), size=3)
    focused = np.zeros_like(norm, dtype=np.float32)
    selected_values = side_signal[display_mask]
    if selected_values.size == 0:
        meta = {
            **empty,
            "peak_ratio": round(float(peak_ratio), 4),
            "peak_value": round(float(side_peak), 4),
            "threshold": round(float(threshold), 5),
        }
        return focused, meta

    high = float(np.percentile(selected_values, 99.0)) if selected_values.size else side_peak
    low = min(float(threshold) * 0.82, high - 1e-6)
    focused[display_mask] = np.clip((side_signal[display_mask] - low) / max(high - low, 1e-6), 0.0, 1.0)
    focused[display_mask] = np.power(focused[display_mask], 0.68)
    focused[focused < 0.08] = 0.0

    coverage_percent = float((focused > 0).sum()) * 100.0 / side_count
    meta = {
        "side": side_name,
        "component_count": 1 if float(focused.max()) > 0.0 else 0,
        "coverage_percent": round(float(coverage_percent), 3),
        "peak_ratio": round(float(peak_ratio), 4),
        "peak_value": round(float(side_peak), 4),
        "threshold": round(float(threshold), 5),
        "displayed": bool(float(focused.max()) > 0.0),
        "voxel_count": int((focused > 0).sum()),
        "seed_component_voxels": int(best_size),
    }
    return focused.astype(np.float32), meta


def restrict_cam_to_substantia_nigra(
    cam: np.ndarray,
    volume: np.ndarray,
    *,
    voxel_spacing: Iterable[float] = (1.0, 1.0, 1.0),
    supplied_mask: np.ndarray | None = None,
    max_components: int = 2,
    target_roi_coverage: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict]:
    raw = np.nan_to_num(np.asarray(cam, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.clip(raw, 0.0, None)
    roi, roi_meta = build_substantia_nigra_roi(volume, voxel_spacing, supplied_mask=supplied_mask)

    empty_meta = {
        **roi_meta,
        "threshold": 0.0,
        "percentile": 100.0,
        "roi_cam_coverage_percent": 0.0,
        "anatomy_coverage_percent": 0.0,
        "component_count": 0,
        "roi_peak_ratio": 0.0,
        "meaningful_activation": False,
        "left_visible": False,
        "right_visible": False,
        "left_peak_ratio": 0.0,
        "right_peak_ratio": 0.0,
        "side_mode": "bilateral_hotspot",
    }
    if raw.ndim != 3 or raw.shape != roi.shape or raw.size == 0 or float(raw.max()) <= 1e-8:
        return np.zeros_like(raw, dtype=np.float32), roi, empty_meta

    positive = raw[raw > 1e-8]
    robust_peak = float(np.percentile(positive, 99.9)) if positive.size else 0.0
    if robust_peak <= 1e-8:
        return np.zeros_like(raw, dtype=np.float32), roi, empty_meta
    norm = np.clip(raw / robust_peak, 0.0, 1.0)

    roi_values = norm[roi]
    roi_values = roi_values[roi_values > 1e-8]
    if roi_values.size == 0:
        return np.zeros_like(raw, dtype=np.float32), roi, empty_meta

    roi_peak = float(roi_values.max())
    global_peak = float(norm.max())
    roi_peak_ratio = roi_peak / max(global_peak, 1e-8)

    left_roi, right_roi = _split_bilateral_roi(roi)
    side_target = float(np.clip(target_roi_coverage * 0.9, 0.16, 0.28))
    left_focus, left_meta = _select_side_hotspot(norm, left_roi, global_peak=global_peak, side_name='left', target_side_coverage=side_target, min_peak_ratio=0.035)
    right_focus, right_meta = _select_side_hotspot(norm, right_roi, global_peak=global_peak, side_name='right', target_side_coverage=side_target, min_peak_ratio=0.035)

    focused = np.maximum(left_focus, right_focus)
    focused[~roi] = 0.0

    anatomy = _visible_anatomy_mask(volume)
    anatomy_count = max(int(anatomy.sum()), 1)
    roi_count = max(int(roi.sum()), 1)
    visible_mask = focused > 0
    left_visible = bool(left_meta.get('displayed'))
    right_visible = bool(right_meta.get('displayed'))
    component_count = int(left_visible) + int(right_visible)
    meaningful = bool(float(focused.max()) > 0.0 and (left_visible or right_visible) and roi_peak_ratio >= 0.10)
    if not meaningful:
        focused[:] = 0.0
        component_count = 0
        left_visible = False
        right_visible = False

    meta = {
        **roi_meta,
        "threshold": round(float(min(left_meta.get('threshold', 0.0) or 0.0, right_meta.get('threshold', 0.0) or 0.0)), 5) if component_count else 0.0,
        "percentile": 0.0,
        "roi_cam_coverage_percent": round(float(visible_mask.sum()) * 100.0 / roi_count, 3) if component_count else 0.0,
        "anatomy_coverage_percent": round(float(visible_mask.sum()) * 100.0 / anatomy_count, 4) if component_count else 0.0,
        "component_count": component_count,
        "roi_peak_ratio": round(float(roi_peak_ratio), 4),
        "meaningful_activation": meaningful,
        "left_visible": left_visible,
        "right_visible": right_visible,
        "left_peak_ratio": left_meta.get('peak_ratio', 0.0),
        "right_peak_ratio": right_meta.get('peak_ratio', 0.0),
        "left_roi_coverage_percent": left_meta.get('coverage_percent', 0.0),
        "right_roi_coverage_percent": right_meta.get('coverage_percent', 0.0),
        "left_voxel_count": left_meta.get('voxel_count', 0),
        "right_voxel_count": right_meta.get('voxel_count', 0),
        "side_mode": "bilateral_hotspot",
        "sides_visible_label": (
            '양측 표시' if (left_visible and right_visible) else '좌측만 표시' if left_visible else '우측만 표시' if right_visible else '표시 없음'
        ),
    }
    return focused.astype(np.float32), roi.astype(bool), meta


def focus_cam_regions(
    cam: np.ndarray,
    volume: np.ndarray | None = None,
    *,
    target_coverage: float = 0.018,
    max_components: int = 3,
) -> tuple[np.ndarray, dict]:
    if volume is None:
        volume = np.ones_like(cam, dtype=np.float32)
    focused, _roi, meta = restrict_cam_to_substantia_nigra(
        cam,
        volume,
        max_components=min(max_components, 2),
        target_roi_coverage=min(max(target_coverage * 14.0, 0.14), 0.30),
    )
    return focused, meta


def _plane_pixel_spacing(axis: int, voxel_spacing: Iterable[float]) -> tuple[float, float]:
    sx, sy, sz = [max(float(value), 1e-6) for value in tuple(voxel_spacing)[:3]]
    if axis == 2:
        return sy, sx
    if axis == 1:
        return sz, sx
    if axis == 0:
        return sz, sy
    raise ValueError(f"지원하지 않는 축입니다: {axis}")


def _spacing_corrected_size(
    shape: tuple[int, int],
    axis: int,
    voxel_spacing: Iterable[float],
    target_long_side: int = 960,
) -> tuple[int, int]:
    row_spacing, col_spacing = _plane_pixel_spacing(axis, voxel_spacing)
    physical_h = max(float(shape[0]) * row_spacing, 1e-6)
    physical_w = max(float(shape[1]) * col_spacing, 1e-6)
    ratio = physical_h / physical_w
    ratio = float(np.clip(ratio, 0.62, 1.85))
    if ratio <= 1.0:
        out_w = target_long_side
        out_h = max(360, int(round(target_long_side * ratio)))
    else:
        out_h = target_long_side
        out_w = max(360, int(round(target_long_side / ratio)))
    return out_w, out_h


def render_cam_overlay_png(
    volume: np.ndarray,
    cam: np.ndarray,
    axis: int,
    *,
    index: int | None = None,
    voxel_spacing: Iterable[float] = (1.0, 1.0, 1.0),
    class_label: str | None = None,
    cam_alpha_max: float = 0.80,
    cam_floor: float = 0.0,
) -> str:
    del cam_floor
    if index is None:
        index = volume.shape[axis] // 2
    index = max(0, min(int(index), volume.shape[axis] - 1))

    base = np.rot90(np.take(volume, index, axis=axis))
    heat = np.rot90(np.take(cam, index, axis=axis))
    base_n = _display_window_mri(base)
    heat_n = np.clip(np.asarray(heat, dtype=np.float32), 0.0, 1.0)
    heat_n *= (base_n > 0.02).astype(np.float32)

    out_w, out_h = _spacing_corrected_size(base_n.shape, axis, voxel_spacing)
    base_img = Image.fromarray((base_n * 255.0).astype(np.uint8), mode='L').resize((out_w, out_h), Image.Resampling.LANCZOS)
    heat_img = Image.fromarray((heat_n * 255.0).astype(np.uint8), mode='L').resize((out_w, out_h), Image.Resampling.BILINEAR)
    base_hi = np.asarray(base_img, dtype=np.float32) / 255.0
    heat_hi = np.asarray(heat_img, dtype=np.float32) / 255.0

    heat_hi = np.where(heat_hi >= 0.10, (heat_hi - 0.10) / 0.90, 0.0)
    heat_hi = np.clip(heat_hi, 0.0, 1.0) ** 0.72

    base_rgb = np.stack([base_hi] * 3, axis=-1) * 255.0
    heat_rgb = _class_palette(heat_hi, class_label=class_label)
    alpha = (heat_hi * cam_alpha_max)[..., None]
    composite = np.clip(base_rgb * (1.0 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)

    output = io.BytesIO()
    Image.fromarray(composite, mode='RGB').save(output, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()


def _axial_hotspot_coordinate(side_cam: np.ndarray, fallback_x: int, fallback_y: int, fallback_z: int) -> tuple[int, int, int, bool]:
    positive = side_cam > 0
    if not np.any(positive):
        return int(fallback_x), int(fallback_y), int(fallback_z), False
    flat_index = int(np.argmax(side_cam))
    coord = np.unravel_index(flat_index, side_cam.shape)
    return int(coord[0]), int(coord[1]), int(coord[2]), True


def _render_crop_overlay(
    volume: np.ndarray,
    cam: np.ndarray,
    *,
    center_xyz: tuple[int, int, int],
    class_label: str | None = None,
    out_size: tuple[int, int] = (360, 360),
    half_width_x: int = 20,
    half_width_y: int = 20,
    cam_alpha_max: float = 0.80,
) -> str:
    x, y, z = [int(v) for v in center_xyz]
    x0 = max(0, x - half_width_x)
    x1 = min(volume.shape[0], x + half_width_x + 1)
    y0 = max(0, y - half_width_y)
    y1 = min(volume.shape[1], y + half_width_y + 1)

    base = np.rot90(np.asarray(volume[x0:x1, y0:y1, z], dtype=np.float32))
    heat = np.rot90(np.asarray(cam[x0:x1, y0:y1, z], dtype=np.float32))
    base_n = _display_window_mri(base)
    heat_n = np.clip(heat, 0.0, 1.0)
    heat_n *= (base_n > 0.02).astype(np.float32)

    base_img = Image.fromarray((base_n * 255.0).astype(np.uint8), mode='L').resize(out_size, Image.Resampling.LANCZOS)
    heat_img = Image.fromarray((heat_n * 255.0).astype(np.uint8), mode='L').resize(out_size, Image.Resampling.BILINEAR)
    base_hi = np.asarray(base_img, dtype=np.float32) / 255.0
    heat_hi = np.asarray(heat_img, dtype=np.float32) / 255.0
    heat_hi = np.where(heat_hi >= 0.08, (heat_hi - 0.08) / 0.92, 0.0)
    heat_hi = np.clip(heat_hi, 0.0, 1.0) ** 0.72

    base_rgb = np.stack([base_hi] * 3, axis=-1) * 255.0
    heat_rgb = _class_palette(heat_hi, class_label=class_label)
    alpha = (heat_hi * cam_alpha_max)[..., None]
    composite = np.clip(base_rgb * (1.0 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)

    output = io.BytesIO()
    Image.fromarray(composite, mode='RGB').save(output, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode()


def render_bilateral_cam_insets_png(
    volume: np.ndarray,
    cam: np.ndarray,
    *,
    class_label: str | None = None,
) -> dict:
    """Return left/right axial hotspot inset images for the substantia nigra region."""
    shape = volume.shape
    mid_x = shape[0] // 2
    mid_y = shape[1] // 2
    fallback_z = int(shape[2] * 0.30)
    left_cam = np.zeros_like(cam, dtype=np.float32)
    right_cam = np.zeros_like(cam, dtype=np.float32)
    left_cam[:mid_x, :, :] = cam[:mid_x, :, :]
    right_cam[mid_x:, :, :] = cam[mid_x:, :, :]

    left_coord = _axial_hotspot_coordinate(left_cam, max(mid_x // 2, 0), mid_y, fallback_z)
    right_coord = _axial_hotspot_coordinate(right_cam, min((mid_x + shape[0]) // 2, shape[0] - 1), mid_y, fallback_z)

    half_x = max(14, int(shape[0] * 0.065))
    half_y = max(14, int(shape[1] * 0.065))
    left_img = _render_crop_overlay(volume, cam, center_xyz=left_coord[:3], class_label=class_label, half_width_x=half_x, half_width_y=half_y)
    right_img = _render_crop_overlay(volume, cam, center_xyz=right_coord[:3], class_label=class_label, half_width_x=half_x, half_width_y=half_y)
    return {
        'left_src': left_img,
        'right_src': right_img,
        'left_has_signal': bool(left_coord[3]),
        'right_has_signal': bool(right_coord[3]),
        'left_xyz': tuple(int(v) for v in left_coord[:3]),
        'right_xyz': tuple(int(v) for v in right_coord[:3]),
    }
