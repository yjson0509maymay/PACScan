"""Cloud-safe NIfTI preprocessing adapted from BRAINTENSOR.

Source workflow:
BRAINTENSOR/01_Preprocessing/스크립트/preparing_ref21order_v2.py

The research pipeline uses FSL BET, ANTs N4, and MNI registration. Those native
tools are not available on Streamlit Community Cloud, so this adapter executes
the deployable stages: validation, canonical orientation, finite-value cleanup,
non-zero min-max normalization, and 56^3 linear resampling.
"""

from __future__ import annotations

import base64
import gzip
import io
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


TARGET_SHAPE = (56, 56, 56)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    message: str
    filename: str
    size_mb: float
    shape: tuple[int, ...] = ()
    spacing: tuple[float, ...] = ()
    orientation: str = "-"


def _nifti_from_bytes(payload: bytes, filename: str) -> nib.Nifti1Image:
    raw = gzip.decompress(payload) if filename.lower().endswith(".gz") else payload
    return nib.Nifti1Image.from_bytes(raw)


def validate_nifti(payload: bytes, filename: str) -> ValidationResult:
    size_mb = len(payload) / 1024 / 1024
    if not filename.lower().endswith((".nii", ".nii.gz")):
        return ValidationResult(False, "지원하지 않는 형식입니다. .nii 또는 .nii.gz 파일을 선택하세요.", filename, size_mb)
    if len(payload) > MAX_UPLOAD_BYTES:
        return ValidationResult(False, "파일이 200MB 제한을 초과했습니다.", filename, size_mb)
    try:
        img = _nifti_from_bytes(payload, filename)
        shape = tuple(int(v) for v in img.shape)
        if len(shape) != 3:
            return ValidationResult(False, f"3D T2 MRI가 필요합니다. 현재 shape: {shape}", filename, size_mb, shape)
        spacing = tuple(round(float(v), 3) for v in img.header.get_zooms()[:3])
        orientation = "".join(nib.aff2axcodes(img.affine))
        probe = np.asanyarray(img.dataobj)
        if not np.isfinite(probe).any():
            return ValidationResult(False, "유효한 영상 voxel을 찾지 못했습니다.", filename, size_mb, shape, spacing, orientation)
        return ValidationResult(True, "유효한 3D NIfTI T2 MRI입니다.", filename, size_mb, shape, spacing, orientation)
    except Exception as exc:
        return ValidationResult(False, f"NIfTI 파일을 읽을 수 없습니다: {exc}", filename, size_mb)


def _normalize_minmax(data: np.ndarray) -> np.ndarray:
    finite = np.isfinite(data)
    mask = finite & (data != 0)
    if not np.any(mask):
        raise ValueError("정규화할 비영(非零) brain voxel이 없습니다.")
    values = data[mask]
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-6:
        raise ValueError("영상 intensity 범위가 너무 작습니다.")
    result = np.zeros_like(data, dtype=np.float32)
    result[mask] = (data[mask] - low) / (high - low)
    return result


def _slice_png(data: np.ndarray, axis: int) -> str:
    index = data.shape[axis] // 2
    plane = np.take(data, index, axis=axis)
    plane = np.rot90(plane)
    finite = plane[np.isfinite(plane)]
    if finite.size:
        lo, hi = np.percentile(finite, [1, 99])
        plane = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)
    pixels = (plane * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def preprocess_nifti(payload: bytes, filename: str) -> dict:
    img = _nifti_from_bytes(payload, filename)
    canonical = nib.as_closest_canonical(img)
    original = canonical.get_fdata(dtype=np.float32)
    original = np.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = _normalize_minmax(original)
    factors = np.asarray(TARGET_SHAPE, dtype=float) / np.asarray(normalized.shape, dtype=float)
    resized = zoom(normalized, factors, order=1).astype(np.float32)
    new_affine = canonical.affine.copy()
    new_affine[:3, :3] = canonical.affine[:3, :3] / factors
    output = nib.Nifti1Image(resized, new_affine)
    output_bytes = gzip.compress(output.to_bytes())
    return {
        "original_shape": tuple(int(v) for v in original.shape),
        "final_shape": TARGET_SHAPE,
        "spacing": tuple(round(float(v), 3) for v in canonical.header.get_zooms()[:3]),
        "orientation": "".join(nib.aff2axcodes(canonical.affine)),
        "original_views": [_slice_png(original, axis) for axis in (2, 1, 0)],
        "processed_views": [_slice_png(resized, axis) for axis in (2, 1, 0)],
        "output_bytes": output_bytes,
        "output_name": filename.removesuffix(".gz").removesuffix(".nii") + "_preprocessed_56.nii.gz",
    }
