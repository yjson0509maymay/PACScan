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
import tempfile
from dataclasses import dataclass
from pathlib import Path

import dicom2nifti
import nibabel as nib
import numpy as np
import pydicom
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


@dataclass(frozen=True)
class DicomFolderResult:
    valid: bool
    message: str
    total_files: int
    dicom_files: int
    series_count: int
    selected_uid: str = ""
    selected_description: str = "-"
    selected_files: int = 0
    patient_id: str = "-"


def inspect_dicom_folder(files: list[tuple[str, bytes]]) -> DicomFolderResult:
    series: dict[str, dict] = {}
    dicom_count = 0
    for name, payload in files:
        try:
            ds = pydicom.dcmread(io.BytesIO(payload), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "SeriesInstanceUID", ""))
            if not uid or not getattr(ds, "SOPClassUID", None):
                continue
            dicom_count += 1
            entry = series.setdefault(uid, {"files": [], "description": "", "protocol": "", "patient_id": "-"})
            entry["files"].append(name)
            entry["description"] = str(getattr(ds, "SeriesDescription", entry["description"]))
            entry["protocol"] = str(getattr(ds, "ProtocolName", entry["protocol"]))
            entry["patient_id"] = str(getattr(ds, "PatientID", entry["patient_id"]))
        except Exception:
            continue
    if not series:
        return DicomFolderResult(False, "폴더에서 유효한 DICOM 시리즈를 찾지 못했습니다.", len(files), 0, 0)

    def score(item: tuple[str, dict]) -> tuple[int, int]:
        text = f"{item[1]['description']} {item[1]['protocol']}".lower()
        t2_score = 10 if "t2" in text else 0
        if "flair" in text or "localizer" in text or "scout" in text:
            t2_score -= 6
        return t2_score, len(item[1]["files"])

    selected_uid, selected = max(series.items(), key=score)
    desc = selected["description"] or selected["protocol"] or "설명 없음"
    has_t2 = "t2" in f"{selected['description']} {selected['protocol']}".lower()
    message = "T2 DICOM 시리즈를 자동 선택했습니다." if has_t2 else "명시적인 T2 표기가 없어 슬라이스 수가 가장 많은 시리즈를 선택했습니다."
    return DicomFolderResult(True, message, len(files), dicom_count, len(series), selected_uid, desc, len(selected["files"]), selected["patient_id"])


def convert_dicom_folder(files: list[tuple[str, bytes]], selected_uid: str) -> tuple[bytes, str]:
    with tempfile.TemporaryDirectory(prefix="neurolens_dicom_") as temp:
        root = Path(temp)
        source, output = root / "source", root / "nifti"
        source.mkdir(); output.mkdir()
        written = 0
        selected_datasets = []
        for _, payload in files:
            try:
                ds = pydicom.dcmread(io.BytesIO(payload), force=True)
                if str(getattr(ds, "SeriesInstanceUID", "")) != selected_uid:
                    continue
                (source / f"slice_{written:05d}.dcm").write_bytes(payload)
                selected_datasets.append(ds)
                written += 1
            except Exception:
                continue
        if written < 2:
            raise ValueError("3D 변환에 필요한 DICOM 슬라이스가 부족합니다.")
        primary_error = None
        try:
            dicom2nifti.convert_directory(str(source), str(output), compression=True, reorient=True)
        except Exception as exc:
            primary_error = exc
        candidates = sorted(output.glob("*.nii.gz")) + sorted(output.glob("*.nii"))
        if not candidates:
            try:
                payload = _stack_dicom_series(selected_datasets)
                return payload, "patient_t2_converted.nii.gz"
            except Exception as fallback_error:
                detail = str(primary_error or "변환 결과 파일 없음")
                raise RuntimeError(
                    "DICOM 시리즈를 NIfTI로 변환하지 못했습니다. "
                    f"기본 변환: {detail} / 보조 변환: {fallback_error}"
                ) from fallback_error
        path = candidates[0]
        payload = path.read_bytes()
        if path.suffix.lower() == ".nii":
            payload = gzip.compress(payload)
        return payload, "patient_t2_converted.nii.gz"


def _stack_dicom_series(datasets: list[pydicom.dataset.Dataset]) -> bytes:
    def slice_position(ds: pydicom.dataset.Dataset) -> float:
        position = getattr(ds, "ImagePositionPatient", None)
        if position is not None and len(position) >= 3:
            return float(position[2])
        return float(getattr(ds, "InstanceNumber", 0))

    ordered = sorted(datasets, key=slice_position)
    slices = []
    expected_shape = None
    for ds in ordered:
        pixels = np.asarray(ds.pixel_array, dtype=np.float32)
        if pixels.ndim != 2:
            raise ValueError(f"2D 슬라이스가 아닙니다: shape={pixels.shape}")
        if expected_shape is None:
            expected_shape = pixels.shape
        elif pixels.shape != expected_shape:
            raise ValueError("DICOM 슬라이스 크기가 서로 다릅니다.")
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(pixels * slope + intercept)
    if len(slices) < 2:
        raise ValueError("픽셀을 읽을 수 있는 DICOM 슬라이스가 부족합니다.")

    volume = np.stack(slices, axis=-1).transpose(1, 0, 2)
    first = ordered[0]
    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    x_spacing = float(pixel_spacing[1])
    y_spacing = float(pixel_spacing[0])
    positions = [slice_position(ds) for ds in ordered]
    nonzero_steps = [
        abs(right - left)
        for left, right in zip(positions, positions[1:])
        if abs(right - left) > 1e-6
    ]
    z_spacing = float(np.median(nonzero_steps)) if nonzero_steps else float(
        getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0))
    )
    affine = np.diag([x_spacing, y_spacing, max(z_spacing, 1e-6), 1.0])
    nifti = nib.Nifti1Image(volume, affine)
    return gzip.compress(nifti.to_bytes())


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
    return _slice_png_at(data, axis, index)


def _slice_png_at(data: np.ndarray, axis: int, index: int) -> str:
    """[2026-07-29 병합, 동료 xai_rendering.py 계열 개선분 반영] robust 대비(1~99
    percentile, 배경 0 제외) + 고해상도 업스케일(LANCZOS) - 기존엔 raw min/max라
    두개골 등 소수 밝은 픽셀에 대비가 눌려 흐릿했음."""
    index = max(0, min(int(index), data.shape[axis] - 1))
    plane = np.rot90(np.take(data, index, axis=axis)).astype(np.float32)
    finite = plane[np.isfinite(plane)]
    if finite.size:
        nonzero = finite[np.abs(finite) > 1e-8]
        sample = nonzero if nonzero.size >= 128 else finite
        lo, hi = np.percentile(sample, [1.0, 99.0])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
        plane = np.clip((plane - lo) / max(hi - lo, 1e-6), 0, 1)
        plane = np.power(plane, 0.92).astype(np.float32)
    else:
        plane = np.zeros_like(plane, dtype=np.float32)

    image = Image.fromarray((plane * 255).astype(np.uint8), mode="L")
    long_side = max(image.size)
    scale = max(1, min(4, int(np.ceil(900 / max(long_side, 1)))))
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def render_nifti_views(payload: bytes, filename: str, indices: tuple[int, int, int]) -> list[str]:
    """Render axial, coronal and sagittal slices at user-selected indexes."""
    img = nib.as_closest_canonical(_nifti_from_bytes(payload, filename))
    data = img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return [
        _slice_png_at(data, axis, index)
        for axis, index in zip((2, 1, 0), indices)
    ]


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

    original_img = nib.Nifti1Image(original.astype(np.float32), canonical.affine, canonical.header.copy())
    original_img.set_data_dtype(np.float32)
    original_bytes = gzip.compress(original_img.to_bytes())

    normalized_header = canonical.header.copy()
    normalized_header.set_data_dtype(np.float32)
    normalized_img = nib.Nifti1Image(normalized.astype(np.float32), canonical.affine, normalized_header)
    normalized_img.set_data_dtype(np.float32)
    normalized_pre_resize_bytes = gzip.compress(normalized_img.to_bytes())
    normalized_pre_resize_name = (
        filename.removesuffix(".gz").removesuffix(".nii") + "_minmax_pre_resize.nii.gz"
    )
    return {
        "original_shape": tuple(int(v) for v in original.shape),
        "final_shape": TARGET_SHAPE,
        "spacing": tuple(round(float(v), 3) for v in canonical.header.get_zooms()[:3]),
        "orientation": "".join(nib.aff2axcodes(canonical.affine)),
        "original_views": [_slice_png(original, axis) for axis in (2, 1, 0)],
        "processed_views": [_slice_png(resized, axis) for axis in (2, 1, 0)],
        "original_bytes": original_bytes,
        "original_name": filename,
        "processed_bytes": output_bytes,
        "normalized_pre_resize_bytes": normalized_pre_resize_bytes,
        "normalized_pre_resize_name": normalized_pre_resize_name,
        "cam_overlay_bytes": normalized_pre_resize_bytes,
        "cam_reference_bytes": normalized_pre_resize_bytes,
        "cam_inverse_transforms": [],
        "processed_name": filename.removesuffix(".gz").removesuffix(".nii") + "_preprocessed_56.nii.gz",
        "output_bytes": output_bytes,
        "output_name": filename.removesuffix(".gz").removesuffix(".nii") + "_preprocessed_56.nii.gz",
    }
