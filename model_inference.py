"""Local bridge from PACScan to the trained BRAINTENSOR Variant3 model.

Mirrors local_pipeline.py's sibling-repo discovery pattern (같은 상위 폴더의
BRAINTENSOR 체크아웃을 찾아 실제 preprocessing 스크립트를 불러오는 방식)와 동일하게,
09_Service/inference.py(체크포인트 로드 + Grad-CAM)를 그대로 재사용한다 - 로직을
PACScan에 복제하지 않고 단일 소스(BRAINTENSOR)를 유지하기 위함.

[전제] 로컬 실행 전용. Streamlit Community Cloud에는 torch가 설치돼 있지 않고
체크포인트(148MB)도 저장소에 없어 이 모듈은 그쪽에서 항상 not-ready로 남는다 -
Cloud 배포는 별도 작업(체크포인트 호스팅 방식 결정 필요, requirements.txt에
torch 추가 필요).
"""

from __future__ import annotations

import base64
import gzip
import importlib.util
import io
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from PIL import Image

from xai_rendering import (
    render_bilateral_cam_insets_png,
    render_cam_overlay_png,
    restrict_cam_to_substantia_nigra,
)

from local_pipeline import _discover_braintensor_script


@dataclass(frozen=True)
class ModelInferenceStatus:
    ready: bool
    checkpoint: str
    message: str


CLASS_LABEL_KR = {"Control": "정상", "Prodromal": "전구기", "PD": "파킨슨병 의심"}
CLASS_NAMES_ENSEMBLE = ["Control", "Prodromal", "PD"]

# [논문 미기재, PACScan 자체 결정] 예측 클래스별 판독 문구 템플릿 - 실제 LLM/RAG
# 서술이 아니라 예측 클래스에 매핑된 고정 문구. 확률·클래스는 실제 모델 출력이고
# 이 문장만 템플릿이라는 점을 UI에서 별도로 명시해야 함.
FINDING_TEMPLATES = {
    "Control": "양측 흑질(Substantia Nigra) 영역을 포함한 기저핵 구조에서 뚜렷한 이상 소견이 관찰되지 않습니다.",
    "Prodromal": "흑질 및 인접 기저핵 영역에서 경미한 신호 변화가 관찰되며, 전구기 파킨슨병 가능성을 배제할 수 없습니다.",
    "PD": "양측 흑질(Substantia Nigra) 영역에서 유의미한 부피 감소 및 신호 변화 소견이 관찰됩니다.",
}


def _braintensor_root(app_root: Path | None = None) -> Path:
    script = _discover_braintensor_script(app_root)
    return script.parents[2]


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def _discover_checkpoint(braintensor_root: Path) -> Path | None:
    """PACSCAN_BRAINTENSOR_CHECKPOINT로 특정 체크포인트를 지정하지 않으면,
    checkpoints/ablation_variant3_*.pt 중 파일명(타임스탬프 포함) 역순 정렬로
    가장 최근(=지금까지 가장 개선된) 것을 자동 선택한다."""
    configured = os.environ.get("PACSCAN_BRAINTENSOR_CHECKPOINT")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    ckpt_dir = braintensor_root / "03_Model_Training" / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    candidates = sorted(ckpt_dir.glob("ablation_variant3_*.pt"), reverse=True)
    return candidates[0] if candidates else None


def model_inference_status(app_root: Path | None = None) -> ModelInferenceStatus:
    root = _braintensor_root(app_root)
    ckpt = _discover_checkpoint(root)
    missing = []
    if not _has_torch():
        missing.append("torch(로컬 파이썬 환경에 설치 필요)")
    if ckpt is None:
        missing.append("Variant3 체크포인트(BRAINTENSOR/03_Model_Training/checkpoints)")
    ready = not missing
    message = "실제 모델 추론 준비 완료" if ready else "필요 환경: " + ", ".join(missing)
    return ModelInferenceStatus(ready, str(ckpt) if ckpt else "", message)


_feature_module = None
_feature_module_root = None
_feature_models = None


def _load_feature_extraction_module(braintensor_root: Path):
    """[2026-07-28 추가] CCA/WOA 앙상블 준비 단계 - 04_Feature_Engineering/
    extract_features.py를 실제 파일 위치 그대로 동적 로드(_load_inference_module과
    동일한 패턴). 이 모듈의 CNN_CKPT/RESNET_CKPT(하드코딩된 특정 체크포인트 파일명)를
    그대로 재사용해야 함 - CCA 변환기가 그 체크포인트들의 FC1/FC2/FC4 특징 분포로
    학습되므로, PACScan 쪽에서 "최신 체크포인트 자동 탐색" 로직(_discover_checkpoint)을
    따로 쓰면 CCA 입력 분포가 달라져 buggy해짐. 반드시 extract_features.py가 가리키는
    바로 그 체크포인트를 써야 함."""
    global _feature_module, _feature_module_root
    if _feature_module is not None and _feature_module_root == braintensor_root:
        return _feature_module
    script = braintensor_root / "04_Feature_Engineering" / "extract_features.py"
    spec = importlib.util.spec_from_file_location("braintensor_extract_features", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"특징 추출 코드를 불러올 수 없습니다: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _feature_module = module
    _feature_module_root = braintensor_root
    return module


def extract_ensemble_features(nifti_bytes: bytes, app_root: Path | None = None) -> dict:
    """[2026-07-28 추가] CNN(Variant3)+ResNet 특징(FV-3/FV-4)을 신규 환자 1명의
    업로드 볼륨에서 추출 - CCA/WOA 앙상블 추론의 1단계."""
    global _feature_models
    root = _braintensor_root(app_root)
    fe = _load_feature_extraction_module(root)
    if _feature_models is None:
        cnn, _ = fe.load_cnn()
        resnet, _ = fe.load_resnet()
        _feature_models = (cnn, resnet)
    cnn, resnet = _feature_models

    raw = gzip.decompress(nifti_bytes) if nifti_bytes[:2] == b"\x1f\x8b" else nifti_bytes
    img = nib.Nifti1Image.from_bytes(raw)
    volume = np.asarray(img.dataobj, dtype=np.float32)

    fv3, fv4 = fe.extract_single(cnn, resnet, volume, device=fe.device)
    return {"fv3": fv3, "fv4": fv4}


_ensemble_artifacts = None


def _load_ensemble_artifacts(braintensor_root: Path):
    """[2026-07-28 추가] 4개 모델(CNN=우리/ResNet=우리/CCA=동료 J/WOA=우리) 앙상블용
    저장된 아티팩트 로드 - 04_Feature_Engineering/build_ensemble_J_cca.py가 만든
    cca_transformer(J의 IndependentPCARCCA)/woa_mask/final_classifier 3종.
    cca_ridge_J 모듈이 sys.path에 있어야 joblib이 IndependentPCARCCA를 역직렬화
    가능(피클은 클래스를 모듈 경로로 참조함)."""
    global _ensemble_artifacts
    if _ensemble_artifacts is not None:
        return _ensemble_artifacts
    import joblib

    fe_dir = braintensor_root / "04_Feature_Engineering"
    if str(fe_dir) not in sys.path:
        sys.path.insert(0, str(fe_dir))
    cca = joblib.load(fe_dir / "ensemble_J_cca_transformer.joblib")
    mask = np.load(fe_dir / "ensemble_J_woa_mask.npy")
    clf = joblib.load(fe_dir / "ensemble_J_final_classifier.joblib")
    _ensemble_artifacts = (cca, mask, clf)
    return _ensemble_artifacts


def ensemble_status(app_root: Path | None = None) -> ModelInferenceStatus:
    """4개 모델 앙상블 아티팩트가 준비돼 있는지 확인(로컬 전용 - 이 파일들은
    BRAINTENSOR 체크아웃에만 있고 git에 커밋되지 않음, Cloud에는 없음)."""
    root = _braintensor_root(app_root)
    fe_dir = root / "04_Feature_Engineering"
    required = ["ensemble_J_cca_transformer.joblib", "ensemble_J_woa_mask.npy", "ensemble_J_final_classifier.joblib"]
    missing = [name for name in required if not (fe_dir / name).is_file()]
    ready = not missing
    message = "앙상블 아티팩트 준비 완료" if ready else f"필요 파일 없음: {', '.join(missing)}"
    return ModelInferenceStatus(ready, str(fe_dir), message)


def run_ensemble_inference(
    nifti_bytes: bytes, app_root: Path | None = None, overlay_nifti_bytes: bytes | None = None,
) -> dict:
    """[2026-07-28 추가] 4개 모델(CNN Variant3=우리, ResNet=우리, CCA=동료 J,
    WOA=우리) 앙상블 최종 추론. 확률/예측 클래스는 이 앙상블 분류기 결과를 쓰고,
    M3d-CAM 시각화는 CNN(Variant3) 자체의 흑질 ROI 제한 CAM을 그대로 사용(앙상블
    분류기 자체는 CCA로 융합된 저차원 특징을 보는 거라 원본 볼륨 공간에 대응되는
    "이 복셀이 중요하다"는 시각화가 없음 - 표준적인 관행대로 시각적 설명은
    CNN 쪽에서, 최종 판단은 앙상블에서 가져오는 구조).

    [로컬 전용] 앙상블 아티팩트가 BRAINTENSOR 체크아웃에만 있고 git에 없어
    Cloud에서는 동작하지 않음 - ensemble_status()로 먼저 확인.
    """
    root = _braintensor_root(app_root)
    cca, mask, clf = _load_ensemble_artifacts(root)

    # CNN(M3d-CAM 시각화용) + fv3/fv4(앙상블 입력용)를 함께 뽑음 - 두 번 추론하지
    # 않도록 extract_ensemble_features()와 _run_local_inference()의 로직을 합침.
    local_status = model_inference_status(app_root)
    if not local_status.ready:
        raise RuntimeError(f"로컬 CNN 모델이 준비되지 않았습니다: {local_status.message}")
    cnn_result = _run_local_inference(nifti_bytes, local_status, app_root, overlay_nifti_bytes)

    features = extract_ensemble_features(nifti_bytes, app_root)
    fv3, fv4 = features["fv3"].reshape(1, -1), features["fv4"].reshape(1, -1)
    z = cca.transform_fused(fv3, fv4, n_components=10, fusion="concat")
    probs = clf.predict_proba(z[:, mask])[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES_ENSEMBLE[pred_idx]

    result = dict(cnn_result)  # cam_views/display_volume 등 시각화 관련 필드는 CNN 결과 그대로 재사용
    if pred_label != cnn_result.get("pred_label"):
        # 앙상블 최종 예측이 CNN 단독 예측과 다르면, 클래스별 색상 팔레트가 실제
        # 표시되는 라벨과 어긋나지 않도록 같은 CAM/ROI로 다시 렌더링한다.
        spacing = tuple(result.get("display_spacing", (1.0, 1.0, 1.0)))
        floor = result.get("cam_floor", 0.0)
        result["cam_views"] = [
            _overlay_slice_png(result["display_volume"], result["display_cam"], axis=2, cam_floor=floor, voxel_spacing=spacing, class_label=pred_label),
            _overlay_slice_png(result["display_volume"], result["display_cam"], axis=1, cam_floor=floor, voxel_spacing=spacing, class_label=pred_label),
            _overlay_slice_png(result["display_volume"], result["display_cam"], axis=0, cam_floor=floor, voxel_spacing=spacing, class_label=pred_label),
        ]
    result.update({
        "normal": round(float(probs[0]) * 100),
        "prodromal": round(float(probs[1]) * 100),
        "pd": round(float(probs[2]) * 100),
        "pred_label": pred_label,
        "pred_label_kr": CLASS_LABEL_KR.get(pred_label, pred_label),
        "finding": FINDING_TEMPLATES.get(pred_label, FINDING_TEMPLATES["PD"]),
        "checkpoint": f"CNN({cnn_result['checkpoint']}) + ResNet(resnet3d_20260722_170643_acc54.3.pt) + CCA(J) + WOA",
        "is_ensemble": True,
    })
    return result


_inference_module = None
_inference_module_root = None


def _load_inference_module(braintensor_root: Path):
    """09_Service/inference.py를 실제 파일 위치 그대로 동적 로드 - 그 파일 내부의
    sys.path 설정(02_Model_Definition/03_Model_Training/09_Service)이 자기
    __file__ 기준으로 계산되므로, BRAINTENSOR 체크아웃 안에서 그대로 잘 동작한다."""
    global _inference_module, _inference_module_root
    if _inference_module is not None and _inference_module_root == braintensor_root:
        return _inference_module
    script = braintensor_root / "09_Service" / "inference.py"
    spec = importlib.util.spec_from_file_location("braintensor_inference", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"추론 코드를 불러올 수 없습니다: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _inference_module = module
    _inference_module_root = braintensor_root
    return module


def _jet_like(gray: np.ndarray) -> np.ndarray:
    """09_Service/app.py의 컬러맵과 동일 - blue->cyan->yellow->red 근사."""
    gray = np.clip(gray, 0.0, 1.0)
    stops_t = np.array([0.0, 0.33, 0.66, 1.0])
    stops_c = np.array(
        [[30, 60, 220], [0, 200, 200], [255, 220, 0], [220, 30, 30]], dtype=np.float32
    )
    r = np.interp(gray, stops_t, stops_c[:, 0])
    g = np.interp(gray, stops_t, stops_c[:, 1])
    b = np.interp(gray, stops_t, stops_c[:, 2])
    return np.stack([r, g, b], axis=-1)


def _display_window_mri(plane: np.ndarray) -> np.ndarray:
    """MRI display-only robust windowing.

    This changes only the PNG visualization, never the tensor used for inference.
    Percentile windowing matches the original-MRI viewer much better than raw min/max,
    which was being dominated by a few very bright skull/background pixels and made the
    brain parenchyma look washed out / blurry.
    """
    plane = np.asarray(plane, dtype=np.float32)
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros_like(plane, dtype=np.float32)

    # Prefer non-background voxels when enough are available. This keeps the black
    # background black while using the useful MRI intensity range for contrast.
    nonzero = finite[np.abs(finite) > 1e-8]
    sample = nonzero if nonzero.size >= 128 else finite
    lo, hi = np.percentile(sample, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(plane, dtype=np.float32)

    windowed = np.clip((plane - lo) / (hi - lo), 0.0, 1.0)
    # Very light display gamma only; no spatial sharpening/filtering is applied.
    # Keeping spatial pixels untouched is important for a medical-image overlay.
    return np.power(windowed, 0.92).astype(np.float32)


def _overlay_slice_png(
    volume: np.ndarray, cam: np.ndarray, axis: int, cam_alpha_max: float = 0.74,
    cam_floor: float = 0.0, index: int | None = None,
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    class_label: str | None = None,
) -> str:
    return render_cam_overlay_png(
        volume, cam, axis, index=index, voxel_spacing=voxel_spacing,
        class_label=class_label, cam_alpha_max=cam_alpha_max, cam_floor=cam_floor,
    )


def _nifti_from_payload(payload: bytes) -> nib.Nifti1Image:
    """Read .nii/.nii.gz bytes without relying on a filename."""
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return nib.Nifti1Image.from_bytes(raw)


def _canonical_volume(payload: bytes) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.as_closest_canonical(_nifti_from_payload(payload))
    volume = np.asarray(img.dataobj, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    return img, volume


def _resample_cam_by_affine(
    cam: np.ndarray,
    source_nifti_bytes: bytes,
    target_img: nib.Nifti1Image,
) -> np.ndarray:
    """Project model-space CAM to another NIfTI grid using physical coordinates."""
    source_img = _nifti_from_payload(source_nifti_bytes)
    cam_img = nib.Nifti1Image(np.asarray(cam, dtype=np.float32), source_img.affine, source_img.header.copy())
    mapped = resample_from_to(cam_img, (target_img.shape, target_img.affine), order=1)
    return np.clip(np.asarray(mapped.dataobj, dtype=np.float32), 0.0, 1.0)



def _discover_sn_mask_bytes(app_root: Path | None = None) -> tuple[bytes | None, str]:
    """Load an optional SN mask NIfTI for visualization.

    Priority: PACSCAN_SN_MASK environment variable, then assets/substantia_nigra_mask.nii.gz
    or assets/pd25_substantia_nigra_mask.nii.gz. The mask must be in the same physical
    coordinate system as the Min-Max pre-resize volume; otherwise omit it and use the
    clearly labelled estimated midbrain ROI fallback.
    """
    candidates: list[Path] = []
    configured = os.environ.get("PACSCAN_SN_MASK")
    if configured:
        candidates.append(Path(configured).expanduser())
    root = (app_root or Path(__file__).resolve().parent).resolve()
    candidates.extend([
        root / "assets" / "substantia_nigra_mask.nii.gz",
        root / "assets" / "pd25_substantia_nigra_mask.nii.gz",
        root / "assets" / "substantia_nigra_mask.nii",
    ])
    for path in candidates:
        if path.is_file():
            return path.read_bytes(), str(path)
    return None, ""


def _resample_binary_mask(mask_bytes: bytes | None, target_img: nib.Nifti1Image) -> np.ndarray | None:
    if not mask_bytes:
        return None
    try:
        mask_img = nib.as_closest_canonical(_nifti_from_payload(mask_bytes))
        mapped = resample_from_to(mask_img, (target_img.shape, target_img.affine), order=0)
        mask = np.asarray(mapped.dataobj, dtype=np.float32) > 0.5
        if int(mask.sum()) < 2:
            return None
        return mask
    except Exception:
        return None

def _apply_antspy_inverse_transforms(
    cam: np.ndarray,
    source_nifti_bytes: bytes,
    original_img: nib.Nifti1Image,
    inverse_transform_paths: list[str],
) -> np.ndarray:
    """Map atlas/preprocessed CAM back to the original DICOM-derived NIfTI space."""
    import ants

    source_img = _nifti_from_payload(source_nifti_bytes)
    cam_img = nib.Nifti1Image(np.asarray(cam, dtype=np.float32), source_img.affine, source_img.header.copy())
    with tempfile.TemporaryDirectory(prefix="pacscan_cam_inverse_") as temp_dir:
        temp = Path(temp_dir)
        fixed_path = temp / "original_canonical.nii.gz"
        cam_path = temp / "cam_model_space.nii.gz"
        nib.save(original_img, str(fixed_path))
        nib.save(cam_img, str(cam_path))
        fixed = ants.image_read(str(fixed_path))
        moving_cam = ants.image_read(str(cam_path))
        mapped = ants.apply_transforms(
            fixed=fixed,
            moving=moving_cam,
            transformlist=[str(Path(path)) for path in inverse_transform_paths],
            interpolator="linear",
        )
        mapped_arr = np.asarray(mapped.numpy(), dtype=np.float32)
    if mapped_arr.shape != original_img.shape:
        raise RuntimeError(
            f"역정합 CAM shape 불일치: mapped={mapped_arr.shape}, original={original_img.shape}"
        )
    mapped_arr = np.clip(mapped_arr, 0.0, 1.0)
    if not np.isfinite(mapped_arr).any() or float(mapped_arr.max()) <= 1e-7:
        raise RuntimeError("역정합 후 CAM이 비어 있습니다.")
    return mapped_arr


def _register_reference_back_to_original(
    cam: np.ndarray,
    source_nifti_bytes: bytes,
    reference_nifti_bytes: bytes,
    original_img: nib.Nifti1Image,
) -> np.ndarray:
    """Fallback for local full preprocessing when saved inverse transforms are unavailable.

    The reference volume is the registration-space, pre-resize image. We first resample the
    56^3 CAM to that grid and then estimate an affine reference->original transform with ANTs.
    """
    import ants

    reference_img = _nifti_from_payload(reference_nifti_bytes)
    cam_reference = _resample_cam_by_affine(cam, source_nifti_bytes, reference_img)
    cam_reference_img = nib.Nifti1Image(
        cam_reference.astype(np.float32), reference_img.affine, reference_img.header.copy()
    )
    with tempfile.TemporaryDirectory(prefix="pacscan_cam_reregister_") as temp_dir:
        temp = Path(temp_dir)
        fixed_path = temp / "original_canonical.nii.gz"
        moving_path = temp / "registration_reference.nii.gz"
        cam_path = temp / "cam_reference.nii.gz"
        nib.save(original_img, str(fixed_path))
        nib.save(reference_img, str(moving_path))
        nib.save(cam_reference_img, str(cam_path))
        fixed = ants.image_read(str(fixed_path))
        moving = ants.image_read(str(moving_path))
        moving_cam = ants.image_read(str(cam_path))
        try:
            reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform="AffineFast",
                random_seed=42,
                verbose=False,
            )
        except Exception:
            reg = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform="Affine",
                random_seed=42,
                verbose=False,
            )
        mapped = ants.apply_transforms(
            fixed=fixed,
            moving=moving_cam,
            transformlist=reg["fwdtransforms"],
            interpolator="linear",
        )
        mapped_arr = np.asarray(mapped.numpy(), dtype=np.float32)
    mapped_arr = np.clip(mapped_arr, 0.0, 1.0)
    if mapped_arr.shape != original_img.shape or float(mapped_arr.max()) <= 1e-7:
        raise RuntimeError("원본 공간 재정합 CAM 생성에 실패했습니다.")
    return mapped_arr


def _attach_original_dicom_overlay(
    model_result: dict,
    source_nifti_bytes: bytes,
    original_nifti_bytes: bytes | None,
    inverse_transform_paths: list[str] | None = None,
    reference_nifti_bytes: bytes | None = None,
) -> dict:
    """Replace the visualization background with the original DICOM-derived volume.

    Prediction still uses the preprocessed 56^3 input. Only the visualization is projected
    back to the original acquisition space.
    """
    if not original_nifti_bytes:
        return model_result

    original_img, original_volume = _canonical_volume(original_nifti_bytes)
    model_cam = np.asarray(model_result.get("display_cam_raw", model_result["display_cam"]), dtype=np.float32)
    projection_method = "nifti_affine"
    projection_warning = ""

    if inverse_transform_paths:
        try:
            original_cam = _apply_antspy_inverse_transforms(
                model_cam, source_nifti_bytes, original_img, inverse_transform_paths
            )
            projection_method = "antspy_saved_inverse"
        except Exception as exc:
            projection_warning = f"저장된 ANTs 역변환 적용 실패: {exc}"
            if reference_nifti_bytes:
                original_cam = _register_reference_back_to_original(
                    model_cam, source_nifti_bytes, reference_nifti_bytes, original_img
                )
                projection_method = "antspy_reregister_fallback"
            else:
                raise
    elif reference_nifti_bytes:
        # Local full preprocessing changed the coordinate system, so affine-only resize is unsafe.
        original_cam = _register_reference_back_to_original(
            model_cam, source_nifti_bytes, reference_nifti_bytes, original_img
        )
        projection_method = "antspy_reregister"
    else:
        # Cloud lightweight preprocessing only canonicalizes/resizes; physical coordinates are preserved.
        original_cam = _resample_cam_by_affine(model_cam, source_nifti_bytes, original_img)

    display_spacing = tuple(float(v) for v in original_img.header.get_zooms()[:3])
    focused_cam, sn_roi, focus_meta = restrict_cam_to_substantia_nigra(
        original_cam, original_volume, voxel_spacing=display_spacing
    )
    cam_floor = 0.0
    pred_label = model_result.get("pred_label")
    cam_views = [
        _overlay_slice_png(original_volume, focused_cam, axis=2, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),
        _overlay_slice_png(original_volume, focused_cam, axis=1, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),
        _overlay_slice_png(original_volume, focused_cam, axis=0, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),
    ]
    model_result.update(
        display_volume=original_volume,
        display_cam=focused_cam,
        display_cam_raw=original_cam,
        display_roi=sn_roi,
        display_spacing=display_spacing,
        cam_focus=focus_meta,
        cam_floor=cam_floor,
        cam_views=cam_views,
        display_space="original_dicom",
        projection_method=projection_method,
        projection_warning=projection_warning,
        cam_render_mode="sn_roi_restricted_spacing_v6",
    )
    return model_result


def _run_local_inference(
    nifti_bytes: bytes, status: ModelInferenceStatus, app_root: Path | None = None,
    overlay_nifti_bytes: bytes | None = None,
) -> dict:
    root = _braintensor_root(app_root)

    import torch  # 이 함수가 실제로 호출될 때만 필요 - 모듈 최상단에서 import하면
    # torch 미설치 환경(app.py의 다른 화면들)에서도 import 실패로 앱 전체가 죽음.
    from scipy.ndimage import zoom as _zoom

    inf = _load_inference_module(root)
    ckpt_path = Path(status.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt_meta = inf.load_model(ckpt_path.name, device=device)

    raw = gzip.decompress(nifti_bytes) if nifti_bytes[:2] == b"\x1f\x8b" else nifti_bytes
    img = nib.Nifti1Image.from_bytes(raw)
    volume = np.asarray(img.dataobj, dtype=np.float32)

    prediction = inf.predict_with_cam(model, volume, device=device)
    probs = prediction["probs"]
    cam = prediction["cam"]
    pred_label = prediction["pred_label"]

    # XAI background is the Min-Max-normalized volume immediately before resize.
    # Resample CAM by NIfTI physical coordinates, not only by array shape, so the 56^3 CAM
    # stays aligned with the higher-resolution pre-resize volume.
    display_volume = volume
    display_cam = cam
    display_img = img
    display_spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    display_space = "model_input_56"
    if overlay_nifti_bytes:
        overlay_raw = gzip.decompress(overlay_nifti_bytes) if overlay_nifti_bytes[:2] == b"\x1f\x8b" else overlay_nifti_bytes
        overlay_img = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(overlay_raw))
        overlay_volume = np.asarray(overlay_img.dataobj, dtype=np.float32)
        overlay_volume = np.nan_to_num(overlay_volume, nan=0.0, posinf=0.0, neginf=0.0)

        cam_img = nib.Nifti1Image(cam.astype(np.float32), img.affine, img.header.copy())
        mapped_cam = resample_from_to(cam_img, (overlay_img.shape, overlay_img.affine), order=1)
        display_cam = np.clip(np.asarray(mapped_cam.dataobj, dtype=np.float32), 0.0, 1.0)
        display_volume = overlay_volume
        display_img = overlay_img
        display_spacing = tuple(float(v) for v in overlay_img.header.get_zooms()[:3])
        display_space = "minmax_pre_resize"

    # [2026-07-28 추가] 이 볼륨 CAM의 상위 활성값만 강조 - 실측 확인 결과 이 모델의
    # CAM은 중앙값이 0.35~0.4로 뇌 전체에 넓게 반응해서, 그대로 칠하면 뇌 전체가
    # 물든 것처럼 보여 사용자가 혼란스러워함(스크린샷으로 확인/피드백 받음). 상위
    # 80퍼센타일을 기준으로 그 아래는 거의 안 보이게 눌러 상대적 강조 영역만 도드라지게 함.
    raw_display_cam = np.asarray(display_cam, dtype=np.float32)
    sn_mask_bytes, sn_mask_path = _discover_sn_mask_bytes(app_root)
    supplied_sn_mask = _resample_binary_mask(sn_mask_bytes, display_img)
    display_cam, sn_roi, focus_meta = restrict_cam_to_substantia_nigra(
        raw_display_cam,
        display_volume,
        voxel_spacing=display_spacing,
        supplied_mask=supplied_sn_mask,
    )
    focus_meta["mask_path"] = sn_mask_path if supplied_sn_mask is not None else ""
    cam_floor = 0.0
    cam_views = [
        _overlay_slice_png(display_volume, display_cam, axis=2, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),  # 축상(axial)
        _overlay_slice_png(display_volume, display_cam, axis=1, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),  # 관상(coronal)
        _overlay_slice_png(display_volume, display_cam, axis=0, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),  # 시상(sagittal)
    ]

    return {
        "normal": round(probs["Control"] * 100),
        "prodromal": round(probs["Prodromal"] * 100),
        "pd": round(probs["PD"] * 100),
        "finding": FINDING_TEMPLATES.get(pred_label, FINDING_TEMPLATES["PD"]),
        "rationale": "흑질(Substantia Nigra) ROI 내부의 M3d-CAM 기여도를 제한적으로 시각화했습니다.",
        "pred_label": pred_label,
        "pred_label_kr": CLASS_LABEL_KR.get(pred_label, pred_label),
        "cam_views": cam_views,
        "checkpoint": ckpt_path.name,
        "test_accuracy": ckpt_meta.get("test_accuracy"),
        # [2026-07-28 추가] AI 분석 탭에 위치/확대 슬라이더를 붙이기 위해 원본
        # 배열도 함께 반환 - render_cam_overlay()가 이걸로 다른 슬라이스를 다시 그림.
        "display_volume": display_volume,
        "display_cam": display_cam,
        "display_cam_raw": raw_display_cam,
        "display_roi": sn_roi,
        "display_spacing": display_spacing,
        "cam_focus": focus_meta,
        "cam_floor": cam_floor,
        "display_space": display_space,
        "projection_method": "nifti_affine_to_minmax_pre_resize" if overlay_nifti_bytes else "none",
        "cam_render_mode": "sn_roi_restricted_spacing_v6",
        "is_ensemble": False,
    }


def run_model_inference(
    nifti_bytes: bytes,
    app_root: Path | None = None,
    overlay_nifti_bytes: bytes | None = None,
    *,
    original_nifti_bytes: bytes | None = None,
    inverse_transform_paths: list[str] | None = None,
    reference_nifti_bytes: bytes | None = None,
) -> dict:
    """Run the trained model (or the 4-model ensemble, when its artifacts are
    available) and render CAM on the requested visualization volume.

    nifti_bytes:
        Preprocessed 56^3 model input.
    original_nifti_bytes:
        DICOM->NIfTI volume before BET/registration/normalization/resizing. When provided,
        CAM is projected back to this acquisition space for visualization.
    inverse_transform_paths:
        ANTs inverse transforms captured during the local full preprocessing registration.
    reference_nifti_bytes:
        Registration-space, pre-resize volume used only as a local fallback when exact saved
        inverse transforms are unavailable.
    overlay_nifti_bytes:
        Preferred visualization background: Min-Max-normalized NIfTI immediately before
        the final 56x56x56 resize. CAM is resampled into this volume's physical space.

    [2026-07-28 추가] 로컬/Cloud 모두 4개 모델(CNN+ResNet+CCA(동료 J)+WOA) 앙상블
    아티팩트가 준비돼 있으면 그걸 우선 사용한다 - app.py는 이 함수 하나만 부르면
    되고 앙상블/단독, 로컬/Cloud 어느 조합인지 신경 쓸 필요 없음(is_ensemble
    플래그로 실제 뭐가 돌았는지만 결과에서 확인).
    """
    local_status = model_inference_status(app_root)
    if local_status.ready:
        use_ensemble = ensemble_status(app_root).ready
        if use_ensemble:
            result = run_ensemble_inference(nifti_bytes, app_root, overlay_nifti_bytes=None)
        else:
            result = _run_local_inference(nifti_bytes, local_status, app_root, overlay_nifti_bytes=None)
        if original_nifti_bytes:
            return _attach_original_dicom_overlay(
                result,
                nifti_bytes,
                original_nifti_bytes,
                inverse_transform_paths=inverse_transform_paths,
                reference_nifti_bytes=reference_nifti_bytes,
            )
        if overlay_nifti_bytes:
            if use_ensemble:
                return run_ensemble_inference(nifti_bytes, app_root, overlay_nifti_bytes=overlay_nifti_bytes)
            return _run_local_inference(nifti_bytes, local_status, app_root, overlay_nifti_bytes)
        return result

    from cloud_model import cloud_model_status, cloud_ensemble_status, run_cloud_inference, run_cloud_ensemble_inference

    cloud_status = cloud_model_status()
    if cloud_status.ready:
        sn_mask_bytes, _sn_mask_path = _discover_sn_mask_bytes(app_root)
        use_cloud_ensemble = cloud_ensemble_status().ready
        if use_cloud_ensemble:
            result = run_cloud_ensemble_inference(nifti_bytes, None, sn_mask_nifti_bytes=sn_mask_bytes)
        else:
            result = run_cloud_inference(nifti_bytes, None, sn_mask_nifti_bytes=sn_mask_bytes)
        if original_nifti_bytes:
            return _attach_original_dicom_overlay(
                result,
                nifti_bytes,
                original_nifti_bytes,
                inverse_transform_paths=None,
                reference_nifti_bytes=None,
            )
        if overlay_nifti_bytes:
            if use_cloud_ensemble:
                return run_cloud_ensemble_inference(nifti_bytes, overlay_nifti_bytes, sn_mask_nifti_bytes=sn_mask_bytes)
            return run_cloud_inference(nifti_bytes, overlay_nifti_bytes, sn_mask_nifti_bytes=sn_mask_bytes)
        return result

    raise RuntimeError(f"로컬 모델({local_status.message}) / Cloud 모델({cloud_status.message}) 둘 다 사용 불가")



def render_cam_insets(model_result: dict) -> dict:
    """Return left/right substantia nigra inset images for the current model_result."""
    return render_bilateral_cam_insets_png(
        np.asarray(model_result["display_volume"], dtype=np.float32),
        np.asarray(model_result["display_cam"], dtype=np.float32),
        class_label=model_result.get("pred_label"),
    )

def render_cam_overlay(model_result: dict, axis: int, index: int) -> str:
    """[2026-07-28 추가] AI 분석 탭의 위치 슬라이더용 - run_model_inference()가
    반환한 display_volume/display_cam으로 임의 슬라이스를 다시 렌더링한다.
    로컬/Cloud 어느 경로든 이 함수(model_inference.py의 _overlay_slice_png) 하나로
    처리 가능 - 렌더링 로직 자체는 두 경로가 동일해서 cloud_model을 따로 부를 필요 없음."""
    return _overlay_slice_png(
        model_result["display_volume"], model_result["display_cam"], axis,
        cam_floor=model_result.get("cam_floor", 0.0), index=index,
        voxel_spacing=tuple(model_result.get("display_spacing", (1.0, 1.0, 1.0))),
        class_label=model_result.get("pred_label"),
    )
