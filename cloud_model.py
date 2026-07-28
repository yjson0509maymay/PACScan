"""Streamlit Cloud용 자체완결 모델 추론 - BRAINTENSOR 저장소 없이도 동작.

[2026-07-27 추가] model_inference.py는 옆(또는 상위) 폴더의 BRAINTENSOR 체크아웃을
찾아 09_Service/inference.py를 그대로 재사용하는 방식인데, Streamlit Community
Cloud에는 PACScan 저장소만 배포되고 BRAINTENSOR 자체가 없음(체크포인트도, 모델
클래스 정의도 없음). 그래서 이 파일은:
  1. CNN3D_Variant3 클래스 정의를 그대로 옮겨옴(BRAINTENSOR의
     02_Model_Definition/ablation_models.py와 동일 - 코드 복제지만, Cloud에는
     BRAINTENSOR 자체가 없어 재사용이 원천적으로 불가능하므로 불가피한 예외).
  2. Grad-CAM(GradCAM3D, 09_Service/gradcam3d.py의 CNN3D_Variant3 대상 부분만)도 함께 옮김.
  3. 체크포인트(148MB, GitHub 100MB 제한 초과)는 Hugging Face Hub에서 실행 시점에
     다운로드(huggingface_hub 캐시로 재실행시 재다운로드 없음).
     저장소: yjson0509maymay/braintensor-variant3 (공개)

model_inference.py의 run_model_inference()가 로컬 BRAINTENSOR 탐색에 실패하면
이 모듈로 자동 폴백함 - app.py는 항상 model_inference.run_model_inference()만
호출하면 되고, 로컬/Cloud 어느 쪽인지는 신경 쓸 필요 없음.
"""

from __future__ import annotations

import base64
import gzip
import io
from dataclasses import dataclass

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import zoom

from xai_rendering import (
    render_cam_overlay_png,
    restrict_cam_to_substantia_nigra,
)

HF_REPO_ID = "yjson0509maymay/braintensor-variant3"
HF_CHECKPOINT_FILENAME = "ablation_variant3_20260726_035445_acc50.0.pt"
# [2026-07-28 추가] 4개 모델 앙상블(CNN+ResNet+CCA(동료 J)+WOA)용 - 같은 HF Hub
# 저장소에 함께 업로드.
HF_RESNET_CHECKPOINT_FILENAME = "resnet3d_20260722_170643_acc54.3.pt"
HF_ENSEMBLE_CCA_FILENAME = "ensemble_J_cca_transformer.joblib"
HF_ENSEMBLE_MASK_FILENAME = "ensemble_J_woa_mask.npy"
HF_ENSEMBLE_CLASSIFIER_FILENAME = "ensemble_J_final_classifier.joblib"

CLASS_LABEL_KR = {"Control": "정상", "Prodromal": "전구기", "PD": "파킨슨병 의심"}
CLASS_NAMES = ["Control", "Prodromal", "PD"]

FINDING_TEMPLATES = {
    "Control": "양측 흑질(Substantia Nigra) 영역을 포함한 기저핵 구조에서 뚜렷한 이상 소견이 관찰되지 않습니다.",
    "Prodromal": "흑질 및 인접 기저핵 영역에서 경미한 신호 변화가 관찰되며, 전구기 파킨슨병 가능성을 배제할 수 없습니다.",
    "PD": "양측 흑질(Substantia Nigra) 영역에서 유의미한 부피 감소 및 신호 변화 소견이 관찰됩니다.",
}


# ============================================================
# CNN3D_Variant3 - BRAINTENSOR의 02_Model_Definition/ablation_models.py와 동일
# (Table3 Study2 그리드서치 확정 구성 - Pooling=Average, Flatten=Global max,
# FC-1/FC-2 병렬 분기, 24-layer)
# ============================================================
class CNN3D_Variant3(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, input_size=56):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm3d(1024)

        self.conv6 = nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm3d(256)

        self.relu = nn.ReLU(inplace=True)

        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        self.flatten = nn.Flatten()
        self.gmp_dim = 256

        self.fc1 = nn.Linear(self.gmp_dim, 1000)
        self.fc2 = nn.Linear(self.gmp_dim, 1000)
        self.classifier = nn.Linear(2000, num_classes)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv2b(x))
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.global_max_pool(x)
        x = self.flatten(x)

        fc1_feat = self.fc1(x)
        fc2_feat = self.fc2(x)
        fv3 = torch.cat([fc1_feat, fc2_feat], dim=1)

        logits = self.classifier(fv3)
        if return_features:
            return logits, {"fc1": fc1_feat, "fc2": fc2_feat, "fv3": fv3}
        return logits


# ============================================================
# ResNet3D - BRAINTENSOR의 02_Model_Definition/final_resnet.py와 동일(Model-2,
# Figure3 사양, 15-layer). [2026-07-28 추가] 4개 모델(CNN+ResNet+CCA+WOA) 앙상블을
# Cloud에서도 쓰기 위해 이식(체크포인트 resnet3d_20260722_170643_acc54.3.pt도
# Hugging Face Hub에 함께 업로드).
# ============================================================
class ResidualUnit3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.unit0 = ResidualUnit3D(in_ch, out_ch, stride=1)
        self.unit1 = ResidualUnit3D(out_ch, out_ch, stride=1)
        self.unit2 = ResidualUnit3D(out_ch, out_ch, stride=1)

    def forward(self, x):
        x = self.unit0(x)
        x = self.unit1(x)
        x = self.unit2(x)
        return x


class ResNet3D(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.block0 = ResidualBlock3D(64, 64)
        self.pool0 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = ResidualBlock3D(64, 128)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block2 = ResidualBlock3D(128, 256)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block3 = ResidualBlock3D(256, 512)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block4 = ResidualBlock3D(512, 512)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.fc4 = nn.Linear(512, 1000)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.pool0(self.block0(x))
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = self.pool4(self.block4(x))
        x = torch.flatten(x, start_dim=1)
        fc4_feat = self.fc4(x)
        logits = self.classifier(self.dropout(fc4_feat))
        if return_features:
            return logits, {"fc4": fc4_feat}
        return logits


# ============================================================
# CCA - 동료 J의 IndependentPCARCCA 구현. cca_ridge_J.py(이 저장소 내 별도 파일,
# 04_Feature_Engineering/cca_ridge_J.py와 동일 내용)에서 그대로 import.
# [중요] 클래스를 여기 인라인으로 재정의하지 않고 반드시 "import"해야 함 - joblib/
# pickle은 클래스를 모듈 경로(예: cca_ridge_J.IndependentPCARCCA)로 참조하므로,
# 저장된 CCA 변환기를 역직렬화하려면 실행 환경에 "cca_ridge_J"라는 이름의 모듈이
# 그대로 있어야 함(클래스 코드만 복사해서 cloud_model 안에 두면 모듈 경로가 달라져
# 역직렬화가 실패함).
# ============================================================
from cca_ridge_J import RidgeCCA, ViewReducer, IndependentPCARCCA  # noqa: F401


def algorithm1_feature_concatenation(x1: np.ndarray, x2: np.ndarray, w1: float = 1.0, w2: float = 1.0) -> np.ndarray:
    """04_Feature_Engineering/extract_features.py의 Algorithm1(FV-3 계산)과 동일 -
    CNN의 FC-1/FC-2에서 논문 Algorithm1(원소별 최댓값 선택)로 FV-3(1000차원)를 만듦."""
    n, d = x1.shape

    def znorm(a):
        mu = a.mean(axis=1, keepdims=True)
        sd = a.std(axis=1, keepdims=True) + 1e-8
        return (a - mu) / sd

    x1n, x2n = znorm(x1), znorm(x2)
    x3 = np.zeros_like(x1)
    for i in range(n):
        used = set()
        for j in range(d):
            wv1, wv2 = w1 * x1n[i, j], w2 * x2n[i, j]
            v1, v2 = x1[i, j], x2[i, j]
            if wv1 > wv2 and v1 not in used:
                x3[i, j] = v1
                used.add(v1)
            elif wv2 >= wv1 and v2 not in used:
                x3[i, j] = v2
                used.add(v2)
            else:
                x3[i, j] = max(v1, v2)
                used.add(x3[i, j])
    return x3


# ============================================================
# Grad-CAM - 09_Service/gradcam3d.py의 CNN3D_Variant3 대상(conv7) 부분만 이식.
# 표준 Grad-CAM(Selvaraju et al., 2017) 구현 - M3d-CAM 패키지 자체는 안 씀(기존
# BRAINTENSOR 쪽 결정과 동일, PACScan UI의 "M3d-CAM" 라벨은 제품 디자인 단계의
# 명칭일 뿐 실제 구현은 항상 Grad-CAM).
# ============================================================
class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._grad_handle = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inputs, output):
        # conv7 뒤에 바로 self.relu(inplace=True)가 적용돼 register_full_backward_hook을
        # 쓰면 "in-place 수정된 view" 오류가 남 - 출력 텐서에 직접 register_hook을
        # 걸어 우회(BRAINTENSOR의 09_Service/gradcam3d.py에서 실측 확인된 방식과 동일).
        self._activations = output
        if self._grad_handle is not None:
            self._grad_handle.remove()
        self._grad_handle = output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self._gradients = grad.detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(logits.argmax(dim=1).item())
        target_idx = pred_idx if class_idx is None else class_idx

        score = logits[:, target_idx]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * self._activations.detach()).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)[0, 0]

        cam_np = cam.cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = cam_np * 0.0

        return cam_np, probs.detach().cpu().numpy()[0], pred_idx

    def close(self):
        self._fwd_handle.remove()
        if self._grad_handle is not None:
            self._grad_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


@dataclass(frozen=True)
class CloudModelStatus:
    ready: bool
    message: str


def cloud_model_status() -> CloudModelStatus:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        return CloudModelStatus(False, "huggingface_hub 미설치")
    return CloudModelStatus(True, "Cloud 모델 추론 준비 완료(Hugging Face Hub)")


_cached_model = None


def _load_model(device: str):
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CHECKPOINT_FILENAME)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CNN3D_Variant3(num_classes=checkpoint.get("num_classes", 3)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _cached_model = (model, checkpoint)
    return _cached_model


def _jet_like(gray: np.ndarray) -> np.ndarray:
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
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return nib.Nifti1Image.from_bytes(raw)


def _resample_binary_mask(mask_bytes: bytes | None, target_img: nib.Nifti1Image) -> np.ndarray | None:
    if not mask_bytes:
        return None
    try:
        mask_img = nib.as_closest_canonical(_nifti_from_payload(mask_bytes))
        mapped = resample_from_to(mask_img, (target_img.shape, target_img.affine), order=0)
        mask = np.asarray(mapped.dataobj, dtype=np.float32) > 0.5
        return mask if int(mask.sum()) >= 2 else None
    except Exception:
        return None

def run_cloud_inference(
    nifti_bytes: bytes,
    overlay_nifti_bytes: bytes | None = None,
    *,
    sn_mask_nifti_bytes: bytes | None = None,
) -> dict:
    """model_inference.run_model_inference()와 동일한 반환 형식.
    overlay_nifti_bytes는 Min-Max 정규화 직후, 56^3 리사이즈 전 NIfTI이다.
    CAM을 NIfTI 물리 좌표 기준으로 이 고해상도 정규화 볼륨에 재표본화한다."""
    status = cloud_model_status()
    if not status.ready:
        raise RuntimeError(status.message)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = _load_model(device)

    raw = gzip.decompress(nifti_bytes) if nifti_bytes[:2] == b"\x1f\x8b" else nifti_bytes
    img = nib.Nifti1Image.from_bytes(raw)
    volume = np.asarray(img.dataobj, dtype=np.float32)

    x = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with GradCAM3D(model, model.conv7) as cam_engine:
        cam, probs, pred_idx = cam_engine(x)

    # conv7 활성화 해상도(14^3)는 원본 볼륨(56^3)보다 작음 - 09_Service/inference.py의
    # predict_with_cam()과 동일하게 scipy.ndimage.zoom으로 볼륨 크기에 맞춰 리사이즈.
    zoom_factors = [s / c for s, c in zip(volume.shape, cam.shape)]
    cam = np.clip(zoom(cam, zoom_factors, order=1), 0.0, 1.0)

    display_volume, display_cam = volume, cam
    display_img = img
    display_spacing = tuple(float(v) for v in img.header.get_zooms()[:3])
    display_space = "model_input_56"
    if overlay_nifti_bytes:
        # Overlay by NIfTI physical coordinates rather than shape-only zoom. This lets the
        # caller pass the actual DICOM-derived original volume safely in the cloud path.
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

    pred_label = CLASS_NAMES[pred_idx]
    raw_display_cam = np.asarray(display_cam, dtype=np.float32)
    supplied_sn_mask = _resample_binary_mask(sn_mask_nifti_bytes, display_img)
    display_cam, sn_roi, focus_meta = restrict_cam_to_substantia_nigra(
        raw_display_cam,
        display_volume,
        voxel_spacing=display_spacing,
        supplied_mask=supplied_sn_mask,
    )
    cam_floor = 0.0
    cam_views = [
        _overlay_slice_png(display_volume, display_cam, axis=2, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),
        _overlay_slice_png(display_volume, display_cam, axis=1, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),
        _overlay_slice_png(display_volume, display_cam, axis=0, cam_floor=cam_floor, voxel_spacing=display_spacing, class_label=pred_label),
    ]

    return {
        "normal": round(float(probs[0]) * 100),
        "prodromal": round(float(probs[1]) * 100),
        "pd": round(float(probs[2]) * 100),
        "finding": FINDING_TEMPLATES.get(pred_label, FINDING_TEMPLATES["PD"]),
        "rationale": "흑질(Substantia Nigra) ROI 내부의 M3d-CAM 기여도를 제한적으로 시각화했습니다.",
        "pred_label": pred_label,
        "pred_label_kr": CLASS_LABEL_KR.get(pred_label, pred_label),
        "cam_views": cam_views,
        "checkpoint": f"{HF_REPO_ID}/{HF_CHECKPOINT_FILENAME} (Hugging Face Hub)",
        "test_accuracy": checkpoint.get("test_accuracy"),
        # [2026-07-28 추가] model_inference.render_cam_overlay()가 재사용 - AI 분석
        # 탭 위치 슬라이더용. local/Cloud 어느 경로든 같은 키 이름으로 반환.
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


_cached_resnet = None
_cached_ensemble_artifacts = None


def _load_resnet_model(device: str):
    global _cached_resnet
    if _cached_resnet is not None:
        return _cached_resnet
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_RESNET_CHECKPOINT_FILENAME)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ResNet3D(num_classes=checkpoint.get("num_classes", 3)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _cached_resnet = (model, checkpoint)
    return _cached_resnet


def _load_ensemble_artifacts():
    """[2026-07-28 추가] 4개 모델 앙상블(CNN+ResNet+CCA(동료 J)+WOA)용 CCA 변환기/
    WOA 마스크/최종 분류기를 Hugging Face Hub에서 다운로드. cca_ridge_J 모듈에서
    import한 클래스로 저장돼 있으므로(cca_ridge_J.py가 이 저장소에도 실제 파일로
    존재), joblib이 이 모듈 경로를 그대로 참조해 정상적으로 역직렬화 가능함."""
    global _cached_ensemble_artifacts
    if _cached_ensemble_artifacts is not None:
        return _cached_ensemble_artifacts
    from huggingface_hub import hf_hub_download
    import joblib

    cca_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_ENSEMBLE_CCA_FILENAME)
    mask_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_ENSEMBLE_MASK_FILENAME)
    clf_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_ENSEMBLE_CLASSIFIER_FILENAME)
    cca = joblib.load(cca_path)
    mask = np.load(mask_path)
    clf = joblib.load(clf_path)
    _cached_ensemble_artifacts = (cca, mask, clf)
    return _cached_ensemble_artifacts


def cloud_ensemble_status() -> CloudModelStatus:
    """4개 모델 앙상블에 필요한 파일들이 HF Hub에서 실제로 받아지는지 확인."""
    try:
        import huggingface_hub  # noqa: F401
        import joblib  # noqa: F401
    except ImportError as exc:
        return CloudModelStatus(False, f"필요 패키지 미설치: {exc}")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HF_REPO_ID, filename=HF_ENSEMBLE_CCA_FILENAME)
        hf_hub_download(repo_id=HF_REPO_ID, filename=HF_RESNET_CHECKPOINT_FILENAME)
    except Exception as exc:
        return CloudModelStatus(False, f"앙상블 아티팩트를 HF Hub에서 찾을 수 없음: {exc}")
    return CloudModelStatus(True, "4개 모델 앙상블 준비 완료(Hugging Face Hub)")


def run_cloud_ensemble_inference(
    nifti_bytes: bytes,
    overlay_nifti_bytes: bytes | None = None,
    *,
    sn_mask_nifti_bytes: bytes | None = None,
) -> dict:
    """[2026-07-28 추가] Cloud에서도 4개 모델(CNN+ResNet+CCA(동료 J)+WOA) 앙상블을
    쓰기 위한 추론. M3d-CAM(흑질 ROI 제한) 시각화는 run_cloud_inference()와 동일하게
    CNN 자체의 CAM을 그대로 쓰고, 최종 확률/예측 클래스만 앙상블 분류기로 교체."""
    cnn_result = run_cloud_inference(
        nifti_bytes, overlay_nifti_bytes, sn_mask_nifti_bytes=sn_mask_nifti_bytes,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_model, _ = _load_model(device)
    resnet_model, _ = _load_resnet_model(device)
    cca, mask, clf = _load_ensemble_artifacts()

    raw = gzip.decompress(nifti_bytes) if nifti_bytes[:2] == b"\x1f\x8b" else nifti_bytes
    img = nib.Nifti1Image.from_bytes(raw)
    volume = np.asarray(img.dataobj, dtype=np.float32)
    x = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        _, cnn_feats = cnn_model(x, return_features=True)
        _, resnet_feats = resnet_model(x, return_features=True)
    fc1 = cnn_feats["fc1"].cpu().numpy()
    fc2 = cnn_feats["fc2"].cpu().numpy()
    fv4 = resnet_feats["fc4"].cpu().numpy()
    fv3 = algorithm1_feature_concatenation(fc1, fc2)

    z = cca.transform_fused(fv3, fv4, n_components=10, fusion="concat")
    probs = clf.predict_proba(z[:, mask])[0]
    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_idx]

    result = dict(cnn_result)
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
        "checkpoint": f"CNN({HF_CHECKPOINT_FILENAME}) + ResNet({HF_RESNET_CHECKPOINT_FILENAME}) + CCA(J) + WOA (Hugging Face Hub)",
        "is_ensemble": True,
    })
    return result
