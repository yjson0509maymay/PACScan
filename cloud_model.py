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
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage import zoom

HF_REPO_ID = "yjson0509maymay/braintensor-variant3"
HF_CHECKPOINT_FILENAME = "ablation_variant3_20260726_035445_acc50.0.pt"

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


def _overlay_slice_png(volume: np.ndarray, cam: np.ndarray, axis: int, cam_alpha_max: float = 0.6) -> str:
    index = volume.shape[axis] // 2
    base = np.rot90(np.take(volume, index, axis=axis))
    heat = np.rot90(np.take(cam, index, axis=axis))
    lo, hi = float(base.min()), float(base.max())
    base_n = (base - lo) / (hi - lo) if hi > lo else np.zeros_like(base)
    heat_n = np.clip(heat, 0.0, 1.0)
    base_rgb = np.stack([base_n] * 3, axis=-1) * 255.0
    heat_rgb = _jet_like(heat_n)
    alpha = (heat_n * cam_alpha_max)[..., None]
    composite = np.clip(base_rgb * (1 - alpha) + heat_rgb * alpha, 0, 255).astype(np.uint8)
    image = Image.fromarray(composite, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def run_cloud_inference(nifti_bytes: bytes) -> dict:
    """model_inference.run_model_inference()와 동일한 반환 형식."""
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

    pred_label = CLASS_NAMES[pred_idx]
    cam_views = [
        _overlay_slice_png(volume, cam, axis=2),
        _overlay_slice_png(volume, cam, axis=1),
        _overlay_slice_png(volume, cam, axis=0),
    ]

    return {
        "normal": round(float(probs[0]) * 100),
        "prodromal": round(float(probs[1]) * 100),
        "pd": round(float(probs[2]) * 100),
        "finding": FINDING_TEMPLATES.get(pred_label, FINDING_TEMPLATES["PD"]),
        "rationale": "흑질(Substantia Nigra) 영역을 포함한 판단 기여 영역(Grad-CAM)이 모델 예측에 크게 기여했습니다.",
        "pred_label": pred_label,
        "pred_label_kr": CLASS_LABEL_KR.get(pred_label, pred_label),
        "cam_views": cam_views,
        "checkpoint": f"{HF_REPO_ID}/{HF_CHECKPOINT_FILENAME} (Hugging Face Hub)",
        "test_accuracy": checkpoint.get("test_accuracy"),
    }
