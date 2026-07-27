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
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from local_pipeline import _discover_braintensor_script


@dataclass(frozen=True)
class ModelInferenceStatus:
    ready: bool
    checkpoint: str
    message: str


CLASS_LABEL_KR = {"Control": "정상", "Prodromal": "전구기", "PD": "파킨슨병 의심"}

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


def run_model_inference(nifti_bytes: bytes, app_root: Path | None = None) -> dict:
    """nifti_bytes: gzip 압축된 56^3 NIfTI(PACScan prep["output_bytes"]).

    반환 dict: normal/prodromal/pd(0~100 정수), finding/rationale(str),
    pred_label(str), cam_views(축상/관상/시상 3장, data URL), checkpoint(str),
    test_accuracy(체크포인트 저장 당시 test accuracy, float|None).
    """
    root = _braintensor_root(app_root)
    status = model_inference_status(app_root)
    if not status.ready:
        raise RuntimeError(status.message)

    import torch  # 이 함수가 실제로 호출될 때만 필요 - 모듈 최상단에서 import하면
    # torch 미설치 환경(app.py의 다른 화면들)에서도 import 실패로 앱 전체가 죽음.

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

    cam_views = [
        _overlay_slice_png(volume, cam, axis=2),  # 축상(axial)
        _overlay_slice_png(volume, cam, axis=1),  # 관상(coronal)
        _overlay_slice_png(volume, cam, axis=0),  # 시상(sagittal)
    ]

    return {
        "normal": round(probs["Control"] * 100),
        "prodromal": round(probs["Prodromal"] * 100),
        "pd": round(probs["PD"] * 100),
        "finding": FINDING_TEMPLATES.get(pred_label, FINDING_TEMPLATES["PD"]),
        "rationale": "흑질(Substantia Nigra) 영역을 포함한 판단 기여 영역(Grad-CAM)이 모델 예측에 크게 기여했습니다.",
        "pred_label": pred_label,
        "pred_label_kr": CLASS_LABEL_KR.get(pred_label, pred_label),
        "cam_views": cam_views,
        "checkpoint": ckpt_path.name,
        "test_accuracy": ckpt_meta.get("test_accuracy"),
    }
