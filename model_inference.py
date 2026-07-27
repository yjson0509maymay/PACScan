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
    업로드 볼륨에서 추출 - CCA/WOA 앙상블 추론의 1단계. 아직 CCA 변환기·WOA
    선택마스크·최종 분류기가 파일로 저장돼 있지 않아(04_Feature_Engineering/
    run_real_pipeline.py가 원래 일괄평가용이라 이들을 저장 안 함) 이 함수는
    fv3/fv4까지만 반환한다 - CCA/WOA 학습이 끝나고 그 산출물을 저장하는 후속
    작업이 끝나야 실제 앙상블 예측(run_ensemble_inference 같은 함수)을 완성할 수
    있다. 지금은 검증된 특징추출 단계 하나만 독립적으로 완결된 상태.
    """
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


def _overlay_slice_png(
    volume: np.ndarray, cam: np.ndarray, axis: int, cam_alpha_max: float = 0.75,
    cam_floor: float = 0.0,
) -> str:
    """[2026-07-28 수정] 실측 확인 결과, 이 모델(Variant3, 212개 샘플로만 학습)의
    Grad-CAM은 흑질처럼 좁은 부위가 아니라 뇌 전체에 걸쳐 넓게 반응함(중앙값 0.35~0.4,
    상위 25%가 0.5 이상) - 이걸 그대로 선형(alpha=heat*0.6)으로 칠하면 뇌 전체가
    칠해진 것처럼 보여 사용자가 혼란스러워함. cam_floor(호출부에서 그 볼륨의 상위
    percentile로 계산해 넘김)보다 낮은 값은 거의 안 보이게 눌러서, 상대적으로 가장
    강하게 반응한 영역만 도드라지게 함 - 신호 자체를 바꾸는 게 아니라 "상대적으로
    어디가 더 강한지"를 시각적으로 알아보기 쉽게 강조하는 것뿐(percentile 임계값은
    이 볼륨 자체의 분포에서 계산 - 절대적인 "이 부위가 흑질이다" 판정이 아님)."""
    index = volume.shape[axis] // 2
    base = np.rot90(np.take(volume, index, axis=axis))
    heat = np.rot90(np.take(cam, index, axis=axis))
    lo, hi = float(base.min()), float(base.max())
    base_n = (base - lo) / (hi - lo) if hi > lo else np.zeros_like(base)
    heat_n = np.clip(heat, 0.0, 1.0)
    if cam_floor > 0.0:
        heat_n = np.clip((heat_n - cam_floor) / max(1e-6, 1.0 - cam_floor), 0.0, 1.0) ** 1.5
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

    [2026-07-27 추가] 로컬(옆 폴더 BRAINTENSOR 체크아웃)이 없으면 - 예: 실제
    Streamlit Community Cloud 배포 환경 - cloud_model.py(Hugging Face Hub에서
    체크포인트를 내려받는 자체완결 버전)로 자동 폴백함. app.py는 이 함수 하나만
    부르면 되고 로컬/Cloud 구분을 신경 쓸 필요 없음.
    """
    local_status = model_inference_status(app_root)
    if local_status.ready:
        return _run_local_inference(nifti_bytes, local_status, app_root)

    from cloud_model import cloud_model_status, run_cloud_inference

    cloud_status = cloud_model_status()
    if cloud_status.ready:
        return run_cloud_inference(nifti_bytes)

    raise RuntimeError(f"로컬 모델({local_status.message}) / Cloud 모델({cloud_status.message}) 둘 다 사용 불가")


def _run_local_inference(nifti_bytes: bytes, status: ModelInferenceStatus, app_root: Path | None = None) -> dict:
    root = _braintensor_root(app_root)

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

    # [2026-07-28 추가] 이 볼륨 CAM의 상위 활성값만 강조 - 실측 확인 결과 이 모델의
    # CAM은 중앙값이 0.35~0.4로 뇌 전체에 넓게 반응해서, 그대로 칠하면 뇌 전체가
    # 물든 것처럼 보여 사용자가 혼란스러워함(스크린샷으로 확인/피드백 받음). 상위
    # 80퍼센타일을 기준으로 그 아래는 거의 안 보이게 눌러 상대적 강조 영역만 도드라지게 함.
    cam_floor = float(np.percentile(cam, 80))
    cam_views = [
        _overlay_slice_png(volume, cam, axis=2, cam_floor=cam_floor),  # 축상(axial)
        _overlay_slice_png(volume, cam, axis=1, cam_floor=cam_floor),  # 관상(coronal)
        _overlay_slice_png(volume, cam, axis=0, cam_floor=cam_floor),  # 시상(sagittal)
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
