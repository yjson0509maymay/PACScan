from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import streamlit as st


ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"


@dataclass(frozen=True)
class Result:
    normal: int = 18
    prodromal: int = 29
    pd: int = 88
    finding: str = "양측 흑질(Substantia Nigra) 영역에서 유의미한 부피 감소 소견이 관찰됩니다."
    rationale: str = (
        "Mid-CAM 모델 분석 결과, 양측 흑질(Substantia Nigra) 영역의 도파민 신경세포 "
        "소실 패턴이 뚜렷하여 파킨슨병 환자의 특징적 해부학적 패턴이 관찰되었습니다."
    )


def data_url(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".") or "png"
    return f"data:image/{ext};base64,{base64.b64encode(path.read_bytes()).decode()}"


def upload_data_url(upload: BinaryIO) -> str:
    mime = getattr(upload, "type", "image/png")
    return f"data:{mime};base64,{base64.b64encode(upload.getvalue()).decode()}"


def infer(_: BinaryIO) -> Result:
    """추후 BRAINTENSOR 추론 함수를 이 어댑터에 연결한다."""
    return Result()


def css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root{--bg:#020a14;--nav:#061426;--card:#0a1728;--card2:#0d1c2f;--line:#1b3048;--blue:#208bff;--muted:#7f93aa;--text:#edf5ff}
*{font-family:Inter,'Noto Sans KR',sans-serif;box-sizing:border-box}
.stApp{background:linear-gradient(145deg,#020812,#061120 52%,#020914);color:var(--text)}
[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"]{display:none!important}
.block-container{max-width:1600px;padding:.55rem .7rem 2rem}
.topbar{height:68px;display:flex;align-items:center;gap:35px;padding:0 20px;background:linear-gradient(90deg,#06162a,#031020);border:1px solid #122841;border-radius:8px;margin-bottom:10px}
.brand{display:flex;align-items:center;gap:10px;font-size:21px;font-weight:800;white-space:nowrap}.brand img{width:106px;max-height:48px;object-fit:contain}.brand b{color:var(--blue)}
.tabs{display:flex;align-self:stretch;gap:28px}.tab{display:flex;align-items:center;padding:0 5px;font-size:16px;font-weight:650;color:#d4e0ed;border-bottom:3px solid transparent}.tab.active{border-color:#228eff}.topmeta{margin-left:auto;text-align:right;font-size:10px;color:#b6c3d2;line-height:1.45;white-space:nowrap}
.panel{background:linear-gradient(145deg,rgba(12,27,47,.98),rgba(5,15,28,.98));border:1px solid var(--line);border-radius:7px;overflow:hidden;box-shadow:0 8px 22px rgba(0,0,0,.14)}
.head{height:44px;padding:0 14px;display:flex;align-items:center;gap:9px;font-weight:700;font-size:15px;border-bottom:1px solid var(--line)}.head .i{color:#2194ff;font-size:19px}
.boxpad{padding:13px}.nav-title{color:#8da0b6;font-size:10px;font-weight:700;letter-spacing:.08em;margin:4px 8px 8px}
.side-item{height:48px;display:flex;align-items:center;gap:10px;padding:0 12px;margin:5px 0;border-radius:5px;font-size:14px;font-weight:650;color:#d4deea}.side-item.active{background:linear-gradient(90deg,#174bb0,#1c61d7);color:#fff}.side-item span{font-size:17px}.side-space{height:115px}
.upload-help{font-size:11px;color:#7f93aa;line-height:1.55;margin:8px 2px 4px}.empty{min-height:520px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;background:radial-gradient(circle,#0a1b30,#020812 60%);color:#71859c}.empty .brain{font-size:55px;filter:grayscale(1);opacity:.55}.empty b{color:#b7c8da;font-size:16px;margin:12px 0 5px}.empty small{line-height:1.7}
.viewer-grid{display:grid;grid-template-columns:minmax(0,4fr) minmax(150px,1.18fr);min-height:460px;background:#01060c}.mainview{position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden;border-right:1px solid #263b53}.mainview img{width:100%;height:100%;max-height:520px;object-fit:contain}.subviews{display:grid;grid-template-rows:1fr 1fr}.subview{position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}.subview:first-child{border-bottom:1px solid #263b53}.subview img{width:100%;height:100%;object-fit:cover}.viewtag{position:absolute;top:8px;left:9px;background:rgba(1,7,14,.72);padding:3px 7px;border-radius:3px;font-weight:700;font-size:11px}.slice{position:absolute;top:12px;left:14px;color:#fff;font-size:11px;line-height:1.6}.slice b{display:block;color:#2496ff}.cross-x,.cross-y{position:absolute;background:rgba(30,133,255,.45);pointer-events:none}.cross-x{height:1px;width:100%;top:50%}.cross-y{width:1px;height:100%;left:50%}
.pinfo,.model{display:grid;grid-template-columns:1fr auto;gap:10px 8px;font-size:10px}.pinfo dt,.model dt{color:#8194aa}.pinfo dd,.model dd{margin:0;color:#e2ebf5;text-align:right}.divider{grid-column:1/-1;height:1px;background:#1e3045;margin:3px 0}
.prob{margin:10px 0 14px}.probline{display:flex;justify-content:space-between;font-size:10px;margin-bottom:6px}.track{height:5px;border-radius:20px;background:#1c2b3d;overflow:hidden}.fill{height:100%;border-radius:20px}.reason{margin-top:12px;padding:10px;border:1px solid #293c54;border-radius:5px;background:rgba(255,255,255,.025);font-size:9px;line-height:1.6;color:#b3c0d0}.reason b{display:block;color:#d8e4f0;font-size:10px;margin-bottom:4px}
.finding{font-size:15px;padding:12px 14px;background:#061325;line-height:1.55}.warning{display:flex;gap:12px;padding:12px 14px;background:linear-gradient(90deg,#2b0f17,#1b0b12);border-top:1px solid #5a202a;color:#e5dce0;font-size:11px;line-height:1.7}.warning .warnicon{color:#ff3b49;font-size:19px}.warning strong{display:block;color:#ffe000}.status{text-align:right;color:#2bdbb2;font-size:9px;margin:9px 3px}
[data-testid="stFileUploader"]{background:#071628;border:1px dashed #2765a5;border-radius:7px;padding:4px}[data-testid="stFileUploader"] section{padding:10px!important}[data-testid="stFileUploader"] small{display:none}.stDownloadButton button{width:100%;background:#0c315e;border:1px solid #216db8;color:#e7f3ff;font-size:11px}.stButton button{width:100%;background:#0d2441;border:1px solid #27517f;color:#dbeafa}.stProgress>div>div{background:#218cff}.stSlider{padding:0 12px 3px;background:#04101d}
@media(max-width:900px){.tabs{gap:7px}.tab{font-size:11px}.topmeta{display:none}.viewer-grid{grid-template-columns:1fr}.subviews{grid-template-columns:1fr 1fr;grid-template-rows:180px}.side-space{height:0}}
</style>
""",
        unsafe_allow_html=True,
    )


def panel(title: str, content: str, icon: str = "▣", extra: str = "") -> None:
    st.markdown(f'<section class="panel" {extra}><div class="head"><span class="i">{icon}</span>{title}</div><div class="boxpad">{content}</div></section>', unsafe_allow_html=True)


st.set_page_config(page_title="NeuroLens AI 분석", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
css()

logo = data_url(ASSETS / "logo.png")
st.markdown(
    f'<header class="topbar"><div class="brand"><img src="{logo}"><span><b>MRI</b> 분석 대시보드</span></div>'
    '<nav class="tabs"><span class="tab active">뉴로렌즈(AI) 분석 결과</span><span class="tab">환자관리</span></nav>'
    '<div class="topmeta">Patient ID. PT-2026-0477<br>Scan Date. 2024.03.28</div></header>',
    unsafe_allow_html=True,
)

nav_col, center_col, info_col = st.columns([.78, 5.15, 1.45], gap="small")

with nav_col:
    st.markdown(
        '<div class="panel boxpad"><div class="nav-title">NEUROLENS</div>'
        '<div class="side-item">▣　진단뷰어</div><div class="side-item active">▤　분석도구</div>'
        '<div class="side-item">▧　임상노트</div><div class="side-item">⚙　설정</div><div class="side-space"></div></div>',
        unsafe_allow_html=True,
    )
    sample_bytes = (ASSETS / "sample_t2_mri.png").read_bytes()
    st.download_button("↓ 예시 MRI 받기", sample_bytes, "NeuroLens_sample_T2_MRI.png", "image/png", use_container_width=True)
    uploaded = st.file_uploader("뇌-MRI(T2) 업로드", type=["png", "jpg", "jpeg"], help="예시 MRI 파일을 내려받은 후 업로드해 보세요.")
    st.markdown('<div class="upload-help">지원 형식: PNG, JPG<br>예시 파일 업로드 시 AI 결과 화면이 활성화됩니다.</div>', unsafe_allow_html=True)

if uploaded is not None and st.session_state.get("analyzed_file") != uploaded.name:
    with center_col:
        with st.status("MRI를 분석하고 있습니다…", expanded=True) as status:
            st.write("영상 무결성 확인")
            time.sleep(.35)
            st.write("3D 특징 및 흑질 영역 분석")
            time.sleep(.35)
            st.write("설명가능성 맵 생성")
            status.update(label="분석이 완료되었습니다.", state="complete", expanded=False)
    st.session_state.analyzed_file = uploaded.name

ready = uploaded is not None
result = infer(uploaded) if ready else None

with center_col:
    st.markdown('<section class="panel"><div class="head"><span class="i">◈</span>Brain MRI　뇌 3D-MRI 병변 시각화</div>', unsafe_allow_html=True)
    if not ready:
        st.markdown('<div class="empty"><div class="brain">🧠</div><b>MRI 분석 대기 중</b><small>왼쪽의 예시 MRI를 내려받아<br>뇌-MRI(T2) 업로드 버튼으로 불러오세요.</small></div>', unsafe_allow_html=True)
    else:
        main_src = upload_data_url(uploaded)
        coronal = data_url(ASSETS / "coronal_result.png")
        sagittal = data_url(ASSETS / "sagittal_result.png")
        st.markdown(
            f'<div class="viewer-grid"><div class="mainview"><img src="{main_src}"><div class="slice">Axial View<b>Slice 124/256</b></div><div class="cross-x"></div><div class="cross-y"></div></div>'
            f'<div class="subviews"><div class="subview"><img src="{coronal}"><span class="viewtag">관상면</span></div>'
            f'<div class="subview"><img src="{sagittal}"><span class="viewtag">시상면</span></div></div></div>',
            unsafe_allow_html=True,
        )
        st.slider("Slice", 1, 256, 124, label_visibility="collapsed")
    st.markdown('</section>', unsafe_allow_html=True)

    if ready and result:
        st.markdown(
            f'<section class="panel" style="margin-top:10px"><div class="head"><span class="i">▤</span>뉴로렌즈(AI) 판독 소견</div>'
            f'<div class="finding">{result.finding}</div><div class="warning"><span class="warnicon">⚠</span><div>'
            f'<strong>파킨슨병 의심 확률({result.pd}%)이 임상적 임계치를 초과하여 높게 분석되었습니다.</strong>'
            '관련 전문의의 심층 검토를 권장합니다.</div></div></section>',
            unsafe_allow_html=True,
        )

with info_col:
    patient = '<dl class="pinfo"><dt>ID</dt><dd>PT-2026-0477</dd><dt>생년월일/나이</dt><dd>김파킨 (M/64)</dd><dt>성별/연령</dt><dd>1960.03.15</dd><span class="divider"></span><dt>주요 병력</dt><dd>고혈압, Medication,<br>Syrtl, 특발성떨림</dd></dl>'
    panel("환자정보", patient, "●")
    st.write("")
    if ready and result:
        probability = (
            f'<div class="prob"><div class="probline"><span>정상 (Normal)</span><b>{result.normal}%</b></div><div class="track"><div class="fill" style="width:{result.normal}%;background:#18a77e"></div></div></div>'
            f'<div class="prob"><div class="probline"><span>경계 (Prodromal)</span><b>{result.prodromal}%</b></div><div class="track"><div class="fill" style="width:{result.prodromal}%;background:#e0a413"></div></div></div>'
            f'<div class="prob"><div class="probline"><span>파킨슨병 (PD)</span><b>{result.pd}%</b></div><div class="track"><div class="fill" style="width:{result.pd}%;background:#ff334b"></div></div></div>'
            f'<div class="reason"><b>ⓘ　판단 근거</b>{result.rationale}<br>고위험군으로 판단합니다.</div>'
        )
        panel("뉴로렌즈 판독요약", probability, "◉")
        st.write("")
        model = '<dl class="model"><dt>딥러닝모델</dt><dd style="color:#489cff">3D-CNN 3D-ResNet</dd><dt>특징융합</dt><dd>OCA</dd><dt>적용 모듈</dt><dd style="color:#489cff">Multi-View Attention</dd></dl>'
        panel("딥러닝 모델", model, "⌁")
    else:
        panel("분석 결과", '<div style="padding:25px 4px;text-align:center;color:#71859c;font-size:11px;line-height:1.7">MRI 업로드 후<br>판독 결과가 표시됩니다.</div>', "◉")
    st.markdown('<div class="status">● System Online · v1.0-prototype</div>', unsafe_allow_html=True)

st.caption("본 정보는 AI 진단 보조 도구이며, 최종 진단은 전문의의 판단을 따릅니다.")
