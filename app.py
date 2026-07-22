from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import streamlit as st

from preprocessing_adapter import preprocess_nifti, validate_nifti


ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"


@dataclass(frozen=True)
class Result:
    normal: int = 18
    prodromal: int = 29
    pd: int = 88
    finding: str = "양측 흑질(Substantia Nigra) 영역에서 유의미한 부피 감소 소견이 관찰됩니다."
    rationale: str = "양측 흑질 영역의 M3d-CAM 활성 패턴이 모델 판단에 크게 기여했습니다."


def data_url(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def panel(title: str, content: str, icon: str = "▣") -> None:
    st.markdown(f'<section class="panel"><div class="head"><span>{icon}</span>{title}</div><div class="pad">{content}</div></section>', unsafe_allow_html=True)


def viewer_html(views: list[str], title: str, badge: str = "") -> str:
    labels = ("Axial View", "Coronal View", "Sagittal View")
    cards = "".join(f'<figure><figcaption>{label}</figcaption><img src="{src}"></figure>' for label, src in zip(labels, views))
    return f'<section class="panel"><div class="head"><span>◈</span>{title}<b class="badge">{badge}</b></div><div class="tri-view">{cards}</div></section>'


def xai_report(result: Result, original_src: str) -> str:
    logo = data_url(ASSETS / "logo.png")
    heatmap = data_url(ASSETS / "sample_t2_mri.png")
    return f'''<section class="report">
      <header><div>NeuroLens <b>XAI</b> <span>Analysis Report</span></div><img src="{logo}"></header>
      <main><div class="report-top"><article><h3>♙ Patient Info</h3><dl><dt>환자 ID</dt><dd>PT-2026-0417</dd><dt>검사일</dt><dd>2024-10-28</dd><dt>검사 유형</dt><dd>T2 MRI</dd><dt>판독 상태</dt><dd class="done">자동 생성 완료</dd></dl></article>
      <article><h3>▣ AI 보조 판독 요약 <small>(AI Model Info &amp; Disclaimer)</small></h3><p>본 리포트는 전처리 영상과 데모 XAI 결과를 바탕으로 자동 생성되었습니다. 모델 연결 전 시연 화면이며 최종 판독 및 진단은 담당 전문의의 판단을 우선합니다.</p><aside>AI 보조 시스템: NeuroLens v1.0</aside></article></div>
      <article class="visual"><h3>▥ XAI 시각화 <small>(M3d-CAM)</small></h3><div class="compare"><figure><figcaption>원본 MRI (T2)</figcaption><img src="{original_src}"></figure><figure><figcaption>AI 분석 결과 (Demo Heatmap)</figcaption><img src="{heatmap}"></figure></div></article>
      <article><h3>▤ AI 진단 확률 요약</h3>{report_bar('정상', result.normal, '#1556c0')}{report_bar('전구기', result.prodromal, '#ff8c00')}{report_bar('파킨슨병 의심', result.pd, '#e91d2b')}</article>
      <article class="narrative"><strong>AI</strong><div><h3>핵심 판독 요약 <small>(RAG/LLM 기반 데모)</small></h3><ul><li>{result.finding}</li><li>파킨슨병 의심 확률이 {result.pd}%로 분석되었습니다.</li><li>임상 증상 및 추가 검사와 종합하여 전문의가 최종 판단해야 합니다.</li></ul></div></article>
      <footer>▣ 생성일시　2024-10-28 14:32 <span>담당 전문의 서명　________________</span></footer></main></section>'''


def report_bar(label: str, value: int, color: str) -> str:
    return f'<div class="rbar"><b>{label}</b><i><em style="width:{value}%;background:{color}"></em></i><strong style="color:{color}">{value}%</strong></div>'


def probability(label: str, value: int, color: str) -> str:
    return f'<div class="prob"><div><span>{label}</span><b>{value}%</b></div><i><em style="width:{value}%;background:{color}"></em></i></div>'


st.set_page_config(page_title="NeuroLens | T2 MRI 분석", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&family=Inter:wght@400;600;700&display=swap');
:root{--bg:#030b16;--panel:#0a182a;--line:#1d324a;--blue:#218cff;--text:#edf5ff;--muted:#8195ac}*{box-sizing:border-box;font-family:Inter,'Noto Sans KR',sans-serif}.stApp{background:radial-gradient(circle at 45% -10%,#10294b,#030b16 42%);color:var(--text)}[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"]{display:none!important}.block-container{max-width:1600px;padding:.6rem .8rem 2rem}
.topbar{min-height:68px;display:flex;align-items:center;gap:30px;padding:10px 20px;background:#06162a;border:1px solid #142a43;border-radius:8px;margin-bottom:10px}.brand{display:flex;align-items:center;gap:12px;font-size:21px;font-weight:800}.brand img{width:105px;max-height:46px;object-fit:contain}.brand b{color:var(--blue)}.topnav{display:flex;gap:28px;align-self:stretch}.topnav span{display:flex;align-items:center;border-bottom:3px solid transparent;font-weight:650}.topnav .active{border-color:var(--blue)}.meta{margin-left:auto;text-align:right;font-size:10px;color:#b7c5d5}
.panel{background:linear-gradient(145deg,#0c1c30,#071424);border:1px solid var(--line);border-radius:8px;overflow:hidden;box-shadow:0 10px 28px rgba(0,0,0,.15)}.head{min-height:44px;padding:10px 14px;display:flex;align-items:center;gap:9px;border-bottom:1px solid var(--line);font-size:15px;font-weight:700}.head>span{color:var(--blue)}.head .badge{margin-left:auto;padding:4px 7px;border:1px solid #1d6bb7;border-radius:30px;color:#58b0ff;font-size:9px}.pad{padding:13px}.side-item{height:45px;display:flex;align-items:center;padding:0 12px;margin:4px 0;border-radius:5px;font-size:13px;font-weight:650}.side-item.active{background:linear-gradient(90deg,#174bad,#2064d8)}.side-gap{height:70px}.hint{font-size:10px;color:var(--muted);line-height:1.6;margin:7px 2px}
.stepper{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}.step{padding:10px;border:1px solid #1c3551;border-radius:7px;background:#071528;color:#71869e;font-size:11px;text-align:center}.step.done{border-color:#1775c9;color:#78bdff;background:#09213d}.step.active{border-color:#2b99ff;color:#fff;box-shadow:0 0 0 1px #2b99ff inset}.empty{min-height:520px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;color:#71859c}.empty b{color:#c5d5e6;font-size:17px;margin:12px}.empty .brain{font-size:55px;filter:grayscale(1);opacity:.6}
.validation{padding:14px;border-left:4px solid #2bdbac;background:#08201f;border-radius:5px;color:#c8eee5;font-size:11px;line-height:1.8}.validation.error{border-color:#ff4150;background:#281018;color:#ffd2d6}.file-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:11px}.file-grid div{padding:9px;background:#071426;border:1px solid #1a3049;border-radius:5px}.file-grid small{display:block;color:#768ba3;font-size:9px}.file-grid b{font-size:11px}
.tri-view{display:grid;grid-template-columns:2fr 1fr 1fr;min-height:470px;background:#01070e}.tri-view figure{margin:0;position:relative;display:flex;align-items:center;justify-content:center;border-right:1px solid #21374f;overflow:hidden}.tri-view figure:last-child{border:0}.tri-view figcaption{position:absolute;top:9px;left:10px;background:#061426cc;padding:4px 7px;border-radius:3px;font-size:10px;z-index:2}.tri-view img{width:100%;height:100%;max-height:520px;object-fit:contain}.qc-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.qc{padding:11px;border:1px solid #1c3853;background:#071729;border-radius:6px}.qc small{display:block;color:#7f95ad;font-size:9px}.qc b{font-size:12px}.qc.ok b{color:#3bd6ae}.demo{padding:8px 12px;background:#3b2a08;border:1px solid #8a6718;border-radius:5px;color:#ffd76a;font-size:10px;margin-bottom:9px}
.prob{margin:10px 0}.prob>div{display:flex;justify-content:space-between;font-size:10px;margin-bottom:5px}.prob i,.rbar i{display:block;height:6px;background:#1c2c40;border-radius:20px;overflow:hidden}.prob em,.rbar em{display:block;height:100%;border-radius:20px}.reason{padding:10px;border:1px solid #263d57;border-radius:5px;color:#b8c7d8;font-size:10px;line-height:1.65}.finding{padding:13px;background:#061426;font-size:14px;line-height:1.6}.warning{padding:12px;background:#2a1017;border-top:1px solid #66232e;color:#ffd3d7;font-size:11px}.warning b{color:#ffe000}
.report{background:#fff;color:#111827;border-radius:8px;overflow:hidden;border:1px solid #b9cce3}.report>header{min-height:80px;display:flex;align-items:center;justify-content:space-between;padding:16px 28px;background:linear-gradient(100deg,#03143c,#001c4f);color:#fff;font-size:25px;font-weight:800;border-bottom:4px solid #238cff}.report header b{color:#2f83ff}.report header span{font-weight:400}.report header img{width:140px;max-height:55px;object-fit:contain}.report main{padding:18px}.report article{border:1px solid #bfd0e3;border-radius:7px;padding:14px;margin-bottom:13px}.report article h3{margin:0 0 10px;color:#082a66;font-size:16px}.report h3 small{font-size:9px}.report-top{display:grid;grid-template-columns:1fr 1.1fr;gap:13px}.report-top article{margin:0}.report dl{display:grid;grid-template-columns:40% 60%;margin:0;font-size:11px}.report dt,.report dd{padding:7px;border-bottom:1px dotted #ccd7e4;margin:0}.report dt{font-weight:700;background:#f6f8fb}.report .done{color:#076dde;font-weight:700}.report p{font-size:11px;line-height:1.8}.report aside{text-align:right;font-size:10px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.compare figure{margin:0;border:1px solid #c6d5e5;padding:8px;border-radius:6px}.compare figcaption{text-align:center;background:#05265a;color:#fff;border-radius:5px;padding:5px;font-size:10px}.compare img{width:100%;height:300px;object-fit:contain;background:#02060b}.rbar{display:grid;grid-template-columns:110px 1fr 45px;gap:10px;align-items:center;padding:6px 8px;font-size:10px}.rbar i{background:#edf0f4}.rbar strong{text-align:right;font-size:13px}.narrative{display:flex;gap:15px}.narrative>strong{width:52px;height:52px;display:flex;align-items:center;justify-content:center;border:3px solid #082a66;border-radius:6px;color:#082a66;font-size:20px}.narrative ul{font-size:10px;line-height:1.8;margin:0;padding-left:18px}.report footer{border-top:2px solid #0b2d68;padding:9px;display:flex;justify-content:space-between;font-size:10px}
[data-testid="stFileUploader"]{background:#071729;border:1px dashed #3779ba;border-radius:7px;padding:3px;color:#eaf4ff!important}[data-testid="stFileUploader"] section{padding:9px!important;background:#071729!important}[data-testid="stFileUploader"] *{color:#dcecff!important}[data-testid="stFileUploader"] button{background:#12345b!important;border:1px solid #357abb!important;color:#fff!important}[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p,[data-testid="stFileUploaderDropzoneInstructions"] span,[data-testid="stFileUploaderDropzoneInstructions"] small{color:#dcecff!important;opacity:1!important}.stButton button,.stDownloadButton button{width:100%;background:#0d315d;border:1px solid #2478c8;color:#eaf5ff!important}.stButton button p,.stDownloadButton button p{color:#eaf5ff!important}.stButton button[kind="primary"]{background:linear-gradient(90deg,#1265d0,#218cff);font-weight:700}[data-testid="stSegmentedControl"]{background:#061426;border:1px solid #1c3856;border-radius:8px;padding:4px}[data-testid="stSegmentedControl"] label,[data-testid="stSegmentedControl"] p{color:#d9e9fa!important;opacity:1!important}[data-testid="stAlert"] *{color:#dcecff!important}[data-testid="stProgress"] p,[data-testid="stStatusWidget"] *{color:#dcecff!important}.status{text-align:right;color:#34d8ad;font-size:9px;margin:8px}
@media(max-width:1100px){div[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important}div[data-testid="stHorizontalBlock"]>div[data-testid="column"]{width:100%!important;flex:1 1 100%!important;min-width:100%!important}.topbar{flex-wrap:wrap}.topnav{order:3;width:100%;height:38px}.side-gap{display:none}.panel.pad{white-space:nowrap;overflow:auto}.side-item{display:inline-flex}.tri-view{grid-template-columns:1fr}.tri-view figure{min-height:320px;border-right:0;border-bottom:1px solid #21374f}.report-top,.compare{grid-template-columns:1fr}}
@media(max-width:650px){.meta{display:none}.brand{font-size:17px}.topnav{gap:13px;font-size:11px}.stepper{grid-template-columns:1fr 1fr}.file-grid,.qc-grid{grid-template-columns:1fr 1fr}.report>header{padding:12px;font-size:16px}.report header img{width:95px}.report main{padding:9px}.compare img{height:230px}.rbar{grid-template-columns:78px 1fr 40px}.report footer{gap:10px}}
</style>''', unsafe_allow_html=True)


logo = data_url(ASSETS / "logo.png")
st.markdown(f'<header class="topbar"><div class="brand"><img src="{logo}"><span><b>MRI</b> 분석 대시보드</span></div><nav class="topnav"><span class="active">뉴로렌즈(AI) 분석 결과</span><span>환자관리</span></nav><div class="meta">Patient ID. PT-2026-0477<br>Scan Date. 2024.10.28</div></header>', unsafe_allow_html=True)

for key, default in {"pipeline_done": False, "view": "원본 MRI", "source_name": None}.items():
    st.session_state.setdefault(key, default)

nav, center, info = st.columns([.82, 5.15, 1.5], gap="small")
with nav:
    st.markdown('<div class="panel pad"><div class="side-item">▣　진단뷰어</div><div class="side-item active">▤　분석도구</div><div class="side-item">▧　임상노트</div><div class="side-item">⚙　설정</div><div class="side-gap"></div></div>', unsafe_allow_html=True)
    sample_path = ASSETS / "sample_t2_mri.nii.gz"
    if sample_path.exists():
        st.download_button("↓ 예시 NIfTI 받기", sample_path.read_bytes(), "NeuroLens_sample_T2_MRI.nii.gz", "application/gzip")
    uploaded = st.file_uploader("T2 MRI NIfTI 업로드", type=["nii", "gz"], help="3D .nii 또는 .nii.gz, 최대 200MB")
    st.markdown('<div class="hint">업로드 즉시 형식과 크기를 검사합니다.<br>실행 버튼은 한 개만 사용합니다.</div>', unsafe_allow_html=True)

payload = uploaded.getvalue() if uploaded else None
validation = validate_nifti(payload, uploaded.name) if uploaded else None
if uploaded and st.session_state.source_name != uploaded.name:
    st.session_state.pipeline_done = False
    st.session_state.source_name = uploaded.name
    st.session_state.pop("prep", None)

step_state = 0 if not uploaded else (4 if st.session_state.pipeline_done else 1)
with center:
    labels = ("1　NIfTI 업로드", "2　전처리", "3　AI 분석", "4　XAI 보고서")
    steps = "".join(f'<div class="step {"done" if i < step_state else "active" if i == step_state else ""}">{label}</div>' for i, label in enumerate(labels))
    st.markdown(f'<div class="stepper">{steps}</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown('<section class="panel"><div class="empty"><div class="brain">🧠</div><b>T2 MRI 분석 대기 중</b><small>왼쪽에서 예시 NIfTI를 내려받거나<br>3D T2 MRI 파일을 업로드하세요.</small></div></section>', unsafe_allow_html=True)
    elif not validation.valid:
        st.markdown(f'<div class="validation error">✕　{validation.message}</div>', unsafe_allow_html=True)
    elif not st.session_state.pipeline_done:
        st.markdown(f'<div class="validation">✓　{validation.message}<div class="file-grid"><div><small>파일명</small><b>{validation.filename}</b></div><div><small>크기</small><b>{validation.size_mb:.2f} MB</b></div><div><small>Volume shape</small><b>{validation.shape}</b></div><div><small>Spacing / 방향</small><b>{validation.spacing} · {validation.orientation}</b></div></div></div>', unsafe_allow_html=True)
        st.write("")
        if st.button("분석 시작", type="primary", use_container_width=True):
            progress = st.progress(0, text="NIfTI 무결성 검사")
            time.sleep(.25); progress.progress(22, text="RAS 방향 표준화")
            time.sleep(.25); progress.progress(48, text="Intensity 정규화")
            prep = preprocess_nifti(payload, uploaded.name)
            progress.progress(76, text="56×56×56 리사이즈 및 QC")
            time.sleep(.25); progress.progress(92, text="AI 분석 화면 준비 (데모 모델)")
            time.sleep(.2); progress.progress(100, text="완료")
            st.session_state.prep = prep
            st.session_state.pipeline_done = True
            st.session_state.view = "전처리 결과"
            st.rerun()
    else:
        prep = st.session_state.prep
        view = st.segmented_control("결과 보기", ["원본 MRI", "전처리 결과", "AI 분석", "XAI 보고서"], default=st.session_state.view, selection_mode="single") or st.session_state.view
        st.session_state.view = view
        result = Result()
        if view == "원본 MRI":
            st.markdown(viewer_html(prep["original_views"], "원본 T2 MRI", "업로드 원본"), unsafe_allow_html=True)
        elif view == "전처리 결과":
            st.markdown(viewer_html(prep["processed_views"], "전처리 결과", "BRAINTENSOR CLOUD ADAPTER"), unsafe_allow_html=True)
            st.markdown(f'<div class="qc-grid"><div class="qc ok"><small>NIfTI 검증</small><b>✓ 통과</b></div><div class="qc ok"><small>Orientation</small><b>✓ {prep["orientation"]}</b></div><div class="qc ok"><small>Intensity</small><b>✓ Min-Max</b></div><div class="qc ok"><small>출력 Shape</small><b>✓ {prep["final_shape"]}</b></div></div>', unsafe_allow_html=True)
            st.download_button("↓ 전처리 NIfTI 다운로드", prep["output_bytes"], prep["output_name"], "application/gzip")
            st.info("Streamlit Cloud에서는 배포 가능한 전처리 단계가 실행됩니다. 연구용 전체 BET·N4·MNI 정합은 FSL/ANTs 실행환경 연결 후 활성화됩니다.")
        elif view == "AI 분석":
            st.markdown('<div class="demo">⚠ 모델 학습 완료 전 디자인 확인용 데모 결과입니다. 실제 진단 결과가 아닙니다.</div>', unsafe_allow_html=True)
            demo_views = [data_url(ASSETS / "sample_t2_mri.png"), data_url(ASSETS / "coronal_result.png"), data_url(ASSETS / "sagittal_result.png")]
            st.markdown(viewer_html(demo_views, "AI 병변 시각화", "M3d-CAM DEMO"), unsafe_allow_html=True)
            st.markdown(f'<section class="panel"><div class="head"><span>▤</span>뉴로렌즈(AI) 판독 소견</div><div class="finding">{result.finding}</div><div class="warning"><b>파킨슨병 의심 확률({result.pd}%)</b> · 모델 연결 전 시연용 수치입니다.</div></section>', unsafe_allow_html=True)
        else:
            original_src = prep["original_views"][0]
            html = xai_report(result, original_src)
            st.markdown(html, unsafe_allow_html=True)
            st.download_button("↓ XAI 보고서 HTML 다운로드", html.encode("utf-8"), "NeuroLens_XAI_Report.html", "text/html")

with info:
    panel("환자정보", '<dl class="pinfo"><b>ID</b>　PT-2026-0477<br><br><b>성명/나이</b>　김파킨 (M/64)<br><br><b>검사</b>　T2 MRI</dl>', "●")
    st.write("")
    if st.session_state.pipeline_done:
        r = Result()
        probs = probability("정상", r.normal, "#1a9d79") + probability("전구기", r.prodromal, "#e5a315") + probability("파킨슨병", r.pd, "#ff334b")
        panel("분석 결과 · DEMO", probs + f'<div class="reason"><b>판단 근거</b><br>{r.rationale}</div>', "◉")
        st.write("")
        panel("전처리 상태", '<div class="reason">✓ NIfTI 검증<br>✓ RAS 표준화<br>✓ Min-Max 정규화<br>✓ 56³ 리사이즈<br>○ BET/N4/MNI 연구환경 대기</div>', "⌁")
    else:
        panel("분석 결과", '<div class="reason">분석 완료 후 결과가 표시됩니다.</div>', "◉")
    st.markdown('<div class="status">● System Online · preprocessing-ready</div>', unsafe_allow_html=True)

st.caption("본 서비스는 AI 진단 보조 프로토타입입니다. 최종 진단은 전문의의 판단을 따릅니다.")
