from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import streamlit as st

from preprocessing_adapter import convert_dicom_folder, inspect_dicom_folder, preprocess_nifti, render_nifti_views, validate_nifti
from local_pipeline import APP_VERSION, local_pipeline_status, run_local_pipeline


ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"


@st.cache_data(ttl=60, show_spinner=False)
def runtime_status():
    return local_pipeline_status(ROOT)


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


def viewer_html(views: list[str], title: str, badge: str = "", zoom_percent: int = 100) -> str:
    labels = ("축상면", "관상면", "시상면")
    cards = "".join(
        f'<figure><figcaption>{label}</figcaption><img src="{src}" style="transform:scale({zoom_percent / 100:.2f})"></figure>'
        for label, src in zip(labels, views)
    )
    return f'<section class="panel"><div class="head"><span>◈</span>{title}<b class="badge">{badge}</b></div><div class="tri-view">{cards}</div></section>'


def relative_slice_index(size: int, position_percent: int) -> int:
    return round((max(size, 1) - 1) * max(0, min(position_percent, 100)) / 100)


def reset_mri_view() -> None:
    for label in ("축상면", "관상면", "시상면"):
        st.session_state[f"linked_{label}_position"] = 50
    st.session_state["linked_mri_zoom"] = 100


def interactive_mri_viewer(prep: dict, source: str) -> str:
    is_original = source == "original"
    shape = tuple(int(value) for value in (prep["original_shape"] if is_original else prep["final_shape"]))
    labels = ("축상면", "관상면", "시상면")
    sizes = (shape[2], shape[1], shape[0])
    control_columns = st.columns(3, gap="small")
    positions = []
    for column, label, size in zip(control_columns, labels, sizes):
        with column:
            positions.append(
                st.slider(
                    f"{label} 연동 위치 (%)",
                    min_value=0,
                    max_value=100,
                    value=50,
                    key=f"linked_{label}_position",
                )
            )
    zoom_column, reset_column = st.columns([4, 1], gap="small")
    with zoom_column:
        zoom_percent = st.slider(
            "영상 확대·축소 (%)",
            min_value=50,
            max_value=250,
            value=100,
            step=10,
            key="linked_mri_zoom",
        )
    with reset_column:
        st.button(
            "↺ 보기 초기화",
            key="reset_mri_view",
            use_container_width=True,
            on_click=reset_mri_view,
        )
    indices = [relative_slice_index(size, position) for size, position in zip(sizes, positions)]
    payload_key = "original_bytes" if is_original else "processed_bytes"
    name_key = "original_name" if is_original else "processed_name"
    fallback_key = "original_views" if is_original else "processed_views"
    if prep.get(payload_key):
        views = render_nifti_views(prep[payload_key], prep[name_key], tuple(indices))
    else:
        views = prep[fallback_key]
    title = "원본 T2 MRI" if is_original else "전처리 결과"
    position_badge = " · ".join(
        f"{label} {index + 1}/{size} ({position}%)"
        for label, index, size, position in zip(labels, indices, sizes, positions)
    )
    badge = f"{position_badge} · 확대 {zoom_percent}%"
    return viewer_html(views, title, badge, zoom_percent)


def xai_report(result: Result, original_src: str, prep: dict, patient_id: str, patient: dict) -> str:
    logo = data_url(ASSETS / "logo.png")
    heatmap = data_url(ASSETS / "sample_t2_mri.png")
    full_pipeline = prep.get("pipeline_mode") == "local_full"
    pipeline_name = "BRAINTENSOR ref21order_v1" if full_pipeline else "PACScan 배포용 경량 전처리"
    pipeline_steps = (
        "DICOM→NIfTI · FSL BET · ANTsPy/PD25 정합 · Min-Max · 56³"
        if full_pipeline
        else "DICOM→NIfTI · 방향 표준화 · Min-Max · 56³"
    )
    run_id = prep.get("run_id", "세션 전용")
    return f'''<section class="report">
      <header><div>NeuroLens <b>XAI</b> <span>분석 보고서</span></div><img src="{logo}"></header>
      <main><div class="report-top"><article><h3>♙ 환자 정보</h3><dl><dt>환자 ID</dt><dd>{patient_id}</dd><dt>검사일</dt><dd>{patient["last_exam"]}</dd><dt>검사 유형</dt><dd>T2 MRI</dd><dt>판독 상태</dt><dd class="done">자동 생성 완료</dd></dl></article>
      <article><h3>▣ AI 모델 <small>(분석 구성 및 현재 상태)</small></h3>
      <div class="model-info">
        <div><b>전처리</b><span>{pipeline_name}<small>{pipeline_steps}<br>실행 ID {run_id}</small></span></div>
        <div><b>예정 모델</b><span>3D-CNN + 3D-ResNet<small>Multi-View Attention · M3d-CAM 설명 시각화</small></span></div>
        <div><b>현재 상태</b><span class="model-demo">모델 연결 전 · 시연용 결과<small>현재 확률과 판독문은 임상 진단 결과가 아닙니다.</small></span></div>
      </div>
      <aside>AI 보조 시스템: NeuroLens / PACScan v{APP_VERSION}</aside></article></div>
      <article class="visual"><h3>▥ XAI 시각화 <small>(M3d-CAM)</small></h3><div class="compare"><figure><figcaption>원본 MRI (T2)</figcaption><img src="{original_src}"></figure><figure><figcaption>AI 분석 결과 (시연용 히트맵)</figcaption><img src="{heatmap}"></figure></div></article>
      <article><h3>▤ AI 진단 확률 요약</h3>{report_bar('정상', result.normal, '#1556c0')}{report_bar('전구기', result.prodromal, '#ff8c00')}{report_bar('파킨슨병 의심', result.pd, '#e91d2b')}</article>
      <article class="narrative"><strong>AI</strong><div><h3>핵심 판독 요약 <small>(RAG/LLM 기반 시연용)</small></h3><ul><li>{result.finding}</li><li>파킨슨병 의심 확률이 {result.pd}%로 분석되었습니다.</li><li>임상 증상 및 추가 검사와 종합하여 전문의가 최종 판단해야 합니다.</li></ul></div></article>
      <footer>▣ 생성일시　2024-10-28 14:32 <span>담당 전문의 서명　________________</span></footer></main></section>'''


def report_bar(label: str, value: int, color: str) -> str:
    return f'<div class="rbar"><b>{label}</b><i><em style="width:{value}%;background:{color}"></em></i><strong style="color:{color}">{value}%</strong></div>'


def probability(label: str, value: int, color: str) -> str:
    return f'<div class="prob"><div><span>{label}</span><b>{value}%</b></div><i><em style="width:{value}%;background:{color}"></em></i></div>'


DEMO_PATIENTS = {
    "PT-DEMO-001": {
        "name": "김뉴로", "sex_age": "M / 64", "birth": "1962.03.15",
        "dicom_patient_id": "PACSCAN-DEMO-001",
        "phone": "010-****-1201", "condition": "파킨슨병 의심", "last_exam": "2026.07.23",
        "history": [("2026.07.23", "T2 MRI", "전처리 완료", "XAI 보고서 생성"),
                    ("2026.06.04", "외래 진료", "경과 관찰", "운동 증상 문진")],
        "note": "최근 보행 속도 저하와 안정 시 떨림을 호소함. MRI 분석 결과와 임상검사를 함께 검토할 예정입니다.",
    },
    "PT-DEMO-002": {
        "name": "이도파", "sex_age": "F / 58", "birth": "1968.11.02",
        "dicom_patient_id": "PACSCAN-DEMO-002",
        "phone": "010-****-3827", "condition": "전구기 추적관찰", "last_exam": "2026.07.18",
        "history": [("2026.07.18", "T2 MRI", "분석 대기", "원본 영상 등록"),
                    ("2026.04.11", "신경학 검사", "추적관찰", "비운동 증상 문진")],
        "note": "후각 저하와 수면행동 증상을 중심으로 추적관찰 중입니다. 다음 방문 시 MRI 비교 판독을 예정합니다.",
    },
    "PT-DEMO-003": {
        "name": "박정상", "sex_age": "M / 61", "birth": "1965.05.27",
        "dicom_patient_id": "PACSCAN-DEMO-003",
        "phone": "010-****-7714", "condition": "정상 대조군", "last_exam": "2026.07.10",
        "history": [("2026.07.10", "T2 MRI", "QC 통과", "정상 범위 시연 결과"),
                    ("2026.03.20", "건강검진", "특이소견 없음", "정기검진")],
        "note": "시연용 정상 대조 환자입니다. 현재 등록된 임상 특이사항은 없습니다.",
    },
}


def mask_patient_id(patient_id: str) -> str:
    if not patient_id or patient_id == "-":
        return "확인 불가"
    if len(patient_id) <= 6:
        return patient_id[:1] + "****" + patient_id[-1:]
    return patient_id[:3] + "****" + patient_id[-3:]


def render_patient_management() -> None:
    st.markdown(
        '<div class="demo-notice">시연용 가상 환자 데이터입니다. 실제 개인정보나 임상 기록을 사용하지 않습니다.</div>',
        unsafe_allow_html=True,
    )
    select_column, action_column = st.columns([4, 1], gap="small")
    with select_column:
        selected_id = st.selectbox(
            "환자 선택",
            list(DEMO_PATIENTS),
            format_func=lambda patient_id: f"{DEMO_PATIENTS[patient_id]['name']} · {patient_id} · {DEMO_PATIENTS[patient_id]['condition']}",
            key="selected_patient_id",
            on_change=sync_selected_patient_query,
        )
    with action_column:
        st.markdown('<div class="patient-action-label">선택 환자 작업</div>', unsafe_allow_html=True)
        st.button(
            "MRI 분석 열기 →",
            key="open_selected_patient_analysis",
            type="primary",
            use_container_width=True,
            on_click=open_selected_patient_analysis,
        )
    patient = DEMO_PATIENTS[selected_id]
    list_col, main_col, note_col = st.columns([1.15, 3.2, 1.45], gap="small")
    with list_col:
        cards = "".join(
            f'''<a href="?page=patients&amp;patient={patient_id}" class="patient-card {"selected" if patient_id == selected_id else ""}">
            <div><b>{item["name"]}</b><span>{item["sex_age"]}</span></div>
            <small>{patient_id}</small><em>{item["condition"]}</em>
            <small>최근 검사 {item["last_exam"]}</small></a>'''
            for patient_id, item in DEMO_PATIENTS.items()
        )
        panel("환자 목록", f'<div class="patient-list">{cards}</div>', "♙")
    with main_col:
        info = (
            f'<div class="patient-profile"><div><small>환자 ID</small><b>{selected_id}</b></div>'
            f'<div><small>성명</small><b>{patient["name"]}</b></div>'
            f'<div><small>성별 / 나이</small><b>{patient["sex_age"]}</b></div>'
            f'<div><small>생년월일</small><b>{patient["birth"]}</b></div>'
            f'<div><small>연락처</small><b>{patient["phone"]}</b></div>'
            f'<div><small>관리 상태</small><b class="patient-condition">{patient["condition"]}</b></div></div>'
        )
        panel("환자 기본정보", info, "●")
        st.write("")
        rows = "".join(
            f"<tr><td>{date}</td><td>{exam}</td><td>{status}</td><td>{summary}</td></tr>"
            for date, exam, status, summary in patient["history"]
        )
        panel("검사 및 분석 이력", f'<div class="history-table"><table><thead><tr><th>일자</th><th>검사</th><th>상태</th><th>요약</th></tr></thead><tbody>{rows}</tbody></table></div>', "▤")
    with note_col:
        panel("최근 임상 메모", f'<div class="clinical-note"><b>{patient["last_exam"]}</b><p>{patient["note"]}</p><small>담당 의료진 · 시연 계정</small></div>', "▧")
        st.write("")
        panel("MRI 분석 관리", '<div class="management-actions"><div><b>최근 상태</b><span>검사 이력 확인</span></div><div><b>보고서</b><span>XAI 보고서 시연</span></div><div><b>연동 상태</b><span class="ok-text">● PACScan 준비 완료</span></div></div>', "◈")


def sync_selected_patient_query() -> None:
    st.query_params["page"] = "patients"
    st.query_params["patient"] = st.session_state["selected_patient_id"]


def open_selected_patient_analysis() -> None:
    st.query_params["page"] = "analysis"
    st.query_params["patient"] = st.session_state["selected_patient_id"]


st.set_page_config(page_title="NeuroLens | T2 MRI 분석", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&family=Inter:wght@400;600;700&display=swap');
:root{--bg:#030b16;--panel:#0a182a;--line:#1d324a;--blue:#218cff;--text:#edf5ff;--muted:#8195ac}*{box-sizing:border-box;font-family:Inter,'Noto Sans KR',sans-serif}.stApp{background:radial-gradient(circle at 45% -10%,#10294b,#030b16 42%);color:var(--text)}[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"]{display:none!important}.block-container{max-width:1600px;padding:.6rem .8rem 2rem}
.topbar{min-height:78px;display:flex;align-items:center;gap:36px;padding:11px 22px;background:#06162a;border:1px solid #142a43;border-radius:8px;margin-bottom:10px}.brand{display:flex;align-items:center;gap:16px;font-size:22px;font-weight:800;flex:0 0 auto}.brand img{width:145px;max-height:58px;object-fit:contain}.brand b{color:var(--blue)}.topnav{display:flex;gap:30px;align-self:stretch}.topnav span{display:flex;align-items:center;border-bottom:3px solid transparent;font-weight:650}.topnav .active{border-color:var(--blue)}.meta{margin-left:auto;text-align:right;font-size:11px;line-height:1.55;color:#b7c5d5;white-space:nowrap}
.panel{background:linear-gradient(145deg,#0c1c30,#071424);border:1px solid var(--line);border-radius:8px;overflow:hidden;box-shadow:0 10px 28px rgba(0,0,0,.15)}.head{min-height:44px;padding:10px 14px;display:flex;align-items:center;gap:9px;border-bottom:1px solid var(--line);font-size:15px;font-weight:700}.head>span{color:var(--blue)}.head .badge{margin-left:auto;padding:4px 7px;border:1px solid #1d6bb7;border-radius:30px;color:#58b0ff;font-size:9px}.pad{padding:13px}.side-item{height:45px;display:flex;align-items:center;padding:0 12px;margin:4px 0;border-radius:5px;font-size:13px;font-weight:650}.side-item.active{background:linear-gradient(90deg,#174bad,#2064d8)}.side-gap{height:70px}.hint{font-size:10px;color:var(--muted);line-height:1.6;margin:7px 2px}
.stepper{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}.step{padding:10px;border:1px solid #1c3551;border-radius:7px;background:#071528;color:#71869e;font-size:11px;text-align:center}.step.done{border-color:#1775c9;color:#78bdff;background:#09213d}.step.active{border-color:#2b99ff;color:#fff;box-shadow:0 0 0 1px #2b99ff inset}.empty{min-height:520px;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;color:#71859c}.empty b{color:#c5d5e6;font-size:17px;margin:12px}.empty .brain{font-size:55px;filter:grayscale(1);opacity:.6}
.validation{padding:14px;border-left:4px solid #2bdbac;background:#08201f;border-radius:5px;color:#c8eee5;font-size:11px;line-height:1.8}.validation.error{border-color:#ff4150;background:#281018;color:#ffd2d6}.file-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:11px}.file-grid div{padding:9px;background:#071426;border:1px solid #1a3049;border-radius:5px}.file-grid small{display:block;color:#768ba3;font-size:9px}.file-grid b{font-size:11px}
.tri-view{display:grid;grid-template-columns:minmax(0,2.45fr) minmax(190px,1fr);grid-template-rows:1fr 1fr;min-height:500px;background:#01070e}.tri-view figure{margin:0;position:relative;display:flex;align-items:center;justify-content:center;overflow:hidden}.tri-view figure:first-child{grid-row:1/3;border-right:1px solid #21374f}.tri-view figure:nth-child(2){border-bottom:1px solid #21374f}.tri-view figcaption{position:absolute;top:9px;left:10px;background:#061426cc;padding:4px 7px;border-radius:3px;font-size:10px;z-index:2}.tri-view img{width:100%;height:100%;max-height:540px;object-fit:contain;transform-origin:center center;transition:transform .18s ease}.tri-view figure:nth-child(n+2) img{max-height:250px}.qc-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.qc{padding:11px;border:1px solid #1c3853;background:#071729;border-radius:6px}.qc small{display:block;color:#7f95ad;font-size:9px}.qc b{font-size:12px}.qc.ok b{color:#3bd6ae}.demo{padding:8px 12px;background:#3b2a08;border:1px solid #8a6718;border-radius:5px;color:#ffd76a;font-size:10px;margin-bottom:9px}
.prob{margin:10px 0}.prob>div{display:flex;justify-content:space-between;font-size:10px;margin-bottom:5px}.prob i,.rbar i{display:block;height:6px;background:#1c2c40;border-radius:20px;overflow:hidden}.prob em,.rbar em{display:block;height:100%;border-radius:20px}.reason{padding:10px;border:1px solid #263d57;border-radius:5px;color:#b8c7d8;font-size:10px;line-height:1.65}.finding{padding:13px;background:#061426;font-size:14px;line-height:1.6}.warning{padding:12px;background:#2a1017;border-top:1px solid #66232e;color:#ffd3d7;font-size:11px}.warning b{color:#ffe000}
.report{background:#fff;color:#111827;border-radius:8px;overflow:hidden;border:1px solid #b9cce3}.report>header{min-height:94px;display:flex;align-items:center;justify-content:space-between;gap:24px;padding:16px 30px;background:linear-gradient(100deg,#03143c,#001c4f);color:#fff;font-size:27px;font-weight:800;border-bottom:4px solid #238cff}.report header b{color:#2f83ff}.report header span{font-weight:400}.report header img{width:175px;max-height:68px;object-fit:contain;flex:0 0 auto}.report main{padding:20px}.report article{border:1px solid #bfd0e3;border-radius:7px;padding:16px;margin-bottom:14px}.report article h3{margin:0 0 12px;color:#082a66;font-size:19px;line-height:1.35}.report h3 small{font-size:12px}.report-top{display:grid;grid-template-columns:1fr 1.1fr;gap:13px}.report-top article{margin:0}.report dl{display:grid;grid-template-columns:40% 60%;margin:0;font-size:14px;line-height:1.5}.report dt,.report dd{padding:10px;border-bottom:1px dotted #ccd7e4;margin:0}.report dt{font-weight:700;background:#f6f8fb}.report .done{color:#076dde;font-weight:700}.report p{font-size:14px;line-height:1.85;margin:0 0 12px}.report aside{text-align:right;font-size:12px;line-height:1.5}.model-info{border-top:1px solid #d7e1ed;margin-bottom:10px}.model-info>div{display:grid;grid-template-columns:90px 1fr;gap:10px;padding:9px 4px;border-bottom:1px dotted #cbd7e5;font-size:13px;line-height:1.45}.model-info>div>b{color:#173c74}.model-info span{font-weight:600}.model-info span small{display:block;margin-top:3px;color:#56677b;font-size:11px;font-weight:400;line-height:1.5}.model-info .model-demo{color:#a06400}.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}.compare figure{margin:0;border:1px solid #c6d5e5;padding:8px;border-radius:6px}.compare figcaption{text-align:center;background:#05265a;color:#fff;border-radius:5px;padding:7px;font-size:13px;font-weight:600}.compare img{width:100%;height:300px;object-fit:contain;background:#02060b}.rbar{display:grid;grid-template-columns:130px 1fr 52px;gap:12px;align-items:center;padding:8px;font-size:14px}.rbar i{background:#edf0f4}.rbar strong{text-align:right;font-size:16px}.narrative{display:flex;gap:16px}.narrative>strong{width:56px;height:56px;display:flex;align-items:center;justify-content:center;border:3px solid #082a66;border-radius:6px;color:#082a66;font-size:21px;flex:0 0 56px}.narrative ul{font-size:15px;line-height:1.85;margin:0;padding-left:21px}.report footer{border-top:2px solid #0b2d68;padding:12px 10px;display:flex;justify-content:space-between;font-size:13px;line-height:1.5}
[data-testid="stFileUploader"]{background:#071729;border:1px dashed #3779ba;border-radius:7px;padding:3px;color:#eaf4ff!important}[data-testid="stFileUploader"] section{padding:9px!important;background:#071729!important}[data-testid="stFileUploader"] *{color:#dcecff!important}[data-testid="stFileUploader"] button{background:#12345b!important;border:1px solid #357abb!important;color:#fff!important}[data-testid="stFileUploader"] small,[data-testid="stFileUploaderDropzoneInstructions"] small{display:none!important}[data-testid="stFileUploaderDropzoneInstructions"] span{color:#dcecff!important;opacity:1!important;font-size:10px!important}[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p{color:#dcecff!important;opacity:1!important}.stButton button,.stDownloadButton button{width:100%;background:#0d315d;border:1px solid #2478c8;color:#eaf5ff!important}.stButton button p,.stDownloadButton button p{color:#eaf5ff!important}.stButton button[kind="primary"]{background:linear-gradient(90deg,#1265d0,#218cff);font-weight:700}[data-testid="stSegmentedControl"]{background:#061426;border:1px solid #1c3856;border-radius:8px;padding:4px}[data-testid="stSegmentedControl"] label,[data-testid="stSegmentedControl"] p{color:#d9e9fa!important;opacity:1!important}[data-testid="stAlert"] *{color:#dcecff!important}[data-testid="stProgress"] p,[data-testid="stStatusWidget"] *{color:#dcecff!important}.status{text-align:right;color:#34d8ad;font-size:9px;margin:8px}.status.idle{color:#8195ac}.status.ready{color:#58b0ff}.status.demo{color:#ffd76a}.status.error{color:#ff7783}
.topnav a{display:flex;align-items:center;border-bottom:3px solid transparent;color:#edf5ff;text-decoration:none;font-weight:650}.topnav a:hover{color:#75baff}.topnav a.active{border-color:var(--blue);color:#fff}.demo-notice{padding:10px 14px;margin-bottom:10px;border:1px solid #70561b;border-radius:7px;background:#2b210b;color:#ffd978;font-size:12px}.patient-list{display:flex;flex-direction:column;gap:8px}.patient-card{display:block;padding:11px;border:1px solid #1c3550;border-radius:7px;background:#071526;color:#edf5ff;text-decoration:none;transition:border-color .15s ease,background .15s ease,transform .15s ease}.patient-card:hover{border-color:#278fff;background:#0b223d;transform:translateY(-1px)}.patient-card.selected{border-color:#278fff;background:#0c294b;box-shadow:0 0 0 1px #278fff inset}.patient-card>div{display:flex;justify-content:space-between;gap:8px}.patient-card b{font-size:13px}.patient-card span,.patient-card small{color:#8195ac;font-size:10px}.patient-card small{display:block;margin-top:4px}.patient-card em{display:block;margin-top:8px;color:#6fbcff;font-size:11px;font-style:normal}.patient-profile{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.patient-profile>div{padding:12px;border:1px solid #1b3550;border-radius:6px;background:#071526}.patient-profile small{display:block;margin-bottom:5px;color:#8195ac;font-size:10px}.patient-profile b{font-size:13px}.patient-condition{color:#63b7ff}.history-table{overflow-x:auto}.history-table table{width:100%;border-collapse:collapse;font-size:12px}.history-table th{text-align:left;padding:9px;background:#102640;color:#8eb4da}.history-table td{padding:11px 9px;border-bottom:1px solid #1d334a;color:#d8e5f2}.clinical-note{font-size:12px;line-height:1.75}.clinical-note>b{color:#63b7ff}.clinical-note p{color:#c1cfdd}.clinical-note small{color:#71869c}.management-actions>div{padding:9px 0;border-bottom:1px solid #1d334a}.management-actions b,.management-actions span{display:block;font-size:11px}.management-actions span{margin-top:4px;color:#91a6ba}.management-actions .ok-text{color:#34d8ad}
.viewer-guide{margin:2px 0 8px;padding:8px 11px;border-left:3px solid #278fff;background:#07182a;color:#a9bdd2;font-size:11px}[data-testid="stSlider"]{padding:6px 10px 2px;border:1px solid #193550;border-radius:7px;background:#07172a}[data-testid="stSlider"] [data-testid="stWidgetLabel"] p{font-size:11px!important;color:#dcecff!important}
.patient-action-label{height:28px;display:flex;align-items:flex-end;color:#9db0c5;font-size:11px;margin-bottom:5px}
@media(max-width:1100px){div[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important}div[data-testid="stHorizontalBlock"]>div[data-testid="column"]{width:100%!important;flex:1 1 100%!important;min-width:100%!important}.topbar{flex-wrap:wrap}.topnav{order:3;width:100%;height:38px}.side-gap{display:none}.panel.pad{white-space:nowrap;overflow:auto}.side-item{display:inline-flex}.tri-view{grid-template-columns:1fr;grid-template-rows:auto}.tri-view figure:first-child{grid-row:auto;border-right:0}.tri-view figure{min-height:320px;border-bottom:1px solid #21374f}.tri-view figure:nth-child(n+2) img{max-height:320px}.report-top,.compare{grid-template-columns:1fr}}
@media(max-width:650px){.topbar{min-height:64px;padding:8px 12px;gap:14px}.meta{display:none}.brand{font-size:17px;gap:9px}.brand img{width:110px;max-height:45px}.topnav{gap:13px;font-size:11px}.stepper{grid-template-columns:1fr 1fr}.file-grid,.qc-grid,.patient-profile{grid-template-columns:1fr 1fr}.report>header{min-height:72px;padding:12px;font-size:17px;gap:12px}.report header img{width:115px;max-height:48px}.report main{padding:9px}.compare img{height:230px}.rbar{grid-template-columns:78px 1fr 40px}.report footer{gap:10px}}
</style>''', unsafe_allow_html=True)


logo = data_url(ASSETS / "logo.png")
page = st.query_params.get("page", "analysis")
if page not in {"analysis", "patients"}:
    page = "analysis"
requested_patient_id = st.query_params.get("patient", "")
if requested_patient_id in DEMO_PATIENTS:
    st.session_state["selected_patient_id"] = requested_patient_id
else:
    st.session_state.setdefault("selected_patient_id", "PT-DEMO-001")
selected_patient_id = st.session_state["selected_patient_id"]
selected_patient = DEMO_PATIENTS[selected_patient_id]
analysis_active = "active" if page == "analysis" else ""
patients_active = "active" if page == "patients" else ""
meta = (
    f"환자 ID. {selected_patient_id}<br>검사일. {selected_patient['last_exam']}"
    if page == "analysis"
    else f"선택 환자. {selected_patient['name']}<br>시연용 데이터"
)
st.markdown(
    f'<header class="topbar"><div class="brand"><img src="{logo}"><span><b>MRI</b> 분석 대시보드</span></div>'
    f'<nav class="topnav"><a class="{analysis_active}" href="?page=analysis&patient={selected_patient_id}">뉴로렌즈(AI) 분석 결과</a>'
    f'<a class="{patients_active}" href="?page=patients&patient={selected_patient_id}">환자관리</a></nav><div class="meta">{meta}</div></header>',
    unsafe_allow_html=True,
)

if page == "patients":
    render_patient_management()
    st.caption("환자관리 화면의 모든 환자정보와 진료기록은 UI 시연을 위해 생성된 가상 데이터입니다.")
    st.stop()

for key, default in {"pipeline_done": False, "view": "원본 MRI", "source_name": None}.items():
    st.session_state.setdefault(key, default)

nav, center, info = st.columns([.82, 5.15, 1.5], gap="small")
with nav:
    st.markdown('<div class="panel pad"><div class="side-item">▣　진단뷰어</div><div class="side-item active">▤　분석도구</div><div class="side-item">▧　임상노트</div><div class="side-item">⚙　설정</div><div class="side-gap"></div></div>', unsafe_allow_html=True)
    sample_path = ASSETS / "PACScan_sample_DICOM_folder.zip"
    if sample_path.exists():
        st.download_button("↓ 예시 DICOM 폴더 받기", sample_path.read_bytes(), "PACScan_sample_DICOM_folder.zip", "application/zip")
    uploaded_files = st.file_uploader("환자 T2 MRI DICOM 폴더 선택", accept_multiple_files="directory", help="환자 한 명의 DICOM 파일이 들어 있는 폴더를 선택하세요.")
    st.markdown('<div class="hint">예시는 ZIP 압축을 푼 뒤 폴더를 선택하세요.<br>DICOM 시리즈를 자동 분류하고 T2를 선택합니다.</div>', unsafe_allow_html=True)
    local_status = runtime_status()
    mode_label = "실제 로컬 전처리" if local_status.ready else "클라우드 경량 전처리"
    st.markdown(f'<div class="reason"><b>실행 모드</b><br>{mode_label}<br><small>PACScan v{APP_VERSION}</small></div>', unsafe_allow_html=True)

file_items = [(file.name, file.getvalue()) for file in uploaded_files] if uploaded_files else []
folder_scan = inspect_dicom_folder(file_items) if file_items else None
folder_signature = "|".join(f"{name}:{len(payload)}" for name, payload in file_items)
if file_items and st.session_state.source_name != folder_signature:
    st.session_state.pipeline_done = False
    st.session_state.source_name = folder_signature
    st.session_state.pop("prep", None)
    st.session_state.pop("folder_scan", None)
    for slice_key in (
        "original_축상면_slice", "original_관상면_slice", "original_시상면_slice",
        "processed_축상면_slice", "processed_관상면_slice", "processed_시상면_slice",
        "linked_축상면_position", "linked_관상면_position", "linked_시상면_position",
        "linked_mri_zoom",
    ):
        st.session_state.pop(slice_key, None)

step_state = 0 if not file_items else (4 if st.session_state.pipeline_done else 1)
with center:
    labels = ("1　DICOM 폴더", "2　변환·전처리", "3　AI 분석", "4　XAI 보고서")
    steps = "".join(f'<div class="step {"done" if i < step_state else "active" if i == step_state else ""}">{label}</div>' for i, label in enumerate(labels))
    if not st.session_state.pipeline_done:
        st.markdown(f'<div class="stepper">{steps}</div>', unsafe_allow_html=True)

    if not file_items:
        st.markdown('<section class="panel"><div class="empty"><div class="brain">🧠</div><b>T2 MRI 분석 대기 중</b><small>왼쪽에서 환자 한 명의 DICOM 폴더를 선택하세요.<br>여러 시리즈가 있어도 T2 시리즈를 자동으로 찾습니다.</small></div></section>', unsafe_allow_html=True)
    elif not folder_scan.valid:
        st.markdown(f'<div class="validation error">✕　{folder_scan.message}</div>', unsafe_allow_html=True)
    elif not st.session_state.pipeline_done:
        expected_dicom_id = selected_patient["dicom_patient_id"]
        patient_id_matches = folder_scan.patient_id == expected_dicom_id
        masked_dicom_id = mask_patient_id(folder_scan.patient_id)
        st.markdown(f'<div class="validation">✓　{folder_scan.message}<div class="file-grid"><div><small>전체 파일</small><b>{folder_scan.total_files}개</b></div><div><small>DICOM / 시리즈</small><b>{folder_scan.dicom_files}개 / {folder_scan.series_count}개</b></div><div><small>선택 T2 시리즈</small><b>{folder_scan.selected_description}</b></div><div><small>DICOM 환자 ID / 슬라이스</small><b>{masked_dicom_id} · {folder_scan.selected_files}장</b></div></div></div>', unsafe_allow_html=True)
        if patient_id_matches:
            st.success(f"선택 환자 확인 완료 · {selected_patient_id}와 업로드 DICOM({masked_dicom_id})이 일치합니다.")
        else:
            st.warning(
                f"환자정보 불일치 · 선택 환자 {selected_patient_id}와 업로드 DICOM({masked_dicom_id})이 다릅니다. "
                "올바른 환자와 검사인지 확인한 뒤 진행하세요."
            )
        st.write("")
        button_label = "분석 시작" if patient_id_matches else "불일치 확인 후 분석 시작"
        if st.button(button_label, type="primary", use_container_width=True):
            progress = st.progress(0, text="DICOM 시리즈 무결성 검사")
            if local_status.ready:
                try:
                    prep = run_local_pipeline(
                        file_items, folder_scan, ROOT,
                        progress=lambda value, text: progress.progress(value, text=text),
                        selected_patient_id=selected_patient_id,
                        patient_id_match=patient_id_matches,
                    )
                except Exception as exc:
                    st.error(f"실제 전처리 실패: {exc}")
                    st.exception(exc)
                    st.stop()
            elif Path(local_status.script).is_file():
                st.error(f"실제 전처리 환경이 아직 완성되지 않았습니다. {local_status.message}")
                st.stop()
            else:
                time.sleep(.2); progress.progress(18, text="T2 시리즈 정렬")
                nifti_payload, nifti_name = convert_dicom_folder(file_items, folder_scan.selected_uid)
                nifti_validation = validate_nifti(nifti_payload, nifti_name)
                if not nifti_validation.valid:
                    st.error(nifti_validation.message)
                    st.stop()
                progress.progress(38, text="DICOM → 3D NIfTI 변환")
                time.sleep(.2); progress.progress(54, text="RAS 방향 표준화")
                time.sleep(.2); progress.progress(68, text="Intensity 정규화")
                prep = preprocess_nifti(nifti_payload, nifti_name)
                prep.update(pipeline_mode="cloud_lightweight", pipeline_version="deployable_v1", run_id="session_only")
                progress.progress(82, text="56×56×56 리사이즈 및 QC")
                time.sleep(.25); progress.progress(100, text="경량 전처리 완료")
            st.session_state.prep = prep
            st.session_state.folder_scan = folder_scan
            st.session_state.pipeline_done = True
            st.session_state.view = "전처리 결과"
            st.rerun()
    else:
        prep = st.session_state.prep
        st.markdown('<div style="color:#9db0c5;font-size:11px;margin:2px 0 7px">분석 완료 · 원하는 결과 화면을 선택해 비교하세요.</div>', unsafe_allow_html=True)
        view_options = ["원본 MRI", "전처리 결과", "AI 분석 (시연용)", "XAI 보고서"]
        if st.session_state.view not in view_options:
            st.session_state.view = "원본 MRI"
        view = st.segmented_control("결과 화면 선택", view_options, default=st.session_state.view, selection_mode="single", label_visibility="collapsed") or st.session_state.view
        st.session_state.view = view
        result = Result()
        if view == "원본 MRI":
            st.markdown('<div class="viewer-guide">원본과 전처리 결과가 같은 상대 위치로 연동됩니다. 슬라이더 위치는 화면을 전환해도 유지됩니다.</div>', unsafe_allow_html=True)
            st.markdown(interactive_mri_viewer(prep, "original"), unsafe_allow_html=True)
        elif view == "전처리 결과":
            st.markdown('<div class="viewer-guide">원본 MRI에서 선택한 상대 위치가 전처리 볼륨의 대응 슬라이스로 자동 변환됩니다.</div>', unsafe_allow_html=True)
            st.markdown(interactive_mri_viewer(prep, "processed"), unsafe_allow_html=True)
            st.markdown(f'<div class="qc-grid"><div class="qc ok"><small>NIfTI 검증</small><b>✓ 통과</b></div><div class="qc ok"><small>Orientation</small><b>✓ {prep["orientation"]}</b></div><div class="qc ok"><small>Intensity</small><b>✓ Min-Max</b></div><div class="qc ok"><small>출력 Shape</small><b>✓ {prep["final_shape"]}</b></div></div>', unsafe_allow_html=True)
            st.download_button("↓ 전처리 NIfTI 다운로드", prep["output_bytes"], prep["output_name"], "application/gzip")
            if prep.get("pipeline_mode") == "local_full":
                st.success(f'BRAINTENSOR {prep.get("pipeline_version")} 실제 전처리 완료 · 실행 ID: {prep.get("run_id")} · {prep.get("elapsed_sec")}초')
            else:
                st.info("Streamlit Cloud 경량 전처리 결과입니다. 로컬 실행 시 BRAINTENSOR의 BET·ANTsPy/PD25 정합·Min-Max·56³ 전체 파이프라인을 사용합니다.")
        elif view == "AI 분석 (시연용)":
            st.markdown('<div class="demo">⚠ 모델 학습 완료 전 디자인 확인용 시연 결과입니다. 실제 진단 결과가 아닙니다.</div>', unsafe_allow_html=True)
            demo_views = [data_url(ASSETS / "sample_t2_mri.png"), data_url(ASSETS / "coronal_result.png"), data_url(ASSETS / "sagittal_result.png")]
            st.markdown(viewer_html(demo_views, "AI 병변 시각화", "M3d-CAM 시연용"), unsafe_allow_html=True)
            st.markdown(f'<section class="panel"><div class="head"><span>▤</span>뉴로렌즈(AI) 판독 소견</div><div class="finding">{result.finding}</div><div class="warning"><b>파킨슨병 의심 확률({result.pd}%)</b> · 모델 연결 전 시연용 수치입니다.</div></section>', unsafe_allow_html=True)
        else:
            original_src = prep["original_views"][0]
            html = xai_report(result, original_src, prep, selected_patient_id, selected_patient)
            st.markdown(html, unsafe_allow_html=True)
            st.download_button("↓ XAI 보고서 HTML 다운로드", html.encode("utf-8"), "NeuroLens_XAI_Report.html", "text/html")

with info:
    panel(
        "환자정보",
        f'<dl class="pinfo"><b>ID</b>　{selected_patient_id}<br><br>'
        f'<b>성명/나이</b>　{selected_patient["name"]} ({selected_patient["sex_age"]})<br><br>'
        f'<b>관리 상태</b>　{selected_patient["condition"]}<br><br><b>검사</b>　T2 MRI</dl>',
        "●",
    )
    st.write("")
    if st.session_state.pipeline_done:
        active_view = st.session_state.view
        prep = st.session_state.prep
        active_scan = st.session_state.get("folder_scan", folder_scan)
        r = Result()
        if active_view == "원본 MRI":
            source_info = (
                f'<div class="reason"><b>선택 시리즈</b><br>{active_scan.selected_description}<br><br>'
                f'<b>DICOM 슬라이스</b><br>{active_scan.selected_files}장<br><br>'
                f'<b>변환 전 원본 크기</b><br>{prep["original_shape"]}<br><br>'
                f'<b>Voxel spacing / 방향</b><br>{prep["spacing"]} · {prep["orientation"]}</div>'
            )
            panel("원본 영상 정보", source_info, "◈")
        elif active_view == "전처리 결과":
            pipeline_label = "BRAINTENSOR ref21order_v1" if prep.get("pipeline_mode") == "local_full" else "클라우드 경량 전처리"
            run_label = prep.get("run_id", "-")
            preprocessing_status = (
                f'<div class="reason"><b>{pipeline_label}</b><br>실행 ID: {run_label}<br><br>'
                '✓ DICOM → NIfTI 변환<br>'
                + ('✓ FSL BET 뇌 추출<br>✓ ANTsPy + PD25 정합<br>' if prep.get("pipeline_mode") == "local_full" else '✓ 영상 방향 표준화<br>') +
                '✓ Min-Max 정규화<br>'
                '✓ 56×56×56 리사이즈<br><br>'
                f'<b>최종 출력</b><br>{prep["final_shape"]}</div>'
            )
            panel("전처리 상태", preprocessing_status, "⌁")
        elif active_view == "AI 분석 (시연용)":
            probs = probability("정상", r.normal, "#1a9d79") + probability("전구기", r.prodromal, "#e5a315") + probability("파킨슨병", r.pd, "#ff334b")
            panel("분석 결과 · 시연용", probs + f'<div class="reason"><b>판단 근거</b><br>{r.rationale}</div>', "◉")
            st.write("")
            panel("AI 모델 상태", '<div class="reason"><b>현재 상태</b><br>모델 연결 전 시연용 결과<br><br><b>예정 모델</b><br>3D-CNN + 3D-ResNet<br>Multi-View Attention</div>', "▣")
        else:
            report_status = (
                '<div class="reason"><b>보고서 생성 완료</b><br><br>'
                '✓ 환자 및 검사 정보<br>'
                '✓ XAI 시각화<br>'
                '✓ 진단 확률 요약<br>'
                '✓ 핵심 판독 요약<br><br>'
                '<b>주의</b><br>현재 AI 결과는 시연용이며 전문의 검토 전 최종 판독문으로 사용할 수 없습니다.</div>'
            )
            panel("XAI 보고서 상태", report_status, "▤")
    else:
        panel("검사 대기", '<div class="reason">DICOM 폴더를 선택하고 분석을 시작하면 현재 화면에 맞는 정보가 표시됩니다.</div>', "◉")
    if not file_items:
        status_class, status_text = "idle", "○ 환자 T2 MRI DICOM 폴더 선택 대기"
    elif not folder_scan.valid:
        status_class, status_text = "error", "● DICOM 확인 필요 · 유효한 T2 시리즈를 찾지 못함"
    elif not st.session_state.pipeline_done:
        status_class, status_text = "ready", "● 원본 DICOM 확인 완료 · 분석 시작 대기"
    elif st.session_state.view == "원본 MRI":
        status_class, status_text = "ready", "● 원본 MRI 불러오기 완료"
    elif st.session_state.view == "전처리 결과":
        mode = "BRAINTENSOR 실제 전처리" if prep.get("pipeline_mode") == "local_full" else "클라우드 경량 전처리"
        status_class, status_text = "", f"● {mode} 완료 · QC 통과"
    elif st.session_state.view == "AI 분석 (시연용)":
        status_class, status_text = "demo", "● AI 분석 시연 화면 · 모델 결과 연결 전"
    else:
        status_class, status_text = "demo", "● XAI 보고서 생성 완료 · 시연용 결과"
    st.markdown(f'<div class="status {status_class}">{status_text}</div>', unsafe_allow_html=True)

st.caption("본 서비스는 AI 진단 보조 프로토타입입니다. 최종 진단은 전문의의 판단을 따릅니다.")
