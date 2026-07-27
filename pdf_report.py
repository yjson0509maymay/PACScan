from __future__ import annotations

import base64
import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#03143C")
BLUE = colors.HexColor("#1678E8")
LIGHT_BLUE = colors.HexColor("#EAF3FF")
LINE = colors.HexColor("#B9CCE3")
TEXT = colors.HexColor("#172033")
MUTED = colors.HexColor("#5D6C80")
FONT_NAME = "PACScanNotoSansKR"


def _register_font(font_path: Path) -> None:
    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(font_path)))


def _image_bytes(source: str | Path) -> bytes:
    if isinstance(source, Path):
        return source.read_bytes()
    if source.startswith("data:"):
        return base64.b64decode(source.split(",", 1)[1])
    return Path(source).read_bytes()


class ProbabilityBar(Flowable):
    def __init__(self, label: str, value: int, color: colors.Color, width: float = 165 * mm):
        super().__init__()
        self.label = label
        self.value = value
        self.bar_color = color
        self.width = width
        self.height = 10 * mm

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFont(FONT_NAME, 10)
        canvas.setFillColor(TEXT)
        canvas.drawString(0, 5.2 * mm, self.label)
        canvas.setFillColor(colors.HexColor("#EDF1F6"))
        canvas.roundRect(34 * mm, 5.4 * mm, 115 * mm, 3.2 * mm, 1.6 * mm, fill=1, stroke=0)
        canvas.setFillColor(self.bar_color)
        canvas.roundRect(
            34 * mm,
            5.4 * mm,
            115 * mm * self.value / 100,
            3.2 * mm,
            1.6 * mm,
            fill=1,
            stroke=0,
        )
        canvas.setFont(FONT_NAME, 11)
        canvas.drawRightString(self.width, 5.2 * mm, f"{self.value}%")


def generate_xai_pdf(
    *,
    result,
    original_src: str,
    prep: dict,
    patient_id: str,
    patient: dict,
    exam_date: str,
    generated_at: str,
    app_version: str,
    assets_dir: Path,
    heatmap_src: str | None = None,
    model_connected: bool = False,
) -> bytes:
    _register_font(assets_dir / "fonts" / "NotoSansKR-VF.ttf")
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=15 * mm,
        title=f"ParkinsLens XAI 판독 결과 - {patient_id}",
        author="ParkinsLens PACScan",
    )
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "KoreanNormal",
        parent=styles["Normal"],
        fontName=FONT_NAME,
        fontSize=9.5,
        leading=15,
        textColor=TEXT,
        wordWrap="CJK",
    )
    small = ParagraphStyle(
        "KoreanSmall",
        parent=normal,
        fontSize=8,
        leading=12,
        textColor=MUTED,
    )
    section = ParagraphStyle(
        "KoreanSection",
        parent=normal,
        fontSize=14,
        leading=19,
        textColor=NAVY,
        spaceAfter=7,
    )
    title = ParagraphStyle(
        "KoreanTitle",
        parent=normal,
        fontSize=22,
        leading=28,
        textColor=colors.white,
    )
    center = ParagraphStyle("KoreanCenter", parent=normal, alignment=TA_CENTER)
    right = ParagraphStyle("KoreanRight", parent=small, alignment=TA_RIGHT)

    logo = Image(io.BytesIO(_image_bytes(assets_dir / "logo.png")), width=31 * mm, height=31 * mm)
    header = Table(
        [[Paragraph("<b>ParkinsLens XAI</b><br/><font size='13'>분석 보고서</font>", title), logo]],
        colWidths=[145 * mm, 31 * mm],
        rowHeights=[35 * mm],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (0, 0), 10 * mm),
                ("RIGHTPADDING", (-1, 0), (-1, 0), 2 * mm),
                ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
            ]
        )
    )

    patient_table = Table(
        [
            [Paragraph("<b>환자 ID</b>", normal), Paragraph(patient_id, normal)],
            [Paragraph("<b>검사일</b>", normal), Paragraph(exam_date, normal)],
            [Paragraph("<b>검사 유형</b>", normal), Paragraph("T2 MRI", normal)],
            [Paragraph("<b>판독 상태</b>", normal), Paragraph("<font color='#076DDE'><b>자동 생성 완료</b></font>", normal)],
        ],
        colWidths=[34 * mm, 54 * mm],
    )
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F4F7FB")),
                ("GRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    full_pipeline = prep.get("pipeline_mode") == "local_full"
    pipeline_name = "BRAINTENSOR ref21order_v1" if full_pipeline else "PACScan 배포용 전처리"
    if model_connected:
        summary_text = (
            "본 보고서는 전처리 결과와 실제 학습된 모델(Variant3)의 추론 결과를 바탕으로 자동 생성되었습니다. "
            "논문 재현 목표 정확도(93.41%)에 아직 못 미치는 연구 프로토타입으로, "
            "최종 판독과 진단은 담당 전문의의 임상적 판단을 우선합니다."
        )
        model_line = "<b>모델</b>　3D-CNN(Variant3) + Grad-CAM"
        status_line = "<b>현재 상태</b>　모델 연결됨 · 실제 추론 결과"
    else:
        summary_text = (
            "본 보고서는 전처리 결과와 현재 시연용 AI 결과를 바탕으로 자동 생성되었습니다. "
            "최종 판독과 진단은 담당 전문의의 임상적 판단을 우선합니다."
        )
        model_line = "<b>예정 모델</b>　3D-CNN + 3D-ResNet / 다중 시점 어텐션"
        status_line = "<b>현재 상태</b>　모델 연결 전 시연용 결과"
    model_info = [
        Paragraph("<b>AI 보조 판독 요약</b>", section),
        Paragraph(summary_text, normal),
        Spacer(1, 3 * mm),
        Paragraph(f"<b>전처리</b>　{pipeline_name}", normal),
        Paragraph(f"<b>실행 ID</b>　{prep.get('run_id', '세션 전용')}", small),
        Paragraph(model_line, normal),
        Paragraph(status_line, normal),
    ]
    top = Table(
        [
            [
                [Paragraph("<b>환자 정보</b>", section), patient_table],
                model_info,
            ]
        ],
        colWidths=[91 * mm, 85 * mm],
    )
    top.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )

    original = Image(io.BytesIO(_image_bytes(original_src)), width=78 * mm, height=67 * mm, kind="proportional")
    heatmap_source = heatmap_src if heatmap_src else (assets_dir / "sample_t2_mri.png")
    heatmap = Image(
        io.BytesIO(_image_bytes(heatmap_source)),
        width=78 * mm,
        height=67 * mm,
        kind="proportional",
    )
    heatmap_caption = "AI 분석 결과 (Grad-CAM 히트맵)" if model_connected else "AI 분석 결과 (시연용 히트맵)"
    image_table = Table(
        [
            [Paragraph("<b>원본 MRI (T2)</b>", center), Paragraph(f"<b>{heatmap_caption}</b>", center)],
            [original, heatmap],
        ],
        colWidths=[88 * mm, 88 * mm],
    )
    image_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, LINE),
                ("ALIGN", (0, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story = [
        header,
        Spacer(1, 5 * mm),
        top,
        Spacer(1, 5 * mm),
        Paragraph("<b>XAI 시각화</b>　<font size='9'>(M3d-CAM)</font>", section),
        image_table,
        PageBreak(),
        Paragraph("<b>AI 진단 확률 요약</b>", section),
        ProbabilityBar("정상", result.normal, colors.HexColor("#1556C0")),
        ProbabilityBar("전구기", result.prodromal, colors.HexColor("#FF8C00")),
        ProbabilityBar("파킨슨병 의심", result.pd, colors.HexColor("#E91D2B")),
        Spacer(1, 7 * mm),
        Paragraph("<b>핵심 판독 요약</b>", section),
        Table(
            [
                [Paragraph("1", center), Paragraph(result.finding, normal)],
                [Paragraph("2", center), Paragraph(f"파킨슨병 의심 확률이 {result.pd}%로 분석되었습니다.", normal)],
                [Paragraph("3", center), Paragraph("임상 증상 및 추가 검사와 종합하여 전문의가 최종 판단해야 합니다.", normal)],
            ],
            colWidths=[12 * mm, 164 * mm],
            style=[
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ],
        ),
        Spacer(1, 8 * mm),
        Paragraph("<b>판단 근거</b>", section),
        Paragraph(result.rationale, normal),
        Spacer(1, 10 * mm),
        Table(
            [
                [Paragraph(f"<b>생성일시</b>　{generated_at}", normal), Paragraph("담당 전문의 서명　________________", right)],
                [Paragraph(f"ParkinsLens / PACScan v{app_version}", small), Paragraph("AI 보조 결과 - 임상 진단용 아님", right)],
            ],
            colWidths=[88 * mm, 88 * mm],
            style=[
                ("LINEABOVE", (0, 0), (-1, 0), 0.8, NAVY),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ],
        ),
    ]
    document.build(story)
    return output.getvalue()
