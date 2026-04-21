from __future__ import annotations

from typing import Dict

import plotly.express as px
import streamlit as st

from rca_prototype.ui.common import section_title, takeaway


def render_page(bundle: Dict) -> None:
    detection = bundle["detection"]
    section_title(
        "라인 생산성은 어디에서 무너지기 시작했는가?",
        "라인 손실은 좁은 시간 구간에 나타났고, 그중 실장기 공정이 가장 큰 비중을 차지합니다. 이 페이지는 synthetic 계획 생산량과 표준 CT를 기준으로 설명합니다.",
    )
    line_time = detection["line_time"].copy()
    line_view = line_time.rename(
        columns={
            "timestamp": "시간",
            "line_actual_output": "실제 생산량",
            "line_expected_output": "계획 생산량",
        }
    )
    fig = px.line(line_view, x="시간", y=["실제 생산량", "계획 생산량"], markers=True, title="시간대별 실제 생산량 vs 계획 생산량")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("특정 시간 구간에 생산량이 계획치 아래로 떨어졌습니다.", "이상 시간대를 공정과 설비 변화와 비교합니다.")
    stage_summary = detection["stage_summary"].copy()
    stage_view = stage_summary.rename(
        columns={
            "stage_name": "공정",
            "output_gap": "생산량 차이",
            "avg_cycle_delay_sec": "평균 지연 시간(초)",
            "actual_output": "실제 생산량",
            "expected_output": "계획 생산량",
        }
    )
    fig2 = px.bar(stage_view, x="공정", y="생산량 차이", color="평균 지연 시간(초)", title="공정별 생산량 차이")
    st.plotly_chart(fig2, use_container_width=True)
    takeaway("실장기 공정이 가장 뚜렷한 공정 병목입니다.", "다음으로 실장기 내부 설비 비교로 내려갑니다.")
    st.dataframe(stage_view[["공정", "실제 생산량", "계획 생산량", "생산량 차이", "평균 지연 시간(초)"]], use_container_width=True, hide_index=True)
