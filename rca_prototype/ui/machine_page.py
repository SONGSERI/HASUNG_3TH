from __future__ import annotations

from typing import Dict

import plotly.express as px
import streamlit as st

from rca_prototype.ui.common import section_title, takeaway


def render_page(bundle: Dict) -> None:
    diagnosis = bundle["diagnosis"]
    section_title(
        "어느 공정과 어느 설비가 책임 구간인가?",
        "문제는 실장기 공정에 국소화되어 있고, 그 안에서도 한 설비가 다른 설비보다 훨씬 나쁩니다.",
    )
    machine_rank = diagnosis["machine_rank"].copy()
    machine_view = machine_rank.rename(
        columns={
            "machine_id": "설비",
            "output_gap": "생산량 차이",
            "pickup_error_count": "흡착 에러 수",
            "cycle_delay_sec": "Cycle 지연(초)",
            "actual_output": "실제 생산량",
            "expected_output": "계획 생산량",
            "confidence": "신뢰도",
            "diagnosis_note": "해석",
        }
    )
    fig = px.bar(machine_view, x="설비", y="생산량 차이", color="흡착 에러 수", text="Cycle 지연(초)", title="실장기 설비별 비교")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("M05 설비가 가장 큰 생산 손실과 흡착 관련 이상을 보입니다.", "같은 설비가 정지 시간과 에러 집중도도 함께 높은지 확인합니다.")
    st.dataframe(
        machine_view[["설비", "실제 생산량", "계획 생산량", "생산량 차이", "Cycle 지연(초)", "흡착 에러 수", "신뢰도", "해석"]],
        use_container_width=True,
        hide_index=True,
    )
