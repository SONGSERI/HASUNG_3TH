from __future__ import annotations

from typing import Dict

import plotly.express as px
import streamlit as st

from rca_prototype.ui.common import section_title, takeaway


def render_page(bundle: Dict) -> None:
    diagnosis = bundle["diagnosis"]
    causes = bundle["causes"]["candidates"]
    material = bundle["data"]["material_context"].copy()
    section_title(
        "가장 가능성이 높은 원인 후보는 무엇인가?",
        "인식 오류나 기계 오류보다 부품 흡착 관련 근거가 더 강하고, 특정 피더·노즐·부품 조합에 집중되어 있습니다.",
    )
    event_mix = diagnosis["event_mix"].melt(id_vars="machine_id", value_vars=["pickup_error_count", "recognition_error_count", "mechanical_error_count"], var_name="error_type", value_name="error_count")
    event_mix["error_type"] = event_mix["error_type"].map(
        {
            "pickup_error_count": "흡착 에러",
            "recognition_error_count": "인식 에러",
            "mechanical_error_count": "기계 에러",
        }
    )
    event_mix = event_mix.rename(columns={"machine_id": "설비", "error_count": "에러 수"})
    fig = px.bar(event_mix, x="설비", y="에러 수", color="error_type", barmode="group", title="설비별 에러 유형 비교")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("이상 설비에서는 흡착 관련 에러가 가장 지배적입니다.", "같은 설비에서 피더와 노즐 집중도도 함께 확인합니다.")
    top_material = (
        material.groupby(["machine_id", "feeder_id", "nozzle_id", "part_number"], as_index=False)
        .agg(pickup_error_count=("pickup_error_count", "sum"))
        .sort_values("pickup_error_count", ascending=False)
        .head(8)
    )
    material_view = top_material.rename(
        columns={
            "machine_id": "설비",
            "feeder_id": "피더",
            "nozzle_id": "노즐",
            "part_number": "부품",
            "pickup_error_count": "흡착 에러 수",
        }
    )
    fig2 = px.bar(material_view, x="피더", y="흡착 에러 수", color="설비", hover_data=["노즐", "부품"], title="피더 / 노즐 / 부품 집중도")
    st.plotly_chart(fig2, use_container_width=True)
    takeaway("이상 패턴은 FDR-05, NZ-03, PN-004 조합에 집중되어 있습니다.", "피더 정렬, 노즐 마모, 부품 공급 안정성을 먼저 점검합니다.")
    cause_view = causes[["rank", "cause_category", "hypothesis", "confidence", "evidence"]].rename(
        columns={
            "rank": "순위",
            "cause_category": "원인 범주",
            "hypothesis": "가설",
            "confidence": "신뢰도",
            "evidence": "근거",
        }
    )
    st.dataframe(cause_view, use_container_width=True, hide_index=True)
