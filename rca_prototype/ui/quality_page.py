from __future__ import annotations

from typing import Dict

import plotly.express as px
import streamlit as st

from rca_prototype.ui.common import section_title, takeaway


def render_page(bundle: Dict) -> None:
    quality = bundle["quality"]
    lots = bundle["lots"]
    section_title(
        "이 설비 이슈가 하류 품질과 LOT에 영향을 주었는가?",
        "실장기 이상 이후 AOI 불량이 증가했고, 일부 LOT는 다른 LOT보다 더 크게 영향을 받았습니다.",
    )
    quality_time = quality["quality_time"].copy()
    quality_view = quality_time.rename(columns={"timestamp": "시간", "aoi_fail_count": "AOI 불량 수"})
    fig = px.line(quality_view, x="시간", y="AOI 불량 수", markers=True, title="시간대별 AOI 불량 추이")
    st.plotly_chart(fig, use_container_width=True)
    takeaway("실장기 이슈 이후 약간의 시차를 두고 AOI 불량이 증가했습니다.", "피크 구간의 배치 관련 AOI 불량 유형을 확인합니다.")
    defect_mix = quality["defect_mix"].copy()
    defect_view = defect_mix.rename(columns={"defect_type": "불량 유형", "defect_count": "불량 수"})
    fig2 = px.bar(defect_view, x="불량 유형", y="불량 수", title="AOI 불량 유형 분포")
    st.plotly_chart(fig2, use_container_width=True)
    takeaway("하류 영향은 배치 관련 불량이 중심입니다.", "영향 LOT의 AOI 이미지와 배치 오프셋을 비교합니다.")
    lot_summary = lots["lot_summary"].copy()
    lot_view = lot_summary.rename(
        columns={
            "lot_id": "LOT",
            "model_id": "모델",
            "affected_machine_count": "영향 설비 수",
            "affected_stage_count": "영향 공정 수",
            "lot_output_gap": "LOT 생산량 차이",
            "lot_defect_rate": "LOT 불량률",
            "impact_type": "영향 유형",
        }
    )
    fig3 = px.scatter(lot_view, x="LOT 생산량 차이", y="LOT 불량률", size="영향 설비 수", color="영향 유형", hover_name="LOT", title="LOT 영향 비교")
    st.plotly_chart(fig3, use_container_width=True)
    takeaway("LOT-002와 LOT-003이 기준 LOT보다 더 큰 영향을 받았습니다.", "이 LOT들에 같은 자재 셋업이 사용됐는지 확인합니다.")
    st.dataframe(lot_view[["LOT", "모델", "영향 설비 수", "영향 공정 수", "LOT 생산량 차이", "LOT 불량률", "영향 유형"]], use_container_width=True, hide_index=True)
