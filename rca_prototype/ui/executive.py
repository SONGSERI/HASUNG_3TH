from __future__ import annotations

from typing import Dict

import streamlit as st

from rca_prototype.ui.common import bullet_block, section_title


def render_page(bundle: Dict) -> None:
    msgs = bundle["messages"]
    findings = bundle["detection"]["top_findings"]
    causes = bundle["causes"]["candidates"]
    quality = bundle["quality"]["quality_effect"]
    lots = bundle["lots"]["lot_finding"]
    st.markdown(
        """
        <style>
        div[data-testid="stMetricValue"]{
            font-size:1.35rem !important;
            line-height:1.25 !important;
        }
        div[data-testid="stMetricLabel"]{
            font-size:.88rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    section_title(
        "이번 RCA 시나리오의 핵심 내용은 무엇인가?",
        msgs["headline"],
    )
    cols = st.columns(4)
    metrics = [
        ("핵심 이슈", findings.iloc[0]["target"] if not findings.empty else "-"),
        ("우선 원인", causes.iloc[0]["cause_category"] if not causes.empty else "-"),
        ("품질 영향", quality.iloc[0]["confidence"] if not quality.empty else "-"),
        ("LOT 영향", lots.iloc[0]["impact_type"] if not lots.empty else "-"),
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)
    if not findings.empty:
        view = findings[["finding_type", "target", "finding", "confidence"]].rename(
            columns={
                "finding_type": "구분",
                "target": "대상",
                "finding": "발견 내용",
                "confidence": "신뢰도",
            }
        )
        st.dataframe(view, use_container_width=True, hide_index=True)
    bullet_block(
        [
            msgs["top_issue"],
            msgs["top_cause"],
            msgs["quality_msg"],
            msgs["lot_msg"],
        ],
        "요약",
    )
