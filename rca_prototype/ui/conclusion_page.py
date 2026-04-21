from __future__ import annotations

from typing import Dict

import pandas as pd
import streamlit as st

from rca_prototype.ui.common import bullet_block, section_title


def render_page(bundle: Dict) -> None:
    causes = bundle["causes"]["candidates"]
    section_title(
        "공장은 무엇부터 점검해야 하는가?",
        "이 프로토타입은 이상 실장기의 흡착 / 피더 / 노즐 조건을 가장 우선적으로 점검해야 한다고 보여줍니다.",
    )
    top_machine = bundle["diagnosis"]["machine_rank"].iloc[0]["machine_id"]
    top_cause = causes.iloc[0]["cause_category"] if not causes.empty else "-"
    confirmed = [
        "실장기 공정이 라인 손실의 핵심 구간입니다.",
        f"{top_machine} 설비가 동급 설비보다 더 나쁩니다.",
        "설비 이상 시간대 이후 AOI 불량이 증가했습니다.",
    ]
    suspected = [
        "부품 흡착 관련 불안정이 가장 강한 원인 후보입니다.",
        "FDR-05, NZ-03, PN-004 조합이 핵심 집중 지점으로 보입니다.",
    ]
    uncertain = [
        "현재는 synthetic 계획 생산량과 표준 CT를 사용한 데모입니다.",
        "실제 전환 시에는 공장 실제 계획량, 표준 CT, routing time으로 교체해 검증해야 합니다.",
    ]
    action_df = pd.DataFrame(
        [
            {"priority": 1, "action": f"실장기 {top_machine} 설비를 가장 먼저 점검", "purpose": "설비 국소 이상 여부 확인"},
            {"priority": 2, "action": "FDR-05 피더 정렬 상태 확인", "purpose": "흡착 관련 집중 여부 검증"},
            {"priority": 3, "action": "NZ-03 노즐 상태 점검", "purpose": "노즐 마모 또는 오염 배제"},
            {"priority": 4, "action": "PN-004 부품 공급 안정성 확인", "purpose": "자재 조건이 흡착 손실을 만들었는지 확인"},
            {"priority": 5, "action": "영향 LOT와 직전 LOT 비교", "purpose": "영향이 국소적인지 확산형인지 확인"},
        ]
    )
    action_view = action_df.rename(columns={"priority": "우선순위", "action": "조치 항목", "purpose": "목적"})
    st.dataframe(action_view, use_container_width=True, hide_index=True)
    bullet_block(confirmed, "확정적으로 볼 수 있는 내용")
    bullet_block(suspected, "가장 유력한 추정 내용")
    bullet_block(uncertain, "추가 확인이 필요한 내용")
