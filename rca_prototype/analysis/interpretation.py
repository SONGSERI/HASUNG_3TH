from __future__ import annotations

from typing import Dict

import pandas as pd


def build_executive_messages(analysis_bundle: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, str]:
    detection = analysis_bundle["detection"]["top_findings"]
    causes = analysis_bundle["causes"]["candidates"]
    quality = analysis_bundle["quality"]["quality_effect"]
    lots = analysis_bundle["lots"]["lot_finding"]
    top_issue = detection.iloc[0]["finding"] if not detection.empty else "라인에서 국소적인 생산성 저하가 관찰됩니다."
    top_cause = causes.iloc[0]["hypothesis"] if not causes.empty else "픽업 관련 원인은 아직 추가 확인이 필요합니다."
    quality_msg = quality.iloc[0]["finding"] if not quality.empty else "하류 품질 영향은 아직 뚜렷하게 확인되지 않았습니다."
    lot_msg = lots.iloc[0]["finding"] if not lots.empty else "LOT 영향은 추가 확인이 필요합니다."
    headline = (
        "이 프로토타입은 실장기 공정 불안정이 라인 손실의 핵심 원인임을 보여주며, "
        "특정 설비에서 부품 흡착 관련 근거가 가장 강하고 AOI 하류 영향도 함께 관찰된다는 시나리오를 설명합니다."
    )
    return {
        "headline": headline,
        "top_issue": top_issue,
        "top_cause": top_cause,
        "quality_msg": quality_msg,
        "lot_msg": lot_msg,
    }
