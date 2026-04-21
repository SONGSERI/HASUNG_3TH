from __future__ import annotations

from typing import Dict

import pandas as pd

from rca_prototype.utils.formatting import confidence_label, safe_float


def analyze_lot_impact(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    lots = data["lot_impact"].copy().sort_values(["lot_output_gap", "lot_defect_rate"], ascending=[True, False]).reset_index(drop=True)
    if lots.empty:
        return {"lot_summary": lots, "lot_finding": pd.DataFrame()}
    defect_mean = lots["lot_defect_rate"].mean()
    output_gap_mean = lots["lot_output_gap"].mean()
    lots["impact_type"] = lots.apply(
        lambda r: "전파형 LOT 이슈"
        if safe_float(r["affected_machine_count"]) >= 2 and safe_float(r["lot_defect_rate"]) > defect_mean
        else "국소 LOT 이슈"
        if safe_float(r["lot_output_gap"]) < output_gap_mean
        else "관찰 필요",
        axis=1,
    )
    top = lots.iloc[0]
    finding = pd.DataFrame(
        [
            {
                "finding": f"{top['lot_id']} LOT이 다른 LOT보다 더 크게 영향을 받았습니다.",
                "why_it_matters": "설비 이슈가 모든 LOT에 동일하게 분포한 것은 아닙니다.",
                "next_check": "영향 LOT와 직전 LOT를 비교하고 같은 부품 셋업을 사용했는지 확인합니다.",
                "confidence": confidence_label(2.0 + (safe_float(top['lot_defect_rate']) > defect_mean) * 0.8),
                "impact_type": top["impact_type"],
            }
        ]
    )
    return {"lot_summary": lots, "lot_finding": finding}
