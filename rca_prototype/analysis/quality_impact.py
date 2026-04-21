from __future__ import annotations

from typing import Dict

import pandas as pd

from rca_prototype.utils.formatting import confidence_label, safe_float


def analyze_quality_impact(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    quality = data["quality_data"].copy()
    machine = data["machine_performance"].copy()
    issue_window = machine.sort_values("pickup_error_count", ascending=False).head(3)["timestamp"].tolist()
    quality_time = (
        quality.groupby(["timestamp"], as_index=False)
        .agg(
            aoi_fail_count=("aoi_fail_count", "sum"),
            defect_rate=("defect_rate", "mean"),
        )
        .sort_values("timestamp")
    )
    baseline_defect = quality_time["aoi_fail_count"].median()
    peak = quality_time.sort_values("aoi_fail_count", ascending=False).head(1)
    peak_row = peak.iloc[0] if not peak.empty else pd.Series(dtype="object")
    quality_effect = pd.DataFrame(
        [
            {
                "finding": "실장기 이상 시간대 이후 AOI 불량이 증가했습니다.",
                "why_it_matters": "이번 이슈는 생산량만 낮춘 것이 아니라 품질에도 영향을 주고 있습니다.",
                "next_check": "배치 관련 AOI 불량 이미지와 영향 LOT를 함께 확인합니다.",
                "confidence": confidence_label(2.3 + (safe_float(peak_row.get("aoi_fail_count", 0)) > baseline_defect) * 1.0),
                "peak_timestamp": peak_row.get("timestamp", "-"),
            }
        ]
    )
    defect_mix = (
        quality.groupby(["defect_type"], as_index=False)
        .agg(defect_count=("defect_count", "sum"))
        .sort_values("defect_count", ascending=False)
    )
    return {
        "quality_time": quality_time,
        "quality_effect": quality_effect,
        "defect_mix": defect_mix,
        "issue_window": pd.DataFrame({"timestamp": issue_window}),
    }
