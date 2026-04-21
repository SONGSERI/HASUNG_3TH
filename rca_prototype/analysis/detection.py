from __future__ import annotations

from typing import Dict

import pandas as pd

from rca_prototype.utils.formatting import confidence_label, safe_float


def analyze_problem_detection(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    stage_perf = data["stage_performance"].copy()
    machine_perf = data["machine_performance"].copy()

    line_time = (
        stage_perf.groupby(["timestamp"], as_index=False)
        .agg(
            line_actual_output=("actual_output", "sum"),
            line_expected_output=("expected_output", "sum"),
            avg_cycle_delay_sec=("cycle_delay_sec", "mean"),
        )
    )
    line_time["line_output_gap"] = line_time["line_actual_output"] - line_time["line_expected_output"]
    gap_threshold = line_time["line_output_gap"].quantile(0.15)
    line_time["is_anomaly"] = line_time["line_output_gap"] <= gap_threshold

    stage_summary = (
        stage_perf.groupby(["stage_id", "stage_name"], as_index=False)
        .agg(
            actual_output=("actual_output", "sum"),
            expected_output=("expected_output", "sum"),
            avg_cycle_delay_sec=("cycle_delay_sec", "mean"),
            avg_stop_time_sec=("stop_time_sec", "mean"),
        )
    )
    stage_summary["output_gap"] = stage_summary["actual_output"] - stage_summary["expected_output"]
    stage_summary["severity_score"] = (
        (-stage_summary["output_gap"]).rank(pct=True).fillna(0) * 0.55
        + stage_summary["avg_cycle_delay_sec"].rank(pct=True).fillna(0) * 0.30
        + stage_summary["avg_stop_time_sec"].rank(pct=True).fillna(0) * 0.15
    )
    stage_summary = stage_summary.sort_values("severity_score", ascending=False).reset_index(drop=True)

    machine_summary = (
        machine_perf.groupby(["machine_id"], as_index=False)
        .agg(
            actual_output=("actual_output", "sum"),
            expected_output=("expected_output", "sum"),
            cycle_delay_sec=("cycle_delay_sec", "mean"),
            stop_time_sec=("stop_time_sec", "mean"),
            pickup_error_count=("pickup_error_count", "sum"),
        )
    )
    machine_summary["output_gap"] = machine_summary["actual_output"] - machine_summary["expected_output"]
    machine_summary["severity_score"] = (
        (-machine_summary["output_gap"]).rank(pct=True).fillna(0) * 0.5
        + machine_summary["cycle_delay_sec"].rank(pct=True).fillna(0) * 0.2
        + machine_summary["stop_time_sec"].rank(pct=True).fillna(0) * 0.1
        + machine_summary["pickup_error_count"].rank(pct=True).fillna(0) * 0.2
    )
    machine_summary = machine_summary.sort_values("severity_score", ascending=False).reset_index(drop=True)

    findings = []
    if not stage_summary.empty:
        top_stage = stage_summary.iloc[0]
        findings.append(
            {
                "finding_type": "공정",
                "target": top_stage["stage_name"],
                "finding": "주요 생산 손실은 Mounter 공정에 집중되어 있습니다." if str(top_stage["stage_id"]) == "MNT" else f"주요 생산 손실은 {top_stage['stage_name']} 공정에 집중되어 있습니다.",
                "why_it_matters": "이 공정이 라인 throughput을 가장 크게 제한할 가능성이 높습니다.",
                "next_check": "이상 공정 내부의 설비별 차이를 비교합니다.",
                "confidence": confidence_label(2.5 + safe_float(top_stage["severity_score"])),
            }
        )
    if not machine_summary.empty:
        top_machine = machine_summary.iloc[0]
        findings.append(
            {
                "finding_type": "설비",
                "target": top_machine["machine_id"],
                "finding": f"{top_machine['machine_id']} 설비가 동급 설비보다 훨씬 불안정합니다.",
                "why_it_matters": "라인 전체 문제가 아니라 특정 설비에 국소화된 이슈일 가능성이 높습니다.",
                "next_check": "pickup 관련 에러와 feeder/nozzle 집중도를 확인합니다.",
                "confidence": confidence_label(2.7 + safe_float(top_machine["severity_score"])),
            }
        )
    anomaly_window = line_time[line_time["is_anomaly"]].sort_values("line_output_gap").head(1)
    if not anomaly_window.empty:
        row = anomaly_window.iloc[0]
        findings.append(
            {
                "finding_type": "시간 구간",
                "target": str(row["timestamp"]),
                "finding": "특정 시간 구간에 라인 생산성이 뚜렷하게 떨어졌습니다.",
                "why_it_matters": "시간 구간이 좁을수록 국소적인 운영 이슈일 가능성이 높습니다.",
                "next_check": "이상 시간대를 설비 에러와 AOI 품질 변화와 연결해 봅니다.",
                "confidence": "높음",
            }
        )

    return {
        "line_time": line_time,
        "stage_summary": stage_summary,
        "machine_summary": machine_summary,
        "top_findings": pd.DataFrame(findings),
    }
