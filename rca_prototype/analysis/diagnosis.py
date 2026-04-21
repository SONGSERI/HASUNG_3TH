from __future__ import annotations

from typing import Dict

import pandas as pd

from rca_prototype.utils.formatting import confidence_label


def analyze_stage_machine(data: Dict[str, pd.DataFrame], detection: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    stage_summary = detection["stage_summary"].copy()
    machine_perf = data["machine_performance"].copy()
    events = data["machine_events"].copy()

    abnormal_stage = stage_summary.head(1).copy()
    machine_rank = (
        machine_perf.groupby(["machine_id"], as_index=False)
        .agg(
            actual_output=("actual_output", "sum"),
            expected_output=("expected_output", "sum"),
            cycle_delay_sec=("cycle_delay_sec", "mean"),
            stop_time_sec=("stop_time_sec", "mean"),
            pickup_error_count=("pickup_error_count", "sum"),
            recognition_error_count=("recognition_error_count", "sum"),
            mechanical_error_count=("mechanical_error_count", "sum"),
        )
    )
    machine_rank["output_gap"] = machine_rank["actual_output"] - machine_rank["expected_output"]
    peer_mean_gap = machine_rank["output_gap"].mean()
    peer_mean_delay = machine_rank["cycle_delay_sec"].mean()
    machine_rank["local_abnormality_score"] = (
        (-machine_rank["output_gap"]).rank(pct=True).fillna(0) * 0.45
        + machine_rank["cycle_delay_sec"].rank(pct=True).fillna(0) * 0.25
        + machine_rank["pickup_error_count"].rank(pct=True).fillna(0) * 0.30
    )
    machine_rank["confidence"] = machine_rank["local_abnormality_score"].apply(lambda v: confidence_label(2.0 + float(v) * 2.0))
    machine_rank["diagnosis_note"] = machine_rank.apply(
        lambda r: "이 설비는 동급 설비보다 훨씬 나쁘고 pickup 관련 손실이 이곳에 집중되어 있습니다."
        if r["output_gap"] < peer_mean_gap and r["cycle_delay_sec"] > peer_mean_delay
        else "이 설비는 기준선에 가깝고 비교 기준으로 볼 수 있습니다.",
        axis=1,
    )
    machine_rank = machine_rank.sort_values("local_abnormality_score", ascending=False).reset_index(drop=True)

    event_mix = (
        events.groupby("machine_id", as_index=False)
        .agg(
            pickup_error_count=("pickup_error_count", "sum"),
            recognition_error_count=("recognition_error_count", "sum"),
            mechanical_error_count=("mechanical_error_count", "sum"),
        )
    )
    return {
        "abnormal_stage": abnormal_stage,
        "machine_rank": machine_rank,
        "event_mix": event_mix,
    }
