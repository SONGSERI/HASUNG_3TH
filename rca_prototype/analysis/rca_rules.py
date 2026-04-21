from __future__ import annotations

from typing import Dict

import pandas as pd

from rca_prototype.utils.constants import CAUSE_ORDER
from rca_prototype.utils.formatting import confidence_label, safe_float


def rank_cause_candidates(data: Dict[str, pd.DataFrame], detection: Dict[str, pd.DataFrame], diagnosis: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    stage_summary = detection["stage_summary"]
    machine_rank = diagnosis["machine_rank"]
    material = data["material_context"].copy()
    lots = data["lot_impact"].copy()

    top_machine = machine_rank.iloc[0] if not machine_rank.empty else pd.Series(dtype="object")
    top_stage = stage_summary.iloc[0] if not stage_summary.empty else pd.Series(dtype="object")
    top_material = (
        material.groupby(["machine_id", "feeder_id", "nozzle_id", "part_number"], as_index=False)
        .agg(
            feeder_error_count=("feeder_error_count", "sum"),
            nozzle_error_count=("nozzle_error_count", "sum"),
            part_related_error_count=("part_related_error_count", "sum"),
            pickup_error_count=("pickup_error_count", "sum"),
        )
        .sort_values("pickup_error_count", ascending=False)
        .head(1)
    )
    top_lot = lots.sort_values(["lot_output_gap", "lot_defect_rate"], ascending=[True, False]).head(1)
    top_mat = top_material.iloc[0] if not top_material.empty else pd.Series(dtype="object")
    top_lot_row = top_lot.iloc[0] if not top_lot.empty else pd.Series(dtype="object")

    material_score = 0.0
    if not top_material.empty:
        material_score += 1.6
        if safe_float(top_mat.get("pickup_error_count", 0)) > safe_float(top_mat.get("feeder_error_count", 0)) + safe_float(top_mat.get("nozzle_error_count", 0)):
            material_score += 1.0
        if safe_float(top_mat.get("feeder_error_count", 0)) > 15:
            material_score += 0.8
        if str(top_mat.get("machine_id", "")) == str(top_machine.get("machine_id", "")):
            material_score += 0.8

    machine_score = 0.0
    if not machine_rank.empty:
        machine_score += 1.4
        if safe_float(top_machine.get("local_abnormality_score", 0)) > 0.8:
            machine_score += 1.0
        if safe_float(top_machine.get("pickup_error_count", 0)) > safe_float(top_machine.get("recognition_error_count", 0)) + safe_float(top_machine.get("mechanical_error_count", 0)):
            machine_score += 0.6

    process_score = 0.0
    if not stage_summary.empty:
        process_score += 1.2
        if str(top_stage.get("stage_id", "")) == "MNT":
            process_score += 0.8
        if safe_float(top_stage.get("avg_cycle_delay_sec", 0)) > 1.0:
            process_score += 0.5
        if machine_rank.shape[0] > 1 and safe_float(machine_rank.iloc[1]["local_abnormality_score"]) < safe_float(top_machine.get("local_abnormality_score", 0)) * 0.7:
            process_score -= 0.3

    lot_score = 0.0
    if not top_lot.empty:
        lot_score += 1.0
        if safe_float(top_lot_row.get("affected_machine_count", 0)) >= 2:
            lot_score += 0.6
        if safe_float(top_lot_row.get("lot_defect_rate", 0)) > lots["lot_defect_rate"].mean():
            lot_score += 0.6

    candidates = pd.DataFrame(
        [
            {
                "cause_category": "자재 / 피더 / 노즐 관련",
                "score": material_score,
                "confidence": confidence_label(material_score),
                "hypothesis": "부품 흡착 관련 에러가 특정 피더, 노즐, 부품 조합에 집중되어 있습니다.",
                "evidence": f"{top_mat.get('machine_id', '-')}, {top_mat.get('feeder_id', '-')}, {top_mat.get('nozzle_id', '-')}, {top_mat.get('part_number', '-')}",
            },
            {
                "cause_category": "설비 관련",
                "score": machine_score,
                "confidence": confidence_label(machine_score),
                "hypothesis": "특정 실장기 한 대가 다른 설비보다 훨씬 불안정합니다.",
                "evidence": f"{top_machine.get('machine_id', '-')} 설비가 생산 차이와 사이클 지연이 가장 나쁩니다.",
            },
            {
                "cause_category": "공정 흐름 관련",
                "score": process_score,
                "confidence": confidence_label(process_score),
                "hypothesis": "실장기 공정이 병목으로 보이지만, 문제는 라인 전체보다 국소적입니다.",
                "evidence": f"{top_stage.get('stage_name', '-')} 공정에서 공정 단위 손실이 가장 크게 나타납니다.",
            },
            {
                "cause_category": "제품 / LOT 관련",
                "score": lot_score,
                "confidence": confidence_label(lot_score),
                "hypothesis": "일부 LOT가 더 크게 영향을 받았지만, 패턴의 시작점은 같은 실장기 이상 시간대로 보입니다.",
                "evidence": f"{top_lot_row.get('lot_id', '-')} LOT에서 영향이 가장 크게 나타납니다.",
            },
        ]
    )
    order_map = {name: idx for idx, name in enumerate(CAUSE_ORDER)}
    candidates["order"] = candidates["cause_category"].map(order_map).fillna(99)
    candidates = candidates.sort_values(["score", "order"], ascending=[False, True]).reset_index(drop=True)
    candidates["rank"] = range(1, len(candidates) + 1)
    return {"candidates": candidates}
