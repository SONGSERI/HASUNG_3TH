from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from rca_prototype.utils.constants import ISSUE_WINDOW, PROCESS_FLOW, RANDOM_SEED, SCENARIO_TITLE


@dataclass(frozen=True)
class SyntheticConfig:
    start: str = "2026-03-23 08:00:00"
    periods: int = 56
    freq: str = "1h"


def _master_data() -> Dict[str, pd.DataFrame]:
    line_id = "LINE-01"
    stage_rows = []
    machine_rows = []
    machine_map = {
        "LD": ["LD01"],
        "PR": ["PR01"],
        "SPI": ["SPI01"],
        "MNT": ["M04", "M05", "M06"],
        "RFL": ["RF01"],
        "AOI": ["AOI01"],
        "ULD": ["UL01"],
    }
    for idx, (stage_id, stage_name) in enumerate(PROCESS_FLOW, start=1):
        stage_rows.append(
            {
                "line_id": line_id,
                "stage_id": stage_id,
                "stage_order": idx,
                "stage_name": stage_name,
            }
        )
        for machine_id in machine_map[stage_id]:
            machine_rows.append(
                {
                    "line_id": line_id,
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "machine_id": machine_id,
                    "machine_group": "Mounter" if stage_id == "MNT" else stage_name.split(" / ")[0],
                }
            )
    model_rows = [
        {"model_id": "MODEL-A", "model_family": "Auto ECU"},
        {"model_id": "MODEL-B", "model_family": "Body Control"},
        {"model_id": "MODEL-C", "model_family": "Gateway"},
    ]
    part_rows = [
        {"part_number": "PN-001", "part_family": "Resistor"},
        {"part_number": "PN-002", "part_family": "Capacitor"},
        {"part_number": "PN-003", "part_family": "IC"},
        {"part_number": "PN-004", "part_family": "Fine Pitch IC"},
        {"part_number": "PN-005", "part_family": "Connector"},
    ]
    feeder_rows = [{"feeder_id": f"FDR-0{i}", "feeder_group": "Auto"} for i in range(1, 7)]
    nozzle_rows = [{"nozzle_id": f"NZ-0{i}", "nozzle_group": "Head-A"} for i in range(1, 5)]
    lot_rows = [
        {"lot_id": "LOT-001", "model_id": "MODEL-A"},
        {"lot_id": "LOT-002", "model_id": "MODEL-B"},
        {"lot_id": "LOT-003", "model_id": "MODEL-C"},
    ]
    return {
        "stage_master": pd.DataFrame(stage_rows),
        "machine_master": pd.DataFrame(machine_rows),
        "model_master": pd.DataFrame(model_rows),
        "part_master": pd.DataFrame(part_rows),
        "feeder_master": pd.DataFrame(feeder_rows),
        "nozzle_master": pd.DataFrame(nozzle_rows),
        "lot_master": pd.DataFrame(lot_rows),
    }


def generate_synthetic_data(seed: int = RANDOM_SEED) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cfg = SyntheticConfig()
    ts_index = pd.date_range(cfg.start, periods=cfg.periods, freq=cfg.freq)
    masters = _master_data()
    line_id = "LINE-01"

    stage_perf_rows: List[Dict] = []
    machine_perf_rows: List[Dict] = []
    event_rows: List[Dict] = []
    material_rows: List[Dict] = []
    quality_rows: List[Dict] = []
    lot_rows: List[Dict] = []

    stage_multiplier = {"LD": 1.0, "PR": 1.0, "SPI": 0.995, "MNT": 0.99, "RFL": 0.99, "AOI": 0.985, "ULD": 0.985}
    mounter_machines = ["M04", "M05", "M06"]
    affected_lots = set(ISSUE_WINDOW["affected_lots"])

    for idx, ts in enumerate(ts_index):
        hour = ts.hour
        day = ts.date()
        if idx < 18:
            lot_id = "LOT-001"
        elif idx < 36:
            lot_id = "LOT-002"
        else:
            lot_id = "LOT-003"
        model_id = "MODEL-A" if lot_id == "LOT-001" else "MODEL-B" if lot_id == "LOT-002" else "MODEL-C"
        issue_active = str(day) == ISSUE_WINDOW["date"] and hour in ISSUE_WINDOW["hours"]
        aoi_lag = str(day) == ISSUE_WINDOW["date"] and hour in [15, 16, 17]
        lot_bias = 1.0 if lot_id not in affected_lots else 1.08

        baseline_line_output = 440 + rng.integers(-6, 7)
        stage_outputs = {}
        for stage_id, stage_name in PROCESS_FLOW:
            expected_output = baseline_line_output * stage_multiplier[stage_id]
            planned_output = expected_output
            cycle_base = {
                "LD": 8.0,
                "PR": 8.4,
                "SPI": 9.2,
                "MNT": 10.8,
                "RFL": 8.8,
                "AOI": 9.6,
                "ULD": 7.8,
            }[stage_id]
            wait_time = 4.0 + rng.uniform(0.0, 0.6)
            stop_time = 3.0 + rng.uniform(0.0, 0.8)
            cycle_delay = rng.uniform(0.0, 0.18)
            actual_output = expected_output - rng.uniform(0, 4)

            if stage_id == "MNT" and issue_active:
                cycle_delay += 4.4 * lot_bias
                stop_time += 16.0
                wait_time += 7.5
                actual_output -= 92 * lot_bias
            elif stage_id == "AOI" and aoi_lag:
                cycle_delay += 1.2
                actual_output -= 26 * lot_bias
            elif stage_id in {"PR", "SPI"}:
                actual_output -= rng.uniform(0, 1.2)

            actual_output = max(actual_output, expected_output * 0.72)
            running_time = 3600 - stop_time - wait_time
            stage_outputs[stage_id] = actual_output
            stage_perf_rows.append(
                {
                    "timestamp": ts,
                    "date": day,
                    "hour_bucket": hour,
                    "line_id": line_id,
                    "stage_id": stage_id,
                    "stage_name": stage_name,
                    "lot_id": lot_id,
                    "model_id": model_id,
                    "actual_output": round(actual_output, 1),
                    "planned_output": round(planned_output, 1),
                    "expected_output": round(expected_output, 1),
                    "output_gap": round(actual_output - expected_output, 1),
                    "cycle_time_sec": round(cycle_base + cycle_delay, 2),
                    "standard_cycle_time_sec": cycle_base,
                    "baseline_cycle_time_sec": cycle_base,
                    "cycle_delay_sec": round(cycle_delay, 2),
                    "routing_time_sec": round(5.0 + (0.8 if stage_id == "MNT" else 0.4), 2),
                    "wait_time_sec": round(wait_time, 2),
                    "stop_time_sec": round(stop_time, 2),
                    "running_time_sec": round(running_time, 2),
                    "utilization_pct": round(running_time / 3600, 4),
                }
            )

        expected_machine_output = (baseline_line_output * stage_multiplier["MNT"]) / len(mounter_machines)
        planned_machine_output = expected_machine_output
        for machine_id in mounter_machines:
            is_bad_machine = machine_id == ISSUE_WINDOW["machine_id"] and issue_active
            mild_peer_effect = machine_id != ISSUE_WINDOW["machine_id"] and issue_active
            cycle_delay = rng.uniform(0.08, 0.25)
            stop_time = 2.6 + rng.uniform(0.3, 1.0)
            wait_time = 2.2 + rng.uniform(0.3, 0.8)
            pickup_error = rng.integers(0, 2)
            recog_error = rng.integers(0, 1)
            mech_error = rng.integers(0, 1)
            feeder_id = f"FDR-0{rng.integers(1, 7)}"
            nozzle_id = f"NZ-0{rng.integers(1, 5)}"
            part_number = f"PN-00{rng.integers(1, 6)}"
            actual_output = expected_machine_output - rng.uniform(1, 4)

            if is_bad_machine:
                feeder_id = ISSUE_WINDOW["feeder_id"]
                nozzle_id = ISSUE_WINDOW["nozzle_id"]
                part_number = ISSUE_WINDOW["part_number"]
                pickup_error = 34 + rng.integers(0, 7)
                recog_error = 2 + rng.integers(0, 2)
                mech_error = 1 + rng.integers(0, 2)
                cycle_delay += 4.8 * lot_bias
                stop_time += 17.0
                wait_time += 7.5
                actual_output -= 56 * lot_bias
            elif mild_peer_effect:
                pickup_error += 1
                cycle_delay += 0.35
                actual_output -= 4

            baseline_cycle = 10.8 if machine_id != "M06" else 11.0
            actual_output = max(actual_output, expected_machine_output * 0.55)
            running_time = 3600 - stop_time - wait_time
            output_gap = actual_output - expected_machine_output
            stop_reason = "Pickup Jam" if is_bad_machine else "Minor Adjustment"
            machine_perf_rows.append(
                {
                    "timestamp": ts,
                    "date": day,
                    "hour_bucket": hour,
                    "line_id": line_id,
                    "stage_id": "MNT",
                    "stage_name": "Mounter",
                    "machine_id": machine_id,
                    "machine_group": "Mounter",
                    "lot_id": lot_id,
                    "model_id": model_id,
                    "actual_output": round(actual_output, 1),
                    "planned_output": round(planned_machine_output, 1),
                    "expected_output": round(expected_machine_output, 1),
                    "output_gap": round(output_gap, 1),
                    "cycle_time_sec": round(baseline_cycle + cycle_delay, 2),
                    "standard_cycle_time_sec": baseline_cycle,
                    "baseline_cycle_time_sec": baseline_cycle,
                    "cycle_delay_sec": round(cycle_delay, 2),
                    "routing_time_sec": 5.8,
                    "wait_time_sec": round(wait_time, 2),
                    "stop_time_sec": round(stop_time, 2),
                    "running_time_sec": round(running_time, 2),
                    "utilization_pct": round(running_time / 3600, 4),
                    "pickup_error_count": int(pickup_error),
                    "recognition_error_count": int(recog_error),
                    "mechanical_error_count": int(mech_error),
                    "stop_count": 1 + int(stop_time > 10),
                    "stop_reason": stop_reason,
                }
            )
            event_rows.append(
                {
                    "event_ts": ts,
                    "date": day,
                    "hour_bucket": hour,
                    "machine_id": machine_id,
                    "line_id": line_id,
                    "stage_id": "MNT",
                    "stage_name": "Mounter",
                    "lot_id": lot_id,
                    "model_id": model_id,
                    "error_type": "Pickup" if pickup_error >= max(recog_error, mech_error) else "Recognition",
                    "error_message": "Pickup instability on feeder/nozzle" if is_bad_machine else "Minor transient error",
                    "error_count": int(pickup_error + recog_error + mech_error),
                    "stop_reason": stop_reason,
                    "stop_count": 1 + int(stop_time > 10),
                    "pickup_error_count": int(pickup_error),
                    "recognition_error_count": int(recog_error),
                    "mechanical_error_count": int(mech_error),
                }
            )
            material_rows.append(
                {
                    "event_ts": ts,
                    "date": day,
                    "hour_bucket": hour,
                    "machine_id": machine_id,
                    "line_id": line_id,
                    "lot_id": lot_id,
                    "model_id": model_id,
                    "feeder_id": feeder_id,
                    "nozzle_id": nozzle_id,
                    "part_number": part_number,
                    "feeder_error_count": int(pickup_error * (0.55 if is_bad_machine else 0.25)),
                    "nozzle_error_count": int(pickup_error * (0.25 if is_bad_machine else 0.15)),
                    "part_related_error_count": int(pickup_error * (0.20 if is_bad_machine else 0.10)),
                    "material_change_count": 2 if is_bad_machine else 0,
                    "pickup_error_count": int(pickup_error),
                }
            )

        aoi_fail = 3 + rng.integers(0, 2)
        defect_type = "Solder" if lot_id == "LOT-001" else "Placement Shift"
        if aoi_lag and lot_id in affected_lots:
            aoi_fail += 18 + rng.integers(0, 5)
            defect_type = "Placement Offset"
        quality_rows.append(
            {
                "timestamp": ts,
                "date": day,
                "hour_bucket": hour,
                "line_id": line_id,
                "lot_id": lot_id,
                "model_id": model_id,
                "spi_fail_count": 2 + rng.integers(0, 2),
                "aoi_fail_count": int(aoi_fail),
                "defect_type": defect_type,
                "defect_count": int(aoi_fail),
                "defect_rate": round(aoi_fail / max(stage_outputs["AOI"], 1), 4),
            }
        )

    stage_perf = pd.DataFrame(stage_perf_rows)
    machine_perf = pd.DataFrame(machine_perf_rows)
    event_df = pd.DataFrame(event_rows)
    material_df = pd.DataFrame(material_rows)
    quality_df = pd.DataFrame(quality_rows)

    lot_impact = (
        machine_perf.groupby(["lot_id", "model_id"], as_index=False)
        .agg(
            affected_machine_count=("machine_id", "nunique"),
            lot_output=("actual_output", "sum"),
            planned_output=("planned_output", "sum"),
            expected_output=("expected_output", "sum"),
        )
        .merge(
            stage_perf.groupby("lot_id", as_index=False).agg(affected_stage_count=("stage_id", "nunique")),
            on="lot_id",
            how="left",
        )
        .merge(
            quality_df.groupby(["lot_id"], as_index=False).agg(aoi_fail_count=("aoi_fail_count", "sum"), defect_rate=("defect_rate", "mean")),
            on="lot_id",
            how="left",
        )
    )
    lot_impact["lot_output_gap"] = lot_impact["lot_output"] - lot_impact["expected_output"]
    lot_impact["lot_defect_rate"] = lot_impact["defect_rate"]

    scenario = pd.DataFrame(
        [
            {
                "scenario_title": SCENARIO_TITLE,
                "issue_date": ISSUE_WINDOW["date"],
                "issue_hours": "14:00-16:00",
                "issue_machine": ISSUE_WINDOW["machine_id"],
                "issue_feeder": ISSUE_WINDOW["feeder_id"],
                "issue_nozzle": ISSUE_WINDOW["nozzle_id"],
                "issue_part": ISSUE_WINDOW["part_number"],
            }
        ]
    )

    return {
        **masters,
        "stage_performance": stage_perf,
        "machine_performance": machine_perf,
        "machine_events": event_df,
        "material_context": material_df,
        "quality_data": quality_df,
        "lot_impact": lot_impact,
        "scenario_info": scenario,
    }
