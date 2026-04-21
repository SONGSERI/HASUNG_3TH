from typing import Dict, List

import numpy as np
import pandas as pd

from utils import _lower, _pick, _pick_dt, _pick_txt, _safe_div


ACTION_TEMPLATES = {
    "SETUP": ["셋업/티칭/피더 세팅 체크리스트 적용", "LOT 시작 직후 불량/정지 증가 여부 확인"],
    "PICKUP_ERROR": ["피더, 노즐, 진공, 스플라이스 점검", "상위 부품과 반복 LOT 확인"],
    "RECOG_ERROR": ["조명, 카메라, 마크 인식 조건 점검", "특정 시간대/LOT 편중 확인"],
    "PLACE_ERROR": ["헤드, 축, 오프셋, 라이브러리 점검", "노즐/피더 조합 편중 확인"],
    "FEEDER_ERROR": ["피더, 릴, 공급 상태 점검", "반복 LOT와 시간대 확인"],
    "TRANSFER_ERROR": ["컨베이어/버퍼/인터록/센서 점검", "WAIT 동시 증가 여부 확인"],
    "WAIT": ["전후공정 병목과 라인 밸런스 점검", "대기 집중 stage 확인"],
}


PROCESS_FLOW = [
    "Printer",
    "SPI",
    "Mounter",
    "AOI",
    "Reflow",
    "AOI_POST",
]

PROCESS_ALIASES = {
    "Printer": ["printer", "print", "screen", "solder paste", "인쇄"],
    "SPI": ["spi", "solder paste inspection", "인쇄검사"],
    "Mounter": ["mounter", "mount", "pick", "place", "실장"],
    "AOI": ["aoi", "inspection", "외관검사"],
    "Reflow": ["reflow", "oven", "리플로우"],
    "AOI_POST": ["post aoi", "final aoi", "moi", "final inspection", "최종검사"],
}

PROCESS_ANALYSIS_RULES = {
    "Printer": {
        "tables": ["fa_2_marking_dtl"],
        "required": {"join": True, "result": False, "quality": False, "machine_stage": False},
    },
    "SPI": {
        "tables": ["fa_24_spi_dtl"],
        "required": {"join": True, "result": True, "quality": False, "machine_stage": False},
    },
    "Mounter": {
        "tables": ["fa_26_34_mounter_dtl"],
        "required": {"join": True, "result": False, "quality": False, "machine_stage": True},
    },
    "AOI": {
        "tables": ["fa_14_aoi_dtl", "fa_42_aoi_dtl"],
        "required": {"join": True, "result": True, "quality": False, "machine_stage": False},
    },
    "Reflow": {
        "tables": [],
        "required": {"join": False, "result": False, "quality": False, "machine_stage": False},
    },
    "AOI_POST": {
        "tables": ["fa_35_moi_dtl"],
        "required": {"join": True, "result": True, "quality": False, "machine_stage": False},
    },
}


def _result_flag(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.upper().str.strip()
    out = pd.Series(["UNKNOWN"] * len(s), index=s.index)
    out[s.isin(["PASS", "OK", "GOOD", "Y", "1"])] = "PASS"
    out[s.str.contains("FAIL|NG|ERR|ERROR|REWORK|BAD", na=False)] = "FAIL"
    return out


def classify_event(tagname: str = "", item: str = "", value: str = "") -> str:
    text = " ".join([str(tagname or ""), str(item or ""), str(value or "")]).upper()
    if any(k in text for k in ["SETUP", "CHANGEOVER", "TEACH", "CALIB", "INITIAL"]):
        return "SETUP"
    if any(k in text for k in ["FEEDER", "FDR", "REEL", "SUPPLY", "FEED"]):
        return "FEEDER_ERROR"
    if any(k in text for k in ["PICKUP", "SUCTION", "FEEDER", "NOZZLE", "VACUUM"]):
        return "PICKUP_ERROR"
    if any(k in text for k in ["RECOG", "RECOGN", "VISION", "MARK", "ALIGN", "CAMERA"]):
        return "RECOG_ERROR"
    if any(k in text for k in ["PLACE", "PLACEMENT", "INSERT", "POSITION"]):
        return "PLACE_ERROR"
    if any(k in text for k in ["TRANSFER", "CONVEYOR", "BUFFER", "INTERLOCK"]):
        return "TRANSFER_ERROR"
    if any(k in text for k in ["WAIT_PRE", "WAIT_POST", "PRE WAIT", "POST WAIT", "WAIT BEFORE", "WAIT AFTER", "UPSTREAM WAIT", "DOWNSTREAM WAIT", "WAIT"]):
        return "WAIT"
    return "OTHER"


def _parse_tag_name(tagname: pd.Series) -> pd.DataFrame:
    parts = tagname.fillna("").astype(str).str.split(".", expand=True)
    out = pd.DataFrame(index=tagname.index)
    p0 = parts[0].fillna("") if 0 in parts.columns else pd.Series([""] * len(tagname), index=tagname.index)
    p1 = parts[1].fillna("") if 1 in parts.columns else pd.Series([""] * len(tagname), index=tagname.index)
    p2 = parts[2].fillna("") if 2 in parts.columns else pd.Series(tagname.fillna("").astype(str), index=tagname.index)
    p3 = parts[3].fillna("") if 3 in parts.columns else pd.Series([""] * len(tagname), index=tagname.index)
    p4 = parts[4].fillna("") if 4 in parts.columns else pd.Series([""] * len(tagname), index=tagname.index)
    out["scope"] = p0.astype(str) + "." + p1.astype(str)
    out["family"] = p2.astype(str)
    out["family"] = out["family"].where(out["family"].ne(""), tagname.fillna("").astype(str))
    out["slot"] = pd.to_numeric(p3, errors="coerce")
    out["metric"] = p4.astype(str)
    out["metric"] = out["metric"].where(out["metric"].ne(""), p3.astype(str))
    out["metric"] = out["metric"].where(out["metric"].ne(""), p2.astype(str))
    out["metric"] = out["metric"].where(out["metric"].ne(""), tagname.fillna("").astype(str))
    return out


def normalize_cause_group(value: str) -> str:
    normalized = str(value or "").strip().lower()
    mapping = {
        "setup": "Setup",
        "process": "Process",
        "quality": "Quality",
        "equipment": "Equipment",
        "waiting": "Waiting",
        "meta": "Meta",
        "proxy": "Proxy",
        "other": "Other",
    }
    return mapping.get(normalized, str(value or "Other").strip() or "Other")


def infer_cause_family(cause_group: str, cause_detail: str) -> str:
    text = f"{cause_group} {cause_detail}".upper()
    if any(k in text for k in ["PICKUP", "FEED", "FEEDER", "REEL"]):
        return "feeder"
    if any(k in text for k in ["NOZZ", "RECOG", "VISION", "CAM", "MARK"]):
        return "vision"
    if any(k in text for k in ["PLACE", "OFFSET", "ALIGN", "COORD", "HEAD"]):
        return "placement"
    if any(k in text for k in ["TRANSFER", "CONVEY", "INTERLOCK", "CVN"]):
        return "transfer"
    if any(k in text for k in ["WAIT_PRE", "UPSTREAM", "BWAIT", "MCFWAIT", "FWAIT"]):
        return "upstream"
    if any(k in text for k in ["WAIT_POST", "DOWNSTREAM", "RWAIT", "MCRWAIT"]):
        return "downstream"
    if any(k in text for k in ["QUALITY", "INSPECT", "DMISS", "BMISS", "FAIL", "BAD"]):
        return "quality"
    if any(k in text for k in ["STOP", "PRD", "SCSTOP", "SCESTOP", "OTHR"]):
        return "process"
    return "unknown"


def classify_tag_event(tag: pd.DataFrame) -> pd.DataFrame:
    if tag.empty:
        return pd.DataFrame(index=tag.index)

    tagname = tag.get("_tagname", pd.Series([""] * len(tag), index=tag.index)).fillna("").astype(str)
    family = tag.get("tag_family", pd.Series([""] * len(tag), index=tag.index)).fillna("").astype(str)
    metric = tag.get("tag_metric", pd.Series([""] * len(tag), index=tag.index)).fillna("").astype(str)
    tag_class = tag.get("tag_class", pd.Series([""] * len(tag), index=tag.index)).fillna("").astype(str)
    tag_value = pd.to_numeric(tag.get("tag_value_num", pd.Series([np.nan] * len(tag), index=tag.index)), errors="coerce")

    event_class = tag.apply(lambda r: classify_event(r.get("_tagname", ""), r.get("tag_metric", ""), r.get("_value", "")), axis=1)
    wait_mask = event_class.eq("WAIT")
    feeder_mask = event_class.eq("FEEDER_ERROR")
    pickup_mask = event_class.eq("PICKUP_ERROR")
    recog_mask = event_class.eq("RECOG_ERROR")
    place_mask = event_class.eq("PLACE_ERROR")
    transfer_mask = event_class.eq("TRANSFER_ERROR")
    setup_mask = event_class.eq("SETUP")
    stop_mask = tag_class.eq("STOP") | tagname.str.contains("STOP|ERR|ERROR|ALARM|FAIL|NG", case=False, na=False)
    inspect_mask = family.eq("InspectionData") | tagname.str.contains("INSPECTION", case=False, na=False)
    meta_mask = family.isin(["Information", "Index"])
    flow_mask = family.isin(["Count", "CycleTime", "Time"])
    other_mask = ~(wait_mask | feeder_mask | pickup_mask | recog_mask | place_mask | transfer_mask | setup_mask | stop_mask | inspect_mask | meta_mask | flow_mask)

    event_class = np.select(
        [setup_mask, feeder_mask, pickup_mask, recog_mask, place_mask, transfer_mask, wait_mask, stop_mask, inspect_mask, meta_mask, flow_mask, other_mask],
        ["SETUP", "FEEDER_ERROR", "PICKUP_ERROR", "RECOG_ERROR", "PLACE_ERROR", "TRANSFER_ERROR", "WAIT", "STOP", "INSPECTION", "META", "FLOW", "OTHER"],
        default="OTHER",
    )
    cause_group = np.select(
        [setup_mask, feeder_mask, pickup_mask, recog_mask, place_mask, transfer_mask, wait_mask, stop_mask, inspect_mask, meta_mask, flow_mask, other_mask],
        ["Setup", "Equipment", "Equipment", "Equipment", "Equipment", "Equipment", "Waiting", "Equipment", "Quality", "Meta", "Proxy", "Other"],
        default="Other",
    )
    cause_group = pd.Series(cause_group, index=tag.index).map(normalize_cause_group)

    cause_detail = np.where(
        setup_mask | feeder_mask | pickup_mask | recog_mask | place_mask | transfer_mask,
        np.where(metric.ne(""), metric, np.where(family.ne(""), family, tagname)),
        np.where(
            wait_mask,
            np.where(tagname.str.contains("WAIT_PRE", case=False, na=False), "WAIT_PRE", np.where(tagname.str.contains("WAIT_POST", case=False, na=False), "WAIT_POST", "WAIT")),
            np.where(
                stop_mask | inspect_mask,
                np.where(metric.ne(""), metric, np.where(family.ne(""), family, tagname)),
                np.where(flow_mask, np.where(metric.ne(""), metric, family), np.where(family.ne(""), family, tagname)),
            ),
        ),
    )

    is_rca_candidate = setup_mask | feeder_mask | wait_mask | pickup_mask | recog_mask | place_mask | transfer_mask | stop_mask | inspect_mask
    proxy_score = np.where(is_rca_candidate, np.where(tag_value.fillna(0).gt(0), tag_value.fillna(0), 1.0), 0.0)

    return pd.DataFrame(
        {
            "event_class": event_class,
            "cause_group": cause_group.astype(str).map(normalize_cause_group),
            "cause_detail": pd.Series(cause_detail, index=tag.index).astype(str).replace({"nan": ""}),
            "is_rca_candidate": is_rca_candidate,
            "proxy_score": pd.to_numeric(pd.Series(proxy_score, index=tag.index), errors="coerce").fillna(0.0),
        },
        index=tag.index,
    )


def build_rca_candidate_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    if shop.empty:
        return pd.DataFrame()
    if "is_rca_candidate" not in shop.columns:
        return pd.DataFrame()
    return shop[shop["is_rca_candidate"].fillna(False)].copy()


def _build_process_fact(raw: Dict[str, pd.DataFrame], name: str, hdr_name: str, dtl_name: str, event_cols: List[str], mach_col, line_col, stage_col, model_col, res_cols, out_col) -> pd.DataFrame:
    dtl = raw.get(dtl_name, pd.DataFrame()).copy()
    hdr = raw.get(hdr_name, pd.DataFrame()).copy()
    if dtl.empty and hdr.empty:
        return pd.DataFrame()
    df = dtl.copy()
    hdr_keys = ["plant_cd", "wc_cd", "file_nm"]
    if not hdr.empty and all(k in df.columns for k in hdr_keys):
        df = df.merge(hdr, on=hdr_keys, how="left", suffixes=("", "_hdr"))
    df["process_name"] = name
    df["event_ts"] = _pick_dt(df, event_cols)
    if "make_dt" in df.columns:
        fallback = pd.to_datetime(df["make_dt"], errors="coerce")
        miss = df["event_ts"].isna()
        df.loc[miss, "event_ts"] = fallback[miss]
    df["approx_event"] = df["event_ts"].isna()
    df["machine_id"] = df[mach_col] if mach_col and mach_col in df.columns else np.nan
    df["line_id"] = df[line_col] if line_col and line_col in df.columns else np.nan
    df["stage_no"] = pd.to_numeric(df[stage_col], errors="coerce") if stage_col and stage_col in df.columns else np.nan
    df["lot_id"] = _pick_txt(df, ["lot_nm", "lot_id"]) if name == "mounter" else _pick_txt(df, ["lot_id"])
    df["model_label"] = _pick_txt(df, [model_col] if model_col else [])
    df["machine_order"] = pd.to_numeric(df["machine_order"], errors="coerce") if "machine_order" in df.columns else np.nan
    df["result_primary"] = _pick_txt(df, res_cols[:1]) if res_cols else "미상"
    df["result_secondary"] = _pick_txt(df, res_cols[1:2]) if len(res_cols) > 1 else "미상"
    df["output_qty"] = pd.to_numeric(df[out_col], errors="coerce") if out_col and out_col in df.columns else 0
    df["is_event_like"] = df["process_name"].isin(["spi", "aoi_14", "aoi_42", "moi"])
    df["is_aggregate_like"] = df["process_name"].eq("mounter")
    return df


def build_shopfloor_fact(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    process_specs = [
        ("mounter", "fa_26_34_mounter_hdr", "fa_26_34_mounter_dtl", ["file_dt", "make_dt"], "mach_cd", "lane", "stage", "lot_nm", ["result"], "output"),
        ("spi", "fa_24_spi_hdr", "fa_24_spi_dtl", ["file_dt", "make_dt"], "mach_cd", "lane", None, "model", ["machineresult", "reviewresult", "callresult", "result"], None),
        ("aoi_14", "fa_14_aoi_hdr", "fa_14_aoi_dtl", ["file_dt", "make_dt"], "mach_cd", "lane", None, "pcbmodel", ["machineresult", "reviewresult", "callresult"], None),
        ("aoi_42", "fa_42_aoi_hdr", "fa_42_aoi_dtl", ["file_dt", "make_dt"], "mach_cd", "lane", None, "pcbmodel", ["machineresult", "reviewresult", "callresult"], None),
        ("moi", "fa_35_moi_hdr", "fa_35_moi_dtl", ["file_dt", "make_dt"], "mach_cd", "lane", None, "pcbmodel", ["machineresult", "reviewresult", "callresult"], None),
        ("marking", "fa_2_marking_hdr", "fa_2_marking_dtl", ["file_dt", "make_dt"], None, None, None, None, [], None),
    ]
    process_facts = [
        _build_process_fact(raw, *spec)
        for spec in process_specs
    ]
    process_facts = [df for df in process_facts if not df.empty]
    shop = pd.concat(process_facts, ignore_index=True, sort=False) if process_facts else pd.DataFrame()
    if not shop.empty:
        shop["day"] = pd.to_datetime(shop["event_ts"], errors="coerce").dt.date
        shop["join_coverage"] = shop["event_ts"].notna().astype(int)
    return shop


def build_tag_event_fact(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    tag = raw.get("_mounter_tag", pd.DataFrame()).copy()
    if tag.empty:
        return pd.DataFrame()
    tag["event_ts"] = _pick_dt(tag, ["_devicedate", "_insertdate"])
    tag["approx_event"] = tag["event_ts"].isna()
    parsed = _parse_tag_name(tag["_tagname"])
    tag["tag_scope"] = parsed["scope"]
    tag["tag_family"] = parsed["family"]
    tag["tag_slot"] = parsed["slot"]
    tag["tag_metric"] = parsed["metric"]
    tag["tag_value_num"] = pd.to_numeric(tag["_value"], errors="coerce")
    tag["tag_class"] = np.where(tag["_tagname"].astype(str).str.contains("WAIT", case=False, na=False), "WAIT", np.where(tag["_tagname"].astype(str).str.contains("STOP|ERR", case=False, na=False), "STOP", "RUN"))
    tag["day"] = pd.to_datetime(tag["event_ts"], errors="coerce").dt.date
    tag["machine_id"] = tag["_equipcode"].astype(str)
    tag["line_id"] = tag["_linecode"].astype(str)
    classification = classify_tag_event(tag)
    tag["event_class"] = classification["event_class"]
    tag["cause_group"] = classification["cause_group"]
    tag["cause_detail"] = classification["cause_detail"]
    tag["is_rca_candidate"] = classification["is_rca_candidate"]
    tag["proxy_score"] = classification["proxy_score"]
    tag["model_label"] = pd.NA
    tag["output_qty"] = 0.0
    tag["process_name"] = "mounter_tag"
    tag["is_event_like"] = tag["event_class"].isin(["WAIT", "STOP", "PICKUP_ERROR", "FEEDER_ERROR", "RECOG_ERROR", "PLACE_ERROR", "TRANSFER_ERROR", "INSPECTION", "SETUP"])
    tag["is_aggregate_like"] = tag["event_class"].isin(["META", "FLOW"])
    tag["join_coverage"] = tag["event_ts"].notna().astype(int)
    tag["traceability_key"] = np.where(tag["tag_family"].eq("Information"), tag["tag_metric"], tag["_equipcode"].astype(str))
    tag["quality_flag"] = np.where(
        tag["event_class"].eq("INSPECTION"),
        np.where(tag["tag_metric"].str.contains("Bad|Fail|NG|Error", case=False, na=False), "FAIL", "PASS"),
        "UNKNOWN",
    )
    tag["result_primary"] = tag["cause_detail"]
    tag["result_secondary"] = tag["tag_value_num"].astype("string")
    return tag


def build_stop_event_fact(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = raw.get("stop_log", pd.DataFrame()).copy()
    if stop.empty:
        return pd.DataFrame()
    lm = raw.get("lot_machine", pd.DataFrame()).copy()
    mach = raw.get("machine", pd.DataFrame()).copy()
    lot = raw.get("lot", pd.DataFrame()).copy()
    file = raw.get("file", pd.DataFrame()).copy()
    if not lm.empty and "lot_machine_id" in stop.columns:
        stop = stop.merge(lm, on="lot_machine_id", how="left", suffixes=("", "_lm"))
    if not mach.empty and "machine_id" in stop.columns:
        stop = stop.merge(mach[["machine_id", "machine_name"]], on="machine_id", how="left")
    if not lot.empty and "lot_id" in stop.columns:
        keep = [c for c in ["lot_id", "lot_name", "model_name"] if c in lot.columns]
        stop = stop.merge(lot[keep], on="lot_id", how="left")
    if not file.empty and "source_file_id" in stop.columns and "file_id" in file.columns:
        stop = stop.merge(file[["file_id", "file_datetime", "file_sequence"]], left_on="source_file_id", right_on="file_id", how="left")
    stop["event_ts"] = _pick_dt(stop, ["recorded_at", "file_datetime"])
    stop["approx_event"] = stop["event_ts"].isna()
    stop["duration_sec"] = pd.to_numeric(stop.get("duration_sec", 0), errors="coerce").fillna(0)
    stop["stop_count"] = pd.to_numeric(stop.get("stop_count", 1), errors="coerce").fillna(1)
    stop["stop_like_reason"] = _pick_txt(stop, ["stop_reason_code"])
    stop["day"] = pd.to_datetime(stop["event_ts"], errors="coerce").dt.date
    stop["is_event_like"] = stop["stop_count"].le(1)
    stop["is_aggregate_like"] = ~stop["is_event_like"]
    stop["model_label"] = _pick_txt(stop, ["model_name", "lot_name", "lot_id"])
    return stop


def build_inspection_fact(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    insp = []
    for name in ["fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"]:
        df = raw.get(name, pd.DataFrame())
        if df.empty:
            continue
        tmp = df.copy()
        if name == "fa_24_spi_dtl":
            tmp["process_name"] = "SPI"
        elif name in {"fa_14_aoi_dtl", "fa_42_aoi_dtl"}:
            tmp["process_name"] = "AOI"
        elif name == "fa_35_moi_dtl":
            tmp["process_name"] = "AOI_POST"
        tmp["event_ts"] = _pick_dt(tmp, ["file_dt", "make_dt"])
        tmp["approx_event"] = tmp["event_ts"].isna()
        tmp["machine_id"] = tmp.get("mach_cd")
        tmp["line_id"] = tmp.get("lane")
        tmp["model_label"] = _pick_txt(tmp, ["model", "pcbmodel"])
        tmp["result_primary"] = _pick_txt(tmp, ["machineresult", "reviewresult"])
        tmp["result_secondary"] = _pick_txt(tmp, ["reviewresult"])
        tmp["quality_flag"] = _result_flag(tmp["result_primary"])
        tmp["day"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.date
        insp.append(tmp)
    return pd.concat(insp, ignore_index=True, sort=False) if insp else pd.DataFrame()


def build_component_fact(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    comp = raw.get("component_pickup_summary", pd.DataFrame()).copy()
    comp_dim = raw.get("component", pd.DataFrame()).copy()
    if comp.empty:
        return pd.DataFrame()
    comp["event_ts"] = pd.to_datetime(comp.get("recorded_at"), errors="coerce")
    comp["day"] = pd.to_datetime(comp["event_ts"], errors="coerce").dt.date
    comp["error_rate"] = comp.apply(lambda r: _safe_div(r.get("error_count", 0), r.get("pickup_count", 0)), axis=1)
    if not comp_dim.empty:
        comp = comp.merge(comp_dim, on="component_id", how="left")
    return comp


def _classify_mounter_item(item: str, row: pd.Series | None = None) -> str:
    text = str(item or "").strip().lower()
    haystack = text
    if row is not None:
        for col in ["item", "result", "value", "val", "status", "state"]:
            if col in row.index and pd.notna(row[col]):
                haystack += f" {str(row[col]).lower()}"
    if any(k in haystack for k in ["feeder", "fdr", "feed"]):
        return "FEEDER"
    if any(k in haystack for k in ["nozzle", "noz"]):
        return "NOZZLE"
    if any(k in haystack for k in ["pickup", "pick up", "pick"]):
        return "PICKUP"
    if any(k in haystack for k in ["recogn", "recog", "vision", "scan", "mark"]):
        return "RECOGNITION"
    if any(k in haystack for k in ["transfer", "conveyor", "buffer", "wait"]):
        return "TRANSFER"
    if any(k in haystack for k in ["part", "reel", "serial", "lot", "product", "pcb"]):
        return "PART"
    if any(k in haystack for k in ["inspect", "judge", "quality", "result", "fail", "ng", "error"]):
        return "QUALITY"
    if any(k in haystack for k in ["count", "time", "cycle", "duration"]):
        return "META"
    if any(k in haystack for k in ["block", "board", "head", "ncadd"]):
        return "PROCESS"
    return "OTHER"


def build_mounter_item_fact(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = raw.get("fa_26_34_mounter_dtl", pd.DataFrame()).copy()
    if df.empty:
        return pd.DataFrame()
    item_col = next((c for c in ["item", "item_name", "key", "param"] if c in df.columns), None)
    if item_col is None:
        return pd.DataFrame()
    value_col = next((c for c in ["value", "val", "result", "status", "state", "data_value", "item_value"] if c in df.columns and c != item_col), None)
    df["event_ts"] = _pick_dt(df, ["file_dt", "make_dt"])
    df["day"] = pd.to_datetime(df["event_ts"], errors="coerce").dt.date
    df["machine_id"] = _pick_txt(df, ["mach_cd"])
    df["line_id"] = _pick_txt(df, ["lane", "line_id"])
    df["stage_no"] = pd.to_numeric(_pick(df, ["stage", "stage_no"]), errors="coerce")
    df["lot_id"] = _pick_txt(df, ["lot_nm", "lot_id"])
    df["model_label"] = _pick_txt(df, ["model_name", "model", "pcbmodel", "lot_nm", "lot_id"])
    df["item_key"] = df[item_col].astype(str)
    if value_col:
        df["item_value"] = df[value_col]
    else:
        df["item_value"] = pd.NA
    if "row_num" in df.columns:
        row_part = pd.to_numeric(df["row_num"], errors="coerce").astype("Int64").astype(str)
    else:
        row_part = pd.Series([""] * len(df), index=df.index)
    df["entity_key"] = (
        _pick_txt(df, ["file_nm"])
        + "|"
        + df["machine_id"].astype(str)
        + "|"
        + df["lot_id"].astype(str)
        + "|"
        + df["stage_no"].astype("string").fillna("")
        + "|"
        + row_part.astype(str)
    )
    df["item_group"] = df.apply(lambda r: _classify_mounter_item(r["item_key"], r), axis=1)
    df["item_numeric_value"] = pd.to_numeric(df["item_value"], errors="coerce")
    df["item_is_numeric"] = df["item_numeric_value"].notna()
    df["item_is_quality_like"] = df["item_group"].eq("QUALITY")
    df["item_is_event_like"] = df["item_group"].isin(["PICKUP", "RECOGNITION", "TRANSFER", "PROCESS"])
    df["join_candidate_machine"] = df["machine_id"].notna()
    df["join_candidate_lot"] = df["lot_id"].notna()
    df["join_candidate_time"] = df["event_ts"].notna()
    df["join_candidate_stage"] = df["stage_no"].notna()
    return df


def _scan_inventory_row(table_name: str, df: pd.DataFrame) -> Dict[str, object]:
    out = {"table_name": table_name, "rows": int(len(df))}
    if df.empty:
        out.update({k: 0 for k in ["distinct_line", "distinct_workcenter", "distinct_machine", "distinct_stage", "distinct_lot", "distinct_model", "distinct_barcode", "item_distinct", "item_groups"]})
        out.update({"min_datetime": pd.NaT, "max_datetime": pd.NaT, "has_result_cols": False, "has_quality_cols": False, "has_join_cols": False})
        return out
    dt_cols = [c for c in df.columns if any(tok in c for tok in ["date", "time", "dt", "datetime", "recorded_at", "insertdate", "devicedate"])]
    dt_series = []
    for c in dt_cols:
        try:
            dt_series.append(pd.to_datetime(df[c], errors="coerce"))
        except Exception:
            pass
    if dt_series:
        merged = pd.concat(dt_series, ignore_index=True)
        out["min_datetime"] = merged.min()
        out["max_datetime"] = merged.max()
    else:
        out["min_datetime"] = pd.NaT
        out["max_datetime"] = pd.NaT
    def distinct(cols):
        vals = set()
        for c in cols:
            if c in df.columns:
                vals.update(df[c].dropna().astype(str).tolist())
        return len(vals)
    out["distinct_line"] = distinct(["line_id", "lane", "_linecode", "workcenter", "wc_cd"])
    out["distinct_workcenter"] = distinct(["workcenter", "wc_cd", "_workcode", "wc"])
    out["distinct_machine"] = distinct(["machine_id", "mach_cd", "_equipcode", "equipcode"])
    out["distinct_stage"] = distinct(["stage_no", "stage", "section", "machine_order"])
    out["distinct_lot"] = distinct(["lot_id", "lot_nm", "lot_name", "panelbarcode", "barcode"])
    out["distinct_model"] = distinct(["model_label", "model_name", "model", "pcbmodel"])
    out["distinct_barcode"] = distinct(["barcode", "panelbarcode", "panel", "bar_code"])
    cols = set(df.columns)
    out["has_result_cols"] = any("result" in c or "status" in c or "judge" in c for c in cols)
    out["has_quality_cols"] = any("quality" in c or "inspect" in c or "fail" in c or "ng" in c or "error" in c for c in cols)
    out["has_join_cols"] = any(c in cols for c in ["machine_id", "mach_cd", "_equipcode", "line_id", "lane", "lot_id", "lot_nm", "model", "pcbmodel", "barcode", "panelbarcode", "stage", "stage_no", "file_nm", "file_dt", "make_dt", "event_ts", "_devicedate"])
    if table_name == "fa_26_34_mounter_dtl":
        item_fact = build_mounter_item_fact({"fa_26_34_mounter_dtl": df})
        out["item_distinct"] = int(item_fact["item_key"].nunique()) if not item_fact.empty else 0
        out["item_groups"] = ", ".join(sorted(item_fact["item_group"].dropna().astype(str).unique().tolist())[:8]) if not item_fact.empty else ""
        out["item_numeric_ratio"] = float(item_fact["item_is_numeric"].mean()) if not item_fact.empty else 0.0
    return out


def build_full_period_data_inventory(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    tables = [
        "fa_26_34_mounter_dtl",
        "fa_26_34_mounter_hdr",
        "_mounter_tag",
        "fa_24_spi_dtl",
        "fa_24_spi_hdr",
        "fa_14_aoi_dtl",
        "fa_14_aoi_hdr",
        "fa_42_aoi_dtl",
        "fa_42_aoi_hdr",
        "fa_35_moi_dtl",
        "fa_35_moi_hdr",
        "fa_2_marking_dtl",
        "fa_2_marking_hdr",
    ]
    rows = []
    for t in tables:
        df = raw.get(t, pd.DataFrame())
        row = _scan_inventory_row(t, df)
        rows.append(row)
    return pd.DataFrame(rows)


def build_table_linkage_matrix(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None, item_fact: pd.DataFrame = None) -> pd.DataFrame:
    inv = inventory if inventory is not None else build_full_period_data_inventory(raw)
    rows = []
    def add(left: str, right: str, linkage: str, possible: bool, evidence: str):
        rows.append({"left_table": left, "right_table": right, "linkage": linkage, "possible": possible, "evidence": evidence})
    def has(table, col_tokens):
        df = raw.get(table, pd.DataFrame())
        if df.empty:
            return False
        cols = set(df.columns)
        return any(any(tok in c for tok in col_tokens) for c in cols)
    def values(table, col_tokens):
        df = raw.get(table, pd.DataFrame())
        if df.empty:
            return set()
        vals = set()
        for c in df.columns:
            if any(tok in c for tok in col_tokens):
                vals.update(df[c].dropna().astype(str).tolist())
        return {v for v in vals if v not in {"", "nan", "None"}}
    pair_specs = [
        ("fa_26_34_mounter_dtl", "_mounter_tag", "machine+time+stage", ["mach_cd", "machine_id", "_equipcode"], ["file_dt", "event_ts", "_devicedate", "_insertdate"]),
        ("fa_26_34_mounter_dtl", "fa_24_spi_dtl", "mounter↔spi", ["mach_cd", "machine_id", "lane"], ["file_dt", "make_dt"]),
        ("fa_26_34_mounter_dtl", "fa_14_aoi_dtl", "mounter↔aoi", ["mach_cd", "machine_id", "lane"], ["file_dt", "make_dt"]),
        ("fa_26_34_mounter_dtl", "fa_42_aoi_dtl", "mounter↔aoi", ["mach_cd", "machine_id", "lane"], ["file_dt", "make_dt"]),
        ("fa_26_34_mounter_dtl", "fa_35_moi_dtl", "mounter↔moi", ["mach_cd", "machine_id", "lane"], ["file_dt", "make_dt"]),
        ("fa_26_34_mounter_dtl", "fa_2_marking_dtl", "mounter↔marking", ["lot_nm", "lot_id", "mach_cd"], ["file_dt", "make_dt"]),
    ]
    for left, right, label, key_tokens, time_tokens in pair_specs:
        left_df = raw.get(left, pd.DataFrame())
        right_df = raw.get(right, pd.DataFrame())
        key_overlap = len(values(left, key_tokens).intersection(values(right, key_tokens)))
        time_overlap = len(values(left, time_tokens).intersection(values(right, time_tokens)))
        possible = not left_df.empty and not right_df.empty and (key_overlap > 0 or time_overlap > 0 or (has(left, key_tokens) and has(right, key_tokens) and has(left, time_tokens) and has(right, time_tokens)))
        evidence = []
        if key_overlap > 0:
            evidence.append(f"key_overlap={key_overlap}")
        if time_overlap > 0:
            evidence.append(f"time_overlap={time_overlap}")
        if not evidence:
            if has(left, key_tokens) and has(right, key_tokens):
                evidence.append("key_columns")
            if has(left, time_tokens) and has(right, time_tokens):
                evidence.append("time_columns")
        add(left, right, label, possible, ", ".join(evidence) if evidence else "no overlap")
    scan_tables = ["fa_26_34_mounter_dtl", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl", "fa_2_marking_dtl", "_mounter_tag"]
    lot_overlap = any(len(values(t, ["lot", "barcode", "panelbarcode"]).intersection(values("fa_26_34_mounter_dtl", ["lot", "barcode", "panelbarcode"]))) > 0 for t in scan_tables if t != "fa_26_34_mounter_dtl")
    machine_overlap = any(len(values(t, ["machine_id", "mach_cd", "_equipcode"]).intersection(values("fa_26_34_mounter_dtl", ["machine_id", "mach_cd", "_equipcode"]))) > 0 for t in scan_tables if t != "fa_26_34_mounter_dtl")
    model_overlap = any(len(values(t, ["model", "model_label", "pcbmodel"]).intersection(values("fa_26_34_mounter_dtl", ["model", "model_label", "pcbmodel"]))) > 0 for t in scan_tables if t != "fa_26_34_mounter_dtl")
    add("lot", "mounter/spi/aoi/moi/marking", "lot-based", lot_overlap, "lot_nm/lot_id/barcode/panelbarcode overlap")
    add("machine", "time-window", "machine+time", machine_overlap, "machine_id/mach_cd/_equipcode overlap")
    add("model", "barcode", "model/barcode", model_overlap, "model_label/model/pcbmodel/barcode overlap")
    return pd.DataFrame(rows)


def build_full_period_analysis_capability(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None, linkage: pd.DataFrame = None, item_fact: pd.DataFrame = None) -> pd.DataFrame:
    inv = inventory if inventory is not None else build_full_period_data_inventory(raw)
    link = linkage if linkage is not None else build_table_linkage_matrix(raw, inv, item_fact)
    item_fact = item_fact if item_fact is not None else build_mounter_item_fact(raw)
    rows = []
    def yesno(cond: bool) -> str:
        return "가능" if cond else "제한"
    def add(capability: str, cond: bool, note: str):
        rows.append({"분석축": capability, "가능여부": yesno(cond), "근거": note})
    add("설비 비교", (inv["distinct_machine"] > 0).any(), "machine 계열 식별자 존재")
    add("공정 비교", bool((inv["distinct_stage"].fillna(0) > 0).any()) or any("stage" in str(x).lower() or "process" in str(x).lower() for x in inv["table_name"]), "stage/process 축 존재")
    add("stage 흐름 분석", any(t in inv["table_name"].tolist() for t in ["fa_26_34_mounter_dtl", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"]), "공정별 stage/lane/order 축")
    add("lot 추적", (inv["distinct_lot"] > 0).any(), "lot_id / lot_nm / barcode")
    add("model 추적", (inv["distinct_model"] > 0).any(), "model_label / model / pcbmodel")
    quality_like = False
    if "has_quality_cols" in inv.columns:
        quality_like = bool(inv["has_quality_cols"].astype(bool).any())
    if "has_result_cols" in inv.columns:
        quality_like = quality_like or bool(inv["has_result_cols"].astype(bool).any())
    add("품질 추적", quality_like, "quality/inspect/fail/result/status 관련 컬럼")
    add("defect ↔ machine 연결", bool(link["possible"].astype(bool).any()), "machine/time link 존재")
    add("defect ↔ lot 연결", bool((inv["distinct_lot"] > 0).any()) and bool(inv["has_result_cols"].astype(bool).any()), "lot + result/quality 축")
    add("tag ↔ quality 연결", ("_mounter_tag" in raw and not raw.get("_mounter_tag", pd.DataFrame()).empty) and (quality_like or bool((inv["has_result_cols"].astype(bool).any()) if "has_result_cols" in inv.columns else False)), "tag stream + result/quality column")
    add("time-window drill-down", (inv["min_datetime"].notna().any() if "min_datetime" in inv.columns else False) or bool(inv["has_join_cols"].astype(bool).any()), "event_ts/file_dt/make_dt 존재")
    add("feeder/nozzle/part level", not item_fact.empty and item_fact["item_group"].isin(["FEEDER", "NOZZLE", "PART", "PICKUP", "RECOGNITION", "TRANSFER"]).any(), "Mounter ITEM normalized row")
    return pd.DataFrame(rows)


def _table_display_name(table_name: str) -> str:
    mapping = {
        "fa_26_34_mounter_dtl": "mounter_dtl",
        "fa_26_34_mounter_hdr": "mounter_hdr",
        "_mounter_tag": "mounter_tag",
        "fa_24_spi_dtl": "spi_dtl",
        "fa_24_spi_hdr": "spi_hdr",
        "fa_14_aoi_dtl": "aoi_dtl",
        "fa_14_aoi_hdr": "aoi_hdr",
        "fa_42_aoi_dtl": "aoi_dtl",
        "fa_42_aoi_hdr": "aoi_hdr",
        "fa_35_moi_dtl": "moi_dtl",
        "fa_35_moi_hdr": "moi_hdr",
        "fa_2_marking_dtl": "marking_dtl",
        "fa_2_marking_hdr": "marking_hdr",
    }
    return mapping.get(str(table_name).lower(), str(table_name))


def _format_preview(values: List[str], limit: int = 4) -> str:
    cleaned = [str(v) for v in values if str(v) not in {"", "nan", "None"}]
    if not cleaned:
        return "-"
    preview = cleaned[:limit]
    if len(cleaned) > limit:
        preview.append("...")
    return ", ".join(preview)


def _collect_values(raw: Dict[str, pd.DataFrame], tokens: List[str]) -> List[str]:
    values: List[str] = []
    for df in raw.values():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for c in df.columns:
            col = str(c).lower()
            if any(tok in col for tok in tokens):
                values.extend(df[c].dropna().astype(str).tolist())
    return sorted({v for v in values if v not in {"", "nan", "None"}})


def _first_nonempty(values: List[str]) -> str:
    for value in values:
        if value not in {"", "nan", "None"}:
            return value
    return "-"


def _inventory_row(inventory: pd.DataFrame, table_name: str) -> pd.Series:
    if inventory is None or inventory.empty or "table_name" not in inventory.columns:
        return pd.Series(dtype="object")
    match = inventory[inventory["table_name"].eq(table_name)]
    if match.empty:
        return pd.Series(dtype="object")
    return match.iloc[0]


def _evaluate_process_analysis(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame, process_name: str) -> Dict[str, object]:
    spec = PROCESS_ANALYSIS_RULES.get(process_name, {"tables": [], "required": {}})
    tables = []
    evidence_cols = []
    reasons = []
    usable = False
    present = False
    required = spec.get("required", {})
    candidate_tables = spec.get("tables", [])

    def _table_usability(row: pd.Series) -> bool:
        has_join = bool(row.get("has_join_cols", False))
        has_result = bool(row.get("has_result_cols", False))
        has_quality = bool(row.get("has_quality_cols", False))
        distinct_machine = int(row.get("distinct_machine", 0) or 0)
        distinct_stage = int(row.get("distinct_stage", 0) or 0)
        rows = int(row.get("rows", 0) or 0)
        if rows <= 0:
            return False
        if required.get("join", False) and not has_join:
            return False
        if required.get("result", False) and not (has_result or has_quality):
            return False
        if required.get("quality", False) and not has_quality:
            return False
        if required.get("machine_stage", False) and not (distinct_machine > 0 and distinct_stage > 0):
            return False
        return True

    for table_name in candidate_tables:
        df = raw.get(table_name, pd.DataFrame())
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        present = True
        tables.append(_table_display_name(table_name))
        inv_row = _inventory_row(inventory, table_name)
        rows = int(inv_row.get("rows", len(df)) or len(df))
        if rows > 0:
            reasons.append("rows")
        if bool(inv_row.get("has_join_cols", False)):
            reasons.append("join")
        if bool(inv_row.get("has_result_cols", False)) or bool(inv_row.get("has_quality_cols", False)):
            reasons.append("result/quality")
        if int(inv_row.get("distinct_machine", 0) or 0) > 0:
            reasons.append("machine")
        if int(inv_row.get("distinct_stage", 0) or 0) > 0:
            reasons.append("stage")
        if _table_usability(inv_row):
            usable = True
        evidence_cols.extend([
            c for c in df.columns
            if any(tok in str(c).lower() for tok in ["machine", "mach", "line", "lane", "stage", "file_dt", "make_dt", "event_ts", "lot", "barcode", "panelbarcode", "result", "quality", "inspect", "fail", "judge", "call", "review", "output"])
        ])

    if not present:
        return {
            "공정": process_name,
            "존재": "없음",
            "판단": "없음",
            "관련 테이블": "-",
            "근거 컬럼": "-",
            "판정근거": "-",
        }

    return {
        "공정": process_name,
        "존재": "✔" if usable else "제한",
        "판단": "분석 가능" if usable else "제한",
        "관련 테이블": _format_preview(sorted(set(tables)), limit=3) if tables else "-",
        "근거 컬럼": _format_preview(sorted(set(str(c) for c in evidence_cols)), limit=6) if evidence_cols else "-",
        "판정근거": _format_preview(sorted(set(reasons)), limit=4) if reasons else "-",
    }


def build_data_scope_summary(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None) -> pd.DataFrame:
    inv = inventory if inventory is not None else build_full_period_data_inventory(raw)
    if inv.empty:
        return pd.DataFrame()
    used = inv[inv["rows"].fillna(0).gt(0)].copy()
    if used.empty:
        return pd.DataFrame()
    period_min = pd.to_datetime(used["min_datetime"], errors="coerce").min() if "min_datetime" in used.columns else pd.NaT
    period_max = pd.to_datetime(used["max_datetime"], errors="coerce").max() if "max_datetime" in used.columns else pd.NaT
    scope_tokens = [
        ("라인 범위", ["line", "_linecode"]),
        ("설비 범위", ["machine_id", "mach_cd", "_equipcode"]),
        ("공정 범위", ["stage", "stage_no", "section", "lane", "process"]),
        ("LOT 범위", ["lot", "barcode", "panelbarcode"]),
        ("모델 범위", ["model", "pcbmodel"]),
    ]
    rows = []
    rows.append({
        "항목": "데이터 기간",
        "존재": "✔" if pd.notna(period_min) and pd.notna(period_max) else "제한",
        "값": f"{period_min.date() if pd.notna(period_min) else '-'} ~ {period_max.date() if pd.notna(period_max) else '-'}",
        "근거": "raw 기간 필드의 min/max",
    })
    rows.append({
        "항목": "분석에 쓰인 데이터",
        "존재": "✔",
        "값": _format_preview([
            "생산 데이터" if any(t in raw and not raw.get(t, pd.DataFrame()).empty for t in ["fa_26_34_mounter_dtl", "fa_26_34_mounter_hdr", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl", "fa_2_marking_dtl"]) else "",
            "이벤트 데이터" if "_mounter_tag" in raw and not raw.get("_mounter_tag", pd.DataFrame()).empty else "",
            "품질/검사 데이터" if any(t in raw and not raw.get(t, pd.DataFrame()).empty for t in ["fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"]) else "",
            "상세추적 ITEM" if "fa_26_34_mounter_dtl" in raw and not raw.get("fa_26_34_mounter_dtl", pd.DataFrame()).empty else "",
        ]),
        "근거": "실제 활용 중인 데이터 영역",
    })
    for label, tokens in scope_tokens:
        values = _collect_values(raw, tokens)
        rows.append({
            "항목": label,
            "존재": "✔" if values else "제한",
            "값": _format_preview(values),
            "근거": ", ".join(sorted({c for df in raw.values() if isinstance(df, pd.DataFrame) and not df.empty for c in df.columns if any(tok in str(c).lower() for tok in tokens)})) or "-",
        })
    return pd.DataFrame(rows)


def build_data_structure_summary(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None, item_fact: pd.DataFrame = None) -> pd.DataFrame:
    inv = inventory if inventory is not None else build_full_period_data_inventory(raw)
    item_fact = item_fact if item_fact is not None else build_mounter_item_fact(raw)
    rows = []

    def add(structure: str, status: str, description: str, evidence: str, tables: str):
        rows.append({
            "구조": structure,
            "존재": status,
            "설명": description,
            "근거 컬럼": evidence,
            "대표 테이블": tables,
        })

    event_cols = set()
    event_present = False
    if "_mounter_tag" in raw and not raw.get("_mounter_tag", pd.DataFrame()).empty:
        tag_df = raw["_mounter_tag"]
        event_present = True
        event_cols.update([c for c in tag_df.columns if any(tok in str(c).lower() for tok in ["devicedate", "insertdate", "event_ts", "tagname", "value"])])
    add(
        "설비 이벤트 데이터",
        "✔" if event_present else "없음",
        "stop / alarm / tag처럼 시간 순서로 보는 데이터",
        ", ".join(sorted(event_cols)) if event_cols else "_devicedate, _insertdate, _tagname, _value",
        "mounter_tag" if event_present else "-",
    )
    if event_present:
        tag_events = raw["_mounter_tag"].copy()
        event_types = tag_events.apply(lambda r: classify_event(r.get("_tagname", ""), r.get("ITEM", ""), r.get("_value", "")), axis=1)
        event_types = sorted([x for x in set(event_types.astype(str).tolist()) if x not in {"OTHER", "", "nan", "None"}])
        add(
            "SMT 이벤트 유형",
            "✔" if event_types else "제한",
            "SETUP / FEEDER_ERROR / PICKUP_ERROR / RECOG_ERROR / PLACE_ERROR / TRANSFER_ERROR / WAIT",
            "classify_event(_tagname, ITEM, _value)",
            _format_preview(event_types) if event_types else "-",
        )

    prod_tables = [t for t in ["fa_26_34_mounter_dtl", "fa_26_34_mounter_hdr", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl", "fa_2_marking_dtl"] if t in raw and not raw.get(t, pd.DataFrame()).empty]
    prod_cols = set()
    for t in prod_tables:
        prod_cols.update([c for c in raw[t].columns if any(tok in str(c).lower() for tok in ["machine", "stage", "line", "lot", "model", "order", "section", "lane"])])
    prod_status = "✔" if prod_tables and prod_cols else ("제한" if prod_tables else "없음")
    add(
        "생산 실적 데이터",
        prod_status,
        "설비-공정-LOT-모델 축을 가진 생산/실적 데이터",
        ", ".join(sorted(prod_cols)) if prod_cols else "machine_id, stage_no, line_id, lot_id, model_label",
        _format_preview([_table_display_name(t) for t in prod_tables]),
    )

    quality_tables = [t for t in ["fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"] if t in raw and not raw.get(t, pd.DataFrame()).empty]
    quality_cols = set()
    for t in quality_tables:
        quality_cols.update([c for c in raw[t].columns if any(tok in str(c).lower() for tok in ["result", "fail", "ng", "review", "call", "judge", "quality", "inspect", "defect"])])
    quality_status = "✔" if quality_tables and quality_cols else ("제한" if quality_tables else "없음")
    add(
        "품질/검사 데이터",
        quality_status,
        "SPI / AOI / MOI의 합격/불합격/결함 신호",
        ", ".join(sorted(quality_cols)) if quality_cols else "result, quality_flag, fail, defect 관련 컬럼",
        _format_preview([_table_display_name(t) for t in quality_tables]),
    )

    item_status = "✔" if not item_fact.empty and item_fact["item_key"].nunique() > 0 else ("제한" if "fa_26_34_mounter_dtl" in raw and not raw.get("fa_26_34_mounter_dtl", pd.DataFrame()).empty else "없음")
    item_groups = _format_preview(sorted(item_fact["item_group"].dropna().astype(str).unique().tolist())) if not item_fact.empty else "-"
    add(
        "상세추적 구조",
        item_status,
        "mounter ITEM key/value를 풀어서 본 상세추적 구조",
        "ITEM, VALUE, item_group, item_numeric_value",
        f"feeder / nozzle / pickup / recognition / transfer / part ({item_groups})" if item_groups != "-" else "feeder / nozzle / pickup / recognition / transfer / part",
    )
    return pd.DataFrame(rows)


def build_data_linkage_summary(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None, linkage: pd.DataFrame = None) -> pd.DataFrame:
    inv = inventory if inventory is not None else build_full_period_data_inventory(raw)
    link = linkage if linkage is not None else build_table_linkage_matrix(raw, inv)
    rows = []

    flow_tables = [
        t for t in [
            "fa_26_34_mounter_dtl",
            "_mounter_tag",
            "fa_24_spi_dtl",
            "fa_14_aoi_dtl",
            "fa_42_aoi_dtl",
            "fa_35_moi_dtl",
            "fa_2_marking_dtl",
        ]
        if t in raw and not raw.get(t, pd.DataFrame()).empty
    ]
    flow_text = " → ".join([_table_display_name(t) for t in flow_tables]) if flow_tables else "-"
    rows.append({
        "연결 축": "현장 흐름",
        "상태": "✔" if flow_tables else "없음",
        "설명": flow_text,
        "근거": "설비 이벤트 → 검사 → 품질 흐름",
    })

    key_rows = [
        ("machine + time", ["machine_id", "mach_cd", "_equipcode"], ["event_ts", "file_dt", "make_dt", "_devicedate", "_insertdate"], "주요 연결 키"),
        ("lot", ["lot_id", "lot_nm", "lot_name", "barcode", "panelbarcode"], ["lot_id", "lot_nm", "lot_name", "barcode", "panelbarcode"], "보조 연결 키"),
        ("model / barcode", ["model", "model_label", "pcbmodel", "barcode", "panelbarcode"], ["model", "model_label", "pcbmodel", "barcode", "panelbarcode"], "부분 연결"),
    ]
    for label, key_tokens, time_tokens, note in key_rows:
        key_cols = sorted({c for df in raw.values() if isinstance(df, pd.DataFrame) and not df.empty for c in df.columns if any(tok in str(c).lower() for tok in key_tokens)})
        time_cols = sorted({c for df in raw.values() if isinstance(df, pd.DataFrame) and not df.empty for c in df.columns if any(tok in str(c).lower() for tok in time_tokens)})
        possible = False
        if not link.empty:
            if label == "machine + time":
                possible = bool(link.loc[link["linkage"].eq("machine+time"), "possible"].any())
            elif label == "lot":
                possible = bool(link.loc[link["linkage"].eq("lot-based"), "possible"].any())
            elif label == "model / barcode":
                possible = bool(link.loc[link["linkage"].eq("model/barcode"), "possible"].any())
        rows.append({
            "연결 축": label,
            "상태": "가능" if possible else ("제한" if key_cols or time_cols else "불가"),
            "설명": note,
            "근거": f"설비/시간/LOT/모델 키: {', '.join(key_cols) if key_cols else '-'} | 시간: {', '.join(time_cols) if time_cols else '-'}",
        })
    pair_rows = [
        ("mounter_dtl ↔ mounter_tag", "machine+time+stage"),
        ("mounter ↔ spi", "mounter↔spi"),
        ("mounter ↔ aoi", "mounter↔aoi"),
        ("mounter ↔ moi", "mounter↔moi"),
    ]
    for label, linkage_name in pair_rows:
        possible = False
        evidence = "-"
        if not link.empty and "linkage" in link.columns:
            subset = link[link["linkage"].eq(linkage_name)]
            if not subset.empty:
                possible = bool(subset["possible"].astype(bool).any())
                evidence = ", ".join(sorted(subset["evidence"].dropna().astype(str).unique().tolist())) or "-"
        rows.append({
            "연결 축": label,
            "상태": "가능" if possible else "제한",
            "설명": "테이블 간 직접 연결",
            "근거": evidence,
        })
    return pd.DataFrame(rows)


def build_data_category_summary(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None, item_fact: pd.DataFrame = None) -> pd.DataFrame:
    inv = inventory if inventory is not None else build_full_period_data_inventory(raw)
    item_fact = item_fact if item_fact is not None else build_mounter_item_fact(raw)
    rows = []

    def add(category: str, status: str, description: str, evidence: str, tables: str):
        rows.append({
            "카테고리": category,
            "존재": status,
            "설명": description,
            "근거 컬럼": evidence,
            "대표 원천": tables,
        })

    prod_present = any(t in raw and not raw.get(t, pd.DataFrame()).empty for t in ["fa_26_34_mounter_dtl", "fa_26_34_mounter_hdr", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl", "fa_2_marking_dtl"])
    prod_evidence = set()
    for t in ["fa_26_34_mounter_dtl", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl", "fa_2_marking_dtl"]:
        if t in raw and not raw.get(t, pd.DataFrame()).empty:
            prod_evidence.update([c for c in raw[t].columns if any(tok in str(c).lower() for tok in ["machine", "stage", "line", "lot", "model"])])
    add(
        "생산 데이터",
        "✔" if prod_present else "없음",
        "생산/실적과 machine / stage / lot / model 축",
        ", ".join(sorted(prod_evidence)) if prod_evidence else "machine_id, stage_no, lot_id, model_label",
        ", ".join([_table_display_name(t) for t in ["fa_26_34_mounter_dtl", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl", "fa_2_marking_dtl"] if t in raw and not raw.get(t, pd.DataFrame()).empty]) or "-",
    )

    event_present = "_mounter_tag" in raw and not raw.get("_mounter_tag", pd.DataFrame()).empty
    event_evidence = ""
    if event_present:
        tag = raw["_mounter_tag"]
        event_evidence = ", ".join([c for c in tag.columns if any(tok in str(c).lower() for tok in ["devicedate", "insertdate", "tagname", "value", "equipcode", "linecode"])])
    add(
        "이벤트 데이터",
        "✔" if event_present else "없음",
        "stop / alarm / tag 형태의 시간 이벤트",
        event_evidence or "_devicedate, _insertdate, _tagname, _value",
        "mounter_tag" if event_present else "-",
    )

    quality_present = any(t in raw and not raw.get(t, pd.DataFrame()).empty for t in ["fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"])
    quality_evidence = set()
    for t in ["fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"]:
        if t in raw and not raw.get(t, pd.DataFrame()).empty:
            quality_evidence.update([c for c in raw[t].columns if any(tok in str(c).lower() for tok in ["result", "fail", "ng", "review", "call", "judge", "quality", "inspect", "defect"])])
    add(
        "품질 데이터",
        "✔" if quality_present else "없음",
        "SPI / AOI / MOI의 result / defect / fail 신호",
        ", ".join(sorted(quality_evidence)) if quality_evidence else "result, quality_flag, fail, defect 관련 컬럼",
        ", ".join([_table_display_name(t) for t in ["fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"] if t in raw and not raw.get(t, pd.DataFrame()).empty]) or "-",
    )

    flow_present = any(t in raw and not raw.get(t, pd.DataFrame()).empty for t in ["fa_26_34_mounter_dtl", "_mounter_tag", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl", "fa_2_marking_dtl"])
    flow_evidence = set()
    for t in ["fa_26_34_mounter_dtl", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl", "fa_2_marking_dtl"]:
        if t in raw and not raw.get(t, pd.DataFrame()).empty:
            flow_evidence.update([c for c in raw[t].columns if any(tok in str(c).lower() for tok in ["stage", "lane", "section", "order", "flow", "process"])])
    add(
        "공정 흐름 데이터",
        "✔" if flow_present else "없음",
        "stage / lane / section / machine_order 기반 흐름",
        ", ".join(sorted(flow_evidence)) if flow_evidence else "stage_no, lane, section, machine_order",
        ", ".join([_table_display_name(t) for t in ["fa_26_34_mounter_dtl", "fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"] if t in raw and not raw.get(t, pd.DataFrame()).empty]) or "-",
    )

    if not item_fact.empty:
        item_status = "핵심 분석 가능 (정규화 필요)"
    elif "fa_26_34_mounter_dtl" in raw and not raw.get("fa_26_34_mounter_dtl", pd.DataFrame()).empty:
        item_status = "제한"
    else:
        item_status = "없음"
    item_groups = _format_preview(sorted(item_fact["item_group"].dropna().astype(str).unique().tolist())) if not item_fact.empty else "-"
    add(
        "상세 추적 데이터",
        item_status,
        "mounter ITEM key/value 기반 상세 추적",
        "ITEM, VALUE, item_group, item_numeric_value",
        f"mounter_dtl ({item_groups})" if item_groups != "-" else "mounter_dtl",
    )
    return pd.DataFrame(rows)


def _detect_process_presence(raw: Dict[str, pd.DataFrame], process_name: str) -> Dict[str, object]:
    aliases = PROCESS_ALIASES.get(process_name, [process_name.lower()])
    tables = []
    evidence_cols = []
    match_basis = []
    for table_name, df in raw.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        table_text = str(table_name).lower()
        cols = [str(c).lower() for c in df.columns]
        text = " ".join(cols)
        table_hit = any(alias in table_text for alias in aliases)
        col_hit = any(alias in text for alias in aliases)
        if table_hit or col_hit:
            tables.append(_table_display_name(table_name))
            if table_hit:
                match_basis.append("table")
            if col_hit:
                match_basis.append("column")
            evidence_cols.extend([c for c in df.columns if any(alias in str(c).lower() for alias in aliases)])
    direct = bool(tables)
    inferred = False
    if not direct and process_name == "AOI_POST":
        inferred = any(t in raw and not raw.get(t, pd.DataFrame()).empty for t in ["fa_35_moi_dtl", "fa_35_moi_hdr"])
        if inferred:
            tables.append("moi_dtl / moi_hdr")
            evidence_cols.extend([c for t in ["fa_35_moi_dtl", "fa_35_moi_hdr"] if t in raw and not raw.get(t, pd.DataFrame()).empty for c in raw[t].columns if any(tok in str(c).lower() for tok in ["machineresult", "reviewresult", "callresult", "pcbmodel", "barcode", "panelbarcode"])])
            match_basis.append("indirect")
    return {
        "공정": process_name,
        "존재": "✔" if direct or inferred else "없음",
        "판단": "직접 확인" if direct else ("간접 해석" if inferred else "없음"),
        "관련 테이블": _format_preview(tables, limit=3) if tables else "-",
        "근거 컬럼": _format_preview(sorted(set(str(c) for c in evidence_cols)), limit=6) if evidence_cols else "-",
        "판정근거": _format_preview(match_basis, limit=3) if match_basis else "-",
    }


def build_process_flow_summary(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None) -> pd.DataFrame:
    inv = inventory if inventory is not None else build_full_period_data_inventory(raw)
    rows = [_evaluate_process_analysis(raw, inv, process_name) for process_name in PROCESS_FLOW]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order_map = {name: i for i, name in enumerate(PROCESS_FLOW)}
    df["_order"] = df["공정"].map(order_map)
    df["공정 순서"] = df["공정"].map(order_map).add(1)
    return df.sort_values("_order").drop(columns=["_order"])


def build_process_coverage(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None) -> pd.DataFrame:
    flow = build_process_flow_summary(raw, inventory)
    if flow.empty:
        return pd.DataFrame()
    covered = flow[flow["판단"].eq("분석 가능")]["공정"].tolist()
    limited = flow[flow["판단"].eq("제한")]["공정"].tolist()
    missing = flow[flow["판단"].eq("없음")]["공정"].tolist()
    rows = [
        {"구분": "분석 가능", "값": ", ".join(covered) if covered else "-", "설명": "시간/설비/결과 키가 갖춰져 실제 분석에 쓰기 가능한 공정"},
        {"구분": "제한", "값": ", ".join(limited) if limited else "-", "설명": "테이블은 있으나 핵심 키 또는 결과 컬럼이 부족한 공정"},
        {"구분": "누락", "값": ", ".join(missing) if missing else "-", "설명": "현재 raw에서 확인되지 않는 공정"},
    ]
    return pd.DataFrame(rows)


def build_rca_capability_summary(raw: Dict[str, pd.DataFrame], inventory: pd.DataFrame = None, linkage: pd.DataFrame = None, item_fact: pd.DataFrame = None) -> pd.DataFrame:
    inv = inventory if inventory is not None else build_full_period_data_inventory(raw)
    link = linkage if linkage is not None else build_table_linkage_matrix(raw, inv)
    item_fact = item_fact if item_fact is not None else build_mounter_item_fact(raw)
    flow = build_process_flow_summary(raw, inv)
    stop_present = "_mounter_tag" in raw and not raw.get("_mounter_tag", pd.DataFrame()).empty
    quality_present = any(t in raw and not raw.get(t, pd.DataFrame()).empty for t in ["fa_24_spi_dtl", "fa_14_aoi_dtl", "fa_42_aoi_dtl", "fa_35_moi_dtl"])
    lot_present = (inv["distinct_lot"] > 0).any() if "distinct_lot" in inv.columns else False
    machine_present = (inv["distinct_machine"] > 0).any() if "distinct_machine" in inv.columns else False
    time_present = bool(inv["min_datetime"].notna().any()) if "min_datetime" in inv.columns else False
    item_present = not item_fact.empty and item_fact["item_group"].isin(["FEEDER", "NOZZLE", "PART", "PICKUP", "RECOGNITION", "TRANSFER"]).any()

    def stat(cond: bool) -> str:
        return "✔" if cond else "제한"

    machine_time = bool(link.loc[link["linkage"].eq("machine+time"), "possible"].any()) if not link.empty else False
    lot_link = bool(link.loc[link["linkage"].eq("lot-based"), "possible"].any()) if not link.empty else False
    rows = [
        {"항목": "설비 단위 분석", "상태": stat(machine_present), "설명": "machine 기준 비교 가능"},
        {"항목": "공정 단위 분석", "상태": stat(not flow.empty and flow["판단"].eq("분석 가능").any()), "설명": "공정 흐름과 결과 키를 함께 비교 가능"},
        {"항목": "LOT 단위 추적", "상태": stat(lot_present), "설명": "lot 기준으로 추적 가능"},
        {"항목": "시간대 분석", "상태": stat(time_present), "설명": "event_ts / file_dt / make_dt 기반"},
        {"항목": "부품 단위 분석", "상태": stat(item_present), "설명": "mounter ITEM 정규화 기반"},
        {"항목": "설비 이벤트 ↔ 품질", "상태": stat(stop_present and quality_present), "설명": "stop/tag와 검사 결과 연결"},
        {"항목": "시간 event ↔ defect", "상태": stat(time_present and quality_present), "설명": "시간대별 defect 매칭"},
        {"항목": "LOT ↔ defect", "상태": stat(lot_present and quality_present), "설명": "LOT 기준 defect 추적"},
        {"항목": "RCA 가능 수준", "상태": "가능" if machine_time and lot_link and quality_present else "제한", "설명": "설비/공정/LOT 단위 RCA 추적"},
    ]
    return pd.DataFrame(rows)


def _attach_dimensions(df: pd.DataFrame, machine_dim: pd.DataFrame, lot_dim: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if not machine_dim.empty and "machine_id" in out.columns and "machine_id" in machine_dim.columns:
        keep = [c for c in ["machine_id", "machine_name", "line_id", "stage_no", "machine_order"] if c in machine_dim.columns]
        if keep:
            md = machine_dim[keep].drop_duplicates(subset=["machine_id"])
            out = out.merge(md, on="machine_id", how="left", suffixes=("", "_machine"))
            for col in ["line_id", "stage_no", "machine_order"]:
                dim_col = f"{col}_machine"
                if dim_col in out.columns:
                    if col in out.columns:
                        out[col] = out[col].combine_first(out[dim_col])
                    else:
                        out[col] = out[dim_col]
                    out = out.drop(columns=[dim_col])
    if not lot_dim.empty and "lot_id" in out.columns and "lot_id" in lot_dim.columns:
        keep = [c for c in ["lot_id", "lot_name", "model_name", "line_id"] if c in lot_dim.columns]
        if keep:
            ld = lot_dim[keep].drop_duplicates(subset=["lot_id"])
            out = out.merge(ld, on="lot_id", how="left", suffixes=("", "_lot"))
            for col in ["lot_name", "model_name", "line_id"]:
                dim_col = f"{col}_lot"
                if dim_col in out.columns:
                    if col in out.columns:
                        out[col] = out[col].combine_first(out[dim_col])
                    else:
                        out[col] = out[dim_col]
                    out = out.drop(columns=[dim_col])
            if "model_name" in out.columns:
                if "model_label" in out.columns:
                    lot_like = out["model_label"].astype(str).str.startswith("LOT", na=False) | out["model_label"].isna()
                    out.loc[lot_like, "model_label"] = out.loc[lot_like, "model_name"]
                    out["model_label"] = out["model_label"].combine_first(out["model_name"])
                else:
                    out["model_label"] = out["model_name"]
    return out


def _pick_metric_column(df: pd.DataFrame, preferred: List[str]) -> str:
    for col in preferred:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").fillna(0).sum() != 0:
            return col
    for col in preferred:
        if col in df.columns:
            return col
    return ""


def build_equipment_overview(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    tag = clean.get("vw_tag_event_fact", pd.DataFrame()).copy()
    comp = clean.get("vw_component_error_fact", pd.DataFrame()).copy()
    if shop.empty and stop.empty and insp.empty and tag.empty and comp.empty:
        return pd.DataFrame()
    frames = []
    if not shop.empty and "machine_id" in shop.columns:
        shop_use = shop.copy()
        if "output_qty" not in shop_use.columns:
            shop_use["output_qty"] = 1.0
        prod_group_cols = [c for c in ["machine_id", "line_id", "stage_no", "machine_order"] if c in shop.columns]
        prod = shop_use.groupby(prod_group_cols, as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            event_rows=("event_ts", "size"),
        )
        frames.append(prod)
    if not stop.empty and "machine_id" in stop.columns:
        stop_use = stop.copy()
        stop_use["avg_stop"] = stop_use["duration_sec"] / stop_use["stop_count"].replace(0, np.nan)
        stop_agg = stop_use.groupby(["machine_id"], as_index=False).agg(
            stop_time=("duration_sec", "sum"),
            stop_count=("stop_count", "sum"),
            avg_stop_time=("avg_stop", "mean"),
        )
        frames.append(stop_agg)
    if not insp.empty and "machine_id" in insp.columns:
        insp_agg = insp.groupby(["machine_id", "line_id"], as_index=False).agg(
            inspect_count=("quality_flag", "size"),
            fail_count=("quality_flag", lambda s: (s == "FAIL").sum()),
        )
        insp_agg["fail_rate"] = insp_agg.apply(lambda r: _safe_div(r["fail_count"], r["inspect_count"]), axis=1)
        frames.append(insp_agg)
    if not tag.empty and "machine_id" in tag.columns:
        tag_agg = tag.groupby(["machine_id"], as_index=False).agg(
            tag_events=("event_ts", "size"),
            setup_events=("event_class", lambda s: (s == "SETUP").sum()),
            feeder_error_events=("event_class", lambda s: (s == "FEEDER_ERROR").sum()),
            pickup_error_events=("event_class", lambda s: (s == "PICKUP_ERROR").sum()),
            recog_error_events=("event_class", lambda s: (s == "RECOG_ERROR").sum()),
            place_error_events=("event_class", lambda s: (s == "PLACE_ERROR").sum()),
            transfer_error_events=("event_class", lambda s: (s == "TRANSFER_ERROR").sum()),
            wait_events=("event_class", lambda s: (s == "WAIT").sum()),
            inspection_events=("event_class", lambda s: (s == "INSPECTION").sum()),
            stop_events=("event_class", lambda s: (s == "STOP").sum()),
        )
        frames.append(tag_agg)
    if not comp.empty and "machine_id" in comp.columns:
        comp_agg = comp.groupby(["machine_id"], as_index=False).agg(
            pickup_count=("pickup_count", "sum"),
            error_count=("error_count", "sum"),
        )
        comp_agg["pickup_error_rate"] = comp_agg.apply(lambda r: _safe_div(r["error_count"], r["pickup_count"]), axis=1)
        frames.append(comp_agg)
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=[c for c in ["machine_id", "line_id"] if c in out.columns and c in frame.columns] or ["machine_id"], how="outer")
    time_frames = []
    for df in [shop, stop, insp, tag]:
        if not df.empty and "machine_id" in df.columns and "event_ts" in df.columns:
            tmp = df[["machine_id", "event_ts"]].copy()
            tmp["event_ts"] = pd.to_datetime(tmp["event_ts"], errors="coerce")
            time_frames.append(tmp.dropna(subset=["machine_id", "event_ts"]))
    if time_frames:
        time_df = pd.concat(time_frames, ignore_index=True, sort=False)
        time_agg = time_df.groupby(["machine_id"], as_index=False).agg(
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        time_agg["observed_span_sec"] = (pd.to_datetime(time_agg["last_event_ts"], errors="coerce") - pd.to_datetime(time_agg["first_event_ts"], errors="coerce")).dt.total_seconds().fillna(0)
        out = out.merge(time_agg, on="machine_id", how="left")
    for col in ["production_rows", "output_qty", "event_rows", "stop_time", "stop_count", "avg_stop_time", "inspect_count", "fail_count", "fail_rate", "tag_events", "setup_events", "feeder_error_events", "pickup_error_events", "recog_error_events", "place_error_events", "transfer_error_events", "wait_events", "inspection_events", "stop_events", "pickup_count", "error_count", "pickup_error_rate", "observed_span_sec"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    for col in ["observed_span_sec", "stop_time", "stop_count", "stop_events"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["avg_stop_time"] = out.apply(lambda r: _safe_div(r.get("stop_time", 0), r.get("stop_count", 0)), axis=1) if "stop_time" in out.columns else 0
    out["failure_count"] = out.apply(lambda r: r.get("stop_count", 0) if r.get("stop_count", 0) > 0 else r.get("stop_events", 0), axis=1)
    out["uptime_sec"] = (out["observed_span_sec"] - out["stop_time"]).clip(lower=0)
    out["mtbf_sec"] = out.apply(lambda r: _safe_div(r.get("uptime_sec", 0), r.get("failure_count", 0)), axis=1)
    out["mttr_sec"] = out.apply(lambda r: _safe_div(r.get("stop_time", 0), r.get("failure_count", 0)), axis=1)
    out["defect_count"] = out.get("fail_count", 0)
    out["defect_rate"] = out.apply(lambda r: _safe_div(r.get("fail_count", 0), r.get("inspect_count", 0)), axis=1)
    event_cols = [c for c in ["tag_events", "stop_events", "inspection_events", "setup_events", "feeder_error_events", "pickup_error_events", "recog_error_events", "place_error_events", "transfer_error_events", "wait_events"] if c in out.columns]
    out["event_density"] = out[event_cols].sum(axis=1) if event_cols else 0
    out["performance_gap"] = out.apply(lambda r: (r.get("stop_time", 0) or 0) + (r.get("fail_count", 0) or 0) * 10, axis=1)
    out["is_outlier"] = False
    if "stop_time" in out.columns and len(out) >= 3:
        q1 = pd.to_numeric(out["stop_time"], errors="coerce").quantile(0.25)
        q3 = pd.to_numeric(out["stop_time"], errors="coerce").quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr if pd.notna(iqr) else np.nan
        out["is_outlier"] = pd.to_numeric(out["stop_time"], errors="coerce").gt(threshold)
    out = out.sort_values(["performance_gap", "defect_rate", "stop_time"], ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def build_process_overview(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    def _norm_stage(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        text = numeric.astype("Int64").astype("string")
        text = text.where(numeric.notna(), series.astype("string"))
        return text.fillna("").str.strip()
    def _norm_line(series: pd.Series) -> pd.Series:
        return series.astype("string").fillna("").str.strip()
    if shop.empty and stop.empty and insp.empty:
        out = pd.DataFrame({"process_name": PROCESS_FLOW})
        out["scope"] = "process"
        out["process_display"] = out["process_name"].map(
            {
                "Printer": "printer",
                "SPI": "spi",
                "Mounter": "mounter",
                "AOI": "aoi",
                "Reflow": "reflow",
                "AOI_POST": "aoi_post",
            }
        ).fillna(out["process_name"].str.lower())
        out["data_status"] = "데이터 없음"
        out["output_status"] = "데이터 없음"
        out["stop_status"] = "데이터 없음"
        out["defect_status"] = "데이터 없음"
        out["production_rows"] = 0
        out["output_qty"] = 0
        out["distinct_machines"] = 0
        out["lot_count"] = 0
        out["stop_time"] = 0
        out["stop_count"] = 0
        out["inspect_count"] = 0
        out["fail_count"] = 0
        out["fail_rate"] = 0
        out["bottleneck_score"] = 0
        out["rank"] = np.arange(1, len(out) + 1)
        out["process_order"] = out["process_name"].map({name: i for i, name in enumerate(PROCESS_FLOW)})
        out["line_id"] = "-"
        out["stage_no"] = "-"
        return out

    def _canonical_process(raw_name: object) -> str | None:
        text = str(raw_name or "").lower()
        if "mounter" in text or "실장" in text:
            return "Mounter"
        if "spi" in text:
            return "SPI"
        if "aoi" in text:
            return "AOI"
        if "moi" in text or "final" in text:
            return "AOI_POST"
        if "print" in text or "printer" in text or "solder paste" in text:
            return "Printer"
        if "reflow" in text or "oven" in text:
            return "Reflow"
        return None

    rows = []
    stage_map = pd.DataFrame()
    machine_map = pd.DataFrame()
    if not shop.empty and {"line_id", "stage_no", "process_name"}.issubset(shop.columns):
        shop = shop.copy()
        shop["line_id"] = _norm_line(shop["line_id"])
        shop["stage_no"] = _norm_stage(shop["stage_no"])
        if "machine_id" in shop.columns:
            shop["machine_id"] = _norm_line(shop["machine_id"])
        stage_map = (
            shop.assign(process_canonical=shop["process_name"].map(_canonical_process))
            .dropna(subset=["process_canonical"])
            .groupby(["line_id", "stage_no"], as_index=False)
            .agg(process_canonical=("process_canonical", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
        )
        if "machine_id" in shop.columns:
            machine_map = (
                shop.assign(process_canonical=shop["process_name"].map(_canonical_process))
                .dropna(subset=["process_canonical"])
                .groupby(["machine_id"], as_index=False)
                .agg(process_canonical=("process_canonical", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
            )

    for process_name in PROCESS_FLOW:
        row: Dict[str, object] = {
            "process_name": process_name,
            "process_display": {
                "Printer": "printer",
                "SPI": "spi",
                "Mounter": "mounter",
                "AOI": "aoi",
                "Reflow": "reflow",
                "AOI_POST": "aoi_post",
            }.get(process_name, process_name.lower()),
            "scope": "process",
            "data_status": "데이터 없음",
            "output_status": "데이터 없음",
            "stop_status": "데이터 없음",
            "defect_status": "데이터 없음",
            "production_rows": 0,
            "output_qty": 0,
            "distinct_machines": 0,
            "lot_count": 0,
            "stop_time": 0,
            "stop_count": 0,
            "inspect_count": 0,
            "fail_count": 0,
            "fail_rate": 0,
            "bottleneck_score": 0,
            "line_id": "-",
            "stage_no": "-",
            "machine_order": np.nan,
        }

        shop_proc = pd.DataFrame()
        if not shop.empty and "process_name" in shop.columns:
            shop_proc = shop.copy()
            shop_proc["process_canonical"] = shop_proc["process_name"].map(_canonical_process)
            shop_proc = shop_proc[shop_proc["process_canonical"].eq(process_name)]
        if not shop_proc.empty:
            agg = shop_proc.groupby(["process_canonical"], as_index=False).agg(
                production_rows=("event_ts", "size"),
                output_qty=("output_qty", "sum"),
                distinct_machines=("machine_id", pd.Series.nunique),
                lot_count=("lot_id", pd.Series.nunique),
                fail_count=("result_primary", lambda s: s.astype(str).str.contains("FAIL|NG|ERR|BAD", case=False, na=False).sum()),
                inspect_count=("result_primary", "size"),
                line_id=("line_id", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
                stage_no=("stage_no", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
                machine_order=("machine_order", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0] if len(s) else np.nan),
            )
            row.update(agg.iloc[0].to_dict())
            row["data_status"] = "데이터 있음"
            row["output_qty"] = float(row.get("production_rows", 0) or 0)
            row["output_status"] = "데이터 있음" if (row.get("production_rows", 0) or 0) > 0 else "데이터 없음"
            row["output_source"] = "throughput"

        stop_proc = pd.DataFrame()
        if not stop.empty:
            stop = stop.copy()
            if "line_id" in stop.columns:
                stop["line_id"] = _norm_line(stop["line_id"])
            if "stage_no" in stop.columns:
                stop["stage_no"] = _norm_stage(stop["stage_no"])
            if "machine_id" in stop.columns:
                stop["machine_id"] = _norm_line(stop["machine_id"])
            if {"line_id", "stage_no"}.issubset(stop.columns) and not stage_map.empty:
                stop_proc = stop.merge(stage_map, on=["line_id", "stage_no"], how="left")
            if stop_proc.empty and "machine_id" in stop.columns and not machine_map.empty:
                stop_proc = stop.merge(machine_map, on=["machine_id"], how="left")
            stop_proc = stop_proc[stop_proc["process_canonical"].eq(process_name)] if not stop_proc.empty else pd.DataFrame()
        if not stop_proc.empty:
            stop_agg = stop_proc.groupby(["process_canonical"], as_index=False).agg(
                stop_time=("duration_sec", "sum"),
                stop_count=("stop_count", "sum"),
            )
            row["stop_time"] = float(stop_agg.iloc[0]["stop_time"])
            row["stop_count"] = float(stop_agg.iloc[0]["stop_count"])
            row["data_status"] = "데이터 있음"
            row["stop_status"] = "데이터 있음"

        insp_proc = pd.DataFrame()
        if not insp.empty and "process_name" in insp.columns:
            insp = insp.copy()
            insp["line_id"] = _norm_line(insp["line_id"]) if "line_id" in insp.columns else "-"
            insp["stage_no"] = _norm_stage(insp["stage_no"]) if "stage_no" in insp.columns else "-"
            insp_proc = insp.copy()
            insp_proc["process_canonical"] = insp_proc["process_name"].map(_canonical_process)
            insp_proc = insp_proc[insp_proc["process_canonical"].eq(process_name)]
        if not insp_proc.empty:
            insp_agg = insp_proc.groupby(["process_canonical"], as_index=False).agg(
                inspect_count=("quality_flag", "size"),
                fail_count=("quality_flag", lambda s: (s == "FAIL").sum()),
            )
            row["inspect_count"] = float(insp_agg.iloc[0]["inspect_count"])
            row["fail_count"] = float(insp_agg.iloc[0]["fail_count"])
            row["data_status"] = "데이터 있음"
            row["defect_status"] = "데이터 있음"

        row["fail_rate"] = _safe_div(row.get("fail_count", 0), row.get("inspect_count", 0))
        row["bottleneck_score"] = (row.get("stop_time", 0) or 0) + (row.get("fail_count", 0) or 0) * 8 + (row.get("lot_count", 0) or 0) * 0.5 + (row.get("production_rows", 0) or 0) * 0.05
        rows.append(row)

    out = pd.DataFrame(rows)
    out["process_order"] = out["process_name"].map({name: i for i, name in enumerate(PROCESS_FLOW)})
    out["rank"] = np.arange(1, len(out) + 1)
    out["process_display"] = out["process_display"].fillna(out["process_name"].str.lower())
    out["stage_no"] = out["stage_no"].astype("string").fillna("-")
    out["line_id"] = out["line_id"].astype("string").fillna("-")
    return out.sort_values("process_order")


def build_lot_analysis_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    if shop.empty and stop.empty and insp.empty:
        return pd.DataFrame()
    rows = []
    if not shop.empty and "lot_id" in shop.columns:
        lot_shop = shop.groupby(["lot_id", "model_label"], as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            machine_count=("machine_id", pd.Series.nunique),
            process_count=("process_name", pd.Series.nunique),
            representative_machine=("machine_id", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            representative_process=("process_name", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
        )
        rows.append(lot_shop)
    if not stop.empty and "lot_id" in stop.columns:
        lot_stop = stop.groupby(["lot_id"], as_index=False).agg(
            stop_time=("duration_sec", "sum"),
            stop_count=("stop_count", "sum"),
            stop_machine_count=("machine_id", pd.Series.nunique),
            stop_process_count=("line_id", pd.Series.nunique),
        )
        rows.append(lot_stop)
    if not insp.empty and "lot_id" in insp.columns:
        lot_insp = insp.groupby(["lot_id"], as_index=False).agg(
            inspect_count=("quality_flag", "size"),
            fail_count=("quality_flag", lambda s: (s == "FAIL").sum()),
        )
        rows.append(lot_insp)
    if not rows:
        return pd.DataFrame()
    out = rows[0]
    for frame in rows[1:]:
        out = out.merge(frame, on=[c for c in ["lot_id"] if c in out.columns and c in frame.columns], how="outer")
    for col in ["production_rows", "output_qty", "machine_count", "process_count", "stop_time", "stop_count", "stop_machine_count", "stop_process_count", "inspect_count", "fail_count"]:
        if col not in out.columns:
            out[col] = 0
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["fail_rate"] = out.apply(lambda r: _safe_div(r.get("fail_count", 0), r.get("inspect_count", 0)), axis=1)
    out["impact_score"] = out.apply(lambda r: (r.get("stop_time", 0) or 0) + (r.get("fail_count", 0) or 0) * 10 + (r.get("production_rows", 0) or 0) * 0.01, axis=1)
    out = out.sort_values(["impact_score", "fail_rate", "stop_time"], ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def build_time_pattern_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    tag = clean.get("vw_tag_event_fact", pd.DataFrame()).copy()
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    if stop.empty and tag.empty and shop.empty:
        return pd.DataFrame()

    def _shift_label(hour: float) -> str:
        if pd.isna(hour):
            return "미상"
        h = int(hour)
        if 6 <= h < 14:
            return "주간(06-14)"
        if 14 <= h < 22:
            return "석간(14-22)"
        return "야간(22-06)"

    hour_frames = []
    if not stop.empty and "event_ts" in stop.columns:
        tmp = stop.copy()
        tmp["hour"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.hour
        tmp["shift"] = tmp["hour"].map(_shift_label)
        hour_frames.append(
            tmp.groupby("hour", as_index=False).agg(
                stop_time=("duration_sec", "sum"),
                stop_count=("stop_count", "sum"),
                stop_events=("event_ts", "size"),
            )
        )
    if not tag.empty and "event_ts" in tag.columns:
        tmp = tag.copy()
        tmp["hour"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.hour
        tmp["shift"] = tmp["hour"].map(_shift_label)
        tmp["error_flag"] = tmp["event_class"].isin(["SETUP", "FEEDER_ERROR", "PICKUP_ERROR", "RECOG_ERROR", "PLACE_ERROR", "TRANSFER_ERROR", "WAIT"]).astype(int)
        hour_frames.append(
            tmp.groupby("hour", as_index=False).agg(
                error_count=("error_flag", "sum"),
                tag_events=("event_ts", "size"),
            )
        )
    if not shop.empty and "event_ts" in shop.columns:
        tmp = shop.copy()
        tmp["hour"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.hour
        tmp["shift"] = tmp["hour"].map(_shift_label)
        hour_frames.append(
            tmp.groupby("hour", as_index=False).agg(
                production_rows=("event_ts", "size"),
            )
        )

    if not hour_frames:
        return pd.DataFrame()
    hour = hour_frames[0]
    for frame in hour_frames[1:]:
        hour = hour.merge(frame, on="hour", how="outer")
    for col in ["stop_time", "stop_count", "stop_events", "error_count", "tag_events", "production_rows"]:
        if col in hour.columns:
            hour[col] = pd.to_numeric(hour[col], errors="coerce").fillna(0)
    hour["shift"] = hour["hour"].map(_shift_label)
    hour["grain"] = "hour"
    hour["bucket"] = hour["hour"].astype("Int64").astype(str).str.zfill(2)
    hour["bucket_order"] = hour["hour"].fillna(99)

    shift = hour.groupby("shift", as_index=False).agg(
        stop_time=("stop_time", "sum"),
        stop_count=("stop_count", "sum"),
        error_count=("error_count", "sum"),
        tag_events=("tag_events", "sum"),
        production_rows=("production_rows", "sum"),
    ) if not hour.empty else pd.DataFrame()
    if not shift.empty:
        order_map = {"주간(06-14)": 0, "석간(14-22)": 1, "야간(22-06)": 2, "미상": 3}
        shift["grain"] = "shift"
        shift["bucket"] = shift["shift"]
        shift["bucket_order"] = shift["shift"].map(order_map).fillna(99)

    out = pd.concat([hour, shift], ignore_index=True, sort=False) if not shift.empty else hour
    if out.empty:
        return out
    for col in ["error_count", "stop_time", "stop_count", "production_rows", "tag_events", "stop_events"]:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out.sort_values(["grain", "bucket_order"])


def build_quality_overview(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    tag = clean.get("vw_tag_event_fact", pd.DataFrame()).copy()
    rows = []
    if not insp.empty:
        by_flag = insp.groupby(["quality_flag"], as_index=False).agg(count=("quality_flag", "size"))
        by_flag["scope"] = "inspection"
        by_flag["metric"] = "defect_flag"
        rows.append(by_flag.rename(columns={"quality_flag": "category"})[["category", "count", "scope", "metric"]])
        by_model = insp.groupby(["model_label", "quality_flag"], as_index=False).agg(count=("quality_flag", "size"))
        by_model["scope"] = "model"
        by_model["metric"] = "inspection"
        rows.append(by_model.rename(columns={"model_label": "category"})[["category", "quality_flag", "count", "scope", "metric"]])
        if "machine_id" in insp.columns:
            by_machine = insp.groupby(["machine_id", "quality_flag"], as_index=False).agg(count=("quality_flag", "size"))
            by_machine["scope"] = "machine"
            by_machine["metric"] = "inspection"
            rows.append(by_machine.rename(columns={"machine_id": "category"})[["category", "quality_flag", "count", "scope", "metric"]])
        if "lot_id" in insp.columns:
            by_lot = insp.groupby(["lot_id", "quality_flag"], as_index=False).agg(count=("quality_flag", "size"))
            by_lot["scope"] = "lot"
            by_lot["metric"] = "inspection"
            rows.append(by_lot.rename(columns={"lot_id": "category"})[["category", "quality_flag", "count", "scope", "metric"]])
    if not stop.empty:
        stop_flag = stop.groupby(["stop_like_reason"], as_index=False).agg(count=("stop_count", "sum"))
        stop_flag["scope"] = "stop"
        stop_flag["metric"] = "downtime"
        rows.append(stop_flag.rename(columns={"stop_like_reason": "category"})[["category", "count", "scope", "metric"]])
    if not tag.empty:
        tag_flag = tag.groupby(["event_class"], as_index=False).agg(count=("event_ts", "size"))
        tag_flag["scope"] = "tag"
        tag_flag["metric"] = "event_density"
        rows.append(tag_flag.rename(columns={"event_class": "category"})[["category", "count", "scope", "metric"]])
    if not shop.empty:
        shop_flag = shop.groupby(["process_name", "result_primary"], as_index=False).agg(count=("event_ts", "size"))
        shop_flag["scope"] = "process"
        shop_flag["metric"] = "result"
        rows.append(shop_flag.rename(columns={"process_name": "category"})[["category", "result_primary", "count", "scope", "metric"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def build_correlation_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    machine = build_equipment_overview(clean)
    if machine.empty:
        return pd.DataFrame()
    numeric_cols = [c for c in ["production_rows", "output_qty", "stop_time", "stop_count", "avg_stop_time", "inspect_count", "fail_count", "fail_rate", "tag_events", "pickup_events", "inspection_events", "wait_events", "stop_events", "pickup_count", "error_count", "pickup_error_rate", "defect_rate", "event_density", "performance_gap"] if c in machine.columns]
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    corr = machine[numeric_cols].corr(numeric_only=True)
    corr.index.name = "metric"
    return corr.reset_index()


def build_analysis_scope_summary(clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    comp = clean.get("vw_component_error_fact", pd.DataFrame())
    period_source = None
    for df in [stop, shop, insp, tag, comp]:
        if not df.empty and "event_ts" in df.columns:
            period_source = df
            break
    period_start = pd.to_datetime(period_source["event_ts"], errors="coerce").min() if period_source is not None else pd.NaT
    period_end = pd.to_datetime(period_source["event_ts"], errors="coerce").max() if period_source is not None else pd.NaT
    lines = sorted(set()
                   .union(*(set(df["line_id"].dropna().astype(str)) for df in [shop, stop, insp, tag, comp] if not df.empty and "line_id" in df.columns)))
    machines = sorted(set()
                      .union(*(set(df["machine_id"].dropna().astype(str)) for df in [shop, stop, insp, tag, comp] if not df.empty and "machine_id" in df.columns)))
    lots = sorted(set()
                  .union(*(set(df["lot_id"].dropna().astype(str)) for df in [shop, stop, insp, tag, comp] if not df.empty and "lot_id" in df.columns)))
    models = sorted(set()
                    .union(*(set(df["model_label"].dropna().astype(str)) for df in [shop, stop, insp, tag, comp] if not df.empty and "model_label" in df.columns)))
    stages = sorted(set()
                    .union(*(set(pd.to_numeric(df["stage_no"], errors="coerce").dropna().astype(int).astype(str)) for df in [shop, stop, insp, tag, comp] if not df.empty and "stage_no" in df.columns)))
    issue_groups = set()
    if not tag.empty and "cause_group" in tag.columns:
        issue_groups.update(tag["cause_group"].dropna().astype(str).tolist())
    if not stop.empty and "stop_like_reason" in stop.columns:
        issue_groups.update(stop["stop_like_reason"].dropna().astype(str).tolist())
    if not insp.empty and "quality_flag" in insp.columns:
        issue_groups.update(insp.loc[insp["quality_flag"].astype(str).eq("FAIL"), "quality_flag"].dropna().astype(str).tolist())
    issue_groups = sorted([x for x in issue_groups if x not in {"Other", "PASS", "UNKNOWN", "미상", ""}])
    focus_hour = "-"
    if not stop.empty and "event_ts" in stop.columns:
        hour = pd.to_datetime(stop["event_ts"], errors="coerce").dt.hour.groupby(pd.to_datetime(stop["event_ts"], errors="coerce").dt.hour).count().sort_values(ascending=False)
        if not hour.empty:
            focus_hour = f"{int(hour.index[0]):02d}시"
    elif not tag.empty and "event_ts" in tag.columns:
        hour = pd.to_datetime(tag["event_ts"], errors="coerce").dt.hour.groupby(pd.to_datetime(tag["event_ts"], errors="coerce").dt.hour).count().sort_values(ascending=False)
        if not hour.empty:
            focus_hour = f"{int(hour.index[0]):02d}시"
    rows = [{
        "항목": "분석 기간",
        "값": f"{period_start.date() if pd.notna(period_start) else '-'} ~ {period_end.date() if pd.notna(period_end) else '-'}",
        "보조": f"대표 시간대 {focus_hour}" if focus_hour != "-" else "대표 시간대 미상",
    }]
    rows.append({
        "항목": "분석 대상 범위",
        "값": f"line {len(lines)} / process-stage {len(stages)} / machine {len(machines)} / lot {len(lots)} / model {len(models)}",
        "보조": f"{', '.join(lines[:3])}{'...' if len(lines) > 3 else ''}" if lines else "line 미상",
    })
    rows.append({
        "항목": "대표 이슈군",
        "값": ", ".join(issue_groups[:4]) if issue_groups else "미상",
        "보조": "stop / defect / cause_group 기반",
    })
    if marts and isinstance(marts, dict):
        loss = marts.get("vw_rca_loss_path_view", pd.DataFrame())
        if not loss.empty:
            top = loss.iloc[0]
            rows.append({
                "항목": "대표 hotspot",
                "값": f"{top.get('machine_id', '-')}, {top.get('line_id', '-')}, {top.get('cause_detail', '-')}",
                "보조": f"{top.get('when', '-')}",
            })
    return pd.DataFrame(rows)


def build_analysis_capability_summary(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    comp = clean.get("vw_component_error_fact", pd.DataFrame())
    rows = []
    def add(cap: str, available: str, note: str):
        rows.append({"분석축": cap, "가능여부": available, "근거": note})
    add("설비 비교", "가능" if any(not df.empty and "machine_id" in df.columns for df in [shop, stop, insp, tag, comp]) else "제한", "machine_id 기준 집계 가능")
    add("공정 병목", "가능" if any(not df.empty and ("stage_no" in df.columns or "process_name" in df.columns) for df in [shop, stop, insp, tag]) else "제한", "stage / process 축 존재")
    add("LOT 추적", "가능" if any(not df.empty and "lot_id" in df.columns for df in [shop, stop, insp, comp]) else "제한", "lot_id 또는 lot_nm 존재")
    add("불량 추적", "가능" if not insp.empty or not stop.empty else "제한", "inspection 또는 stop reason 사용")
    add("시간 추세", "가능" if any(not df.empty and "event_ts" in df.columns for df in [shop, stop, insp, tag, comp]) else "제한", "event_ts 기준 추세 가능")
    add("feeder/nozzle/part", "가능" if not comp.empty or not tag.empty else "제한", "component_pickup_summary 또는 tag proxy 사용")
    return pd.DataFrame(rows)


def build_analysis_focus_summary(clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    comp = clean.get("vw_component_error_fact", pd.DataFrame())
    rows = []
    if marts and isinstance(marts, dict) and not marts.get("vw_rca_loss_path_view", pd.DataFrame()).empty:
        top = marts["vw_rca_loss_path_view"].iloc[0]
        rows.extend([
            {"구분": "대표 hotspot machine", "값": str(top.get("machine_id", "-")), "근거": top.get("path_key", "-")},
            {"구분": "대표 hotspot process/stage", "값": f"{top.get('line_id', '-')}/Stage {top.get('stage_no', '-')}", "근거": top.get("where", "-")},
            {"구분": "대표 이상 시간대", "값": str(top.get("when", "-")), "근거": f"impact {top.get('impact', 0):.0f}"},
            {"구분": "대표 이슈군", "값": str(top.get("what", "-")), "근거": f"repeat {top.get('repeat', '-')}",},
        ])
    else:
        if not stop.empty and "machine_id" in stop.columns:
            m = stop.groupby("machine_id", as_index=False).agg(impact=("duration_sec", "sum")).sort_values("impact", ascending=False)
            if not m.empty:
                top = m.iloc[0]
                rows.append({"구분": "대표 hotspot machine", "값": str(top["machine_id"]), "근거": f"stop {top['impact']:.0f}s"})
        if not insp.empty and "quality_flag" in insp.columns:
            f = insp.groupby(["machine_id"], as_index=False).agg(fail=("quality_flag", lambda s: (s == "FAIL").sum())).sort_values("fail", ascending=False)
            if not f.empty:
                top = f.iloc[0]
                rows.append({"구분": "대표 이슈군", "값": str(top["machine_id"]), "근거": f"fail {top['fail']:.0f}"})
        if not tag.empty and "event_ts" in tag.columns:
            h = pd.to_datetime(tag["event_ts"], errors="coerce").dt.hour.value_counts().sort_values(ascending=False)
            if not h.empty:
                rows.append({"구분": "대표 이상 시간대", "값": f"{int(h.index[0]):02d}시", "근거": f"count {int(h.iloc[0])}"})
        if not comp.empty and "machine_id" in comp.columns:
            c = comp.groupby("machine_id", as_index=False).agg(error=("error_count", "sum")).sort_values("error", ascending=False)
            if not c.empty:
                top = c.iloc[0]
                rows.append({"구분": "대표 hotspot process/stage", "값": str(top["machine_id"]), "근거": f"error {top['error']:.0f}"})
    return pd.DataFrame(rows)


def build_rca_loss_path_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    proxy = build_rca_candidate_view(clean)
    source = stop if not stop.empty else proxy
    if source.empty:
        return pd.DataFrame()
    tmp = source.copy()
    metric_col = "duration_sec" if not stop.empty and "duration_sec" in tmp.columns else "proxy_score"
    tmp[metric_col] = pd.to_numeric(tmp.get(metric_col, 0), errors="coerce").fillna(0)
    if not stop.empty:
        tmp["cause_group"] = "Stop"
        tmp["cause_detail"] = _pick_txt(tmp, ["stop_like_reason"])
    if "event_ts" in tmp.columns:
        tmp["hour"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.hour
        tmp["day"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.date
    else:
        tmp["hour"] = pd.NA
        tmp["day"] = pd.NA
    if "stage_no" in tmp.columns:
        tmp["stage_no"] = pd.to_numeric(tmp["stage_no"], errors="coerce")
    group_cols = [c for c in ["line_id", "stage_no", "machine_id", "cause_group", "cause_detail"] if c in tmp.columns]
    if not group_cols:
        return pd.DataFrame()

    def _mode_or_top(series: pd.Series, metric: pd.Series):
        valid = pd.DataFrame({"value": series, "metric": metric}).dropna(subset=["value"])
        if valid.empty:
            return "-"
        ranked = valid.groupby("value", as_index=False).agg(metric=("metric", "sum")).sort_values("metric", ascending=False)
        return ranked.iloc[0]["value"]

    rows = []
    for keys, grp in tmp.groupby(group_cols, dropna=False):
        key_map = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        impact = float(grp[metric_col].sum())
        if not stop.empty and "stop_count" in grp.columns:
            events = int(pd.to_numeric(grp["stop_count"], errors="coerce").fillna(0).sum())
        else:
            events = int(len(grp))
        lot_count = int(grp["lot_id"].nunique()) if "lot_id" in grp.columns else 0
        day_count = int(grp["day"].nunique()) if "day" in grp.columns else 0
        hour_count = int(grp["hour"].nunique()) if "hour" in grp.columns else 0
        top_hour = _mode_or_top(grp["hour"], grp[metric_col]) if "hour" in grp.columns else "-"
        top_day = _mode_or_top(grp["day"], grp[metric_col]) if "day" in grp.columns else "-"
        repeat_score = _safe_div(events, max(lot_count, 1)) + _safe_div(events, max(hour_count, 1))
        path_key = " | ".join([str(key_map.get(c, "-")) for c in ["line_id", "stage_no", "machine_id", "cause_group", "cause_detail"]])
        impact_formula = "정지시간 합계(sec)" if not stop.empty else "proxy score 합계"
        impact_unit = "sec" if not stop.empty else "score"
        rows.append({
            "path_key": path_key,
            "line_id": key_map.get("line_id", "-"),
            "stage_no": key_map.get("stage_no", "-"),
            "machine_id": key_map.get("machine_id", "-"),
            "cause_group": key_map.get("cause_group", "Proxy"),
            "cause_detail": key_map.get("cause_detail", "-"),
            "cause_family": infer_cause_family(str(key_map.get("cause_group", "Proxy")), str(key_map.get("cause_detail", "-"))),
            "event_group": "STOP" if not stop.empty else "PROXY",
            "metric_type": "stop_time_sec" if not stop.empty else "proxy_score",
            "impact_unit": impact_unit,
            "impact": impact,
            "events": events,
            "lot_count": lot_count,
            "day_count": day_count,
            "hour_count": hour_count,
            "when": f"{top_day} / {int(top_hour):02d}시" if pd.notna(top_hour) and top_hour != "-" else str(top_day),
            "where": f"{key_map.get('line_id', '-')} / Stage {key_map.get('stage_no', '-')} / {key_map.get('machine_id', '-')}",
            "how_much": f"{impact:,.0f} {impact_unit} / {events}건",
            "repeat": f"{events}건 / LOT {lot_count} / 시간대 {hour_count}",
            "what": f"{key_map.get('cause_group', 'Proxy')} / {key_map.get('cause_detail', '-')}",
            "impact_formula": impact_formula,
            "repeat_score": repeat_score,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["impact", "repeat_score", "events"], ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def build_rca_card_summary(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    proxy = build_rca_candidate_view(clean)
    rows = []
    loss_path = build_rca_loss_path_view(clean)
    if not loss_path.empty:
        top = loss_path.iloc[0]
        rows.extend([
            {
                "card_key": "when",
                "headline_value": str(top.get("when", "-")),
                "sub_label": "손실경로 기준 집중 시점",
                "evidence": f"rank {int(top.get('rank', 0))}, path {top.get('path_key', '-')}",
                "metric_value": float(top.get("impact", 0)),
                "detail_key": "when",
            },
            {
                "card_key": "where",
                "headline_value": str(top.get("where", "-")),
                "sub_label": "손실경로 기준 위치",
                "evidence": f"설비 {top.get('machine_id', '-')}",
                "metric_value": float(top.get("impact", 0)),
                "detail_key": "where",
            },
            {
                "card_key": "how_much",
                "headline_value": str(top.get("how_much", "-")),
                "sub_label": "손실경로 영향 규모",
                "evidence": f"events {top.get('events', 0)}, LOT {top.get('lot_count', 0)}",
                "metric_value": float(top.get("impact", 0)),
                "detail_key": "how_much",
            },
            {
                "card_key": "repeat",
                "headline_value": str(top.get("repeat", "-")),
                "sub_label": "반복 경로",
                "evidence": f"반복점수 {top.get('repeat_score', 0):.2f}",
                "metric_value": float(top.get("repeat_score", 0)),
                "detail_key": "repeat",
            },
            {
                "card_key": "what",
                "headline_value": str(top.get("what", "-")),
                "sub_label": "손실경로 원인 후보",
                "evidence": f"cause {top.get('cause_group', 'Proxy')}",
                "metric_value": float(top.get("impact", 0)),
                "detail_key": "what",
            },
        ])
    else:
        if not stop.empty and "event_ts" in stop.columns:
            tmp = stop.copy()
            tmp["hour"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.hour
            hourly = tmp.groupby("hour", as_index=False).agg(loss_time=("duration_sec", "sum"), events=("stop_count", "sum")).sort_values("loss_time", ascending=False)
            if not hourly.empty:
                top = hourly.iloc[0]
                rows.append({
                    "card_key": "when",
                    "headline_value": f"{int(top['hour']):02d}시",
                    "sub_label": "가장 집중된 시간대",
                    "evidence": f"손실시간 {top['loss_time']:.0f}초, 건수 {top['events']:.0f}",
                    "metric_value": float(top["loss_time"]),
                    "detail_key": "hour",
                })
        elif not proxy.empty:
            tmp = proxy.copy()
            tmp["hour"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.hour
            hourly = tmp.groupby("hour", as_index=False).agg(proxy_score=("proxy_score", "sum"), events=("proxy_score", "size")).sort_values("proxy_score", ascending=False)
            if not hourly.empty:
                top = hourly.iloc[0]
                rows.append({
                    "card_key": "when",
                    "headline_value": f"{int(top['hour']):02d}시",
                    "sub_label": "가장 집중된 시간대",
                    "evidence": f"proxy_score {top['proxy_score']:.1f}, 건수 {top['events']:.0f}",
                    "metric_value": float(top["proxy_score"]),
                    "detail_key": "hour",
                })
        machine_view = build_equipment_overview(clean)
        if not machine_view.empty:
            top_machine = machine_view.iloc[0]
            rows.append({
                "card_key": "where",
                "headline_value": str(top_machine.get("machine_id", "-")),
                "sub_label": "가장 영향이 큰 설비",
                "evidence": f"line {top_machine.get('line_id', '-')}, stage {top_machine.get('stage_no', '-')}",
                "metric_value": float(top_machine.get("performance_gap", 0)),
                "detail_key": "machine",
            })
        if not stop.empty:
            total_loss = float(stop["duration_sec"].sum())
            total_events = float(stop["stop_count"].sum()) if "stop_count" in stop.columns else float(len(stop))
            rows.append({
                "card_key": "how_much",
                "headline_value": f"{total_loss:,.0f}초",
                "sub_label": "총 손실시간",
                "evidence": f"정지건수 {total_events:,.0f}, 영향 설비 {stop['machine_id'].nunique() if 'machine_id' in stop.columns else 0}",
                "metric_value": total_loss,
                "detail_key": "loss",
            })
        elif not proxy.empty:
            total_proxy = float(proxy["proxy_score"].sum())
            rows.append({
                "card_key": "how_much",
                "headline_value": f"{total_proxy:,.1f}",
                "sub_label": "총 proxy 강도",
                "evidence": f"candidate event {len(proxy):,}",
                "metric_value": total_proxy,
                "detail_key": "loss",
            })
        repeat = build_rca_repeat_pattern_view(clean)
        if not repeat.empty:
            top_repeat = repeat.iloc[0]
            rows.append({
                "card_key": "repeat",
                "headline_value": str(top_repeat.get("pattern", "-")),
                "sub_label": "가장 반복되는 조합",
                "evidence": f"반복 {top_repeat.get('events', 0):.0f}건, 누적 {top_repeat.get('impact', 0):.0f}",
                "metric_value": float(top_repeat.get("impact", 0)),
                "detail_key": "repeat",
            })
        cause = build_rca_hotspot_view(clean)
        if not cause.empty:
            cause_only = cause[cause["hotspot_type"].eq("cause")] if "hotspot_type" in cause.columns else cause
            top_cause = cause_only.iloc[0] if not cause_only.empty else cause.iloc[0]
            rows.append({
                "card_key": "what",
                "headline_value": f"{top_cause.get('cause_group', 'Proxy')} / {top_cause.get('cause_detail', '-')}",
                "sub_label": "상위 후보군",
                "evidence": f"집중도 {top_cause.get('impact', 0):.0f}, 건수 {top_cause.get('events', 0):.0f}",
                "metric_value": float(top_cause.get("impact", 0)),
                "detail_key": "cause",
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_rca_timeline_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    proxy = build_rca_candidate_view(clean)
    source = stop if not stop.empty else proxy
    if source.empty or "event_ts" not in source.columns:
        return pd.DataFrame()
    tmp = source.copy()
    if not stop.empty:
        tmp["cause_group"] = "Stop"
        tmp["cause_detail"] = _pick_txt(tmp, ["stop_like_reason"])
    if "stage_no" in tmp.columns:
        tmp["stage_no"] = pd.to_numeric(tmp["stage_no"], errors="coerce")
    tmp["hour"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.hour
    tmp["day"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.date
    metric = "duration_sec" if not stop.empty else "proxy_score"
    group_cols = [c for c in ["day", "hour", "line_id", "stage_no", "machine_id", "cause_group", "cause_detail"] if c in tmp.columns]
    out = tmp.groupby(group_cols, as_index=False).agg(metric_value=(metric, "sum"), events=("event_ts", "size"))
    out["path_key"] = out.apply(lambda r: " | ".join([str(r.get(c, "-")) for c in ["line_id", "stage_no", "machine_id", "cause_group", "cause_detail"]]), axis=1)
    out["metric_type"] = "stop_time_sec" if not stop.empty else "proxy_score"
    out["event_group"] = "STOP" if not stop.empty else "PROXY"
    out["impact_unit"] = "sec" if not stop.empty else "score"
    out["cause_family"] = out.apply(lambda r: infer_cause_family(str(r.get("cause_group", "")), str(r.get("cause_detail", ""))), axis=1)
    return out.sort_values(["day", "hour"], ascending=True)


def build_rca_hotspot_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    proxy = build_rca_candidate_view(clean)
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    rows = []
    if not stop.empty:
        reason_hotspot = stop.groupby(["stop_like_reason"], as_index=False).agg(impact=("duration_sec", "sum"), events=("stop_count", "sum"))
        reason_hotspot["hotspot_type"] = "cause"
        reason_hotspot["cause_group"] = "Stop"
        reason_hotspot["cause_detail"] = reason_hotspot["stop_like_reason"]
        reason_hotspot["cause_family"] = reason_hotspot.apply(lambda r: infer_cause_family(str(r.get("cause_group", "")), str(r.get("cause_detail", ""))), axis=1)
        reason_hotspot["event_group"] = "STOP"
        reason_hotspot["metric_type"] = "stop_time_sec"
        reason_hotspot["impact_unit"] = "sec"
        reason_hotspot["path_key"] = reason_hotspot["cause_detail"].apply(lambda x: f"- | - | - | Stop | {x}")
        rows.append(reason_hotspot.drop(columns=["stop_like_reason"]))
        by_machine = stop.groupby(["machine_id", "line_id"], as_index=False).agg(impact=("duration_sec", "sum"), events=("stop_count", "sum"))
        by_machine["hotspot_type"] = "machine"
        by_machine["cause_group"] = "Stop"
        by_machine["cause_detail"] = "stop_time"
        by_machine["cause_family"] = "process"
        by_machine["event_group"] = "STOP"
        by_machine["metric_type"] = "stop_time_sec"
        by_machine["impact_unit"] = "sec"
        by_machine["path_key"] = by_machine.apply(lambda r: f"{r.get('line_id', '-')} | - | {r.get('machine_id', '-')} | Stop | stop_time", axis=1)
        rows.append(by_machine)
        if "stage_no" in stop.columns:
            by_stage = stop.groupby(["line_id", "stage_no"], as_index=False).agg(impact=("duration_sec", "sum"), events=("stop_count", "sum"))
            by_stage["hotspot_type"] = "stage"
            by_stage["cause_group"] = "Stop"
            by_stage["cause_detail"] = "stage_time"
            by_stage["cause_family"] = "process"
            by_stage["event_group"] = "STOP"
            by_stage["metric_type"] = "stop_time_sec"
            by_stage["impact_unit"] = "sec"
            by_stage["path_key"] = by_stage.apply(lambda r: f"{r.get('line_id', '-')} | {r.get('stage_no', '-')} | - | Stop | stage_time", axis=1)
            rows.append(by_stage)
    elif not proxy.empty:
        by_machine = proxy.groupby(["machine_id", "line_id", "cause_group", "cause_detail"], as_index=False).agg(impact=("proxy_score", "sum"), events=("proxy_score", "size"))
        by_machine["hotspot_type"] = "machine"
        by_machine["cause_family"] = by_machine.apply(lambda r: infer_cause_family(str(r.get("cause_group", "")), str(r.get("cause_detail", ""))), axis=1)
        by_machine["event_group"] = np.where(by_machine["cause_group"].astype(str).str.contains("Quality", case=False, na=False), "QUALITY", np.where(by_machine["cause_group"].astype(str).str.contains("Stop", case=False, na=False), "STOP", "PROXY"))
        by_machine["metric_type"] = "proxy_score"
        by_machine["impact_unit"] = "score"
        by_machine["path_key"] = by_machine.apply(lambda r: f"{r.get('line_id', '-')} | - | {r.get('machine_id', '-')} | {r.get('cause_group', 'Proxy')} | {r.get('cause_detail', '-')}", axis=1)
        rows.append(by_machine)
    if not insp.empty:
        by_defect = insp.groupby(["machine_id", "line_id", "model_label", "quality_flag"], as_index=False).agg(impact=("quality_flag", "size"), events=("quality_flag", "size"))
        by_defect["hotspot_type"] = "defect"
        by_defect["cause_group"] = "Quality"
        by_defect["cause_detail"] = by_defect["quality_flag"]
        by_defect["cause_family"] = "quality"
        by_defect["event_group"] = "QUALITY"
        by_defect["metric_type"] = "defect_count"
        by_defect["impact_unit"] = "count"
        by_defect["path_key"] = by_defect.apply(lambda r: f"{r.get('line_id', '-')} | - | {r.get('machine_id', '-')} | Quality | {r.get('quality_flag', '-')}", axis=1)
        rows.append(by_defect.drop(columns=["quality_flag"]))
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "impact" in out.columns:
        out = out.sort_values(["impact", "events"], ascending=False)
    return out


def build_rca_repeat_pattern_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    proxy = build_rca_candidate_view(clean)
    source = stop if not stop.empty else proxy
    if source.empty:
        return pd.DataFrame()
    tmp = source.copy()
    if not stop.empty:
        tmp["cause_group"] = "Stop"
        tmp["cause_detail"] = _pick_txt(tmp, ["stop_like_reason"])
    if "event_ts" in tmp.columns:
        tmp["hour"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.hour
        tmp["day"] = pd.to_datetime(tmp["event_ts"], errors="coerce").dt.date
    metric = "duration_sec" if not stop.empty else "proxy_score"
    group_cols = [c for c in ["line_id", "stage_no", "machine_id", "cause_group", "cause_detail", "hour", "lot_id"] if c in tmp.columns]
    if not group_cols:
        return pd.DataFrame()
    if not stop.empty and "stop_count" in tmp.columns:
        out = tmp.groupby(group_cols, as_index=False).agg(impact=(metric, "sum"), events=("stop_count", "sum"))
    else:
        out = tmp.groupby(group_cols, as_index=False).agg(impact=(metric, "sum"), events=("event_ts" if "event_ts" in tmp.columns else metric, "size"))
    out["path_key"] = out.apply(lambda r: " | ".join([str(r.get(c, "-")) for c in ["line_id", "stage_no", "machine_id", "cause_group", "cause_detail"]]), axis=1)
    out["metric_type"] = "stop_time_sec" if not stop.empty else "proxy_score"
    out["event_group"] = "STOP" if not stop.empty else "PROXY"
    out["impact_unit"] = "sec" if not stop.empty else "score"
    out["cause_family"] = out.apply(lambda r: infer_cause_family(str(r.get("cause_group", "")), str(r.get("cause_detail", ""))), axis=1)
    def _pattern(row):
        items = []
        for c in ["machine_id", "cause_group", "cause_detail", "hour", "lot_id"]:
            if c in out.columns and pd.notna(row.get(c)):
                items.append(str(row.get(c)))
        return " / ".join(items) if items else "-"
    out["pattern"] = out.apply(_pattern, axis=1)
    out = out.sort_values(["impact", "events"], ascending=False)
    return out


def build_rca_drilldown_view(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    proxy = build_rca_candidate_view(clean)
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    frames = []
    if not stop.empty:
        tmp = stop.copy()
        tmp["source_type"] = "stop"
        tmp["metric_value"] = tmp.get("duration_sec", 0)
        tmp["defect_flag"] = "STOP"
        frames.append(tmp)
    if not proxy.empty:
        tmp = proxy.copy()
        tmp["source_type"] = "proxy"
        tmp["metric_value"] = tmp.get("proxy_score", 0)
        tmp["defect_flag"] = tmp.get("quality_flag", "UNKNOWN")
        frames.append(tmp)
    if not insp.empty:
        tmp = insp.copy()
        tmp["source_type"] = "inspection"
        tmp["metric_value"] = 1.0
        tmp["defect_flag"] = tmp.get("quality_flag", "UNKNOWN")
        frames.append(tmp)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "event_ts" in out.columns:
        out["hour"] = pd.to_datetime(out["event_ts"], errors="coerce").dt.hour
        out["day"] = pd.to_datetime(out["event_ts"], errors="coerce").dt.date
    if "cause_group" not in out.columns:
        out["cause_group"] = np.where(out["source_type"].eq("stop"), "Stop", np.where(out["source_type"].eq("inspection"), "Quality", "Proxy"))
    if "cause_detail" not in out.columns:
        out["cause_detail"] = np.where(out["source_type"].eq("stop"), out.get("stop_like_reason", "UNKNOWN"), np.where(out["source_type"].eq("inspection"), out.get("quality_flag", "UNKNOWN"), out.get("cause_detail", "UNKNOWN")))
    if "path_key" not in out.columns:
        out["path_key"] = out.apply(lambda r: " | ".join([str(r.get(c, "-")) for c in ["line_id", "stage_no", "machine_id", "cause_group", "cause_detail"]]), axis=1)
    out["metric_type"] = np.where(out["source_type"].eq("stop"), "stop_time_sec", np.where(out["source_type"].eq("inspection"), "defect_count", "proxy_score"))
    out["event_group"] = np.where(out["source_type"].eq("stop"), "STOP", np.where(out["source_type"].eq("inspection"), "QUALITY", "PROXY"))
    out["impact_unit"] = np.where(out["source_type"].eq("stop"), "sec", np.where(out["source_type"].eq("inspection"), "count", "score"))
    out["cause_family"] = out.apply(lambda r: infer_cause_family(str(r.get("cause_group", "")), str(r.get("cause_detail", ""))), axis=1)
    return out


def build_clean_views(raw: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    meta = raw.get("_meta", {}) if isinstance(raw.get("_meta", {}), dict) else {}
    raw = {k: _lower(v) for k, v in raw.items() if isinstance(v, pd.DataFrame)}
    item_fact = build_mounter_item_fact(raw)
    if not item_fact.empty:
        item_fact = item_fact.sort_values(["event_ts", "machine_id", "lot_id", "stage_no"], ascending=True)
    out = {
        "vw_mounter_item_fact": item_fact if not item_fact.empty else pd.DataFrame(),
        "vw_shopfloor_event_fact": item_fact.copy() if not item_fact.empty else pd.DataFrame(),
        "vw_mounter_event_fact": item_fact.copy() if not item_fact.empty else pd.DataFrame(),
        "vw_machine_dim": raw.get("machine", pd.DataFrame()).copy(),
        "vw_lot_dim": raw.get("lot", pd.DataFrame()).copy(),
        "_meta": meta,
    }
    return out


def build_feature_marts(clean: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    item = clean.get("vw_mounter_item_fact", pd.DataFrame()).copy()
    if item.empty:
        empty_keys = [
            "vw_mounter_summary",
            "vw_equipment_overview",
            "vw_process_overview",
            "vw_lot_analysis",
            "vw_time_pattern_view",
            "vw_priority_view",
        ]
        return {k: pd.DataFrame() for k in empty_keys}

    item["event_ts"] = pd.to_datetime(item.get("event_ts"), errors="coerce")
    item["day"] = pd.to_datetime(item["event_ts"], errors="coerce").dt.date
    item["hour"] = pd.to_datetime(item["event_ts"], errors="coerce").dt.hour
    item["shift"] = item["hour"].map(lambda h: "주간(06-14)" if pd.notna(h) and 6 <= int(h) < 14 else "석간(14-22)" if pd.notna(h) and 14 <= int(h) < 22 else "야간(22-06)" if pd.notna(h) else "미상")
    if "output_qty" in item.columns:
        item["output_qty"] = pd.to_numeric(item["output_qty"], errors="coerce").fillna(1)
    else:
        item["output_qty"] = pd.Series([1] * len(item), index=item.index)
    item["stage_no"] = pd.to_numeric(item.get("stage_no"), errors="coerce")

    def _cycle_frame(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        if df.empty or not all(c in df.columns for c in group_cols + ["event_ts"]):
            return pd.DataFrame(columns=group_cols + ["avg_cycle_sec", "cycle_std_sec"])
        tmp = df[group_cols + ["event_ts"]].copy().dropna(subset=["event_ts"])
        if tmp.empty:
            return pd.DataFrame(columns=group_cols + ["avg_cycle_sec", "cycle_std_sec"])
        tmp = tmp.sort_values(group_cols + ["event_ts"])
        tmp["cycle_sec"] = tmp.groupby(group_cols)["event_ts"].diff().dt.total_seconds()
        out = tmp.groupby(group_cols, as_index=False).agg(avg_cycle_sec=("cycle_sec", "mean"), cycle_std_sec=("cycle_sec", "std"))
        out["avg_cycle_sec"] = pd.to_numeric(out["avg_cycle_sec"], errors="coerce").fillna(0)
        out["cycle_std_sec"] = pd.to_numeric(out["cycle_std_sec"], errors="coerce").fillna(0)
        return out

    total_output = float(item["output_qty"].sum()) or 1.0
    total_rows = len(item)
    active_days = item["day"].nunique(dropna=True)
    period_start = pd.to_datetime(item["event_ts"], errors="coerce").min()
    period_end = pd.to_datetime(item["event_ts"], errors="coerce").max()
    total_machines = item["machine_id"].nunique(dropna=True) if "machine_id" in item.columns else 0
    total_stages = item[["line_id", "stage_no"]].drop_duplicates().shape[0] if {"line_id", "stage_no"}.issubset(item.columns) else 0
    total_lots = item["lot_id"].nunique(dropna=True) if "lot_id" in item.columns else 0

    machine = pd.DataFrame()
    if "machine_id" in item.columns:
        machine = item.groupby(["machine_id", "line_id", "stage_no"], as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            lot_count=("lot_id", "nunique"),
            model_count=("model_label", "nunique"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        cycle = _cycle_frame(item, ["machine_id"])
        machine = machine.merge(cycle, on="machine_id", how="left")
        machine["observed_span_sec"] = (pd.to_datetime(machine["last_event_ts"], errors="coerce") - pd.to_datetime(machine["first_event_ts"], errors="coerce")).dt.total_seconds().fillna(0)
        machine["output_per_hour"] = machine.apply(lambda r: _safe_div(r.get("output_qty", 0), max(r.get("observed_span_sec", 0), 1) / 3600), axis=1)
        machine["output_share"] = machine["output_qty"] / total_output
        out_norm = pd.to_numeric(machine["output_qty"], errors="coerce")
        cyc_norm = pd.to_numeric(machine.get("cycle_std_sec", 0), errors="coerce")
        span_norm = pd.to_numeric(machine.get("observed_span_sec", 0), errors="coerce")
        out_rank = out_norm.rank(pct=True, ascending=True).fillna(0)
        cyc_rank = cyc_norm.rank(pct=True, ascending=True).fillna(0)
        span_rank = span_norm.rank(pct=True, ascending=True).fillna(0)
        machine["bottleneck_score"] = (1 - out_rank) * 0.45 + cyc_rank * 0.35 + span_rank * 0.20
        q1_out = out_norm.quantile(0.25) if not machine.empty else 0
        q3_cycle = cyc_norm.quantile(0.75) if not machine.empty else 0
        q3_share = machine["output_share"].quantile(0.75) if not machine.empty else 0
        def _machine_type(row):
            if row.get("output_qty", 0) <= q1_out and row.get("cycle_std_sec", 0) >= q3_cycle:
                return "생산성 손실형"
            if row.get("output_share", 0) >= q3_share and row.get("output_qty", 0) > 0:
                return "집중형"
            if row.get("cycle_std_sec", 0) >= q3_cycle:
                return "안정성 문제"
            return "주의"
        machine["problem_type"] = machine.apply(_machine_type, axis=1)
        machine["reasoning"] = machine.apply(lambda r: f"출력 {r.get('output_qty', 0):.0f}, cycle 편차 {r.get('cycle_std_sec', 0):.1f}s, 점유율 {r.get('output_share', 0) * 100:.1f}%", axis=1)
        machine["recommended_action"] = machine.apply(lambda r: "라인 밸런스 및 자재 보급 타이밍 점검" if r.get("problem_type") == "생산성 손실형" else "부하 집중도와 전환 구간 점검" if r.get("problem_type") == "집중형" else "작업 표준화 및 cycle 변동 원인 확인" if r.get("problem_type") == "안정성 문제" else "상위 설비와 비교해 변동성 확인", axis=1)
        machine["status_label"] = machine["bottleneck_score"].apply(lambda v: "개선필요" if v >= machine["bottleneck_score"].quantile(0.75) else "주의" if v >= machine["bottleneck_score"].quantile(0.40) else "정상")
        machine["confidence"] = np.where(machine["production_rows"].ge(10), "Actual", "Estimated")
        machine["rank"] = np.arange(1, len(machine) + 1)

    process = pd.DataFrame()
    if {"line_id", "stage_no"}.issubset(item.columns):
        process = item.groupby(["line_id", "stage_no"], as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            machine_count=("machine_id", "nunique"),
            lot_count=("lot_id", "nunique"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        cycle = _cycle_frame(item, ["line_id", "stage_no"])
        process = process.merge(cycle, on=["line_id", "stage_no"], how="left")
        process["observed_span_sec"] = (pd.to_datetime(process["last_event_ts"], errors="coerce") - pd.to_datetime(process["first_event_ts"], errors="coerce")).dt.total_seconds().fillna(0)
        process["output_per_hour"] = process.apply(lambda r: _safe_div(r.get("output_qty", 0), max(r.get("observed_span_sec", 0), 1) / 3600), axis=1)
        process["output_share"] = process["output_qty"] / total_output
        out_rank = pd.to_numeric(process["output_qty"], errors="coerce").rank(pct=True, ascending=True).fillna(0)
        cyc_rank = pd.to_numeric(process.get("cycle_std_sec", 0), errors="coerce").rank(pct=True, ascending=True).fillna(0)
        span_rank = pd.to_numeric(process.get("observed_span_sec", 0), errors="coerce").rank(pct=True, ascending=True).fillna(0)
        process["bottleneck_score"] = (1 - out_rank) * 0.5 + cyc_rank * 0.3 + span_rank * 0.2
        q1_out = pd.to_numeric(process["output_qty"], errors="coerce").quantile(0.25) if not process.empty else 0
        q3_cycle = pd.to_numeric(process.get("cycle_std_sec", 0), errors="coerce").quantile(0.75) if not process.empty else 0
        def _process_type(row):
            if row.get("output_qty", 0) <= q1_out and row.get("cycle_std_sec", 0) >= q3_cycle:
                return "정지 병목"
            if row.get("cycle_std_sec", 0) >= q3_cycle:
                return "흐름 병목"
            if row.get("output_qty", 0) <= q1_out:
                return "주의"
            return "정상"
        process["process_display"] = process.apply(lambda r: f"Line-{r.get('line_id', '-')} / Stage-{int(r.get('stage_no', 0)) if pd.notna(r.get('stage_no', np.nan)) else '-'}", axis=1)
        process["problem_type"] = process.apply(_process_type, axis=1)
        process["reasoning"] = process.apply(lambda r: f"출력 {r.get('output_qty', 0):.0f}, cycle 편차 {r.get('cycle_std_sec', 0):.1f}s, machine {r.get('machine_count', 0):.0f}개", axis=1)
        process["recommended_action"] = process.apply(lambda r: "라인 밸런스와 보급 타이밍 조정" if r.get("problem_type") == "정지 병목" else "전후 stage 연결과 전환 시간 확인" if r.get("problem_type") == "흐름 병목" else "생산량 변동 원인과 기준선 재설정", axis=1)
        process["confidence"] = np.where(process["production_rows"].ge(10), "Actual", "Estimated")
        process["rank"] = np.arange(1, len(process) + 1)

    lot = pd.DataFrame()
    if "lot_id" in item.columns:
        lot = item.groupby(["lot_id", "model_label"], as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            machine_count=("machine_id", "nunique"),
            stage_count=("stage_no", "nunique"),
            line_count=("line_id", "nunique"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        cycle = _cycle_frame(item, ["lot_id"])
        lot = lot.merge(cycle, on=["lot_id"], how="left")
        lot["observed_span_sec"] = (pd.to_datetime(lot["last_event_ts"], errors="coerce") - pd.to_datetime(lot["first_event_ts"], errors="coerce")).dt.total_seconds().fillna(0)
        lot["output_per_hour"] = lot.apply(lambda r: _safe_div(r.get("output_qty", 0), max(r.get("observed_span_sec", 0), 1) / 3600), axis=1)
        lot["spread_score"] = lot[["machine_count", "stage_count", "line_count"]].sum(axis=1)
        lot["concentration_score"] = 1 / (1 + lot["spread_score"])
        lot["priority_score"] = lot["spread_score"] * 0.6 + lot["concentration_score"] * 0.4
        machine_q = lot["machine_count"].quantile(0.25) if not lot.empty else 0
        stage_q = lot["stage_count"].quantile(0.75) if not lot.empty else 0
        def _lot_type(row):
            if row.get("machine_count", 0) <= machine_q and row.get("stage_count", 0) <= 1:
                return "국소 LOT"
            if row.get("stage_count", 0) >= stage_q or row.get("machine_count", 0) >= stage_q:
                return "전파 LOT"
            if row.get("output_per_hour", 0) < lot["output_per_hour"].quantile(0.25):
                return "저생산 LOT"
            return "주의"
        lot["problem_type"] = lot.apply(_lot_type, axis=1)
        lot["reasoning"] = lot.apply(lambda r: f"설비 {r.get('machine_count', 0):.0f}개 / 공정 {r.get('stage_count', 0):.0f}개 / 집중도 {r.get('concentration_score', 0) * 100:.1f}%", axis=1)
        lot["recommended_action"] = lot.apply(lambda r: "대표 설비와 stage를 먼저 확인" if r.get("problem_type") == "국소 LOT" else "전파 범위와 연속 lot 비교" if r.get("problem_type") == "전파 LOT" else "생산 밀도와 스케줄 확인", axis=1)
        lot["confidence"] = np.where(lot["production_rows"].ge(10), "Actual", "Estimated")
        lot["rank"] = np.arange(1, len(lot) + 1)

    time_view = pd.DataFrame()
    if not item.empty:
        time_view = item.groupby(["hour", "shift"], as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            machine_count=("machine_id", "nunique"),
            lot_count=("lot_id", "nunique"),
        )
        time_view["bucket"] = time_view["hour"].astype("Int64").astype(str).str.zfill(2)
        time_view["bucket_order"] = time_view["hour"].fillna(99)
        time_view["grain"] = "hour"
        shift_order = {"주간(06-14)": 0, "석간(14-22)": 1, "야간(22-06)": 2, "미상": 3}
        shift_view = time_view.groupby("shift", as_index=False).agg(
            production_rows=("production_rows", "sum"),
            output_qty=("output_qty", "sum"),
            machine_count=("machine_count", "sum"),
            lot_count=("lot_count", "sum"),
        )
        if not shift_view.empty:
            shift_view["grain"] = "shift"
            shift_view["bucket"] = shift_view["shift"]
            shift_view["bucket_order"] = shift_view["shift"].map(shift_order).fillna(99)
            time_view = pd.concat([time_view, shift_view], ignore_index=True, sort=False)
        time_view = time_view.sort_values(["grain", "bucket_order"])

    summary = pd.DataFrame([
        {"항목": "총 생산기록", "값": f"{total_rows:,}", "근거": "mounter_dtl 행 수"},
        {"항목": "총 출력", "값": f"{int(total_output):,}", "근거": "output 합계"},
        {"항목": "설비 수", "값": f"{total_machines:,}", "근거": "machine_id distinct"},
        {"항목": "공정 수", "값": f"{total_stages:,}", "근거": "line/stage distinct"},
        {"항목": "LOT 수", "값": f"{total_lots:,}", "근거": "lot_id distinct"},
        {"항목": "활성 일수", "값": f"{active_days:,}", "근거": "event_ts distinct day"},
        {"항목": "활성 기간", "값": f"{period_start.date() if pd.notna(period_start) else '-'} ~ {period_end.date() if pd.notna(period_end) else '-'}", "근거": "event_ts min ~ max"},
    ])

    priority_rows = []
    if not machine.empty:
        for _, r in machine.sort_values(["bottleneck_score", "output_qty"], ascending=[False, True]).head(5).iterrows():
            priority_rows.append({
                "대상유형": "설비",
                "대상": str(r.get("machine_id", "-")),
                "문제유형": r.get("problem_type", "주의"),
                "근거 KPI": f"output {r.get('output_qty', 0):.0f} / cycle {r.get('cycle_std_sec', 0):.1f}s",
                "예상 영향": f"라인 {r.get('line_id', '-')}, stage {r.get('stage_no', '-')}",
                "추천 액션": r.get("recommended_action", "-"),
                "기대 효과": "기준 변동 축소",
                "priority_score": float(r.get("bottleneck_score", 0)),
            })
    if not process.empty:
        for _, r in process.sort_values(["bottleneck_score", "output_qty"], ascending=[False, True]).head(5).iterrows():
            priority_rows.append({
                "대상유형": "공정",
                "대상": str(r.get("process_display", "-")),
                "문제유형": r.get("problem_type", "주의"),
                "근거 KPI": f"output {r.get('output_qty', 0):.0f} / cycle {r.get('cycle_std_sec', 0):.1f}s",
                "예상 영향": f"machine {r.get('machine_count', 0):.0f}개",
                "추천 액션": r.get("recommended_action", "-"),
                "기대 효과": "흐름 안정화",
                "priority_score": float(r.get("bottleneck_score", 0)),
            })
    if not lot.empty:
        for _, r in lot.sort_values(["priority_score", "production_rows"], ascending=[False, False]).head(5).iterrows():
            priority_rows.append({
                "대상유형": "LOT",
                "대상": str(r.get("lot_id", "-")),
                "문제유형": r.get("problem_type", "주의"),
                "근거 KPI": f"spread {r.get('spread_score', 0):.0f} / density {r.get('output_per_hour', 0):.1f}",
                "예상 영향": f"machine {r.get('machine_count', 0):.0f}개 / stage {r.get('stage_count', 0):.0f}개",
                "추천 액션": r.get("recommended_action", "-"),
                "기대 효과": "확산 범위 축소",
                "priority_score": float(r.get("priority_score", 0)),
            })
    priority = pd.DataFrame(priority_rows)
    if not priority.empty:
        priority = priority.sort_values("priority_score", ascending=False).reset_index(drop=True)
        priority["순위"] = np.arange(1, len(priority) + 1)
        priority = priority[["순위", "대상유형", "대상", "문제유형", "근거 KPI", "예상 영향", "추천 액션", "기대 효과", "priority_score"]]

    return {
        "vw_mounter_summary": summary,
        "vw_equipment_overview": machine,
        "vw_process_overview": process,
        "vw_lot_analysis": lot,
        "vw_time_pattern_view": time_view,
        "vw_priority_view": priority,
    }


def _filter_by_period(dfs: Dict[str, pd.DataFrame], period: str) -> Dict[str, pd.DataFrame]:
    if period == "전체":
        return {k: v.copy() if isinstance(v, pd.DataFrame) else v for k, v in dfs.items()}
    days = 7 if period == "최근 7일" else 30
    end = pd.Timestamp.now().normalize()
    start = end - pd.Timedelta(days=days - 1)
    out = {}
    for name, df in dfs.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            out[name] = df.copy() if isinstance(df, pd.DataFrame) else df
            continue
        tmp = df.copy()
        ts_col = "event_ts" if "event_ts" in tmp.columns else "day" if "day" in tmp.columns else "recorded_at" if "recorded_at" in tmp.columns else None
        if ts_col:
            ts = pd.to_datetime(tmp[ts_col], errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)
            normalized = ts.dt.normalize()
            if getattr(normalized.dt, "tz", None) is not None:
                normalized = normalized.dt.tz_convert(None)
            tmp = tmp[normalized.between(start, end)]
        out[name] = tmp
    return out
