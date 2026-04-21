import pandas as pd
import streamlit as st

from data_layer import (
    generate_pickup_rca_sample_data,
    generate_sample_data,
    load_mount_demo_snapshot,
    load_raw_data,
)
from transform import build_clean_views, build_component_fact, build_feature_marts, build_inspection_fact, build_stop_event_fact, build_tag_event_fact
from ui_tabs import _css, render_equipment_screen, render_rca, render_rca_workflow, render_summary
from rca_prototype import render_ai_demo_tab


def _build_rca_demo_clean(raw):
    clean = build_clean_views(raw)
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    if shop.empty and "fa_26_34_mounter_dtl" in raw:
        shop = clean.get("vw_mounter_item_fact", pd.DataFrame()).copy()
    clean["vw_shopfloor_event_fact"] = shop
    clean["vw_stop_event_fact"] = build_stop_event_fact(raw)
    clean["vw_inspection_event_fact"] = build_inspection_fact(raw)
    clean["vw_tag_event_fact"] = build_tag_event_fact(raw)
    clean["vw_component_error_fact"] = build_component_fact(raw)
    return clean


def _filter_frames_to_date(frames, target_date: str):
    target = pd.Timestamp(target_date).date()
    out = {}
    for key, value in frames.items():
        if not isinstance(value, pd.DataFrame):
            out[key] = value
            continue
        if value.empty:
            out[key] = value.copy()
            continue
        tmp = value.copy()
        candidate_cols = [c for c in ["event_ts", "day", "recorded_at", "file_dt", "make_dt", "_devicedate", "_insertdate"] if c in tmp.columns]
        if not candidate_cols:
            out[key] = tmp
            continue
        mask = pd.Series(False, index=tmp.index)
        for col in candidate_cols:
            ts = pd.to_datetime(tmp[col], errors="coerce")
            mask = mask | ts.dt.date.eq(target)
        out[key] = tmp[mask].copy()
    return out


def main():
    st.set_page_config(layout="wide", page_title="분석 워크벤치", page_icon="📈")
    _css()
    st.markdown('<div class="hero"><h1>분석 워크벤치</h1></div>', unsafe_allow_html=True)
    sample = False
    try:
        raw = load_raw_data("전체")
        if not raw or all(isinstance(v, pd.DataFrame) and v.empty for k, v in raw.items() if k != "_meta"):
            raw = generate_sample_data()
            sample = True
    except Exception:
        raw = generate_sample_data()
        sample = True
    clean = build_clean_views(raw)
    marts = build_feature_marts(clean)
    mount_tab_date = "2026-03-16"
    raw_mount = load_mount_demo_snapshot()
    if not raw_mount:
        raw_mount = _filter_frames_to_date(raw, mount_tab_date)
        raw_mount["_meta"] = {
            **(raw_mount.get("_meta", {}) if isinstance(raw_mount.get("_meta", {}), dict) else {}),
            "source": "filtered_live",
            "mount_tab_date": mount_tab_date,
        }
    clean_mount = build_clean_views(raw_mount)
    marts_mount = build_feature_marts(clean_mount)
    rca_raw = raw
    rca_source = clean
    rca_marts = marts
    has_rca_data = any(
        not clean.get(key, pd.DataFrame()).empty
        for key in ["vw_stop_event_fact", "vw_inspection_event_fact", "vw_tag_event_fact", "vw_component_error_fact"]
    )
    rca_sample_mode = not has_rca_data
    if rca_sample_mode:
        rca_raw = generate_pickup_rca_sample_data()
        rca_source = _build_rca_demo_clean(rca_raw)
        rca_marts = build_feature_marts(rca_source)
    tabs = st.tabs(["문제진단(MOUNT)", "경로분석(MOUNT)", "원인분석(샘플 시나리오)", "생성형AI 데모"])
    with tabs[0]:
        mount_source = raw_mount.get("_meta", {}).get("source", "unknown") if isinstance(raw_mount.get("_meta", {}), dict) else "unknown"
        st.caption(f"이 탭은 {mount_tab_date} 기준 고정 데이터를 사용합니다. 현재 소스는 `{mount_source}`이며, 2026-03-14 데이터는 정합성 이슈로 제외했습니다.")
        render_summary(raw_mount, clean_mount, marts_mount, sample)
        render_equipment_screen(clean_mount, marts_mount, mode="full")
    with tabs[1]:
        render_rca(rca_source, rca_marts, rca_sample_mode)
    with tabs[2]:
        render_rca_workflow(rca_source, rca_marts, rca_sample_mode)
    with tabs[3]:
        render_ai_demo_tab()

if __name__ == "__main__":
    main()
