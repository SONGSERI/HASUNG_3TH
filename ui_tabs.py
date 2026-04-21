from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_layer import generate_sample_data
from transform import (
    ACTION_TEMPLATES,
    build_full_period_data_inventory,
    build_mounter_item_fact,
    build_component_fact,
    build_data_category_summary,
    build_data_linkage_summary,
    build_data_scope_summary,
    build_data_structure_summary,
    build_analysis_capability_summary,
    build_analysis_focus_summary,
    build_analysis_scope_summary,
    build_process_coverage,
    build_process_flow_summary,
    build_rca_capability_summary,
    build_table_linkage_matrix,
    build_equipment_overview,
    build_lot_analysis_view,
    build_process_overview,
    build_quality_overview,
    build_rca_card_summary,
    build_rca_candidate_view,
    build_rca_drilldown_view,
    build_rca_hotspot_view,
    build_rca_loss_path_view,
    build_rca_repeat_pattern_view,
    build_time_pattern_view,
    build_rca_timeline_view,
)
from utils import _fmt_sec, _safe_div
from rca_prototype import render_prototype_tab


DARK_TEMPLATE = "plotly_white"
PRIMARY = "#2563eb"
SECONDARY = "#f97316"


def _css() -> None:
    st.markdown(
        """
        <style>
        .block-container{max-width:1620px;padding-top:0.9rem;padding-bottom:2rem}
        .stApp{background:
            radial-gradient(circle at top right, rgba(37,99,235,.06), transparent 24%),
            linear-gradient(180deg,#e9edf3 0%,#eef2f6 45%,#e5e7eb 100%)}
        .hero{position:relative;padding:1.2rem 1.3rem;border-radius:24px;background:
            radial-gradient(circle at top right, rgba(96,165,250,.18), transparent 26%),
            radial-gradient(circle at left bottom, rgba(245,158,11,.12), transparent 22%),
            linear-gradient(135deg,#1f2937,#111827);border:1px solid rgba(148,163,184,.22);margin-bottom:1rem;overflow:hidden;box-shadow:0 18px 42px rgba(15,23,42,.16)}
        .hero:after{content:"";position:absolute;inset:0;border-radius:24px;background:linear-gradient(90deg,rgba(255,255,255,.08),transparent 24%,transparent 76%,rgba(255,255,255,.04));pointer-events:none}
        .hero h1{margin:0;color:#f8fafc;font-size:2.05rem;letter-spacing:-.02em}
        .hero p{margin:.42rem 0 0;color:#cbd5e1}
        .box{padding:1rem 1.05rem;border-radius:22px;background:linear-gradient(180deg,#faf8f2,#f4f1e8);border:1px solid rgba(148,163,184,.26);margin:.8rem 0 1rem;box-shadow:0 12px 30px rgba(15,23,42,.08)}
        .card{padding:.9rem 1rem;border-radius:18px;background:linear-gradient(180deg,#fbfaf6,#f3efe5);border:1px solid rgba(148,163,184,.24);box-shadow:0 10px 24px rgba(15,23,42,.07);transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease}
        .card:hover{transform:translateY(-1px);border-color:rgba(37,99,235,.24);box-shadow:0 14px 30px rgba(15,23,42,.10)}
        .card .k{color:#64748b;font-size:.82rem;font-weight:600}
        .card .v{color:#0f172a;font-size:1.8rem;font-weight:800;margin-top:.15rem}
        .card .f{color:#475569;font-size:.82rem}
        .pill{display:inline-block;padding:.24rem .55rem;border-radius:999px;background:#e8eef8;border:1px solid rgba(37,99,235,.16);margin:.15rem .2rem 0 0;color:#1e3a8a;font-size:.82rem}
        div[data-testid="stExpander"]{border:1px solid rgba(148,163,184,.24);border-radius:18px;overflow:hidden;background:#f8f5ee;margin:.35rem 0 .6rem 0;box-shadow:0 8px 22px rgba(15,23,42,.05)}
        div[data-testid="stExpander"] details{background:transparent}
        div[data-testid="stExpander"] summary{padding:.2rem .35rem;font-weight:700;color:#0f172a}
        div[data-testid="stExpander"] summary:hover{background:rgba(15,23,42,.03)}
        div[data-testid="stExpander"] div[data-testid="stVerticalBlock"]{padding-top:.35rem}
        .stMarkdown, .stCaption, .stInfo, .stAlert{color:#0f172a}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card(label: str, value: str, foot: str = "", accent: str = PRIMARY) -> str:
    return f'<div class="card" style="border-top:3px solid {accent}"><div class="k">{label}</div><div class="v">{value}</div><div class="f">{foot}</div></div>'


def _basis_card(label: str, value: str, foot: str = "", accent: str = PRIMARY) -> str:
    return _card(label, value, foot, accent)


def _section_header(title: str, question: str, accent: str) -> str:
    return f'''
    <div style="margin:1rem 0 .75rem 0;padding:.82rem .95rem;border-radius:16px;border:1px solid rgba(148,163,184,.24);background:linear-gradient(90deg, rgba(249,247,241,.98), rgba(242,238,230,.96));border-left:4px solid {accent};box-shadow:0 8px 20px rgba(15,23,42,.05);">
        <div style="font-size:1rem;font-weight:700;color:#0f172a">{title}</div>
        <div style="margin-top:.15rem;font-size:.82rem;color:#475569">{question}</div>
    </div>
    '''


def _story_box(title: str, lines: List[str], tone: str = "neutral") -> None:
    palette = {
        "neutral": {
            "bg": "linear-gradient(135deg,#f8f4ea,#efe8dc)",
            "border": "rgba(148,163,184,.24)",
            "title": "#334155",
            "body": "#0f172a",
        },
        "dark": {
            "bg": "linear-gradient(135deg,#253247,#17202d)",
            "border": "rgba(148,163,184,.24)",
            "title": "#bfdbfe",
            "body": "#f8fafc",
        },
        "accent": {
            "bg": "linear-gradient(135deg,#eef5ff,#e4ecf8)",
            "border": "rgba(37,99,235,.18)",
            "title": "#1d4ed8",
            "body": "#0f172a",
        },
    }
    style = palette.get(tone, palette["neutral"])
    body = "".join([f"<div style='margin-top:.22rem'>{line}</div>" for line in lines])
    st.markdown(
        f"""
        <div style="margin:.35rem 0 .95rem 0;padding:.9rem 1rem;border-radius:18px;border:1px solid {style['border']};background:{style['bg']};box-shadow:0 10px 24px rgba(15,23,42,.06);">
            <div style="font-size:.78rem;color:{style['title']};margin-bottom:.22rem;font-weight:700;letter-spacing:.03em;">{title}</div>
            <div style="font-size:.94rem;line-height:1.68;font-weight:600;color:{style['body']};">
                {body}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_style(fig, title: str, height: int = None):
    fig.update_layout(
        template=DARK_TEMPLATE,
        title=title,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#0f172a"),
        margin=dict(l=10, r=10, t=45, b=10),
        title_font=dict(color="#0f172a"),
        legend=dict(font=dict(color="#0f172a")),
    )
    if height is not None:
        fig.update_layout(height=height)
    return fig


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _status_badge(status: str) -> str:
    mapping = {
        "개선필요": "🔴 개선필요",
        "주의": "🟠 주의",
        "정상": "🔵 정상",
        "심각": "🔴 심각",
        "생산성 손실형": "🟠 생산성 손실형",
        "품질 집중형": "🟠 품질 집중형",
        "구조적 문제": "🔴 구조적 문제",
        "공정 연계형": "🟠 공정 연계형",
        "국소 LOT 문제": "🟠 국소 LOT 문제",
        "전파 LOT 문제": "🔴 전파 LOT 문제",
        "우선 점검 LOT": "🟠 우선 점검 LOT",
        "복합 문제형": "🔴 복합 문제형",
        "정상": "🔵 정상",
        "복합 병목": "🔴 복합 병목",
        "정지 병목": "🟠 정지 병목",
        "품질 병목": "🟠 품질 병목",
        "흐름 병목": "🟠 흐름 병목",
        "안정성 문제": "🟠 안정성 문제",
        "국소 문제": "🟠 국소 문제",
    }
    return mapping.get(str(status), f"🔵 {status}")


def _confidence_label(has_direct: bool, has_proxy: bool) -> str:
    if has_direct:
        return "Actual"
    if has_proxy:
        return "Estimated"
    return "Low confidence"


def _quadrant_label(stop_val: float, defect_val: float, stop_th: float, defect_th: float) -> str:
    if stop_val >= stop_th and defect_val >= defect_th:
        return "심각"
    if stop_val >= stop_th and defect_val < defect_th:
        return "생산성 손실형"
    if stop_val < stop_th and defect_val >= defect_th:
        return "품질 집중형"
    return "정상"


def _problem_type_from_signals(stop_val: float, defect_val: float, wait_val: float, repeat_val: float, stop_th: float, defect_th: float, wait_th: float) -> str:
    if stop_val >= stop_th and defect_val >= defect_th:
        return "복합 문제형"
    if wait_val >= wait_th:
        return "공정 연계형"
    if stop_val >= stop_th and defect_val < defect_th:
        return "생산성 손실형"
    if stop_val < stop_th and defect_val >= defect_th:
        return "품질 집중형"
    if repeat_val > 0.6:
        return "구조적 문제"
    return "정상"


def _reason_action_hint(reason_text: str) -> str:
    text = str(reason_text or "").upper()
    if any(tok in text for tok in ["PICK", "FEED", "NOZZ", "REEL"]):
        return "nozzle / feeder / reel check"
    if any(tok in text for tok in ["RECOG", "VISION", "MARK", "CAM"]):
        return "vision / camera / lighting tuning"
    if any(tok in text for tok in ["PLACE", "OFFSET", "ALIGN"]):
        return "placement offset / calibration"
    if "WAIT_PRE" in text or "BWAIT" in text or "MCFWAIT" in text:
        return "upstream balance / feeder timing"
    if "WAIT_POST" in text or "RWAIT" in text or "MCRWAIT" in text:
        return "downstream congestion / buffer"
    if "TRANSFER" in text or "CNV" in text:
        return "conveyor / interlock issue"
    return "reason code refinement / detailed check"


def _classification_basis_machine(row: pd.Series, stop_th: float, defect_th: float, wait_th: float) -> str:
    stop_time = _safe_float(row.get("stop_time", 0))
    defect_rate = _safe_float(row.get("defect_rate", 0))
    wait_count = _safe_float(row.get("wait_count", 0))
    retry_rate = _safe_float(row.get("retry_rate", 0))
    if stop_time >= stop_th and defect_rate >= defect_th:
        return "정지 상위구간 + 불량 상위구간 동시 충족"
    if wait_count >= wait_th:
        return "WAIT 비중이 높아 전후공정 연계가 우선"
    if stop_time >= stop_th and defect_rate < defect_th:
        return "정지는 높고 불량은 낮아 생산성 손실형"
    if defect_rate >= defect_th and stop_time < stop_th:
        return "불량은 높고 정지는 낮아 품질 집중형"
    if retry_rate > 0:
        return "재작업 proxy가 보여 안정성 저하 가능성"
    return "기준치 내 변동으로 주의 단계"


def _classification_basis_process(row: pd.Series, output_th: float, stop_th: float, wait_th: float, ct_th: float) -> str:
    output_qty = _safe_float(row.get("output_qty", 0))
    stop_time = _safe_float(row.get("stop_time", 0))
    wait_count = _safe_float(row.get("wait_count", 0))
    ct_std = _safe_float(row.get("ct_std_sec", 0))
    fail_rate = _safe_float(row.get("fail_rate", row.get("defect_rate", 0)))
    if output_qty <= output_th and stop_time >= stop_th:
        return "출력 하락 + 정지 증가로 전형적 병목"
    if stop_time >= stop_th and fail_rate >= 0.5:
        return "정지와 불량이 함께 높아 복합 병목"
    if wait_count >= wait_th:
        return "WAIT가 높아 라인 밸런스/연계 문제"
    if fail_rate >= 0.5:
        return "불량 비중이 높아 품질 병목"
    if ct_std >= ct_th:
        return "CT 편차가 커 공정 안정성 이슈"
    return "현재 기준에서는 주의 수준"


def _classification_basis_lot(row: pd.Series, machine_th: float, process_th: float, impact_th: float) -> str:
    machine_count = _safe_float(row.get("machine_count", 0))
    process_count = _safe_float(row.get("process_count", 0))
    impact_score = _safe_float(row.get("impact_score", 0))
    stop_time = _safe_float(row.get("stop_time", 0))
    if machine_count <= machine_th and stop_time >= impact_th:
        return "특정 설비 한정 영향이 커 국소 LOT 문제"
    if process_count >= process_th:
        return "다수 공정으로 퍼져 구조적 전파 가능성"
    if impact_score >= impact_th:
        return "영향 점수가 높아 우선 점검 LOT"
    return "영향 범위가 제한된 주의 LOT"


def _status_score(value: str) -> int:
    v = str(value)
    if v == "분석 가능":
        return 2
    if v in {"직접 확인", "간접 해석"}:
        return 2 if v == "직접 확인" else 1
    if v == "✔" or v.startswith("가능"):
        return 2
    if str(value) == "제한":
        return 1
    return 0


def _status_chart(df: pd.DataFrame, label_col: str, status_col: str, title: str, color_map: Dict[str, str] = None):
    if df.empty or label_col not in df.columns or status_col not in df.columns:
        return None
    view = df[[label_col, status_col]].copy()
    view["score"] = view[status_col].astype(str).map(_status_score)
    if color_map is None:
        color_map = {
            "분석 가능": "#22c55e",
            "제한": "#f59e0b",
            "없음": "#64748b",
            "직접 확인": "#22c55e",
            "간접 해석": "#f59e0b",
            "✔": "#22c55e",
            "가능": "#22c55e",
            "불가": "#ef4444",
        }
    view["color"] = view[status_col].astype(str).map(lambda x: color_map.get(x, "#64748b"))
    fig = px.bar(
        view,
        x="score",
        y=label_col,
        orientation="h",
        text=status_col,
        color=status_col,
        color_discrete_map=color_map,
    )
    fig.update_traces(textposition="outside")
    status_values = set(view[status_col].astype(str).tolist())
    if {"분석 가능", "제한", "없음"} & status_values:
        fig.update_xaxes(range=[0, 2.3], tickmode="array", tickvals=[0, 1, 2], ticktext=["없음", "제한", "분석 가능"])
    elif any(view[status_col].astype(str).isin(["직접 확인", "간접 해석"])):
        fig.update_xaxes(range=[0, 2.3], tickmode="array", tickvals=[0, 1, 2], ticktext=["없음", "간접 해석", "직접 확인"])
    else:
        fig.update_xaxes(range=[0, 2.3], tickmode="array", tickvals=[0, 1, 2], ticktext=["없음", "제한", "가능"])
    fig.update_yaxes(categoryorder="total ascending")
    return _plot_style(fig, title)


def _top_reason_series(stop: pd.DataFrame) -> pd.Series:
    if stop.empty or "stop_like_reason" not in stop.columns:
        return pd.Series(dtype="object")
    return stop["stop_like_reason"].fillna("미상").astype(str)


def _compute_reliability_indicators(stop: pd.DataFrame) -> Dict[str, float]:
    total = len(stop)
    if total == 0:
        return {
            "total": 0,
            "aggregated_ratio": 0,
            "duplicate_ratio": 0,
            "coverage_ratio": 0,
            "approx_ratio": 0,
            "aggregated_count": 0,
            "duplicate_count": 0,
            "coverage_count": 0,
            "approx_count": 0,
        }
    aggregated_count = int(stop.get("stop_count", 0).fillna(0).gt(1).sum()) if "stop_count" in stop.columns else 0
    duplicate_count = int(stop.duplicated(subset=[c for c in ["lot_id", "machine_id", "stop_like_reason"] if c in stop.columns], keep=False).sum()) if any(c in stop.columns for c in ["lot_id", "machine_id", "stop_like_reason"]) else 0
    coverage_count = int(stop.get("join_coverage", pd.Series([0] * total)).fillna(0).gt(0).sum()) if "join_coverage" in stop.columns else 0
    approx_count = int(stop.get("approx_event", pd.Series([False] * total)).fillna(False).sum()) if "approx_event" in stop.columns else 0
    return {
        "total": total,
        "aggregated_ratio": _safe_div(aggregated_count, total),
        "duplicate_ratio": _safe_div(duplicate_count, total),
        "coverage_ratio": _safe_div(coverage_count, total),
        "approx_ratio": _safe_div(approx_count, total),
        "aggregated_count": aggregated_count,
        "duplicate_count": duplicate_count,
        "coverage_count": coverage_count,
        "approx_count": approx_count,
    }


def _loss_path_priority(stop: pd.DataFrame) -> pd.DataFrame:
    if stop.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["line_id", "stage_no", "machine_id", "stop_like_reason"] if c in stop.columns]
    if len(group_cols) < 2:
        return pd.DataFrame()
    loss = (
        stop.groupby(group_cols, as_index=False)
        .agg(
            loss_time=("duration_sec", "sum"),
            event_count=("stop_count", "sum"),
        )
        .sort_values("loss_time", ascending=False)
    )
    total_loss = loss["loss_time"].sum() or 1
    loss["share"] = loss["loss_time"] / total_loss
    loss["avg_duration"] = loss.apply(lambda row: _safe_div(row["loss_time"], row["event_count"]), axis=1)
    return loss.head(5)


def _render_reliability_badge(indicators: Dict[str, float]):
    if not indicators or indicators.get("total", 0) == 0:
        return
    text = (
        f"이벤트형/누적형(추정): stop_count>1 {indicators['aggregated_ratio'] * 100:.1f}% "
        f"(n={indicators['aggregated_count']}/{indicators['total']}) · "
        f"중복 {indicators['duplicate_ratio'] * 100:.1f}% (n={indicators['duplicate_count']}) · "
        f"FILE coverage {indicators['coverage_ratio'] * 100:.1f}% (n={indicators['coverage_count']}) · "
        f"시간 근사 {indicators['approx_ratio'] * 100:.1f}% (n={indicators['approx_count']})"
    )
    st.markdown(
        f"""
        <div style='padding:8px 12px;border-radius:14px;background:#121826;color:#cfcfcf;font-size:12px;margin:0 0 .8rem 0;'>
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _quality_status(null_ratio: float) -> str:
    if null_ratio <= 0.05:
        return "정상"
    if null_ratio <= 0.20:
        return "부분 사용 가능"
    return "제한적 분석"


def _quality_color(status: str) -> str:
    return {
        "정상": "#0f9d58",
        "부분 사용 가능": "#f5b642",
        "제한적 분석": "#ef553b",
    }.get(status, "#97a2b3")


def _non_empty_options(series_list: List[pd.Series], fallback: str = "전체") -> List[str]:
    values: List[str] = []
    for series in series_list:
        if series is None or series.empty:
            continue
        values.extend([str(v) for v in series.dropna().astype(str).tolist() if str(v) != ""])
    deduped = sorted(set(values))
    return [fallback] + deduped if deduped else [fallback]


def _apply_selection(df: pd.DataFrame, filters: Dict[str, str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if filters.get("line") and filters["line"] != "전체" and "line_id" in out.columns:
        out = out[out["line_id"].astype(str) == filters["line"]]
    if filters.get("stage") and filters["stage"] != "전체" and "stage_no" in out.columns:
        out = out[out["stage_no"].astype(str) == filters["stage"]]
    if filters.get("machine") and filters["machine"] != "전체" and "machine_id" in out.columns:
        out = out[out["machine_id"].astype(str) == filters["machine"]]
    if filters.get("lot") and filters["lot"] != "전체" and "lot_id" in out.columns:
        out = out[out["lot_id"].astype(str) == filters["lot"]]
    if filters.get("model") and filters["model"] != "전체" and "model_label" in out.columns:
        out = out[out["model_label"].astype(str) == filters["model"]]
    defect = filters.get("defect")
    if defect and defect != "전체":
        mask = pd.Series(False, index=out.index)
        if "quality_flag" in out.columns:
            mask = mask | out["quality_flag"].astype(str).eq(defect)
        if "result_primary" in out.columns:
            mask = mask | out["result_primary"].astype(str).eq(defect)
        if "cause_detail" in out.columns:
            mask = mask | out["cause_detail"].astype(str).eq(defect)
        out = out[mask] if mask.any() else out.iloc[0:0]
    return out


def _build_filter_panel(clean: Dict[str, pd.DataFrame], prefix: str) -> Dict[str, str]:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    machine_series = []
    lot_series = []
    model_series = []
    line_series = []
    stage_series = []
    defect_series = []
    for df in [shop, stop, insp, tag]:
        if not df.empty:
            if "machine_id" in df.columns:
                machine_series.append(df["machine_id"])
            if "lot_id" in df.columns:
                lot_series.append(df["lot_id"])
            if "model_label" in df.columns:
                model_series.append(df["model_label"])
            if "line_id" in df.columns:
                line_series.append(df["line_id"])
            if "stage_no" in df.columns:
                stage_series.append(df["stage_no"].astype(str))
            if "quality_flag" in df.columns:
                defect_series.append(df["quality_flag"])
            if "result_primary" in df.columns:
                defect_series.append(df["result_primary"])
            if "stop_like_reason" in df.columns:
                defect_series.append(df["stop_like_reason"])
            if "cause_detail" in df.columns:
                defect_series.append(df["cause_detail"])
    cols = st.columns(6)
    options = {
        "line": _non_empty_options(line_series),
        "stage": _non_empty_options(stage_series),
        "machine": _non_empty_options(machine_series),
        "lot": _non_empty_options(lot_series),
        "model": _non_empty_options(model_series),
        "defect": _non_empty_options(defect_series),
    }
    labels = {
        "line": "line",
        "stage": "stage",
        "machine": "machine_id",
        "lot": "lot_id",
        "model": "model_label",
        "defect": "defect/result_flag",
    }
    filters = {}
    for col, key in zip(cols, ["line", "stage", "machine", "lot", "model", "defect"]):
        with col:
            filters[key] = st.selectbox(labels[key], options[key], key=f"{prefix}_{key}_filter")
    return filters


def _render_card_row(card_df: pd.DataFrame) -> None:
    if card_df.empty:
        st.info("카드 요약을 만들 수 있는 데이터가 충분하지 않습니다.")
        return
    order = {"when": 0, "where": 1, "how_much": 2, "repeat": 3, "what": 4}
    view = card_df.copy()
    view["_order"] = view["card_key"].map(order).fillna(99)
    view = view.sort_values("_order")
    cols = st.columns(min(5, len(view)))
    title_map = {"when": "언제", "where": "어디", "how_much": "얼마나", "repeat": "반복", "what": "무엇"}
    for col, (_, row) in zip(cols, view.iterrows()):
        with col:
            st.markdown(_card(title_map.get(str(row.get("card_key", "-")), str(row.get("card_key", "-")).upper()), str(row.get("headline_value", "-")), f"{row.get('sub_label', '')} · {row.get('evidence', '')}", PRIMARY), unsafe_allow_html=True)


def _field_null_ratio(df: pd.DataFrame, col: str) -> float:
    if df.empty or col not in df.columns:
        return 1.0
    return _safe_div(df[col].isna().sum(), len(df))


def _build_quality_summary(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    rows = []

    def add_row(label: str, ratio: float, meaning: str):
        rows.append({
            "항목": label,
            "null율": float(ratio),
            "상태": _quality_status(float(ratio)),
            "해석": meaning,
        })

    if not stop.empty:
        add_row("timestamp", _field_null_ratio(stop, "event_ts"), "정지/알람 시각")
        add_row("설비 식별자", _field_null_ratio(stop, "machine_id"), "정지 대상 설비")
        add_row("정지사유", _field_null_ratio(stop, "stop_like_reason"), "정지 원인 분류")
        add_row("runtime/downtime", _field_null_ratio(stop, "duration_sec"), "정지시간/가동손실")
        add_row("알람", _field_null_ratio(stop, "stop_like_reason"), "알람/정지 이벤트")
    elif not tag.empty:
        add_row("timestamp", _field_null_ratio(tag, "event_ts"), "태그 이벤트 시각")
        add_row("설비 식별자", _field_null_ratio(tag, "machine_id"), "장비별 추적 키")
        add_row("runtime/downtime", _field_null_ratio(tag, "tag_value_num"), "wait/run proxy")
        add_row("알람", _field_null_ratio(tag, "tag_metric"), "태그 기반 이상 신호")
    elif not shop.empty:
        add_row("timestamp", _field_null_ratio(shop, "event_ts"), "공정 이벤트 시각")
        add_row("설비 식별자", _field_null_ratio(shop, "machine_id"), "공정 장비 식별")
        add_row("runtime/downtime", _field_null_ratio(shop, "output_qty"), "산출량 proxy")
        add_row("알람", _field_null_ratio(shop, "result_primary"), "검사 결과 proxy")
    if not insp.empty:
        add_row("검사 timestamp", _field_null_ratio(insp, "event_ts"), "검사/품질 이벤트")
        add_row("모델", _field_null_ratio(insp, "model_label"), "품질 분석 기준")

    return pd.DataFrame(rows)


def _quality_summary(clean: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    rows = []
    if not shop.empty:
        for label in ["timestamp", "설비 식별자", "model"]:
            rows.append({"항목": label, "null율": 0.0, "상태": "정상", "해석": "커버됨"})
    if not stop.empty:
        rows.append({"항목": "정지사유", "null율": _safe_div(stop["stop_like_reason"].isna().sum(), len(stop)), "상태": "정상" if _safe_div(stop["stop_like_reason"].isna().sum(), len(stop)) <= 0.05 else "주의", "해석": "정지 / 알람 분석"})
        rows.append({"항목": "timestamp", "null율": _safe_div(stop["event_ts"].isna().sum(), len(stop)), "상태": "정상" if _safe_div(stop["event_ts"].isna().sum(), len(stop)) <= 0.05 else "주의", "해석": "이벤트 시각"})
    if not insp.empty:
        rows.append({"항목": "inspection", "null율": _safe_div(insp["event_ts"].isna().sum(), len(insp)), "상태": "정상" if _safe_div(insp["event_ts"].isna().sum(), len(insp)) <= 0.05 else "주의", "해석": "검사 시각"})
    return pd.DataFrame(rows)


def _pickup_item_summary(item: pd.DataFrame) -> pd.DataFrame:
    if item.empty or "item_key" not in item.columns:
        return pd.DataFrame()
    view = item.copy()
    view["item_key"] = view["item_key"].fillna("").astype(str)
    view["item_group"] = view["item_group"].fillna("").astype(str) if "item_group" in view.columns else ""
    pickup_mask = view["item_key"].str.contains("PICK|NOZZ|FEED|REEL|VAC|SUCTION|SUCK", case=False, na=False)
    if "item_group" in view.columns:
        pickup_mask = pickup_mask | view["item_group"].str.contains("PICK|FEED|NOZZ|REEL", case=False, na=False)
    pickup = view[pickup_mask].copy()
    if pickup.empty:
        return pd.DataFrame()
    source = pickup
    out = source.groupby(["item_group", "item_key"], as_index=False).agg(
        건수=("item_key", "size"),
        설비수=("machine_id", "nunique") if "machine_id" in source.columns else ("item_key", "size"),
        공정수=("stage_no", "nunique") if "stage_no" in source.columns else ("item_key", "size"),
        LOT수=("lot_id", "nunique") if "lot_id" in source.columns else ("item_key", "size"),
    )
    out = out.sort_values(["건수", "설비수", "공정수"], ascending=[False, False, False]).head(10).reset_index(drop=True)
    out["비고"] = "pickup 관련 항목"
    return out


def _core_category_status(raw: Dict[str, pd.DataFrame], item: pd.DataFrame) -> pd.DataFrame:
    frames = [item] + [df for df in raw.values() if isinstance(df, pd.DataFrame)]

    def _present_cols(aliases: List[str]) -> List[str]:
        found = []
        for alias in aliases:
            alias_low = str(alias).lower()
            for df in frames:
                if df.empty:
                    continue
                for col in df.columns:
                    col_low = str(col).lower()
                    if col_low == alias_low or col_low.endswith(f"_{alias_low}") or alias_low in col_low:
                        found.append(str(col))
        return sorted(set(found))

    specs = [
        {
            "카테고리": "PRODUCTION",
            "설명": "생산 실적",
            "핵심": ["Actual", "Output", "Prod", "Mount", "Pickup", "Board"],
            "보조": ["Machine", "Stage", "Lane", "LotName"],
            "note": "실제 생산량과 기본 흐름을 확인하는 축",
            "aliases": {
                "Actual": ["actual"],
                "Output": ["output", "output_qty"],
                "Prod": ["prod"],
                "Mount": ["mount"],
                "Pickup": ["pickup"],
                "Board": ["board"],
                "Machine": ["machine_id", "mach_cd"],
                "Stage": ["stage_no", "stage"],
                "Lane": ["line_id", "lane"],
                "LotName": ["lot_nm", "lot_id", "lot_name"],
            },
        },
        {
            "카테고리": "QUALITY",
            "설명": "불량 / 미스",
            "핵심": ["PMiss", "DMiss", "HMiss", "MMiss", "RMiss", "RetryBoard"],
            "보조": ["BadParts", "BadBoard", "LotRetryBoard", "Result"],
            "note": "미스/불량/재작업을 확인하는 축",
            "aliases": {
                "PMiss": ["pmiss"],
                "DMiss": ["dmiss"],
                "HMiss": ["hmiss"],
                "MMiss": ["mmiss"],
                "RMiss": ["rmiss"],
                "RetryBoard": ["retryboard", "retry_board"],
                "BadParts": ["badparts"],
                "BadBoard": ["badboard"],
                "LotRetryBoard": ["lotretryboard"],
                "Result": ["result"],
            },
        },
        {
            "카테고리": "EQUIPMENT_STATE",
            "설명": "설비 상태",
            "핵심": ["TotalStop", "PowerON", "Bwait", "Pwait", "Cwait", "CnvStop"],
            "보조": ["Idle", "PRDStop", "SCStop", "SCEStop"],
            "note": "정지/대기/가동 상태를 확인하는 축",
            "aliases": {
                "TotalStop": ["totalstop"],
                "PowerON": ["poweron"],
                "Bwait": ["bwait"],
                "Pwait": ["pwait"],
                "Cwait": ["cwait"],
                "CnvStop": ["cnvstop"],
                "Idle": ["idle"],
                "PRDStop": ["prdstop"],
                "SCStop": ["scstop"],
                "SCEStop": ["scestop"],
            },
        },
        {
            "카테고리": "TIME_METRIC",
            "설명": "시간",
            "핵심": ["CTime1", "CTime2", "CTime3", "Date"],
            "보조": ["TDispense", "TPriming", "Diff"],
            "note": "사이클/공정 시간 변동을 보는 축",
            "aliases": {
                "CTime1": ["ctime1"],
                "CTime2": ["ctime2"],
                "CTime3": ["ctime3"],
                "TDispense": ["tdispense"],
                "TPriming": ["tpriming"],
                "Diff": ["diff"],
                "Date": ["event_ts", "file_dt", "make_dt", "date"],
            },
        },
        {
            "카테고리": "COMPONENT",
            "설명": "부품 / 자재",
            "핵심": ["PartsName", "ReelID", "NozzleName", "Vendor"],
            "보조": ["UseF", "UseR", "FAdd", "FSAdd", "NCAdd", "NHAdd"],
            "note": "자재/노즐/릴 연계 여부를 보는 축",
            "aliases": {
                "PartsName": ["partsname"],
                "ReelID": ["reelid"],
                "NozzleName": ["nozzlename"],
                "Vendor": ["vendor"],
                "UseF": ["usef"],
                "UseR": ["user"],  # fallback lower-case match on useR
                "FAdd": ["fadd"],
                "FSAdd": ["fsadd"],
                "NCAdd": ["ncadd"],
                "NHAdd": ["nhadd"],
            },
        },
        {
            "카테고리": "TRACEABILITY",
            "설명": "추적",
            "핵심": ["Serial", "TGSerial", "BLKSerial", "BcrStatus", "Code", "MJSID"],
            "보조": ["SerialStatus", "BLKCode"],
            "note": "시리얼/바코드/시스템 추적 축",
            "aliases": {
                "Serial": ["serial"],
                "TGSerial": ["tgserial"],
                "BLKSerial": ["blkserial"],
                "SerialStatus": ["serialstatus"],
                "BcrStatus": ["bcrstatus"],
                "Code": ["code"],
                "BLKCode": ["blkcode"],
                "MJSID": ["mjsid"],
            },
        },
        {
            "카테고리": "IDENTITY",
            "설명": "기준 정보",
            "핵심": ["Machine", "Stage", "Lane", "LName", "LotName", "ProductID", "PlanID", "Rev"],
            "보조": ["MasterWO", "SubWO"],
            "note": "설비/공정/LOT/제품 기준 축",
            "aliases": {
                "Machine": ["machine_id", "mach_cd", "machine"],
                "Stage": ["stage_no", "stage"],
                "Lane": ["line_id", "lane"],
                "LName": ["lname", "line_name"],
                "LotName": ["lot_nm", "lot_name", "lot_id"],
                "ProductID": ["productid", "model_label", "model", "pcbmodel"],
                "PlanID": ["planid"],
                "MasterWO": ["masterwo"],
                "SubWO": ["subwo"],
                "Rev": ["rev"],
            },
        },
        {
            "카테고리": "OPERATION_META",
            "설명": "운영",
            "핵심": ["Author", "AuthorType", "Comment", "Change", "DataEdit", "Simulation", "Version"],
            "보조": ["UnitAdjust", "Format"],
            "note": "운영 이력/수정/버전 관리 축",
            "aliases": {
                "Author": ["author"],
                "AuthorType": ["authortype"],
                "Comment": ["comment"],
                "Change": ["change"],
                "DataEdit": ["dataedit"],
                "UnitAdjust": ["unitadjust"],
                "Simulation": ["simulation"],
                "Format": ["format"],
                "Version": ["version"],
            },
        },
    ]

    rows = []
    for spec in specs:
        matched_core = []
        matched_proxy = []
        rep_cols = []
        for item_name in spec["핵심"]:
            cols = _present_cols(spec["aliases"].get(item_name, [item_name]))
            if cols:
                matched_core.append(item_name)
                rep_cols.extend(cols[:2])
        for item_name in spec["보조"]:
            cols = _present_cols(spec["aliases"].get(item_name, [item_name]))
            if cols:
                matched_proxy.append(item_name)
                rep_cols.extend(cols[:2])
        total = len(spec["핵심"]) + len(spec["보조"])
        hit = len(matched_core) + len(matched_proxy)
        if hit == 0:
            status = "없음"
        elif len(matched_core) >= max(3, len(spec["핵심"]) // 2):
            status = "충분"
        else:
            status = "부분"
        rows.append({
            "카테고리": spec["카테고리"],
            "설명": spec["설명"],
            "핵심 확인 항목": " / ".join(spec["핵심"]),
            "현재 상태": status,
            "확인된 항목": " / ".join(matched_core + matched_proxy) if (matched_core or matched_proxy) else "-",
            "대표 컬럼": ", ".join(sorted(set(rep_cols))) if rep_cols else "-",
            "판단": spec["note"],
            "핵심 커버리지": f"{hit}/{total}",
        })
    return pd.DataFrame(rows)


def _problem_type_guide() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "문제유형": "생산성 손실형",
            "설명": "정지는 많지만 불량은 상대적으로 낮은 상태",
            "판정 기준": "stop/대기 손실이 output 대비 높음",
            "개선 포인트": "라인 밸런스, 보급 타이밍, 전환 구간 점검",
        },
        {
            "문제유형": "품질 문제형",
            "설명": "불량/미스가 높지만 정지는 상대적으로 낮은 상태",
            "판정 기준": "defect/retry가 output 대비 높음",
            "개선 포인트": "노즐, 피더, 비전, 헤드, 자재 조건 점검",
        },
        {
            "문제유형": "복합 문제형",
            "설명": "정지와 불량이 함께 높은 상태",
            "판정 기준": "stop_time과 defect_rate가 동시에 상위권",
            "개선 포인트": "설비 조건과 공정 조건을 동시에 조치",
        },
        {
            "문제유형": "정지 병목",
            "설명": "특정 공정의 정지가 흐름 전체를 막는 상태",
            "판정 기준": "output 저하 + stop_time 상위 + wait 동반",
            "개선 포인트": "전후 stage 연결, 전환 시간, 보급 타이밍 점검",
        },
        {
            "문제유형": "흐름 병목",
            "설명": "정지보다는 대기/연계 불균형이 주된 상태",
            "판정 기준": "wait 비중이 높고 output 편차가 큼",
            "개선 포인트": "upstream/downstream buffer와 라인 밸런스 점검",
        },
        {
            "문제유형": "국소 LOT",
            "설명": "특정 LOT에만 영향이 집중된 상태",
            "판정 기준": "machine_count, stage_count가 낮음",
            "개선 포인트": "해당 LOT의 자재/작업 조건/trace 우선 확인",
        },
        {
            "문제유형": "전파 LOT",
            "설명": "한 LOT의 영향이 여러 설비/공정으로 퍼진 상태",
            "판정 기준": "machine_count 또는 stage_count가 넓음",
            "개선 포인트": "대표 설비와 대표 공정을 먼저 잡고 확산 차단",
        },
        {
            "문제유형": "주의",
            "설명": "현재는 경계 수준이지만 추적이 필요한 상태",
            "판정 기준": "절대/상대 지표가 기준선 부근",
            "개선 포인트": "추이 모니터링과 반복성 확인",
        },
    ])


def _build_structure_summary(inventory: pd.DataFrame) -> pd.DataFrame:
    if inventory.empty:
        return pd.DataFrame()
    view = inventory.copy()

    def _structure_type(row: pd.Series) -> str:
        table_name = str(row.get("table_name", "")).lower()
        if table_name == "fa_26_34_mounter_dtl":
            if int(row.get("item_distinct", 0) or 0) > 0:
                return "정형 + ITEM 정규화"
            return "정형"
        if bool(row.get("has_quality_cols", False)) or bool(row.get("has_result_cols", False)):
            return "검사/결과형"
        if bool(row.get("has_join_cols", False)):
            return "이벤트형"
        return "기초 raw"

    view["structure_type"] = view.apply(_structure_type, axis=1)
    view["analysis_ready"] = np.where(
        view["has_join_cols"].fillna(False) | view["has_result_cols"].fillna(False) | view["has_quality_cols"].fillna(False),
        "가능",
        "제한",
    )
    show_cols = [c for c in [
        "table_name",
        "structure_type",
        "analysis_ready",
        "has_join_cols",
        "has_result_cols",
        "has_quality_cols",
        "item_distinct",
        "item_groups",
        "item_numeric_ratio",
    ] if c in view.columns]
    if not show_cols:
        return pd.DataFrame()
    return view[show_cols].sort_values(["analysis_ready", "table_name"], ascending=[False, True])


def render_summary(raw: Dict[str, pd.DataFrame], clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame], sample_mode: bool):
    item = clean.get("vw_mounter_item_fact", pd.DataFrame()).copy()
    summary = marts.get("vw_mounter_summary", pd.DataFrame()).copy()
    equipment = marts.get("vw_equipment_overview", pd.DataFrame()).copy()
    process = marts.get("vw_process_overview", pd.DataFrame()).copy()
    lot = marts.get("vw_lot_analysis", pd.DataFrame()).copy()
    time_view = marts.get("vw_time_pattern_view", pd.DataFrame()).copy()
    priority = marts.get("vw_priority_view", pd.DataFrame()).copy()
    core_status = _core_category_status(raw, item)
    problem_guide = _problem_type_guide()

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("### 1. 데이터 이해")
    st.caption("이 화면은 현재 확보된 데이터가 어느 수준까지 설명력을 가지는지, 그리고 지금 당장 무엇을 볼 수 있는지를 고객 관점에서 정리한 화면입니다.")
    source = raw.get("_meta", {}).get("source", "unknown") if isinstance(raw.get("_meta", {}), dict) else "unknown"
    st.info(f"현재 데모는 `{source}` 데이터를 사용하며, 핵심 설명 범위는 mounter 중심 데이터입니다.")
    _story_box(
        "이 화면에서 답하려는 질문",
        [
            "지금 확보된 데이터만으로 어디까지 설명할 수 있는가",
            "바로 활용 가능한 분석과 추가 보완이 필요한 분석은 무엇인가",
            "고객이 이번 데모에서 기대해도 되는 범위는 어디까지인가",
        ],
        tone="accent",
    )

    if summary.empty:
        st.info("요약 데이터를 만들 수 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    period_start = pd.to_datetime(item["event_ts"], errors="coerce").min() if not item.empty and "event_ts" in item.columns else pd.NaT
    period_end = pd.to_datetime(item["event_ts"], errors="coerce").max() if not item.empty and "event_ts" in item.columns else pd.NaT
    period_text = f"{period_start:%Y-%m-%d} ~ {period_end:%Y-%m-%d}" if pd.notna(period_start) and pd.notna(period_end) else "-"

    card_cols = st.columns(6)
    card_map = {str(r["항목"]): r for _, r in summary.iterrows()}
    cards = [
        ("총 생산기록", card_map.get("총 생산기록", {}).get("값", "-"), "mounter 행 수"),
        ("출력 proxy", card_map.get("총 출력", {}).get("값", "-"), "기록건수 기반 합계"),
        ("활성 설비 수", card_map.get("설비 수", {}).get("값", "-"), "machine_id distinct"),
        ("활성 공정 수", card_map.get("공정 수", {}).get("값", "-"), "line/stage distinct"),
        ("활성 LOT 수", card_map.get("LOT 수", {}).get("값", "-"), "lot_id distinct"),
        ("활성 일수", card_map.get("활성 일수", {}).get("값", "-"), period_text),
    ]
    for col, (label, value, foot) in zip(card_cols, cards):
        with col:
            st.markdown(_card(label, str(value), foot), unsafe_allow_html=True)

    st.markdown("#### 한눈에 보는 데이터 규모")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("#### 현재 바로 설명 가능한 데이터")
    if not core_status.empty:
        st.dataframe(core_status, use_container_width=True, hide_index=True)
    else:
        st.info("현재 화면 활용 가능 데이터를 요약할 수 없습니다.")

    st.markdown("#### 화면에서 쓰는 문제 분류 기준")
    st.dataframe(problem_guide, use_container_width=True, hide_index=True)

    _story_box(
        "데이터 해석 가이드",
        [
            "설비는 생산량이 낮거나 흐름 변동이 큰 곳부터 우선 확인합니다.",
            "공정은 어느 구간에서 병목이 생기는지 보면서 전체 흐름 관점으로 해석합니다.",
            "LOT는 문제가 한 지점에 머무는지, 여러 설비와 공정으로 퍼지는지를 먼저 봅니다.",
        ],
        tone="neutral",
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_equipment(clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame]):
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    comp = clean.get("vw_component_error_fact", pd.DataFrame())
    filters = _build_filter_panel(clean, "equipment")
    filtered_clean = {
        **clean,
        "vw_shopfloor_event_fact": _apply_selection(shop, filters),
        "vw_stop_event_fact": _apply_selection(stop, filters),
        "vw_inspection_event_fact": _apply_selection(insp, filters),
        "vw_tag_event_fact": _apply_selection(tag, filters),
        "vw_component_error_fact": _apply_selection(comp, filters),
    }
    equipment = build_equipment_overview(filtered_clean)
    process = build_process_overview(filtered_clean)
    process_all = build_process_overview(clean)
    lot = build_lot_analysis_view(filtered_clean)
    filtered_stop = filtered_clean.get("vw_stop_event_fact", pd.DataFrame())
    filtered_tag = filtered_clean.get("vw_tag_event_fact", pd.DataFrame())
    filtered_shop = filtered_clean.get("vw_shopfloor_event_fact", pd.DataFrame())

    def _safe_float(v, default: float = 0.0) -> float:
        try:
            if pd.isna(v):
                return default
            return float(v)
        except Exception:
            return default

    def _fmt_pct(v) -> str:
        return f"{_safe_float(v) * 100:.1f}%"

    def _fmt_stop(v) -> str:
        return _fmt_sec(_safe_float(v))

    def _label_machine(row: pd.Series) -> str:
        machine_id = str(row.get("machine_id", "-"))
        line_id = str(row.get("line_id", "-"))
        stage_no = str(row.get("stage_no", "-"))
        stop_text = _fmt_stop(row.get("stop_time", 0))
        defect_text = _fmt_pct(row.get("defect_rate", 0))
        mtbf_text = _fmt_stop(row.get("mtbf_sec", 0))
        mttr_text = _fmt_stop(row.get("mttr_sec", 0))
        outlier_text = " · 이상치" if bool(row.get("is_outlier", False)) else ""
        return f"{machine_id} · line {line_id} · stage {stage_no} · 정지 {stop_text} · 불량률 {defect_text} · MTBF {mtbf_text} · MTTR {mttr_text}{outlier_text}"

    def _label_process(row: pd.Series) -> str:
        name = str(row.get("process_display", row.get("process_name", "-")))
        line_id = str(row.get("line_id", "-"))
        stage_no = str(row.get("stage_no", "-"))
        output_status = str(row.get("output_status", "데이터 있음"))
        stop_status = str(row.get("stop_status", "데이터 있음"))
        defect_status = str(row.get("defect_status", "데이터 있음"))
        output_qty = f"{_safe_float(row.get('output_qty', 0), 0):.0f}" if output_status == "데이터 있음" else "데이터 없음"
        stop_text = _fmt_stop(row.get("stop_time", 0)) if stop_status == "데이터 있음" else "데이터 없음"
        defect_text = _fmt_pct(row.get("fail_rate", row.get("defect_rate", 0))) if defect_status == "데이터 있음" else "데이터 없음"
        hint = str(row.get("bottleneck_hint", "평균권"))
        status = str(row.get("data_status", "데이터 있음"))
        status_text = "" if status == "데이터 있음" else f" · {status}"
        return f"{name} · line {line_id} · stage {stage_no} · output {output_qty} · 정지 {stop_text} · 불량률 {defect_text} · {hint}{status_text}"

    def _label_lot(row: pd.Series) -> str:
        lot_id = str(row.get("lot_id", "-"))
        model_label = str(row.get("model_label", "-"))
        impact_text = f"{_safe_float(row.get('impact_score', 0), 0):.1f}"
        stop_text = _fmt_stop(row.get("stop_time", 0))
        defect_text = _fmt_pct(row.get("fail_rate", 0))
        return f"{lot_id} · model {model_label} · 영향도 {impact_text} · 정지 {stop_text} · 불량률 {defect_text}"

    def _display_process_name(value: object) -> str:
        name = str(value or "-").strip()
        mapping = {
            "mounter": "Mounter",
            "spi": "SPI",
            "aoi_14": "AOI",
            "aoi_42": "AOI",
            "moi": "MOI",
            "marking": "Marking",
            "stop": "Stop",
            "inspection": "Inspection",
            "mounter_tag": "Mounter Tag",
        }
        return mapping.get(name.lower(), name)

    def _process_analysis_basis(row: pd.Series) -> str:
        process_name = str(row.get("process_name", "")).lower()
        process_display = str(row.get("process_display", row.get("process_name", "-"))).upper()
        if process_name == "spi" or process_display == "SPI":
            return "시간 + 설비 + 결과"
        if process_name in {"aoi_14", "aoi_42", "aoi"} or process_display == "AOI":
            return "시간 + 설비 + 결과"
        if process_name in {"moi", "aoi_post"} or process_display in {"MOI", "AOI_POST"}:
            return "최종검사 결과 + 설비"
        if process_name == "mounter" or process_display == "MOUNTER":
            return "설비 + stage + LOT"
        if process_name == "printer" or process_display == "PRINTER":
            return "공정 이벤트 + 산출"
        if process_name == "reflow" or process_display == "REFLOW":
            return "현재 데이터 없음"
        return "공정별 기준 별도"

    def _process_analysis_content(row: pd.Series) -> str:
        process_name = str(row.get("process_name", "")).lower()
        process_display = str(row.get("process_display", row.get("process_name", "-"))).upper()
        if process_name == "spi" or process_display == "SPI":
            return "machineresult/reviewresult, panelbarcode/model, line/lane/time"
        if process_name in {"aoi_14", "aoi_42", "aoi"} or process_display == "AOI":
            return "machineresult/reviewresult, barcode/panelbarcode, line/lane/time"
        if process_name in {"moi", "aoi_post"} or process_display in {"MOI", "AOI_POST"}:
            return "최종검사 result, barcode/panelbarcode, line/time"
        if process_name == "mounter" or process_display == "MOUNTER":
            return "output, stop, lot, item, tag event"
        if process_name == "printer" or process_display == "PRINTER":
            return "현재 raw는 marking proxy 중심, 시간/라인/LOT 흐름"
        if process_name == "reflow" or process_display == "REFLOW":
            return "현재 raw 데이터 없음"
        return "공정별 분석 내용 별도"

    def _process_analysis_status(row: pd.Series) -> str:
        process_name = str(row.get("process_name", "")).lower()
        process_display = str(row.get("process_display", row.get("process_name", "-"))).upper()
        if process_name == "reflow" or process_display == "REFLOW":
            return "없음"
        if process_name == "printer" or process_display == "PRINTER":
            return "제한"
        if process_name == "mounter" or process_display == "MOUNTER":
            return "분석 가능"
        if process_name in {"spi", "aoi_14", "aoi_42", "aoi", "moi", "aoi_post"} or process_display in {"SPI", "AOI", "MOI", "AOI_POST"}:
            return "분석 가능" if row.get("output_status", "") == "데이터 있음" or row.get("defect_status", "") == "데이터 있음" else "제한"
        return "제한"

    def _build_process_analysis_table(process_df: pd.DataFrame) -> pd.DataFrame:
        if process_df.empty:
            return pd.DataFrame()
        base = process_df.copy()
        if "process_name" not in base.columns:
            return pd.DataFrame()
        rows = []
        analysis_map = {
            "Printer": {
                "분석축": "공정 이벤트 / 산출",
                "분석내용": "marking proxy, 시간대 분포, line 흐름",
                "판정기준": "현재 raw는 제한적",
            },
            "SPI": {
                "분석축": "검사결과 / 불량률",
                "분석내용": "machineresult/reviewresult, inspect_count, fail_rate, line/stage",
                "판정기준": "inspect_count와 fail_rate가 있으면 분석 가능",
            },
            "Mounter": {
                "분석축": "정지 / 출력 / LOT",
                "분석내용": "output_qty, stop_time, stop_count, lot_count, machine_order",
                "판정기준": "stop_time과 output_qty가 모두 있어야 해석 가능",
            },
            "AOI": {
                "분석축": "검사결과 / 품질",
                "분석내용": "inspect_count, fail_rate, line/stage, model/bcode proxy",
                "판정기준": "inspection result와 fail signal 기반",
            },
            "AOI_POST": {
                "분석축": "최종검사 / 재작업",
                "분석내용": "final inspection, fail_rate, barcode/panelbarcode, line",
                "판정기준": "MOI/최종검사 데이터가 있어야 분석 가능",
            },
            "Reflow": {
                "분석축": "현재 데이터 없음",
                "분석내용": "-",
                "판정기준": "원천 테이블 부재",
            },
        }
        for _, row in base.iterrows():
            pname = str(row.get("process_name", ""))
            spec = analysis_map.get(pname, {"분석축": "공정별 기준 별도", "분석내용": "-", "판정기준": "기준 미정"})
            stop_time = _safe_float(row.get("stop_time", 0))
            output_qty = _safe_float(row.get("output_qty", 0))
            fail_rate = _safe_float(row.get("fail_rate", 0))
            inspect_count = _safe_float(row.get("inspect_count", 0))
            if pname == "Printer":
                status = "제한" if output_qty <= 0 else "분석 가능"
                headline = "marking proxy 중심"
            elif pname == "SPI":
                status = "분석 가능" if inspect_count > 0 else "제한"
                headline = "검사 결과 중심"
            elif pname == "Mounter":
                status = "분석 가능" if stop_time > 0 or output_qty > 0 else "제한"
                headline = "정지/출력 중심"
            elif pname == "AOI":
                status = "분석 가능" if inspect_count > 0 or fail_rate > 0 else "제한"
                headline = "검사/품질 중심"
            elif pname == "AOI_POST":
                status = "분석 가능" if inspect_count > 0 or fail_rate > 0 else "제한"
                headline = "최종검사 중심"
            elif pname == "Reflow":
                status = "없음"
                headline = "데이터 없음"
            else:
                status = "제한"
                headline = "미정"
            rows.append({
                "공정": row.get("process_display", pname.lower()),
                "분석 상태": status,
                "핵심 초점": headline,
                "분석축": spec["분석축"],
                "분석내용": spec["분석내용"],
                "판정기준": spec["판정기준"],
                "output_qty": output_qty,
                "stop_time": stop_time,
                "inspect_count": inspect_count,
                "fail_rate": fail_rate,
                "line_id": row.get("line_id", "-"),
                "stage_no": row.get("stage_no", "-"),
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            order = {name.lower(): i for i, name in enumerate(["Printer", "SPI", "Mounter", "AOI", "Reflow", "AOI_POST"])}
            out["_order"] = out["공정"].astype(str).str.lower().map(order).fillna(99)
            out = out.sort_values("_order").drop(columns=["_order"])
        return out

    machine_focus = equipment.copy()
    if not machine_focus.empty:
        machine_focus = machine_focus.sort_values(
            ["stop_time", "defect_rate", "event_density"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    process_focus = process[process["scope"].eq("process")] if not process.empty and "scope" in process.columns else process.copy()
    process_universe = process_all[process_all["scope"].eq("process")] if not process_all.empty and "scope" in process_all.columns else process_all.copy()
    if process_focus.empty and not process_universe.empty:
        process_focus = process_universe.copy()
        process_focus["selection_note"] = "선택 필터 결과가 비어 전체 공정 기준으로 표시"
    else:
        process_focus["selection_note"] = ""
    if not process_focus.empty:
        process_focus["process_display"] = process_focus["process_display"].fillna(process_focus["process_name"].map(_display_process_name))
        if "stage_no" in process_focus.columns:
            process_focus["stage_no"] = process_focus["stage_no"].astype("string").fillna("-").str.strip()
        if "line_id" in process_focus.columns:
            process_focus["line_id"] = process_focus["line_id"].astype("string").fillna("-").str.strip()
        process_focus["analysis_basis"] = process_focus.apply(_process_analysis_basis, axis=1)
        for col in ["stop_time", "stop_count", "inspect_count", "fail_count", "output_qty"]:
            if col not in process_focus.columns:
                process_focus[col] = 0
            process_focus[col] = pd.to_numeric(process_focus[col], errors="coerce").fillna(0)
        process_focus["fail_rate"] = process_focus.apply(lambda r: _safe_div(r.get("fail_count", 0), r.get("inspect_count", 0)), axis=1)
        for col in ["output_status", "stop_status", "defect_status"]:
            if col not in process_focus.columns:
                process_focus[col] = "데이터 없음"
            process_focus[col] = process_focus[col].fillna("데이터 없음")
        process_focus = process_focus.sort_values(
            ["stop_time", "fail_rate", "output_qty"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        stop_q3 = pd.to_numeric(process_focus["stop_time"], errors="coerce").quantile(0.75)
        output_q1 = pd.to_numeric(process_focus["output_qty"], errors="coerce").quantile(0.25)
        defect_q3 = pd.to_numeric(process_focus["fail_rate"], errors="coerce").quantile(0.75)

        def _bottleneck_hint(row: pd.Series) -> str:
            hints = []
            if pd.notna(stop_q3) and row.get("stop_time", 0) >= stop_q3 and stop_q3 > 0:
                hints.append("정지↑")
            if pd.notna(output_q1) and row.get("output_qty", 0) <= output_q1:
                hints.append("산출↓")
            if pd.notna(defect_q3) and row.get("fail_rate", 0) >= defect_q3 and defect_q3 > 0:
                hints.append("불량↑")
            return " / ".join(hints) if hints else "평균권"

        process_focus["bottleneck_hint"] = process_focus.apply(_bottleneck_hint, axis=1)
    lot_focus = lot.copy()
    if not lot_focus.empty:
        lot_focus = lot_focus.sort_values(
            ["impact_score", "fail_rate", "stop_time"],
            ascending=[False, False, False],
            kind="mergesort",
        )
    time_view = build_time_pattern_view(filtered_clean)
    time_hour = time_view[time_view["grain"].eq("hour")].copy() if not time_view.empty and "grain" in time_view.columns else pd.DataFrame()
    time_shift = time_view[time_view["grain"].eq("shift")].copy() if not time_view.empty and "grain" in time_view.columns else pd.DataFrame()

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("### 설비/공정 분석")
    st.caption("이 탭은 RCA가 아니라 문제 위치를 찾는 화면입니다. 설비는 어디가 문제인지, 공정은 어디가 막히는지, LOT는 어디에 영향이 번지는지 확인합니다.")
    st.markdown("#### 분류별 분석 기준")
    criteria_view = pd.DataFrame([
        {"분류": "설비", "기준 키": "machine_id", "판정 포인트": "정지와 불량이 설비별로 갈리는가"},
        {"분류": "공정", "기준 키": "process_display / line_id / stage_no", "판정 포인트": "공정별 output / stop / defect가 함께 보이는가"},
        {"분류": "LOT", "기준 키": "lot_id / model_label", "판정 포인트": "LOT 영향이 공정/설비로 전파되는가"},
    ])
    st.dataframe(criteria_view, use_container_width=True, hide_index=True)
    _render_reliability_badge(_compute_reliability_indicators(filtered_stop))
    st.markdown(
        """
        <div style="margin:.6rem 0 1rem 0;padding:.75rem .9rem;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);color:#cdd5df;">
            현재 탭은 <b>문제 위치 탐색</b> 용도이며, 아래 순서로 읽습니다: <b>설비 관점 → 공정 관점 → LOT 관점 → 시간 관점</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    if machine_focus.empty and process_focus.empty and lot_focus.empty:
        st.warning("설비/공정 분석을 위한 데이터가 부족합니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    top_machine = machine_focus.iloc[0] if not machine_focus.empty else pd.Series(dtype="object")
    top_process = process_focus.iloc[0] if not process_focus.empty else pd.Series(dtype="object")
    top_lot = lot_focus.iloc[0] if not lot_focus.empty else pd.Series(dtype="object")
    major_error = "-"
    if not filtered_tag.empty and "event_class" in filtered_tag.columns:
        error_counts = filtered_tag["event_class"].astype(str).value_counts()
        error_counts = error_counts[~error_counts.index.isin(["OTHER", "META", "FLOW", "INSPECTION", "STOP"])]
        if not error_counts.empty:
            major_error = str(error_counts.index[0]).replace("_", " ")
    summary_parts = []
    if not machine_focus.empty:
        summary_parts.append(f"문제 설비 `{top_machine.get('machine_id', '-')}`")
    if not process_focus.empty:
        summary_parts.append(f"문제 공정 `{top_process.get('process_display', top_process.get('process_name', '-'))}`")
    if not lot_focus.empty:
        summary_parts.append(f"영향 LOT `{top_lot.get('lot_id', '-')}`")
    summary_text = " · ".join(summary_parts) if summary_parts else "현재 데이터에서 먼저 볼 위치를 정하기 어렵습니다."
    if summary_text and major_error != "-":
        line_id = str(top_machine.get("line_id", "-")) if not machine_focus.empty else "-"
        summary_text = f"{line_id} {top_machine.get('machine_id', '-') if not machine_focus.empty else '-'}에서 {major_error}가 두드러지고, {top_process.get('process_display', top_process.get('process_name', '-')) if not process_focus.empty else '-'} 공정이 막히며, {top_lot.get('lot_id', '-') if not lot_focus.empty else '-'} LOT 영향이 큽니다."
    st.markdown("#### 한줄 요약")
    st.markdown(
        f"""
        <div style="padding:10px 14px;border-radius:14px;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.08);margin:0 0 .85rem 0;color:#e6edf7;">
            {summary_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_card("문제 설비", str(top_machine.get("machine_id", "-")), _label_machine(top_machine), PRIMARY), unsafe_allow_html=True)
    with c2:
        proc_label = str(top_process.get("process_display", top_process.get("process_name", "-")))
        if "stage_no" in top_process.index:
            proc_label = f"{proc_label} / stage {top_process.get('stage_no', '-')}"
        st.markdown(_card("문제 공정", proc_label, _label_process(top_process), SECONDARY), unsafe_allow_html=True)
    with c3:
        st.markdown(_card("영향 LOT", str(top_lot.get("lot_id", "-")), _label_lot(top_lot), "#22c55e"), unsafe_allow_html=True)
    with c4:
        st.markdown(_card("주요 오류 유형", major_error, "WAIT / PICKUP / FEEDER / RECOG / PLACE / TRANSFER / SETUP", "#8b5cf6"), unsafe_allow_html=True)

    def _avg_compare_label(label: str, value: float, mean_value: float, unit: str = "") -> str:
        return f"{label} {value:.2f}{unit} vs 평균 {mean_value:.2f}{unit} ({'+' if value - mean_value >= 0 else ''}{value - mean_value:.2f}{unit})"

    machine_mean_stop = pd.to_numeric(machine_focus["stop_time"], errors="coerce").fillna(0).mean() if not machine_focus.empty and "stop_time" in machine_focus.columns else 0
    machine_mean_defect = pd.to_numeric(machine_focus["defect_rate"], errors="coerce").fillna(0).mean() if not machine_focus.empty and "defect_rate" in machine_focus.columns else 0
    process_mean_stop = pd.to_numeric(process_focus["stop_time"], errors="coerce").fillna(0).mean() if not process_focus.empty and "stop_time" in process_focus.columns else 0
    process_mean_output = pd.to_numeric(process_focus["output_qty"], errors="coerce").fillna(0).mean() if not process_focus.empty and "output_qty" in process_focus.columns else 0
    process_all_mean_stop = pd.to_numeric(process_universe["stop_time"], errors="coerce").fillna(0).mean() if not process_universe.empty and "stop_time" in process_universe.columns else 0
    process_all_mean_output = pd.to_numeric(process_universe["output_qty"], errors="coerce").fillna(0).mean() if not process_universe.empty and "output_qty" in process_universe.columns else 0
    lot_mean_impact = pd.to_numeric(lot_focus["impact_score"], errors="coerce").fillna(0).mean() if not lot_focus.empty and "impact_score" in lot_focus.columns else 0
    lot_mean_defect = pd.to_numeric(lot_focus["fail_rate"], errors="coerce").fillna(0).mean() if not lot_focus.empty and "fail_rate" in lot_focus.columns else 0

    if not machine_focus.empty and "stop_time" in machine_focus.columns:
        total_stop = pd.to_numeric(machine_focus["stop_time"], errors="coerce").fillna(0).sum()
        top_n = max(1, int(np.ceil(len(machine_focus) * 0.2)))
        pareto_share = _safe_div(pd.to_numeric(machine_focus.head(top_n)["stop_time"], errors="coerce").fillna(0).sum(), total_stop)
        outlier_count = int(machine_focus["is_outlier"].fillna(False).sum()) if "is_outlier" in machine_focus.columns else 0
        st.caption(f"Pareto: 상위 20% 설비가 전체 stop_time의 {pareto_share * 100:.1f}%를 차지. outlier 설비 {outlier_count}개 탐지.")
        st.caption(_avg_compare_label("선택 설비 stop_time", _safe_float(top_machine.get("stop_time", 0)), machine_mean_stop, "s"))
        st.caption(_avg_compare_label("선택 설비 defect_rate", _safe_float(top_machine.get("defect_rate", 0)) * 100, machine_mean_defect * 100, "%"))
        stop_high = _safe_float(top_machine.get("stop_time", 0)) > machine_mean_stop
        defect_high = _safe_float(top_machine.get("defect_rate", 0)) > machine_mean_defect
        machine_interpretation = "복합형" if stop_high and defect_high else "정지형" if stop_high else "품질형" if defect_high else "안정형"
        if stop_high and not defect_high:
            machine_interpretation = "정지형"
        elif not stop_high and defect_high:
            machine_interpretation = "품질형"
        machine_summary = pd.DataFrame([
            {
                "지표": "정지 리스크",
                "선택": f"{_safe_float(top_machine.get('stop_time', 0)):.2f}s",
                "평균": f"{machine_mean_stop:.2f}s",
                "차이": f"{_safe_float(top_machine.get('stop_time', 0)) - machine_mean_stop:.2f}s",
                "판정": "높음" if stop_high else "낮음",
                "해석": "정지 시간이 평균보다 높은지 확인",
            },
            {
                "지표": "불량 리스크",
                "선택": f"{_safe_float(top_machine.get('defect_rate', 0)) * 100:.2f}%",
                "평균": f"{machine_mean_defect * 100:.2f}%",
                "차이": f"{(_safe_float(top_machine.get('defect_rate', 0)) - machine_mean_defect) * 100:.2f}%",
                "판정": "높음" if defect_high else "낮음",
                "해석": "불량률이 평균보다 높은지 확인",
            },
            {
                "지표": "복합 문제 여부",
                "선택": f"{pareto_share * 100:.1f}%",
                "평균": f"{(1 / max(len(machine_focus), 1)) * 100:.1f}%",
                "차이": f"{(pareto_share - (1 / max(len(machine_focus), 1))) * 100:.1f}%",
                "판정": "문제 설비" if stop_high and defect_high else "단일 리스크",
                "해석": "정지와 불량이 함께 높은 설비만 문제 설비로 본다",
            },
            {
                "지표": "설비 유형",
                "선택": machine_interpretation,
                "평균": "전체 평균",
                "차이": "-",
                "판정": "최종 해석",
                "해석": "정지형 / 품질형 / 복합형 / 안정형으로 묶음",
            },
        ])
        st.dataframe(machine_summary, use_container_width=True, hide_index=True)

    st.markdown(_section_header("설비 관점", "어느 설비가 문제인가?", PRIMARY), unsafe_allow_html=True)
    if not machine_focus.empty:
        top_machine_view = machine_focus.head(10).copy()
        machine_order_view = top_machine_view.copy()
        if "machine_order" in machine_order_view.columns:
            machine_order_view = machine_order_view.sort_values(["machine_order", "machine_id"], ascending=[True, True], kind="mergesort")
        else:
            machine_order_view = machine_order_view.sort_values(["machine_id"], ascending=[True], kind="mergesort")
        machine_axis_order = machine_order_view["machine_id"].astype(str).tolist()
        if not filtered_tag.empty and "event_class" in filtered_tag.columns:
            overall_error = (
                filtered_tag.groupby("event_class", as_index=False)
                .agg(count=("event_class", "size"))
                .sort_values("count", ascending=False)
            )
            overall_error = overall_error[~overall_error["event_class"].astype(str).isin(["OTHER", "META", "FLOW", "INSPECTION", "STOP"])]
        else:
            overall_error = pd.DataFrame()
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        machine_order_view,
                        x="machine_id",
                        y="stop_time",
                        color="line_id" if "line_id" in top_machine_view.columns else None,
                        text="stop_time",
                        category_orders={"machine_id": machine_axis_order},
                    ),
                    "설비별 stop_time",
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        machine_order_view,
                        x="machine_id",
                        y="defect_rate",
                        color="line_id" if "line_id" in top_machine_view.columns else None,
                        text="defect_rate",
                        category_orders={"machine_id": machine_axis_order},
                    ),
                    "설비별 defect_rate",
                ),
                use_container_width=True,
            )
        if not overall_error.empty:
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        overall_error.head(8),
                        x="event_class",
                        y="count",
                        text="count",
                    ),
                    "주요 오류 유형",
                ),
                use_container_width=True,
            )
        machine_compare = pd.DataFrame([
            {"항목": "stop_time", "선택": _safe_float(top_machine.get("stop_time", 0)), "전체 평균": machine_mean_stop, "차이": _safe_float(top_machine.get("stop_time", 0)) - machine_mean_stop},
            {"항목": "defect_rate(%)", "선택": _safe_float(top_machine.get("defect_rate", 0)) * 100, "전체 평균": machine_mean_defect * 100, "차이": (_safe_float(top_machine.get("defect_rate", 0)) - machine_mean_defect) * 100},
            {"항목": "MTBF(s)", "선택": _safe_float(top_machine.get("mtbf_sec", 0)), "전체 평균": pd.to_numeric(machine_focus.get("mtbf_sec", pd.Series([0])), errors="coerce").fillna(0).mean() if "mtbf_sec" in machine_focus.columns else 0, "차이": _safe_float(top_machine.get("mtbf_sec", 0)) - (pd.to_numeric(machine_focus.get("mtbf_sec", pd.Series([0])), errors="coerce").fillna(0).mean() if "mtbf_sec" in machine_focus.columns else 0)},
            {"항목": "MTTR(s)", "선택": _safe_float(top_machine.get("mttr_sec", 0)), "전체 평균": pd.to_numeric(machine_focus.get("mttr_sec", pd.Series([0])), errors="coerce").fillna(0).mean() if "mttr_sec" in machine_focus.columns else 0, "차이": _safe_float(top_machine.get("mttr_sec", 0)) - (pd.to_numeric(machine_focus.get("mttr_sec", pd.Series([0])), errors="coerce").fillna(0).mean() if "mttr_sec" in machine_focus.columns else 0)},
        ])
        st.dataframe(machine_compare, use_container_width=True, hide_index=True)
        st.plotly_chart(
            _plot_style(
                px.scatter(
                    top_machine_view,
                    x="stop_time",
                    y="defect_rate",
                    size="event_density" if "event_density" in top_machine_view.columns else None,
                    color="line_id" if "line_id" in top_machine_view.columns else None,
                    hover_name="machine_id",
                ),
                "정지시간 vs 불량률",
            ),
            use_container_width=True,
        )
        machine_error_cols = [c for c in ["setup_events", "feeder_error_events", "pickup_error_events", "recog_error_events", "place_error_events", "transfer_error_events", "wait_events"] if c in top_machine_view.columns]
        if machine_error_cols:
            error_long = top_machine_view[["machine_id"] + machine_error_cols].melt(id_vars="machine_id", var_name="error_type", value_name="count")
            error_long["count"] = pd.to_numeric(error_long["count"], errors="coerce").fillna(0)
            error_long = error_long[error_long["count"] > 0]
            if not error_long.empty:
                st.plotly_chart(
                    _plot_style(
                        px.bar(
                            error_long,
                            x="machine_id",
                            y="count",
                            color="error_type",
                            barmode="stack",
                        ),
                        "설비별 에러 유형 분포",
                    ),
                    use_container_width=True,
                )
        st.dataframe(
            top_machine_view[
                [c for c in ["rank", "machine_id", "line_id", "stage_no", "output_qty", "stop_time", "stop_count", "mtbf_sec", "mttr_sec", "defect_rate", "inspect_count", "fail_count", "event_density", "is_outlier"] if c in top_machine_view.columns]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("설비 비교 데이터를 만들 수 없습니다.")

    st.markdown(_section_header("공정 관점", "어느 공정이 막히는가?", SECONDARY), unsafe_allow_html=True)
    if not process_focus.empty:
        top_process_view = process_focus.head(10).copy()
        process_order_map = {"aoi_post": 0, "aoi": 1, "mounter": 2, "spi": 3, "reflow": 4, "printer": 5}
        top_process_view["process_key"] = top_process_view["process_display"].astype(str).str.lower().map(lambda x: x if x in process_order_map else x)
        top_process_view["process_order_ui"] = top_process_view["process_key"].map(lambda x: process_order_map.get(str(x), 99))
        top_process_view = top_process_view.sort_values(["process_order_ui", "stop_time", "fail_rate", "output_qty"], ascending=[True, False, False, False], kind="mergesort")
        top_process_view["process_label"] = top_process_view.apply(
            lambda row: f"{row.get('process_display', row.get('process_name', '-'))}<br>L{row.get('line_id', '-')}/S{row.get('stage_no', '-')}", axis=1
        )
        top_process_view["output_text"] = np.where(top_process_view["output_status"].astype(str).eq("데이터 있음"), top_process_view["output_qty"].round(0).astype(int).astype(str), "데이터 없음")
        top_process_view["stop_text"] = np.where(top_process_view["stop_status"].astype(str).eq("데이터 있음"), top_process_view["stop_time"].round(0).astype(int).astype(str), "데이터 없음")
        top_process_view["defect_text"] = np.where(top_process_view["defect_status"].astype(str).eq("데이터 있음"), (top_process_view["fail_rate"] * 100).round(1).astype(str) + "%", "데이터 없음")
        top_process_view["basis_text"] = top_process_view.get("analysis_basis", pd.Series(["공정별 기준 별도"] * len(top_process_view), index=top_process_view.index)).fillna("공정별 기준 별도")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        top_process_view.sort_values("process_order_ui", ascending=True),
                        x="process_display",
                        y="output_qty",
                        color="line_id" if "line_id" in top_process_view.columns else None,
                        text="output_text",
                    ),
                    "공정별 output(throughput)",
                ),
                use_container_width=True,
            )
        with p2:
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        top_process_view.sort_values("process_order_ui", ascending=True),
                        x="process_display",
                        y="stop_time",
                        color="line_id" if "line_id" in top_process_view.columns else None,
                        text="stop_text",
                    ),
                    "공정별 stop_time",
                ),
                use_container_width=True,
            )
        with p3:
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        top_process_view.sort_values("process_order_ui", ascending=True),
                        x="process_display",
                        y="fail_rate",
                        color="line_id" if "line_id" in top_process_view.columns else None,
                        text="defect_text",
                    ),
                    "공정별 defect_rate",
                ),
                use_container_width=True,
            )
        process_compare = pd.DataFrame([
            {"항목": "output(throughput)", "선택": f"{_safe_float(top_process.get('output_qty', 0)):.0f}" if str(top_process.get("output_status", "데이터 없음")) == "데이터 있음" else "데이터 없음", "전체 평균": f"{process_all_mean_output:.0f}" if process_all_mean_output > 0 else "데이터 없음", "차이": "-" if str(top_process.get("output_status", "데이터 없음")) != "데이터 있음" else f"{_safe_float(top_process.get('output_qty', 0)) - process_all_mean_output:.0f}"},
            {"항목": "stop_time", "선택": _fmt_stop(top_process.get("stop_time", 0)) if str(top_process.get("stop_status", "데이터 없음")) == "데이터 있음" else "데이터 없음", "전체 평균": _fmt_stop(process_all_mean_stop) if process_all_mean_stop > 0 else "데이터 없음", "차이": "-" if str(top_process.get("stop_status", "데이터 없음")) != "데이터 있음" else f"{_safe_float(top_process.get('stop_time', 0)) - process_all_mean_stop:.0f}s"},
            {"항목": "defect_rate(%)", "선택": _fmt_pct(top_process.get("fail_rate", 0)) if str(top_process.get("defect_status", "데이터 없음")) == "데이터 있음" else "데이터 없음", "전체 평균": _fmt_pct(pd.to_numeric(process_universe.get("fail_rate", pd.Series([0])), errors="coerce").fillna(0).mean()) if "fail_rate" in process_universe.columns and pd.to_numeric(process_universe.get("fail_rate", pd.Series([0])), errors="coerce").fillna(0).mean() > 0 else "데이터 없음", "차이": "-" if str(top_process.get("defect_status", "데이터 없음")) != "데이터 있음" else f"{(_safe_float(top_process.get('fail_rate', 0)) - (pd.to_numeric(process_universe.get('fail_rate', pd.Series([0])), errors='coerce').fillna(0).mean() if 'fail_rate' in process_universe.columns else 0)) * 100:.1f}%"},
        ])
        st.dataframe(process_compare, use_container_width=True, hide_index=True)
        process_analysis = _build_process_analysis_table(process_universe if not process_universe.empty else process_focus)
        if not process_analysis.empty:
            st.markdown("#### 공정별 분석 결과")
            st.dataframe(
                process_analysis[
                    [c for c in ["공정", "분석 상태", "핵심 초점", "분석축", "분석내용", "판정기준", "line_id", "stage_no", "output_qty", "stop_time", "inspect_count", "fail_rate"] if c in process_analysis.columns]
                ],
                use_container_width=True,
                hide_index=True,
            )
        if process_focus["selection_note"].astype(str).str.len().fillna(0).gt(0).any():
            st.info("선택 필터에서 공정 데이터가 비어 있어 전체 공정 기준으로 표시합니다.")
        st.dataframe(
            top_process_view[
                [c for c in ["rank", "process_display", "basis_text", "line_id", "stage_no", "machine_order", "output_qty", "output_status", "stop_time", "stop_status", "stop_count", "fail_count", "defect_status", "fail_rate", "bottleneck_hint", "production_rows", "distinct_machines", "lot_count"] if c in top_process_view.columns]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("공정 비교 데이터를 만들 수 없습니다.")

    st.markdown(_section_header("LOT 관점", "어느 LOT에 영향이 번졌는가?", "#22c55e"), unsafe_allow_html=True)
    if not lot_focus.empty:
        lot_view = lot_focus.head(10).copy()
        lot_compare = pd.DataFrame([
            {"항목": "impact_score", "선택": _safe_float(top_lot.get("impact_score", 0)), "전체 평균": lot_mean_impact, "차이": _safe_float(top_lot.get("impact_score", 0)) - lot_mean_impact},
            {"항목": "defect_rate(%)", "선택": _safe_float(top_lot.get("fail_rate", 0)) * 100, "전체 평균": lot_mean_defect * 100, "차이": (_safe_float(top_lot.get("fail_rate", 0)) - lot_mean_defect) * 100},
        ])
        st.dataframe(lot_compare, use_container_width=True, hide_index=True)
        l1, l2 = st.columns(2)
        with l1:
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        lot_view.sort_values("impact_score", ascending=False),
                        x="lot_id",
                        y="impact_score",
                        color="model_label" if "model_label" in lot_view.columns else None,
                        text="impact_score",
                    ),
                    "LOT 영향도",
                ),
                use_container_width=True,
            )
        with l2:
            st.plotly_chart(
                _plot_style(
                    px.scatter(
                        lot_view,
                        x="production_rows" if "production_rows" in lot_view.columns else "output_qty",
                        y="fail_rate",
                        size="stop_time" if "stop_time" in lot_view.columns else None,
                        color="model_label" if "model_label" in lot_view.columns else None,
                        hover_name="lot_id",
                    ),
                    "LOT vs defect",
                ),
                use_container_width=True,
            )
        st.plotly_chart(
            _plot_style(
                px.scatter(
                    lot_view,
                    x="production_rows" if "production_rows" in lot_view.columns else "output_qty",
                    y="stop_time",
                    size="impact_score" if "impact_score" in lot_view.columns else None,
                    color="model_label" if "model_label" in lot_view.columns else None,
                    hover_name="lot_id",
                ),
                "LOT vs 생산량",
            ),
            use_container_width=True,
        )
        st.dataframe(
            lot_view[
                [c for c in ["rank", "lot_id", "model_label", "production_rows", "output_qty", "stop_time", "stop_count", "inspect_count", "fail_count", "fail_rate", "impact_score", "machine_count", "process_count", "representative_machine", "representative_process"] if c in lot_view.columns]
            ],
            use_container_width=True,
            hide_index=True,
        )
        if filters.get("lot") and filters["lot"] != "전체":
            lot_detail = _apply_selection(shop, filters)
            if not lot_detail.empty:
                st.caption(f"선택 LOT `{filters['lot']}` 드릴다운")
                st.dataframe(
                    lot_detail[[c for c in ["event_ts", "process_name", "machine_id", "stage_no", "line_id", "model_label", "result_primary", "output_qty"] if c in lot_detail.columns]].head(50),
                    use_container_width=True,
                    hide_index=True,
                )
    else:
        st.info("LOT 영향 데이터를 만들 수 없습니다.")

    st.markdown(_section_header("시간 관점", "언제 문제가 집중됐는가?", "#10b981"), unsafe_allow_html=True)
    if not time_view.empty:
        if not time_hour.empty:
            hour_plot = time_hour.sort_values("bucket_order") if "bucket_order" in time_hour.columns else time_hour
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        hour_plot,
                        x="bucket",
                        y="stop_time" if "stop_time" in hour_plot.columns else "error_count",
                        text="stop_time" if "stop_time" in hour_plot.columns else "error_count",
                        color="shift" if "shift" in hour_plot.columns else None,
                    ),
                    "시간대별 stop_time / error_count",
                ),
                use_container_width=True,
            )
        if not time_shift.empty:
            shift_cols = [c for c in ["shift", "stop_time", "stop_count", "error_count", "production_rows"] if c in time_shift.columns]
            st.dataframe(time_shift[shift_cols].sort_values("shift"), use_container_width=True, hide_index=True)
            st.plotly_chart(
                _plot_style(
                    px.bar(
                        time_shift.sort_values("stop_time", ascending=False),
                        x="shift",
                        y="stop_time" if "stop_time" in time_shift.columns else "error_count",
                        text="stop_time" if "stop_time" in time_shift.columns else "error_count",
                        color="shift" if "shift" in time_shift.columns else None,
                    ),
                    "shift별 stop_time",
                ),
                use_container_width=True,
            )
    else:
        st.info("시간 패턴 데이터를 만들 수 없습니다.")
    st.markdown("#### 해석 메모")
    st.markdown("- 설비는 `stop_time`, `defect_rate`, 에러 유형 분포를 같이 봅니다.")
    st.markdown("- 공정은 `output`, `stop_time`, `defect_rate`, 병목 사유를 같이 봅니다.")
    st.markdown("- LOT는 영향도와 생산량/불량의 동시 변화를 봅니다.")
    st.markdown("- 시간은 hour / shift별 stop_time과 error_count 집중 구간을 봅니다.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_rca(clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame], sample_mode: bool):
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    comp = clean.get("vw_component_error_fact", pd.DataFrame())
    filters = _build_filter_panel(clean, "rca")
    filtered_clean = {
        **clean,
        "vw_shopfloor_event_fact": _apply_selection(shop, filters),
        "vw_stop_event_fact": _apply_selection(stop, filters),
        "vw_inspection_event_fact": _apply_selection(insp, filters),
        "vw_tag_event_fact": _apply_selection(tag, filters),
        "vw_component_error_fact": _apply_selection(comp, filters),
    }
    loss_paths = build_rca_loss_path_view(filtered_clean)
    card_summary = build_rca_card_summary(filtered_clean)
    timeline = build_rca_timeline_view(filtered_clean)
    hotspot = build_rca_hotspot_view(filtered_clean)
    repeat = build_rca_repeat_pattern_view(filtered_clean)
    drilldown = build_rca_drilldown_view(filtered_clean)
    proxy_candidates = build_rca_candidate_view(filtered_clean)
    st.markdown('<div class="box">', unsafe_allow_html=True)

    def _rca_panel(title: str, method_lines: List[str], interpretation_lines: List[str], tone: str = "info"):
        icon = {"info": "ℹ️", "warn": "⚠️", "accent": "🔎"}.get(tone, "ℹ️")
        with st.expander(f"{icon} {title}", expanded=False):
            st.markdown("**화면이 보는 기준**")
            st.markdown("\n".join([f"- {line}" for line in method_lines]))
            st.markdown("**고객 해석 포인트**")
            st.markdown("\n".join([f"- {line}" for line in interpretation_lines]))

    _story_box(
        "영향도 읽는 법",
        [
            "정지 데이터 기준: impact = 정지시간 합계(sec)",
            "품질 데이터 기준: impact = 불량 건수 합계 또는 품질 점수 합계",
            "대체 지표 기준: impact = proxy score 또는 metric_value 합계",
            "따라서 impact는 반드시 집계 기준, 단위, 발생 건수를 함께 보면서 해석해야 합니다.",
        ],
        tone="neutral",
    )

    _render_reliability_badge(_compute_reliability_indicators(stop))
    st.markdown("#### 화면 용어 해석")
    _rca_panel(
        "용어를 이렇게 읽으면 됩니다",
        [
            "측정 기준은 시간이냐, 건수냐, 점수냐를 구분합니다.",
            "이벤트 분류는 정지인지, 품질인지, 성능 이슈인지를 구분합니다.",
            "원인 코드와 원인군은 무엇이 발생했고 어느 영역을 점검해야 하는지 나눠 보여줍니다.",
        ],
        [
            "고객에게는 숫자 자체보다 무엇을 더한 값인지가 더 중요합니다.",
            "영향도는 항상 집계 기준, 단위, 발생 건수를 함께 봐야 혼동이 없습니다.",
        ],
        tone="info",
    )
    consistency = pd.DataFrame([
        {"화면 용어": "metric_type", "쉽게 말하면": "시간/건수/점수 같은 측정 기준", "의미": "숫자의 단위", "예시": "stage_time / stop_time / defect_count"},
        {"화면 용어": "event_group", "쉽게 말하면": "정지/품질/성능 이슈 구분", "의미": "이벤트 분류", "예시": "STOP / PERFORMANCE / QUALITY"},
        {"화면 용어": "cause_code", "쉽게 말하면": "발생한 현상의 코드", "의미": "원천 이벤트 코드", "예시": "TRANSFER_ERR / PICKUP_ERR / RECOG_ERR / PLACE_ERR"},
        {"화면 용어": "cause_family", "쉽게 말하면": "우선 점검해야 할 영역", "의미": "조치 영역", "예시": "feeder / nozzle / transfer / vision / upstream / downstream"},
        {"화면 용어": "impact", "쉽게 말하면": "문제의 크기", "의미": "경로별 누적 합계", "예시": "집계 기준 / 단위 / 발생 건수 함께 확인"},
    ])
    st.dataframe(consistency, use_container_width=True, hide_index=True)

    if all(df.empty for df in [shop, stop, insp, tag, comp, proxy_candidates, loss_paths, hotspot, repeat, drilldown]):
        st.warning("RCA 분석을 위한 데이터가 부족합니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    anomaly_rows = []
    if not loss_paths.empty:
        top_prod = loss_paths.iloc[0]
        anomaly_rows.append({
            "이상유형": "생산성 이상",
            "집중 축": str(top_prod.get("where", "-")),
            "대표 후보": str(top_prod.get("what", "-")),
            "근거": str(top_prod.get("how_much", "-")),
            "다음 확인": str(top_prod.get("path_key", "-")),
            "우선순위 규칙": "라인 손실 직접성 + 영향도 + 반복 여부",
            "우선순위 점수": 3,
        })
    elif not card_summary.empty:
        top_prod = card_summary.iloc[0]
        anomaly_rows.append({
            "이상유형": "생산성 이상",
            "집중 축": str(top_prod.get("headline_value", "-")),
            "대표 후보": str(top_prod.get("sub_label", "-")),
            "근거": str(top_prod.get("evidence", "-")),
            "다음 확인": str(top_prod.get("detail_key", "-")),
            "우선순위 규칙": "라인 손실 직접성 + 영향 범위",
            "우선순위 점수": 3,
        })
    else:
        anomaly_rows.append({"이상유형": "생산성 이상", "집중 축": "데이터 부족", "대표 후보": "-", "근거": "-", "다음 확인": "-", "우선순위 규칙": "데이터 부족", "우선순위 점수": 0})

    quality_hotspot = hotspot[hotspot["hotspot_type"].eq("defect")] if not hotspot.empty and "hotspot_type" in hotspot.columns else pd.DataFrame()
    if not quality_hotspot.empty:
        top_quality = quality_hotspot.iloc[0]
        anomaly_rows.append({
            "이상유형": "품질 이상",
            "집중 축": f"{top_quality.get('machine_id', '-') } / {top_quality.get('model_label', '-')}",
            "대표 후보": f"{top_quality.get('cause_group', '-') } / {top_quality.get('cause_detail', '-')}",
            "근거": f"impact {top_quality.get('impact', 0):.0f}",
            "다음 확인": "불량 모델 / LOT / 검사 조건",
            "우선순위 규칙": "품질 영향 + 모델/Lot 집중",
            "우선순위 점수": 1 if float(top_quality.get("impact", 0)) > 0 else 0,
        })
    elif not proxy_candidates.empty:
        top_quality = proxy_candidates.iloc[0]
        anomaly_rows.append({
            "이상유형": "품질 이상",
            "집중 축": f"{top_quality.get('line_id', '-') } / {top_quality.get('machine_id', '-')}",
            "대표 후보": f"{top_quality.get('cause_group', '-') } / {top_quality.get('cause_detail', '-')}",
            "근거": f"proxy {top_quality.get('proxy_score', 0):.0f}",
            "다음 확인": "불량/검사 proxy 데이터",
            "우선순위 규칙": "품질 proxy + 반복 후보",
            "우선순위 점수": 1,
        })
    else:
        anomaly_rows.append({"이상유형": "품질 이상", "집중 축": "데이터 부족", "대표 후보": "-", "근거": "-", "다음 확인": "-", "우선순위 규칙": "데이터 부족", "우선순위 점수": 0})

    machine_hotspot = hotspot[hotspot["hotspot_type"].eq("machine")] if not hotspot.empty and "hotspot_type" in hotspot.columns else pd.DataFrame()
    if not machine_hotspot.empty:
        top_machine = machine_hotspot.iloc[0]
        anomaly_rows.append({
            "이상유형": "설비 이상",
            "집중 축": f"{top_machine.get('line_id', '-') } / {top_machine.get('machine_id', '-')}",
            "대표 후보": f"{top_machine.get('cause_group', '-') } / {top_machine.get('cause_detail', '-')}",
            "근거": f"impact {top_machine.get('impact', 0):.0f}",
            "다음 확인": "설비 / 헤드 / 노즐 / 피더",
            "우선순위 규칙": "국소 설비성 + 반복 정지/오류",
            "우선순위 점수": 2 if float(top_machine.get("impact", 0)) > 0 else 0,
        })
    elif not loss_paths.empty:
        top_machine = loss_paths.iloc[0]
        anomaly_rows.append({
            "이상유형": "설비 이상",
            "집중 축": f"{top_machine.get('line_id', '-') } / {top_machine.get('machine_id', '-')}",
            "대표 후보": f"{top_machine.get('cause_group', '-') } / {top_machine.get('cause_detail', '-')}",
            "근거": f"impact {top_machine.get('impact', 0):.0f}",
            "다음 확인": "설비 / 헤드 / 노즐 / 피더",
            "우선순위 규칙": "국소 설비성 + 정지 이력",
            "우선순위 점수": 2,
        })
    else:
        anomaly_rows.append({"이상유형": "설비 이상", "집중 축": "데이터 부족", "대표 후보": "-", "근거": "-", "다음 확인": "-", "우선순위 규칙": "데이터 부족", "우선순위 점수": 0})

    anomaly_candidate_view = pd.DataFrame(anomaly_rows).sort_values(["우선순위 점수", "이상유형"], ascending=[False, True]).reset_index(drop=True)
    anomaly_candidate_view["rank"] = np.arange(1, len(anomaly_candidate_view) + 1)

    st.markdown("#### 상위 이상 후보")
    st.markdown("- 시나리오 흐름에서 먼저 따라가야 할 이상 후보입니다.")
    st.caption("영향도는 경로별 누적 합계이며, 정지시간 합계 또는 품질/이상 점수 합계로 읽습니다.")
    top_candidate_hint = "-"
    if not anomaly_candidate_view.empty:
        top_row = anomaly_candidate_view.iloc[0]
        top_candidate_hint = f"{top_row.get('이상유형', '-')} / {top_row.get('대표 후보', '-')} / {top_row.get('근거', '-')}"
    _rca_panel(
        "상위 이상 후보 해석",
        [
            "먼저 따라가야 할 후보를 상단에 둡니다.",
            "후보는 loss path 원천행이 아니라 이상유형별 대표 후보로 다시 묶습니다.",
            "rank는 단위가 다른 impact를 직접 비교하지 않고, 생산 손실 직접성, 국소 설비성, 품질 영향 우선순위로 정합니다.",
            "이 표는 결론이 아니라 드릴다운 시작점입니다.",
        ],
        [
            f"현재 가장 먼저 볼 후보는 {top_candidate_hint}입니다.",
            "이 후보가 어느 라인, 어느 설비, 어느 LOT까지 퍼지는지 아래 단계에서 좁힙니다.",
        ],
        tone="info",
    )
    if not anomaly_candidate_view.empty:
        rank_note = pd.DataFrame([
            {"rank 산정 기준": "1순위", "설명": "라인 손실에 직접 연결되는 생산성 이상을 우선 확인"},
            {"rank 산정 기준": "2순위", "설명": "특정 설비에 국소화된 설비 이상을 확인"},
            {"rank 산정 기준": "3순위", "설명": "품질/LOT 영향 후보를 확인"},
            {"rank 산정 기준": "해석 원칙", "설명": "단위가 다른 impact 값은 직접 비교하지 않고 이상유형별 대표 후보로 본다"},
        ])
        st.dataframe(rank_note, use_container_width=True, hide_index=True)
        candidate_cols = ["rank", "이상유형", "집중 축", "대표 후보", "근거", "우선순위 규칙", "다음 확인"]
        st.dataframe(anomaly_candidate_view[candidate_cols], use_container_width=True, hide_index=True)
    elif not card_summary.empty:
        routing_source = card_summary.head(5).copy()
        routing_source["이상유형"] = routing_source["card_key"].map({"when": "시간 이상", "where": "범위 이상", "how_much": "영향도 이상", "repeat": "반복 이상", "what": "원인 후보"}).fillna("이상")
        routing_source["후보명"] = routing_source["card_key"]
        routing_source["영향도"] = routing_source["headline_value"].astype(str)
        routing_source["rank 기준"] = routing_source["sub_label"].astype(str)
        routing_source["다음 확인"] = routing_source["detail_key"].astype(str)
        routing_cols = [c for c in ["card_key", "이상유형", "후보명", "headline_value", "sub_label", "영향도", "rank 기준", "다음 확인"] if c in routing_source.columns]
        st.dataframe(routing_source[routing_cols], use_container_width=True, hide_index=True)
    elif not proxy_candidates.empty:
        st.dataframe(proxy_candidates.head(5), use_container_width=True, hide_index=True)
    else:
        st.info("상위 이상 후보를 만들 데이터가 부족합니다.")

    st.markdown("#### 0단계. 이상 탐지")
    st.markdown("- 생산성 / 품질 / 설비 중 무엇이 먼저 깨졌는지 먼저 분리합니다.")
    if not stop.empty:
        total_stop = float(stop["duration_sec"].sum()) if "duration_sec" in stop.columns else 0.0
        total_count = int(stop["stop_count"].sum()) if "stop_count" in stop.columns else len(stop)
        aggregated_count = int((stop["stop_count"] > 1).sum()) if "stop_count" in stop.columns else 0
        event_count = max(len(stop) - aggregated_count, 0)
        macro_avg = float(stop[stop["stop_count"] > 1]["duration_sec"].mean()) if "stop_count" in stop.columns and not stop[stop["stop_count"] > 1].empty else 0.0
        micro_avg = float(stop[stop["stop_count"] <= 1]["duration_sec"].mean()) if "stop_count" in stop.columns and not stop[stop["stop_count"] <= 1].empty else 0.0
        dominant_label = "데이터 부족"
        if not loss_paths.empty:
            dominant_label = "생산성 이상"
        elif not hotspot.empty and "hotspot_type" in hotspot.columns:
            hotspot_types = hotspot["hotspot_type"].astype(str).value_counts()
            if hotspot_types.get("defect", 0) >= max(hotspot_types.get("machine", 0), hotspot_types.get("stop", 0)):
                dominant_label = "품질 이상"
            elif hotspot_types.get("machine", 0) > 0:
                dominant_label = "설비 이상"
        _rca_panel(
            "0단계 해석",
            [
                "정지 / 품질 / 설비 이상 중 어디가 먼저 깨졌는지 봅니다.",
                "총 정지, 평균 정지, 누적형/이벤트형 비중은 현재 상태를 빠르게 요약합니다.",
                "micro / macro 값은 정지 형태가 작은 이벤트인지 누적형인지 구분하는 보조 기준입니다.",
            ],
            [
                f"현재 정지 요약은 총 정지 {_fmt_sec(total_stop)}, {total_count:,}회입니다.",
                f"누적형 비중 {_safe_div(aggregated_count, len(stop)) * 100:.1f}%, 이벤트형 비중 {_safe_div(event_count, len(stop)) * 100:.1f}%입니다.",
                f"가장 먼저 의심해야 할 큰 축은 {dominant_label}입니다.",
            ],
            tone="warn",
        )
        st.markdown(
            f"- 정지 요약: 총 정지 {_fmt_sec(total_stop)}, {total_count:,}회, 누적형 {_safe_div(aggregated_count, len(stop)) * 100:.1f}%, 이벤트형 {_safe_div(event_count, len(stop)) * 100:.1f}%, 평균 정지 {_fmt_sec(_safe_div(total_stop, total_count or 1))} (micro {_fmt_sec(micro_avg)} / macro {_fmt_sec(macro_avg)})"
        )
    else:
        st.info("정지 감지에 사용할 데이터가 부족합니다.")

    st.dataframe(anomaly_candidate_view.rename(columns={"이상유형": "이상 유형"})[["이상 유형", "집중 축", "대표 후보", "근거", "다음 확인"]], use_container_width=True, hide_index=True)

    st.markdown("#### 1단계. 영향 범위 확인")
    st.markdown("- 어느 라인 / 공정 / 설비 / 모델 / LOT / 부품에 영향이 집중되는지 먼저 봅니다.")
    scope_frames = []
    if not drilldown.empty:
        scope_defs = [
            ("라인/공정", ["line_id", "stage_no"]),
            ("설비", ["line_id", "stage_no", "machine_id"]),
            ("모델/LOT", ["model_label", "lot_id"]),
            ("부품", [c for c in ["part_no", "part_number", "item_key"] if c in drilldown.columns]),
        ]
        for scope_name, cols in scope_defs:
            cols = [c for c in cols if c in drilldown.columns]
            if not cols:
                continue
            grp_cols = cols + [c for c in ["source_type", "metric_type", "impact_unit"] if c in drilldown.columns]
            scope_df = drilldown.groupby(grp_cols, as_index=False).agg(impact=("metric_value", "sum"), events=("metric_value", "size"))
            scope_df["영역"] = scope_name
            scope_df["대상"] = scope_df.apply(lambda r: " / ".join(str(r.get(c, "-")) for c in cols), axis=1)
            scope_df["측정기준"] = scope_df.apply(lambda r: f"{r.get('metric_type', '-') } / {r.get('impact_unit', '-')}", axis=1)
            scope_frames.append(scope_df[["영역", "대상", "source_type", "metric_type", "impact_unit", "측정기준", "impact", "events"]].sort_values(["impact", "events"], ascending=False).head(5))
    if scope_frames:
        scope_summary = pd.concat(scope_frames, ignore_index=True, sort=False)
        scope_view = scope_summary.rename(columns={"impact": "impact", "events": "events"})
        if not scope_view.empty:
            scope_view["우선순위 점수"] = (
                pd.to_numeric(scope_view["impact"], errors="coerce").rank(pct=True, ascending=False).fillna(0) * 0.6
                + pd.to_numeric(scope_view["events"], errors="coerce").rank(pct=True, ascending=False).fillna(0) * 0.4
            )
            scope_view = scope_view.sort_values(["우선순위 점수", "impact", "events"], ascending=[False, False, False]).reset_index(drop=True)
            scope_view["우선순위"] = np.arange(1, len(scope_view) + 1)
            scope_top = scope_view.iloc[0]
            _rca_panel(
                "1단계 해석",
                [
                    "라인 / 공정 / 설비 / 모델 / LOT / 부품 중 어디에 몰리는지 봅니다.",
                    "impact와 event count가 큰 축을 먼저 찾되, 측정기준이 같은 행끼리만 비교합니다.",
                    "표는 우선순위 점수 기준으로 먼저 정렬합니다.",
                    "한 곳에 몰리면 국소 문제, 여러 곳에 퍼지면 전파 문제로 봅니다.",
                ],
                [
                    f"현재 가장 큰 범위는 {scope_top.get('영역', '-')} / {scope_top.get('대상', '-')}, impact {scope_top.get('impact', 0):.0f} {scope_top.get('impact_unit', '-')}, events {scope_top.get('events', 0)}건입니다.",
                    f"이 값은 {scope_top.get('측정기준', '-')} 기준이므로 다른 단위와 직접 비교하면 안 됩니다.",
                ],
                tone="accent",
            )
        c1, c2 = st.columns([0.62, 0.38])
        with c1:
            scope_cols = [c for c in ["우선순위", "영역", "대상", "측정기준", "impact", "events", "우선순위 점수"] if c in scope_view.columns]
            st.dataframe(scope_view[scope_cols].head(12), use_container_width=True, hide_index=True)
        with c2:
            plot_scope = scope_view.head(10).copy()
            fig_scope = px.bar(plot_scope, x="대상", y="impact", color="측정기준", text="impact", facet_row="영역")
            fig_scope.update_layout(yaxis_title="impact")
            st.plotly_chart(_plot_style(fig_scope, "영향 범위 상위"), use_container_width=True)
    else:
        st.info("라인 / 공정 / 설비 / 모델 / LOT 범위를 만들 데이터가 부족합니다.")

    st.markdown("#### 2단계. 원인 도메인 분해")
    st.markdown("- Stop / Placement / Quality 도메인으로 나누고, 에러가 어디에서 시작됐는지 봅니다.")
    _rca_panel(
        "2단계 해석",
        [
            "원인을 Stop / Placement / Quality로 먼저 나눕니다.",
            "정지계열은 stop 원인, Placement는 픽업/인식/장착 분석에 집중하고, Quality는 검사/불량으로 봅니다.",
            "도메인을 맞힌 뒤 세부 조합을 봐야 조치가 정확합니다.",
        ],
        [
            "이 구간은 문제가 정지인지, 픽업/인식/장착인지, 품질인지 분류하는 단계입니다.",
            "도메인이 달라지면 담당 부서와 조치 방식도 달라집니다.",
        ],
        tone="warn",
    )
    domain_rows = []
    if not stop.empty:
        reason_col = "stop_like_reason" if "stop_like_reason" in stop.columns else "stop_reason_code" if "stop_reason_code" in stop.columns else None
        if reason_col:
            stop_summary = stop.groupby(reason_col, as_index=False).agg(impact=("duration_sec", "sum"), events=("stop_count", "sum") if "stop_count" in stop.columns else (reason_col, "size")).sort_values(["impact", "events"], ascending=False)
            if not stop_summary.empty:
                top_stop = stop_summary.iloc[0]
                domain_rows.append({
                    "도메인": "Stop",
                    "대표 후보": str(top_stop.get(reason_col, "-")),
                    "기준": "정지시간 합계(sec)",
                    "현재 해석": f"impact {top_stop.get('impact', 0):.0f} sec / event count {top_stop.get('events', 0):.0f}건",
                    "다음 확인": "line / stage / machine별 정지 집중",
                })
    if not comp.empty:
        comp_summary = comp.copy()
        if "error_rate" not in comp_summary.columns:
            comp_summary["error_rate"] = comp_summary.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
        comp_summary = comp_summary.sort_values(["error_rate", "error_count"], ascending=False)
        if not comp_summary.empty:
            top_place = comp_summary.iloc[0]
            domain_rows.append({
                "도메인": "Placement",
                "대표 후보": " / ".join([str(top_place.get(c, "-")) for c in ["part_number", "feeder_id", "nozzle_serial"] if c in comp_summary.columns]),
                "기준": "error_rate = error_count / pickup_count",
                "현재 해석": f"error_rate {top_place.get('error_rate', 0):.2%} / error_count {top_place.get('error_count', 0):.0f}",
                "다음 확인": "반복 LOT와 반복 설비 여부",
            })
    quality_hotspot = hotspot[hotspot["hotspot_type"].eq("defect")] if not hotspot.empty and "hotspot_type" in hotspot.columns else pd.DataFrame()
    if not quality_hotspot.empty:
        top_quality = quality_hotspot.iloc[0]
        domain_rows.append({
            "도메인": "Quality",
            "대표 후보": f"{top_quality.get('machine_id', '-')} / {top_quality.get('model_label', '-')}",
            "기준": "불량 건수(count)",
            "현재 해석": f"impact {top_quality.get('impact', 0):.0f} count / event count {top_quality.get('events', 0):.0f}건",
            "다음 확인": "불량 모델과 LOT 집중",
        })
    if domain_rows:
        st.markdown("##### 2.0 도메인 요약")
        st.dataframe(pd.DataFrame(domain_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Stop / Placement / Quality 도메인을 나눌 원천 데이터가 부족합니다.")
    st.markdown("##### 2.1 에러 / 피더 정보")
    st.markdown("- 영향이 시작된 부품, 피더, 노즐 조합을 먼저 확인합니다.")
    feeder_like = comp.copy()
    if not feeder_like.empty:
        feeder_cols = [c for c in ["machine_id", "part_number", "feeder_id", "feeder_serial", "nozzle_serial", "lot_id", "error_count", "pickup_count"] if c in feeder_like.columns]
        if feeder_cols:
            feeder_like["error_rate"] = feeder_like.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1) if "error_rate" not in feeder_like.columns else feeder_like["error_rate"]
            feeder_summary = feeder_like[feeder_cols + (["error_rate"] if "error_rate" not in feeder_cols else [])].copy()
            feeder_summary = feeder_summary.sort_values(["error_rate", "error_count", "pickup_count"], ascending=False)
            top_feeder = feeder_summary.iloc[0]
            _rca_panel(
                "2.1 해석",
                [
                    "에러율이 높은 부품·피더·노즐 조합부터 봅니다.",
                    "같은 조합이 반복되면 자재, 피더, 노즐 쪽을 먼저 의심합니다.",
                    "같은 LOT에서만 높으면 자재 편중, 여러 LOT에 걸치면 장비 조건 문제 가능성이 큽니다.",
                ],
                [
                    f"현재 가장 높은 조합은 {top_feeder.get('part_number', '-')} / {top_feeder.get('feeder_id', '-')} / {top_feeder.get('nozzle_serial', '-') }입니다.",
                    f"error_rate {top_feeder.get('error_rate', 0):.2%}, error_count {top_feeder.get('error_count', 0):.0f}, pickup_count {top_feeder.get('pickup_count', 0):.0f}건입니다.",
                    "이 조합이 여러 LOT에서 반복되는지 확인해야 합니다.",
                ],
                tone="accent",
            )
            st.dataframe(feeder_summary.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("에러 / 피더 정보로 좁힐 수 있는 데이터가 부족합니다.")
    else:
        st.info("에러 / 피더 정보로 좁힐 수 있는 데이터가 부족합니다.")

    st.markdown("##### 2.2 피더 / 노즐 핫스팟")
    st.markdown("- 반복되는 부품·장치 조합을 찾아 같은 패턴인지 확인합니다.")
    if not feeder_like.empty:
        hotspot_cols = [c for c in ["part_number", "feeder_id", "feeder_serial", "nozzle_serial", "machine_id"] if c in feeder_like.columns]
        if hotspot_cols:
            top_components = feeder_like.groupby(hotspot_cols, as_index=False).agg(error_count=("error_count", "sum"), pickup_count=("pickup_count", "sum"))
            top_components["error_rate"] = top_components.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
            top_components = top_components.sort_values(["error_rate", "error_count", "pickup_count"], ascending=False).head(10)
            top_comp = top_components.iloc[0]
            _rca_panel(
                "2.2 해석",
                [
                    "자주 문제되는 조합을 찾습니다.",
                    "error_rate가 높고 반복되는 조합은 우선 점검 대상입니다.",
                    "분산이 크면 특정 LOT에 편중된 문제인지도 같이 확인합니다.",
                ],
                [
                    f"현재 가장 높은 조합은 {top_comp.get('part_number', '-')} / {top_comp.get('feeder_id', '-')} / {top_comp.get('nozzle_serial', '-') }입니다.",
                    f"error_rate {top_comp.get('error_rate', 0):.2%}, error_count {top_comp.get('error_count', 0):.0f}건입니다.",
                    "같은 조합이 여러 설비나 여러 LOT에서 반복되면 구조적 문제로 봐야 합니다.",
                ],
                tone="info",
            )
            st.dataframe(top_components, use_container_width=True, hide_index=True)
            if "part_number" in feeder_like.columns and "lot_id" in feeder_like.columns:
                part_lot = feeder_like.groupby(["part_number", "lot_id"], as_index=False).agg(error_count=("error_count", "sum"), pickup_count=("pickup_count", "sum"))
                part_lot["error_rate"] = part_lot.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
                part_variance = part_lot.groupby("part_number")["error_rate"].var().reset_index(name="variance").dropna()
                if not part_variance.empty:
                    top_part = part_variance.sort_values("variance", ascending=False).iloc[0]
                    top_lot_row = part_lot[part_lot["part_number"] == top_part["part_number"]].sort_values("error_rate", ascending=False).head(1)
                    lot_label = top_lot_row.iloc[0]["lot_id"] if not top_lot_row.empty else "Unknown"
                    st.markdown(f"- `{top_part['part_number']}` LOT {lot_label}의 error_rate 분산 {top_part['variance']:.4f} → 특정 LOT 편중 가능성")
        else:
            st.info("피더 / 노즐 / 부품 핫스팟을 찾을 데이터가 부족합니다.")

    st.markdown("#### 3단계. 설비-자재-품질 교차검증")
    st.markdown("- 픽업 오류와 정지는 함께 움직이는지, 같은 설비 / 모델 / LOT에서 반복되는지 봅니다.")
    st.markdown("- 숫자 상관계수보다 동시간대 동반 발생 비중과 반복도를 우선 봅니다.")
    if "machine_id" in feeder_like.columns and not stop.empty and "machine_id" in stop.columns:
        corr_df = feeder_like.copy()
        corr_df["pickup_count"] = pd.to_numeric(corr_df.get("pickup_count", 0), errors="coerce").fillna(0)
        corr_df["error_count"] = pd.to_numeric(corr_df.get("error_count", 0), errors="coerce").fillna(0)
        corr_df = corr_df.groupby("machine_id", as_index=False).agg(pickup_error_count=("error_count", "sum"), pickup_count=("pickup_count", "sum"))
        corr_df["rate"] = corr_df.apply(lambda r: _safe_div(r.get("pickup_error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
        stops_per_machine = stop.groupby("machine_id", as_index=False)["duration_sec"].sum()
        total_stop = float(stops_per_machine["duration_sec"].sum()) or 1.0
        stops_per_machine["stop_share"] = stops_per_machine["duration_sec"] / total_stop
        corr_df = corr_df.merge(stops_per_machine[["machine_id", "stop_share"]], on="machine_id", how="left").fillna(0)
        reason_col = "stop_reason_group" if "stop_reason_group" in stop.columns else "stop_like_reason" if "stop_like_reason" in stop.columns else "stop_reason_code"
        if reason_col in stop.columns:
            stop_group = stop.groupby(["machine_id", reason_col], as_index=False)["duration_sec"].sum()
            if not stop_group.empty:
                idx = stop_group.groupby("machine_id")["duration_sec"].idxmax().dropna()
                stop_group = stop_group.loc[idx]
                corr_df = corr_df.merge(stop_group[["machine_id", reason_col]], on="machine_id", how="left")
                if reason_col != "stop_reason_group":
                    corr_df = corr_df.rename(columns={reason_col: "stop_reason_group"})
        corr_df["co_occurrence_rate"] = corr_df.apply(lambda r: min(float(r.get("rate", 0)), float(r.get("stop_share", 0))), axis=1)
        corr_df["repeat_ratio"] = corr_df.apply(lambda r: _safe_div(r.get("pickup_error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
        corr_df = corr_df.sort_values(["co_occurrence_rate", "stop_share"], ascending=False)
        if not corr_df.empty:
            rate_max = max(float(corr_df["rate"].max()), 0.01)
            stop_max = max(float(corr_df["stop_share"].max()), 0.01)
            top_corr = corr_df.iloc[0]
            _rca_panel(
                "3단계 해석",
                [
                    "픽업 오류와 정지가 같이 움직이는지 봅니다.",
                    "상관계수보다 동시간대 동반 발생 비중과 반복도를 먼저 봅니다.",
                    "같은 설비, 같은 모델, 같은 LOT에서 반복되면 구조적 문제 가능성이 큽니다.",
                ],
                [
                    f"현재 가장 눈에 띄는 설비는 {top_corr.get('machine_id', '-')}입니다.",
                    f"pickup error rate {top_corr.get('rate', 0):.2%}, stop share {top_corr.get('stop_share', 0):.2%}, co-occurrence {top_corr.get('co_occurrence_rate', 0):.2%}입니다.",
                    "이 설비는 픽업 오류와 정지가 함께 움직이는지 추가 확인이 필요합니다.",
                ],
                tone="accent",
            )
            fig_corr = px.scatter(
                corr_df,
                x="rate",
                y="stop_share",
                size="pickup_count",
                color="stop_reason_group" if "stop_reason_group" in corr_df.columns else None,
                template=DARK_TEMPLATE,
                title="Pickup vs Stop Correlation",
                hover_name="machine_id",
                hover_data={"pickup_count": True, "pickup_error_count": True, "rate": ":.1%", "stop_share": ":.1%"},
            )
            fig_corr.update_xaxes(range=[0, rate_max * 1.15], tickformat=".0%")
            fig_corr.update_yaxes(range=[0, stop_max * 1.15], tickformat=".0%")
            st.plotly_chart(_plot_style(fig_corr, "픽업 오류와 정지 동반 발생", 360), use_container_width=True)
            st.dataframe(
                corr_df[[c for c in ["machine_id", "pickup_error_count", "pickup_count", "rate", "stop_share", "co_occurrence_rate", "stop_reason_group"] if c in corr_df.columns]].head(10),
                use_container_width=True,
                hide_index=True,
            )
        if not drilldown.empty:
            cross_cols = [c for c in ["line_id", "stage_no", "machine_id", "model_label", "lot_id"] if c in drilldown.columns]
            if cross_cols:
                cross_df = drilldown.groupby(cross_cols, as_index=False).agg(impact=("metric_value", "sum"), event_count=("metric_value", "size")).sort_values(["impact", "event_count"], ascending=False)
                st.dataframe(cross_df.head(10), use_container_width=True, hide_index=True)
    else:
        st.info("픽업 오류와 정지 상관을 만들 데이터가 부족합니다.")

    st.markdown("#### 4단계. 반복성 / 재현성 확인")
    st.markdown("- 같은 시간대, 같은 설비, 같은 LOT, 같은 부품 조합에서 반복되는지 확인합니다.")
    if not repeat.empty:
        repeat_view = repeat.rename(columns={"impact": "impact", "events": "events"})
        top_repeat = repeat_view.iloc[0]
        _rca_panel(
            "4단계 해석",
            [
                "같은 문제가 반복되는지를 봅니다.",
                "반복되면 일회성보다 구조적 문제로 봅니다.",
                "시간대, 설비, LOT, 부품 중 무엇이 반복되는지 잡아야 조치가 정확해집니다.",
            ],
            [
                f"현재 가장 반복되는 패턴은 {top_repeat.get('pattern', '-')}입니다.",
                f"impact {top_repeat.get('impact', 0):.0f}, events {top_repeat.get('events', 0)}건입니다.",
                "이 패턴이 동일 shift, 동일 LOT, 동일 부품에서 반복되는지 확인해야 합니다.",
            ],
            tone="warn",
        )
        fig_repeat = px.bar(repeat_view.head(10), x="pattern", y="impact", color="cause_group" if "cause_group" in repeat_view.columns else None, text="impact")
        fig_repeat.update_layout(yaxis_title="impact")
        st.plotly_chart(_plot_style(fig_repeat, "반복: 재발 패턴"), use_container_width=True)
        st.dataframe(repeat_view[[c for c in ["path_key", "pattern", "machine_id", "cause_group", "cause_detail", "hour", "lot_id", "events", "impact"] if c in repeat_view.columns]].head(15), use_container_width=True, hide_index=True)
    else:
        st.info("반복성 / 재현성 패턴을 만들 데이터가 부족합니다.")

    selected = None
    active_key = None
    if not loss_paths.empty or not card_summary.empty or not proxy_candidates.empty:
        st.markdown("#### 5단계. 조치 우선순위")
        if not loss_paths.empty:
            top_priority = loss_paths.iloc[0]
            _rca_panel(
                "5단계 해석",
                [
                    "무엇부터 할지 순서를 정합니다.",
                    "반복도, 영향 크기, 확산 범위를 같이 봅니다.",
                    "고객 설명에서는 이걸 먼저 하면 손실이 가장 빨리 줄어든다고 말하면 됩니다.",
                ],
                [
                    f"현재 우선순위 1위는 {top_priority.get('path_key', '-')}입니다.",
                    f"impact {top_priority.get('impact', 0):.0f}, events {top_priority.get('events', 0)}건, formula {top_priority.get('impact_formula', '-') }입니다.",
                    "이 항목이 실제로 반복되는지, 여러 LOT로 퍼지는지 확인해야 합니다.",
                ],
                tone="accent",
            )
            path_options = ["전체"] + (loss_paths["path_key"].astype(str).tolist() if "path_key" in loss_paths.columns else [f"rank {int(r.get('rank', i + 1))}" for i, (_, r) in enumerate(loss_paths.iterrows())])
            selected_key = st.selectbox("손실경로 선택", path_options, index=0, key="rca_loss_path_select")
            active_key = None if selected_key == "전체" else str(selected_key)
            selected_rows = loss_paths.head(1) if active_key is None else (loss_paths[loss_paths["path_key"].astype(str).eq(active_key)] if "path_key" in loss_paths.columns else loss_paths.head(1))
            selected = selected_rows.iloc[0] if not selected_rows.empty else loss_paths.iloc[0]
            st.markdown(f"- 현재 드릴다운 대상: `{selected.get('path_key', '-')}`")
            c1, c2, c3, c4, c5 = st.columns(5)
            cards = [
                ("언제", str(selected.get("when", "-")), f"rank {int(selected.get('rank', 0))}"),
                ("어디", str(selected.get("where", "-")), f"path {selected.get('machine_id', '-') }"),
                ("얼마나", str(selected.get("how_much", "-")), f"{selected.get('impact_formula', '정지시간 합계(sec)')} / {selected.get('impact', 0):.0f} {selected.get('impact_unit', '')}".strip()),
                ("반복", str(selected.get("repeat", "-")), f"score {selected.get('repeat_score', 0):.2f}"),
                ("무엇", str(selected.get("what", "-")), f"{selected.get('cause_group', 'Proxy')}"),
            ]
            for col, (label, value, foot) in zip([c1, c2, c3, c4, c5], cards):
                with col:
                    st.markdown(_card(label, value, foot), unsafe_allow_html=True)
            st.markdown("##### 손실경로 우선순위")
            table_view = loss_paths.head(10).rename(columns={"impact": "impact", "events": "events", "repeat_score": "repeat_score"})
            display_cols = [c for c in ["rank", "metric_type", "event_group", "cause_family", "path_key", "when", "where", "how_much", "impact_formula", "repeat", "what", "impact", "events"] if c in table_view.columns]
            st.dataframe(table_view[display_cols], use_container_width=True, hide_index=True)
        elif not card_summary.empty:
            st.markdown("- 손실경로가 부족해 후보 카드로 우선순위를 봅니다.")
            _render_card_row(card_summary.sort_values("card_key") if not card_summary.empty else card_summary)
        else:
            st.dataframe(proxy_candidates.head(10), use_container_width=True, hide_index=True)

    if active_key is not None:
        if "path_key" in drilldown.columns:
            drilldown = drilldown[drilldown["path_key"].astype(str).eq(active_key)].copy()
        if "path_key" in repeat.columns:
            repeat = repeat[repeat["path_key"].astype(str).eq(active_key)].copy()
        if "path_key" in hotspot.columns:
            hotspot = hotspot[hotspot["path_key"].astype(str).eq(active_key)].copy()
    st.markdown("#### 6단계. 카드 상세 연결")
    if not timeline.empty:
        if active_key is not None and "path_key" in timeline.columns:
            tl = timeline[timeline["path_key"].astype(str).eq(active_key)].copy()
        else:
            tl = timeline.copy()
        if not tl.empty:
            tl_top = tl.groupby(["hour"], as_index=False).agg(metric_value=("metric_value", "sum")).sort_values("metric_value", ascending=False).iloc[0]
            _rca_panel(
                "6단계 해석",
                [
                    "선택한 후보가 시간대별로 어떻게 변하는지 봅니다.",
                    "여기서 추세가 보이면 우연이 아니라 반복 패턴으로 설명할 수 있습니다.",
                    "상세 표는 고객에게 원인 후보를 뒷받침하는 증거입니다.",
                ],
                [
                    f"현재 가장 강한 시간대는 {int(tl_top.get('hour', 0)):02d}시이며 metric_value {tl_top.get('metric_value', 0):.0f}입니다.",
                    "같은 시간대가 반복되면 교대, 자재보급, 셋업 시점을 의심할 수 있습니다.",
                ],
                tone="info",
            )
            st.plotly_chart(_plot_style(px.line(tl.groupby(["hour"], as_index=False).agg(metric_value=("metric_value", "sum")), x="hour", y="metric_value", markers=True), "언제: 시간대별 추이"), use_container_width=True)
        else:
            st.info("선택 손실경로에 대한 시간대 추이를 만들 수 있는 데이터가 부족합니다.")
    else:
        st.info("시간대 추이를 만들 수 있는 데이터가 부족합니다.")
    if not hotspot.empty:
        machine_hotspot = hotspot[hotspot["hotspot_type"].eq("machine")] if "hotspot_type" in hotspot.columns else hotspot
        if not machine_hotspot.empty:
            machine_plot = machine_hotspot.copy().head(10)
            if "line_id" in machine_plot.columns:
                machine_plot["대상"] = machine_plot.apply(lambda r: f"{r.get('machine_id', '-')}\n{r.get('line_id', '-')}", axis=1)
            else:
                machine_plot["대상"] = machine_plot["machine_id"].astype(str)
            machine_view = machine_plot.rename(columns={"impact": "impact"})
            fig_machine = px.bar(machine_view, x="대상", y="impact", text="impact")
            fig_machine.update_layout(yaxis_title="impact")
            st.plotly_chart(_plot_style(fig_machine, "어디: hotspot 설비/라인"), use_container_width=True)
        if "hotspot_type" in hotspot.columns and "defect" in hotspot["hotspot_type"].astype(str).unique():
            defect_hotspot = hotspot[hotspot["hotspot_type"].eq("defect")]
            if not defect_hotspot.empty:
                defect_plot = defect_hotspot.head(10).copy()
                if "model_label" in defect_plot.columns:
                    defect_plot["대상"] = defect_plot.apply(lambda r: f"{r.get('machine_id', '-')}\n{r.get('model_label', '-')}", axis=1)
                else:
                    defect_plot["대상"] = defect_plot["machine_id"].astype(str)
                defect_view = defect_plot.rename(columns={"impact": "impact"})
                fig_defect = px.bar(defect_view, x="대상", y="impact", text="impact")
                fig_defect.update_layout(yaxis_title="impact")
                st.plotly_chart(_plot_style(fig_defect, "어디: defect hotspot"), use_container_width=True)
    if not card_summary.empty and "how_much" in card_summary["card_key"].values:
        hm = card_summary.loc[card_summary["card_key"].eq("how_much")].iloc[0]
        st.markdown(f"- 얼마나: `{hm['headline_value']}` · {hm['evidence']}")
        scale_df = drilldown.copy()
        if not scale_df.empty:
            scale_df = scale_df.groupby(["source_type"], as_index=False).agg(impact=("metric_value", "sum"))
            fig_scale = px.bar(scale_df, x="source_type", y="impact", text="impact")
            fig_scale.update_layout(yaxis_title="impact")
            st.plotly_chart(_plot_style(fig_scale, "영향 규모 분포"), use_container_width=True)
    if not hotspot.empty:
        cause_view = hotspot[hotspot["cause_group"].notna()] if "cause_group" in hotspot.columns else hotspot
        if not cause_view.empty:
            if {"cause_group", "cause_detail"}.issubset(cause_view.columns):
                cause_plot = (
                    cause_view.groupby(["cause_group", "cause_detail"], as_index=False)
                    .agg(impact=("impact", "sum"), events=("events", "sum"))
                    .sort_values(["impact", "events"], ascending=False)
                )
                cause_plot["대상"] = cause_plot.apply(lambda r: f"{r.get('cause_group', '-')}\n{r.get('cause_detail', '-')}", axis=1)
            else:
                cause_plot = cause_view.head(10).copy()
                cause_plot["대상"] = cause_plot.get("cause_detail", pd.Series(["-"] * len(cause_plot), index=cause_plot.index)).astype(str)
            fig_cause = px.bar(cause_plot.head(10), x="대상", y="impact", text="impact")
            fig_cause.update_layout(yaxis_title="impact")
            st.plotly_chart(_plot_style(fig_cause, "무엇: cause_group / cause_detail"), use_container_width=True)
            st.dataframe(cause_plot[[c for c in ["cause_group", "cause_detail", "impact", "events"] if c in cause_plot.columns]].head(15), use_container_width=True, hide_index=True)

    st.markdown("#### 드릴다운 테이블")
    if not drilldown.empty:
        drill_top = drilldown.iloc[0]
        _rca_panel(
            "드릴다운 해석",
            [
                "행 단위 이벤트를 확인합니다.",
                "결론보다 증거 확인에 초점을 둡니다.",
                "선택한 후보가 실제로 어디에서 발생했는지 마지막으로 검증합니다.",
            ],
            [
                f"가장 먼저 볼 행은 {drill_top.get('line_id', '-')} / Stage {drill_top.get('stage_no', '-')} / {drill_top.get('machine_id', '-') }입니다.",
                f"metric_value {drill_top.get('metric_value', 0):.0f}, source {drill_top.get('source_type', '-') }입니다.",
            ],
            tone="info",
        )
        st.dataframe(drilldown[[c for c in ["event_ts", "day", "hour", "source_type", "line_id", "stage_no", "machine_id", "lot_id", "model_label", "cause_group", "cause_detail", "result_primary", "quality_flag", "metric_value", "path_key"] if c in drilldown.columns]].head(50), use_container_width=True, hide_index=True)
    else:
        st.info("드릴다운용 상세 데이터가 충분하지 않습니다.")

    st.markdown("#### 7단계. 최종 조치 제안")
    st.markdown("- 마지막에는 원인 후보를 실행 조치로 바꿔서 다음 행동을 정합니다.")
    action_rows = []
    if selected is not None:
        selected_text = " ".join(str(selected.get(k, "")) for k in ["what", "where", "cause_group", "cause_detail"])
        action_rows.append({
            "우선순위": 1,
            "대상": str(selected.get("path_key", selected.get("machine_id", "-"))),
            "해석": "대표 원인 후보",
            "추천 조치": _reason_action_hint(selected_text),
            "다음 확인": str(selected.get("where", "대표 설비/공정")),
            "담당팀": "설비 + 생산",
            "추가 데이터": "상세 stop log / feeder log / same lot 비교",
        })
    if not repeat.empty:
        top_repeat = repeat.sort_values("impact", ascending=False).iloc[0]
        repeat_text = " ".join(str(top_repeat.get(k, "")) for k in ["pattern", "cause_group", "cause_detail"])
        action_rows.append({
            "우선순위": len(action_rows) + 1,
            "대상": str(top_repeat.get("pattern", "repeat")),
            "해석": "반복 발생 패턴",
            "추천 조치": _reason_action_hint(repeat_text),
            "다음 확인": str(top_repeat.get("machine_id", "반복 발생 설비")),
            "담당팀": "설비 + 공정",
            "추가 데이터": "동일 shift / 동일 LOT / 동일 model 비교",
        })
    if not hotspot.empty:
        cause_view = hotspot[hotspot["cause_group"].notna()] if "cause_group" in hotspot.columns else hotspot
        if not cause_view.empty:
            if {"cause_group", "cause_detail"}.issubset(cause_view.columns):
                cause_action = (
                    cause_view.groupby(["cause_group", "cause_detail"], as_index=False)
                    .agg(impact=("impact", "sum"), events=("events", "sum"))
                    .sort_values(["impact", "events"], ascending=False)
                )
            else:
                cause_action = cause_view.head(3).copy()
                if "impact" not in cause_action.columns:
                    cause_action["impact"] = 0
            for rank, (_, row) in enumerate(cause_action.head(3).iterrows(), start=1):
                cause_text = " ".join(str(row.get(k, "")) for k in ["cause_group", "cause_detail"])
                stage_label = "우선 조치 대상" if rank == 1 else "동반 확인 대상" if rank == 2 else "보조 확인 대상"
                action_rows.append({
                    "우선순위": len(action_rows) + 1,
                    "대상": f"{row.get('cause_group', '-')}\n{row.get('cause_detail', '-')}",
                    "해석": stage_label,
                    "추천 조치": _reason_action_hint(cause_text),
                    "다음 확인": "관련 설비 / 부품 / LOT 재확인",
                    "담당팀": "설비 + 품질",
                    "추가 데이터": "관련 설비 로그 / 부품 lot / 타임라인",
                })
                if len(action_rows) >= 5:
                    break
    if action_rows:
        action_df = pd.DataFrame(action_rows)
        action_df["조치 수준"] = action_df["우선순위"].apply(lambda v: "최우선" if v == 1 else "우선" if v == 2 else "보조")
        top_action = action_df.iloc[0]
        _rca_panel(
            "7단계 해석",
            [
                "분석 결과를 실행 조치로 바꿉니다.",
                "누가, 무엇을, 어떤 순서로 확인해야 하는지까지 적어야 바로 움직일 수 있습니다.",
                "추천 조치는 코드 이름보다 현장 행동으로 읽히는 것이 중요합니다.",
            ],
            [
                f"현재 가장 먼저 할 조치는 {top_action.get('대상', '-')}입니다.",
                f"해석은 {top_action.get('해석', '-')}이며, 추천 조치는 {top_action.get('추천 조치', '-')}입니다.",
                "이 조치는 impact와 반복성, 확산 범위를 함께 보고 정한 우선순위입니다.",
            ],
            tone="accent",
        )
        action_cols = [c for c in ["우선순위", "대상", "해석", "추천 조치", "다음 확인", "담당팀", "추가 데이터", "조치 수준"] if c in action_df.columns]
        st.dataframe(action_df[action_cols], use_container_width=True, hide_index=True)
        st.markdown("##### 조치 해석")
        st.markdown("- `nozzle / feeder / reel check`면 흡착·공급 경로를 먼저 봅니다.")
        st.markdown("- `vision / camera / lighting tuning`이면 인식 조건을 먼저 봅니다.")
        st.markdown("- `upstream balance / feeder timing`이면 전공정 흐름부터 봅니다.")
        st.markdown("- `downstream congestion / buffer`이면 후공정 적체를 먼저 봅니다.")
        st.markdown("- `conveyor / interlock issue`이면 이송·인터락을 먼저 봅니다.")
    else:
        st.info("최종 조치 제안을 만들 수 있는 데이터가 부족합니다.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_rca(clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame], sample_mode: bool):
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    comp = clean.get("vw_component_error_fact", pd.DataFrame())
    filters = _build_filter_panel(clean, "rca")
    filtered_clean = {
        **clean,
        "vw_shopfloor_event_fact": _apply_selection(shop, filters),
        "vw_stop_event_fact": _apply_selection(stop, filters),
        "vw_inspection_event_fact": _apply_selection(insp, filters),
        "vw_tag_event_fact": _apply_selection(tag, filters),
        "vw_component_error_fact": _apply_selection(comp, filters),
    }
    stop = filtered_clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    insp = filtered_clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    comp = filtered_clean.get("vw_component_error_fact", pd.DataFrame()).copy()
    meta = filtered_clean.get("_meta", {}) if isinstance(filtered_clean.get("_meta", {}), dict) else {}

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("### 경로분석(MOUNT)")
    st.caption("추적할 가치가 높은 TOP 5 경로를 먼저 고르고, 선정 이유를 설명한 뒤, 대표 경로 1개를 시나리오로 해석합니다.")
    if sample_mode:
        st.info("기준 시나리오: 2026-03-24 14:00~18:00 / Line-1 / Stage-1 / M05 / LOT002 / FDR-5 / PN-0004 / NOZ204")

    if comp.empty and stop.empty and insp.empty and tag.empty:
        st.warning("경로분석에 사용할 데이터가 부족합니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for df in [comp, stop, insp]:
        if not df.empty and "event_ts" in df.columns:
            df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")
            df["time_bucket"] = df["event_ts"].dt.floor("H")
            df["day"] = df["event_ts"].dt.date

    if not comp.empty:
        for col in ["pickup_count", "error_count", "pickup_error_count"]:
            comp[col] = pd.to_numeric(comp.get(col, 0), errors="coerce").fillna(0)
        comp["error_rate"] = comp.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
        comp = comp.sort_values(["pickup_error_count", "error_rate", "error_count"], ascending=False).reset_index(drop=True)
    if not stop.empty:
        stop["duration_sec"] = pd.to_numeric(stop.get("duration_sec", 0), errors="coerce").fillna(0)
        stop["stop_count"] = pd.to_numeric(stop.get("stop_count", 0), errors="coerce").fillna(0)
    if not insp.empty and "quality_flag" in insp.columns:
        insp["is_fail"] = insp["quality_flag"].astype(str).str.upper().eq("FAIL").astype(int)

    top_comp = comp.iloc[0] if not comp.empty else pd.Series(dtype="object")
    primary_machine = str(top_comp.get("machine_id", meta.get("machine_id", "-")))
    primary_lot = str(top_comp.get("lot_id", meta.get("lot_id", "-")))
    primary_line = str(top_comp.get("line_id", top_comp.get("line", "Line-1")))
    primary_stage = int(pd.to_numeric(top_comp.get("stage_no", 1), errors="coerce")) if pd.notna(top_comp.get("stage_no", 1)) else 1
    primary_part = str(top_comp.get("part_number", meta.get("part_number", "-")))
    primary_feeder = str(top_comp.get("feeder_id", meta.get("feeder_id", "-")))
    primary_nozzle = str(top_comp.get("nozzle_serial", meta.get("nozzle_serial", "-")))

    stop_machine = stop[stop["machine_id"].astype(str).eq(primary_machine)].copy() if not stop.empty and "machine_id" in stop.columns else pd.DataFrame()
    comp_machine = comp[comp["machine_id"].astype(str).eq(primary_machine)].copy() if not comp.empty and "machine_id" in comp.columns else pd.DataFrame()
    quality_lot = insp[insp["lot_id"].astype(str).eq(primary_lot)].copy() if not insp.empty and "lot_id" in insp.columns else pd.DataFrame()

    peak_loss_bucket = None
    if not stop_machine.empty and "time_bucket" in stop_machine.columns:
        peak_row = stop_machine.groupby("time_bucket", as_index=False).agg(duration_sec=("duration_sec", "sum")).sort_values("duration_sec", ascending=False).head(1)
        if not peak_row.empty:
            peak_loss_bucket = peak_row.iloc[0]["time_bucket"]

    def _safe_bucket_label(ts) -> str:
        try:
            if pd.isna(ts):
                return "-"
            return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "-"

    def _score_alignment(row: pd.Series) -> float:
        score = 0.0
        if str(row.get("machine_id", "-")) == primary_machine:
            score += 0.40
        if str(row.get("lot_id", "-")) == primary_lot:
            score += 0.20
        if str(row.get("part_number", "-")) == primary_part:
            score += 0.10
        if str(row.get("cause_family", "")).lower() in {"pickup", "material", "quality_lag"}:
            score += 0.15
        bucket = row.get("first_seen_ts", pd.NaT)
        if peak_loss_bucket is not None and pd.notna(bucket):
            diff_hour = abs((pd.Timestamp(bucket) - pd.Timestamp(peak_loss_bucket)).total_seconds()) / 3600
            score += 0.15 if diff_hour <= 1 else 0.08 if diff_hour <= 2 else 0.0
        return min(score, 1.0)

    def _confidence(row: pd.Series) -> str:
        score = float(row.get("route_priority_score", 0))
        return "High" if score >= 0.78 else "Medium" if score >= 0.58 else "Low"

    route_rows = []
    if not comp.empty:
        mat = (
            comp.groupby(["line_id", "stage_no", "machine_id", "lot_id", "part_number", "feeder_id", "nozzle_serial"], as_index=False)
            .agg(
                impact_value=("error_count", "sum"),
                pickup_sum=("pickup_count", "sum"),
                repeat_count=("event_ts", "size"),
                unique_day_count=("day", "nunique"),
                unique_lot_count=("lot_id", "nunique"),
                affected_time_bucket_count=("time_bucket", "nunique"),
                first_seen_ts=("event_ts", "min"),
                last_seen_ts=("event_ts", "max"),
            )
        )
        mat["impact_value"] = mat.apply(lambda r: _safe_div(r.get("impact_value", 0), max(r.get("pickup_sum", 0), 1)), axis=1)
        mat["impact_unit"] = "rate"
        mat["metric_family"] = "pickup_error_rate"
        mat["event_group"] = "PLACEMENT"
        mat["cause_family"] = "pickup"
        mat["cause_detail"] = mat.apply(lambda r: f"{r.get('feeder_id', '-')}/{r.get('part_number', '-')}/{r.get('nozzle_serial', '-')}", axis=1)
        mat["path_id"] = mat.apply(lambda r: f"R-PICK-{r.get('machine_id', '-')}-{str(r.get('feeder_id', '-'))}", axis=1)
        route_rows.append(mat)
    if not stop.empty:
        stop_route = (
            stop.groupby(["line_id", "stage_no", "machine_id", "lot_id", "stop_like_reason"], as_index=False)
            .agg(
                impact_value=("duration_sec", "sum"),
                repeat_count=("stop_count", "sum"),
                unique_day_count=("day", "nunique"),
                unique_lot_count=("lot_id", "nunique"),
                affected_time_bucket_count=("time_bucket", "nunique"),
                first_seen_ts=("event_ts", "min"),
                last_seen_ts=("event_ts", "max"),
            )
        )
        stop_route["impact_unit"] = "sec"
        stop_route["metric_family"] = "stop_time"
        stop_route["event_group"] = "STOP"
        stop_route["cause_family"] = np.where(stop_route["stop_like_reason"].astype(str).str.contains("PICKUP", case=False, na=False), "pickup", "stop")
        stop_route["cause_detail"] = stop_route["stop_like_reason"].astype(str)
        stop_route["part_number"] = primary_part
        stop_route["feeder_id"] = primary_feeder
        stop_route["nozzle_serial"] = primary_nozzle
        stop_route["path_id"] = stop_route.apply(lambda r: f"R-STOP-{r.get('machine_id', '-')}-{r.get('stop_like_reason', '-')}", axis=1)
        route_rows.append(stop_route.drop(columns=["stop_like_reason"]))
    if not quality_lot.empty:
        q = (
            quality_lot.groupby(["lot_id"], as_index=False)
            .agg(
                impact_value=("is_fail", "sum"),
                repeat_count=("is_fail", "sum"),
                unique_day_count=("day", "nunique"),
                unique_lot_count=("lot_id", "nunique"),
                affected_time_bucket_count=("time_bucket", "nunique"),
                first_seen_ts=("event_ts", "min"),
                last_seen_ts=("event_ts", "max"),
            )
        )
        q["line_id"] = primary_line
        q["stage_no"] = primary_stage
        q["machine_id"] = "AOI01"
        q["impact_unit"] = "count"
        q["metric_family"] = "defect_count"
        q["event_group"] = "QUALITY"
        q["cause_family"] = "quality_lag"
        q["cause_detail"] = "AOI FAIL after pickup instability"
        q["part_number"] = primary_part
        q["feeder_id"] = primary_feeder
        q["nozzle_serial"] = primary_nozzle
        q["path_id"] = q.apply(lambda r: f"R-QUAL-{r.get('lot_id', '-')}", axis=1)
        route_rows.append(q)
    if not comp_machine.empty and not stop_machine.empty:
        merged = (
            comp_machine.groupby(["machine_id", "lot_id", "time_bucket"], as_index=False)
            .agg(pickup_error_count=("pickup_error_count", "sum"), error_count=("error_count", "sum"))
            .merge(
                stop_machine.groupby(["machine_id", "lot_id", "time_bucket"], as_index=False).agg(stop_time_sec=("duration_sec", "sum"), stop_events=("stop_count", "sum")),
                on=["machine_id", "lot_id", "time_bucket"],
                how="inner",
            )
        )
        if not merged.empty:
            link = merged.groupby(["machine_id", "lot_id"], as_index=False).agg(
                impact_value=("time_bucket", "size"),
                repeat_count=("time_bucket", "size"),
                unique_day_count=("time_bucket", lambda s: pd.to_datetime(s, errors="coerce").dt.date.nunique()),
                unique_lot_count=("lot_id", "nunique"),
                affected_time_bucket_count=("time_bucket", "nunique"),
                first_seen_ts=("time_bucket", "min"),
                last_seen_ts=("time_bucket", "max"),
                pickup_peak=("pickup_error_count", "max"),
                stop_peak=("stop_time_sec", "max"),
            )
            link["line_id"] = primary_line
            link["stage_no"] = primary_stage
            link["impact_unit"] = "buckets"
            link["metric_family"] = "route_link"
            link["event_group"] = "TRACE"
            link["cause_family"] = "pickup"
            link["cause_detail"] = "pickup error overlaps stop window"
            link["part_number"] = primary_part
            link["feeder_id"] = primary_feeder
            link["nozzle_serial"] = primary_nozzle
            link["path_id"] = link.apply(lambda r: f"R-LINK-{r.get('machine_id', '-')}-{r.get('lot_id', '-')}", axis=1)
            route_rows.append(link)
    if not stop_machine.empty and not quality_lot.empty:
        lag = (
            stop_machine.groupby("time_bucket", as_index=False).agg(stop_time_sec=("duration_sec", "sum"))
            .merge(quality_lot.groupby("time_bucket", as_index=False).agg(fail_count=("is_fail", "sum")), on="time_bucket", how="left")
            .fillna(0)
        )
        lag = lag[(lag["stop_time_sec"] > 0) | (lag["fail_count"] > 0)]
        if not lag.empty:
            lag_route = pd.DataFrame(
                [
                    {
                        "line_id": primary_line,
                        "stage_no": primary_stage,
                        "machine_id": primary_machine,
                        "lot_id": primary_lot,
                        "part_number": primary_part,
                        "feeder_id": primary_feeder,
                        "nozzle_serial": primary_nozzle,
                        "impact_value": float(lag["fail_count"].sum()),
                        "impact_unit": "count",
                        "metric_family": "quality_lag",
                        "event_group": "TRACE",
                        "cause_family": "quality_lag",
                        "cause_detail": "stop window followed by AOI fail",
                        "repeat_count": int((lag["stop_time_sec"] > 0).sum()),
                        "unique_day_count": int(pd.to_datetime(lag["time_bucket"], errors="coerce").dt.date.nunique()),
                        "unique_lot_count": 1,
                        "affected_time_bucket_count": int(lag["time_bucket"].nunique()),
                        "first_seen_ts": lag["time_bucket"].min(),
                        "last_seen_ts": lag["time_bucket"].max(),
                        "path_id": f"R-LAG-{primary_machine}-{primary_lot}",
                    }
                ]
            )
            route_rows.append(lag_route)

    route_df = pd.concat(route_rows, ignore_index=True, sort=False) if route_rows else pd.DataFrame()
    if route_df.empty:
        st.warning("Route candidate를 만들 수 있는 데이터가 부족합니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    def _route_meaning(row: pd.Series) -> str:
        metric_family = str(row.get("metric_family", ""))
        cause_family = str(row.get("cause_family", ""))
        if metric_family == "pickup_error_rate":
            return "Pickup error rate 상위 경로"
        if metric_family == "stop_time":
            return "Stop time 상위 경로"
        if metric_family == "defect_count":
            return "AOI fail count 상위 경로"
        if metric_family == "route_link":
            return "Pickup-stop 동시간대 중첩 경로"
        if metric_family == "quality_lag":
            return "Stop 이후 AOI fail 전이 경로"
        if cause_family == "pickup":
            return "Pickup 계열 추적 경로"
        return "대표 RCA 추적 경로"

    route_df["route_meaning"] = route_df.apply(_route_meaning, axis=1)
    route_df["where"] = route_df.apply(lambda r: f"{r.get('line_id', '-') } / Stage-{int(pd.to_numeric(r.get('stage_no', 0), errors='coerce')) if pd.notna(pd.to_numeric(r.get('stage_no', 0), errors='coerce')) else '-'} / {r.get('machine_id', '-')}", axis=1)
    route_df["when"] = route_df.apply(lambda r: f"{_safe_bucket_label(r.get('first_seen_ts'))} ~ {_safe_bucket_label(r.get('last_seen_ts'))}", axis=1)
    route_df["main_symptom"] = route_df.apply(lambda r: f"{r.get('event_group', '-')} / {r.get('cause_family', '-')} / {r.get('cause_detail', '-')}", axis=1)
    route_df["affected_lot_count"] = pd.to_numeric(route_df.get("unique_lot_count", 0), errors="coerce").fillna(0)
    route_df["alignment_with_primary_issue"] = route_df.apply(_score_alignment, axis=1)
    route_df["repeated_same_machine"] = route_df["machine_id"].astype(str).eq(primary_machine).astype(int)
    route_df["repeated_same_stage"] = pd.to_numeric(route_df["stage_no"], errors="coerce").eq(primary_stage).astype(int)
    route_df["repeated_same_cause_detail"] = route_df["cause_detail"].astype(str).eq(route_df["cause_detail"].mode().iloc[0] if not route_df["cause_detail"].mode().empty else "").astype(int)

    route_df["impact_family_rank"] = route_df.groupby("metric_family")["impact_value"].rank(pct=True, ascending=False).fillna(0)
    route_df["recurrence_rank"] = (
        route_df["repeat_count"].rank(pct=True, ascending=False).fillna(0) * 0.6
        + route_df["affected_time_bucket_count"].rank(pct=True, ascending=False).fillna(0) * 0.4
    )
    route_df["concentration_rank"] = (
        route_df["repeated_same_machine"] * 0.45
        + route_df["repeated_same_stage"] * 0.20
        + route_df["alignment_with_primary_issue"] * 0.35
    )
    route_df["temporal_rank"] = route_df["alignment_with_primary_issue"]
    route_df["inspection_value_rank"] = np.where(
        route_df["cause_family"].astype(str).isin(["pickup", "quality_lag"]),
        1.0,
        0.65,
    )
    route_df["route_priority_score"] = (
        route_df["impact_family_rank"] * 0.25
        + route_df["recurrence_rank"] * 0.20
        + route_df["concentration_rank"] * 0.20
        + route_df["temporal_rank"] * 0.20
        + route_df["inspection_value_rank"] * 0.15
    )
    route_df["path_confidence"] = route_df.apply(_confidence, axis=1)

    def _selection_reason(row: pd.Series) -> str:
        metric_family = str(row.get("metric_family", ""))
        impact_value = float(pd.to_numeric(row.get("impact_value", 0), errors="coerce") or 0)
        impact_unit = str(row.get("impact_unit", "-"))
        repeat_count = int(pd.to_numeric(row.get("repeat_count", 0), errors="coerce") or 0)
        lot_count = int(pd.to_numeric(row.get("affected_lot_count", 0), errors="coerce") or 0)
        bucket_count = int(pd.to_numeric(row.get("affected_time_bucket_count", 0), errors="coerce") or 0)
        alignment = float(pd.to_numeric(row.get("alignment_with_primary_issue", 0), errors="coerce") or 0)

        if metric_family == "route_link":
            return f"대표 설비·LOT 축과 정합성이 {alignment:.2f}이고 pickup-stop 중첩이 {bucket_count}개 구간에서 확인돼 우선 추적합니다."
        if metric_family == "quality_lag":
            return f"정지 이후 AOI fail 영향이 {impact_value:.0f} {impact_unit}이고 {bucket_count}개 구간에서 이어져 후행 영향 확인 가치가 높습니다."
        return f"영향 {impact_value:.2f} {impact_unit}, 반복 {repeat_count}회, 영향 LOT {lot_count}개, 대표 축 정합성 {alignment:.2f}라 우선 추적합니다."

    def _recommended_check(row: pd.Series) -> str:
        family = str(row.get("cause_family", ""))
        if family == "pickup":
            return "feeder 정렬, nozzle 상태, reel 장력, same lot 반복 여부 확인"
        if family == "quality_lag":
            return "AOI FAIL 시점과 stop window 연결, 영향 LOT 범위 확인"
        return "정지 원인 로그와 동시간대 설비 상태 확인"

    route_df["selection_reason"] = route_df.apply(_selection_reason, axis=1)
    route_df["recommended_check"] = route_df.apply(_recommended_check, axis=1)
    route_df = route_df.sort_values(["route_priority_score", "alignment_with_primary_issue", "repeat_count"], ascending=False).reset_index(drop=True)
    route_df["path_id"] = [f"PATH-{idx:02d}" for idx in range(1, len(route_df) + 1)]
    top5 = route_df.head(5).copy()
    top5["rank"] = np.arange(1, len(top5) + 1)
    top5["impact_summary"] = top5.apply(lambda r: f"{r.get('impact_value', 0):.2f} {r.get('impact_unit', '-')}", axis=1)
    top5["recurrence_summary"] = top5.apply(lambda r: f"repeat {int(r.get('repeat_count', 0))} / lot {int(r.get('affected_lot_count', 0))} / bucket {int(r.get('affected_time_bucket_count', 0))}", axis=1)
    pickup_peak_series = pd.to_numeric(top5["pickup_peak"], errors="coerce").fillna(0) if "pickup_peak" in top5.columns else pd.Series(0.0, index=top5.index)
    top5["chart_label"] = top5.apply(
        lambda r: f"#{int(r.get('rank', 0))} {r.get('path_id', '-')}" if pd.notna(r.get("rank")) else str(r.get("path_id", "-")),
        axis=1,
    )
    top5["stop_loss_sec"] = np.where(top5["metric_family"].astype(str).eq("stop_time"), pd.to_numeric(top5["impact_value"], errors="coerce").fillna(0), 0.0)
    top5["quality_impact_count"] = np.where(
        top5["metric_family"].astype(str).isin(["defect_count", "quality_lag"]),
        pd.to_numeric(top5["impact_value"], errors="coerce").fillna(0),
        0.0,
    )
    top5["pickup_instability"] = np.where(
        top5["metric_family"].astype(str).eq("pickup_error_rate"),
        pd.to_numeric(top5["impact_value"], errors="coerce").fillna(0),
        pickup_peak_series,
    )
    top5["route_overlap"] = np.where(
        top5["metric_family"].astype(str).eq("route_link"),
        pd.to_numeric(top5["impact_value"], errors="coerce").fillna(0),
        pd.to_numeric(top5["affected_time_bucket_count"], errors="coerce").fillna(0),
    )
    top5["lot_spread"] = pd.to_numeric(top5["affected_lot_count"], errors="coerce").fillna(0)
    top5["recurrence_strength"] = pd.to_numeric(top5["repeat_count"], errors="coerce").fillna(0)
    top5["alignment_pct"] = pd.to_numeric(top5["alignment_with_primary_issue"], errors="coerce").fillna(0) * 100
    top5["priority_pct"] = pd.to_numeric(top5["route_priority_score"], errors="coerce").fillna(0) * 100

    metric_candidates = {
        "정지 손실": top5["stop_loss_sec"],
        "품질 영향": top5["quality_impact_count"],
        "반복성": top5["recurrence_strength"],
        "LOT 확산": top5["lot_spread"],
        "대표 이슈 정합성": top5["alignment_pct"],
    }
    for metric_name, metric_series in metric_candidates.items():
        max_value = float(metric_series.max()) if len(metric_series) else 0.0
        top5[f"{metric_name}_점수"] = metric_series.apply(lambda v: _safe_div(v, max_value) * 100 if max_value > 0 else 0.0)

    bottleneck_score_cols = ["정지 손실_점수", "품질 영향_점수", "반복성_점수", "LOT 확산_점수", "대표 이슈 정합성_점수"]
    top5["bottleneck_index"] = (
        top5["정지 손실_점수"] * 0.34
        + top5["품질 영향_점수"] * 0.22
        + top5["반복성_점수"] * 0.18
        + top5["LOT 확산_점수"] * 0.10
        + top5["대표 이슈 정합성_점수"] * 0.16
    )

    def _dominant_bottleneck(row: pd.Series) -> str:
        stop_score = float(row.get("정지 손실_점수", 0))
        quality_score = float(row.get("품질 영향_점수", 0))
        flow_score = float(row.get("반복성_점수", 0)) * 0.55 + float(row.get("LOT 확산_점수", 0)) * 0.45
        pickup_score = max(float(row.get("pickup_instability", 0)) * 100, float(row.get("대표 이슈 정합성_점수", 0)) * 0.85)
        candidates = {
            "정지 병목": stop_score,
            "품질 병목": quality_score,
            "흐름 병목": flow_score,
            "실장 불안정": pickup_score,
        }
        return max(candidates.items(), key=lambda item: item[1])[0]

    def _smt_action_point(row: pd.Series) -> str:
        dominant = str(row.get("dominant_bottleneck", ""))
        if dominant == "정지 병목":
            return f"stop {float(row.get('stop_loss_sec', 0)):.0f}초가 가장 커 즉시 정지 원인 로그 확인"
        if dominant == "품질 병목":
            return f"AOI 영향 {float(row.get('quality_impact_count', 0)):.0f}건으로 하류 격리 판단 필요"
        if dominant == "흐름 병목":
            return f"반복 {int(row.get('recurrence_strength', 0))}회, LOT {int(row.get('lot_spread', 0))}개로 라인 밸런스 확인"
        return f"pickup 축 정합성 {float(row.get('alignment_pct', 0)):.0f}점으로 feeder/nozzle 우선 점검"

    top5["dominant_bottleneck"] = top5.apply(_dominant_bottleneck, axis=1)
    top5["smt_action_point"] = top5.apply(_smt_action_point, axis=1)

    st.markdown("#### A. TOP 5 추적 우선 경로")
    st.caption("대표 경로를 깊게 보기 전에, SMT 공정 관점에서 어느 경로가 정지·품질·흐름 병목을 만드는지 먼저 비교합니다.")

    chart_cols = st.columns([1.2, 1.0])
    with chart_cols[0]:
        compare_heatmap = top5[["chart_label"] + bottleneck_score_cols].copy()
        compare_heatmap = compare_heatmap.rename(
            columns={
                "정지 손실_점수": "정지 손실",
                "품질 영향_점수": "품질 영향",
                "반복성_점수": "반복성",
                "LOT 확산_점수": "LOT 확산",
                "대표 이슈 정합성_점수": "대표 이슈 정합성",
            }
        )
        heatmap_matrix = compare_heatmap.set_index("chart_label")[["정지 손실", "품질 영향", "반복성", "LOT 확산", "대표 이슈 정합성"]]
        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=heatmap_matrix.values,
                x=list(heatmap_matrix.columns),
                y=list(heatmap_matrix.index),
                colorscale=[
                    [0.0, "#f8efe3"],
                    [0.35, "#f7c58d"],
                    [0.65, "#ef8a3d"],
                    [1.0, "#b45309"],
                ],
                zmin=0,
                zmax=100,
                text=np.round(heatmap_matrix.values, 0),
                texttemplate="%{text:.0f}",
                hovertemplate="경로 %{y}<br>%{x}: %{z:.1f}점<extra></extra>",
                colorbar=dict(title="병목 강도", ticksuffix="점"),
            )
        )
        fig_heatmap.update_layout(xaxis_title="비교 축", yaxis_title="경로")
        st.plotly_chart(_plot_style(fig_heatmap, "Top5 병목 비교 맵", 340), use_container_width=True)
        st.caption("해석 기준: 값이 높을수록 해당 경로가 그 축에서 더 강한 병목 신호를 보입니다.")

    with chart_cols[1]:
        priority_plot = top5.sort_values(["bottleneck_index", "priority_pct"], ascending=True).copy()
        fig_priority = px.bar(
            priority_plot,
            x="bottleneck_index",
            y="chart_label",
            color="dominant_bottleneck",
            orientation="h",
            text="priority_pct",
            title="Top5 추적 우선순위와 병목 타입",
            color_discrete_map={
                "정지 병목": "#c2410c",
                "품질 병목": "#dc2626",
                "흐름 병목": "#0369a1",
                "실장 불안정": "#2563eb",
            },
        )
        fig_priority.update_traces(texttemplate="우선순위 %{text:.0f}점", textposition="outside")
        fig_priority.update_layout(xaxis_title="종합 병목 지수", yaxis_title="경로", showlegend=True)
        st.plotly_chart(_plot_style(fig_priority, "Top5 추적 우선순위와 병목 타입", 340), use_container_width=True)
        st.caption("해석 기준: 막대가 길수록 먼저 추적해야 하며, 색은 먼저 확인할 병목 성격을 뜻합니다.")

    top_route = top5.sort_values(["bottleneck_index", "priority_pct"], ascending=False).iloc[0]
    gap_vs_second = float(top_route.get("bottleneck_index", 0)) - float(top5.sort_values(["bottleneck_index", "priority_pct"], ascending=False).iloc[1].get("bottleneck_index", 0)) if len(top5) > 1 else 0.0
    st.markdown(
        "\n".join(
            [
                f"- 최우선 경로는 `{top_route.get('path_id', '-')}`이며, 현재 판정은 `{top_route.get('dominant_bottleneck', '-')}`입니다.",
                f"- 이 경로는 정지 {float(top_route.get('stop_loss_sec', 0)):.0f}초, 품질 영향 {float(top_route.get('quality_impact_count', 0)):.0f}건, 반복 {int(top_route.get('recurrence_strength', 0))}회로 다른 후보 대비 차이가 큽니다.",
                f"- 2위 경로와의 병목 지수 차이는 {gap_vs_second:.1f}점이며, 우선 액션은 {top_route.get('smt_action_point', '-') }입니다.",
            ]
        )
    )

    top5_view = top5.rename(
        columns={
            "path_id": "경로",
            "route_meaning": "경로 의미",
            "where": "주요 위치",
            "main_symptom": "주요 증상",
            "metric_family": "영향 유형",
            "impact_summary": "영향 값",
            "repeat_count": "반복 횟수",
            "path_confidence": "신뢰도",
            "dominant_bottleneck": "병목 판정",
            "smt_action_point": "SMT 우선 확인",
            "selection_reason": "선정 이유",
        }
    )
    st.dataframe(
        top5_view[["rank", "경로", "경로 의미", "주요 위치", "when", "주요 증상", "영향 유형", "영향 값", "반복 횟수", "병목 판정", "신뢰도", "SMT 우선 확인", "선정 이유"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("요약: 상위 5개 경로를 병목 타입까지 같이 보여주므로, 어느 경로가 정지 병목인지 품질 병목인지 바로 구분할 수 있습니다.")

    representative = top5[top5["cause_family"].astype(str).eq("pickup")].head(1)
    representative = representative if not representative.empty else top5.head(1)
    rep = representative.iloc[0]

    st.markdown("#### B. 대표 경로 상세 분석")
    st.markdown(
        _card(
            "Rank 1 대표 경로",
            f"{rep.get('path_id', '-')} · {rep.get('route_meaning', '-')}",
            f"{rep.get('where', '-')} · {rep.get('path_confidence', '-')}",
        ),
        unsafe_allow_html=True,
    )
    st.caption("대표 경로가 선택된 이유를 공정/스테이지와 동급 설비 비교 근거로 다시 확인합니다.")

    rep_stage = int(pd.to_numeric(rep.get("stage_no", primary_stage), errors="coerce") or primary_stage)
    rep_machine = str(rep.get("machine_id", primary_machine))
    rep_line = str(rep.get("line_id", primary_line))
    benchmark_rows = []
    stage_benchmark = pd.DataFrame()
    if not comp.empty:
        stage_benchmark = comp.groupby(["line_id", "stage_no"], as_index=False).agg(
            error_count=("error_count", "sum"),
            pickup_count=("pickup_count", "sum"),
        )
        stage_benchmark["error_rate"] = stage_benchmark.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
    if not stop.empty:
        stage_stop = stop.groupby(["line_id", "stage_no"], as_index=False).agg(stop_time_sec=("duration_sec", "sum"))
        stage_benchmark = stage_stop if stage_benchmark.empty else stage_benchmark.merge(stage_stop, on=["line_id", "stage_no"], how="outer")
    if not stage_benchmark.empty:
        stage_benchmark = stage_benchmark.fillna(0)
        stage_benchmark["stage_score"] = pd.to_numeric(stage_benchmark.get("error_rate", 0), errors="coerce").fillna(0) * 100 + pd.to_numeric(stage_benchmark.get("stop_time_sec", 0), errors="coerce").fillna(0)
        stage_benchmark = stage_benchmark.sort_values("stage_score", ascending=False)
        stage_avg_error = float(stage_benchmark["error_rate"].mean()) if "error_rate" in stage_benchmark.columns and not stage_benchmark.empty else 0.0
        stage_avg_stop = float(stage_benchmark["stop_time_sec"].mean()) if "stop_time_sec" in stage_benchmark.columns and not stage_benchmark.empty else 0.0
        top_stage_row = stage_benchmark.head(1)
        rep_stage_row = stage_benchmark[
            stage_benchmark["line_id"].astype(str).eq(rep_line)
            & pd.to_numeric(stage_benchmark["stage_no"], errors="coerce").eq(rep_stage)
        ].head(1)
        if not rep_stage_row.empty:
            rep_stage_row = rep_stage_row.iloc[0]
            top_stage_no = int(pd.to_numeric(top_stage_row.iloc[0].get("stage_no", 0), errors="coerce") or 0) if not top_stage_row.empty else 0
            benchmark_rows.append(
                {
                    "비교 레벨": "공정/스테이지",
                    "대표 후보": f"{rep_line} / Stage-{rep_stage}",
                    "비교 기준": "pickup error rate / stop time",
                    "대표 경로 값": f"{float(rep_stage_row.get('error_rate', 0)):.2%} / {float(rep_stage_row.get('stop_time_sec', 0)):.0f} sec",
                    "비교 평균": f"{stage_avg_error:.2%} / {stage_avg_stop:.0f} sec",
                    "판정": "대표 공정/스테이지" if rep_stage == top_stage_no else "상위 공정/스테이지",
                    "선정 의미": f"{rep_line} / Stage-{rep_stage}의 pickup error와 stop time이 함께 높습니다.",
                }
            )

    machine_benchmark = pd.DataFrame()
    if not comp.empty:
        machine_benchmark = comp.groupby("machine_id", as_index=False).agg(
            pickup_error_count=("pickup_error_count", "sum"),
            error_count=("error_count", "sum"),
            pickup_count=("pickup_count", "sum"),
        )
        machine_benchmark["error_rate"] = machine_benchmark.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
    if not stop.empty:
        stop_benchmark = stop.groupby("machine_id", as_index=False).agg(stop_time_sec=("duration_sec", "sum"), stop_events=("stop_count", "sum"))
        machine_benchmark = stop_benchmark if machine_benchmark.empty else machine_benchmark.merge(stop_benchmark, on="machine_id", how="outer")
    if not machine_benchmark.empty:
        machine_benchmark = machine_benchmark.fillna(0)
        peer_rows = machine_benchmark[machine_benchmark["machine_id"].astype(str) != rep_machine].copy()
        peer_mean_error = float(peer_rows["error_rate"].mean()) if not peer_rows.empty and "error_rate" in peer_rows.columns else 0.0
        peer_mean_stop = float(peer_rows["stop_time_sec"].mean()) if not peer_rows.empty and "stop_time_sec" in peer_rows.columns else 0.0
        this_row = machine_benchmark[machine_benchmark["machine_id"].astype(str).eq(rep_machine)].head(1)
        if not this_row.empty:
            this_row = this_row.iloc[0]
            benchmark_rows.append(
                {
                    "비교 레벨": "설비",
                    "대표 후보": f"{rep_machine} pickup error rate",
                    "비교 기준": "동급 설비 대비 pickup error rate",
                    "대표 경로 값": f"{float(this_row.get('error_rate', 0)):.2%}",
                    "비교 평균": f"{peer_mean_error:.2%}",
                    "판정": f"{_safe_div(float(this_row.get('error_rate', 0)), max(peer_mean_error, 1e-9)):.1f}배",
                    "선정 의미": f"{rep_machine}의 pickup 계열 이상이 peer 설비보다 뚜렷합니다.",
                }
            )
            benchmark_rows.append(
                {
                    "비교 레벨": "설비",
                    "대표 후보": f"{rep_machine} stop time",
                    "비교 기준": "동급 설비 대비 stop time",
                    "대표 경로 값": f"{float(this_row.get('stop_time_sec', 0)):.0f} sec",
                    "비교 평균": f"{peer_mean_stop:.0f} sec",
                    "판정": f"{_safe_div(float(this_row.get('stop_time_sec', 0)), max(peer_mean_stop, 1e-9)):.1f}배",
                    "선정 의미": "pickup 이상이 실제 정지 손실로 이어진 설비라는 점을 보여줍니다.",
                }
            )

    lot_compare = pd.DataFrame()
    if not comp.empty and "lot_id" in comp.columns:
        lot_compare = comp.groupby("lot_id", as_index=False).agg(
            error_count=("error_count", "sum"),
            pickup_count=("pickup_count", "sum"),
        )
        lot_compare["pickup_success_rate"] = 1 - lot_compare.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
    if not stop.empty and "lot_id" in stop.columns:
        stop_by_lot = stop.groupby("lot_id", as_index=False).agg(stop_time_sec=("duration_sec", "sum"))
        lot_compare = stop_by_lot if lot_compare.empty else lot_compare.merge(stop_by_lot, on="lot_id", how="outer")
    if not insp.empty and "lot_id" in insp.columns and "is_fail" in insp.columns:
        quality_by_lot = insp.groupby("lot_id", as_index=False).agg(fail_count=("is_fail", "sum"), inspect_count=("is_fail", "size"))
        quality_by_lot["fail_rate"] = quality_by_lot.apply(lambda r: _safe_div(r.get("fail_count", 0), max(r.get("inspect_count", 0), 1)), axis=1)
        lot_compare = quality_by_lot if lot_compare.empty else lot_compare.merge(quality_by_lot, on="lot_id", how="outer")
    if not lot_compare.empty:
        lot_compare = lot_compare.fillna(0)

    if benchmark_rows:
        st.markdown("##### 대표 경로 선정 근거")
        st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)
    else:
        st.info("대표 경로 선정 근거를 만들 비교 데이터가 부족합니다.")

    evidence_cols = st.columns(3)
    with evidence_cols[0]:
        if not stage_benchmark.empty:
            stage_plot = stage_benchmark.copy()
            stage_plot["stage_label"] = stage_plot.apply(
                lambda r: f"{r.get('line_id', '-')}/Stage-{int(pd.to_numeric(r.get('stage_no', 0), errors='coerce')) if pd.notna(pd.to_numeric(r.get('stage_no', 0), errors='coerce')) else '-'}",
                axis=1,
            )
            stage_plot["대표 공정/스테이지"] = np.where(
                stage_plot["line_id"].astype(str).eq(rep_line)
                & pd.to_numeric(stage_plot["stage_no"], errors="coerce").eq(rep_stage),
                "대표 공정/스테이지",
                "비교 공정/스테이지",
            )
            stage_long = stage_plot.melt(
                id_vars=["stage_label", "대표 공정/스테이지"],
                value_vars=[c for c in ["error_rate", "stop_time_sec"] if c in stage_plot.columns],
                var_name="지표",
                value_name="값",
            )
            stage_long["지표"] = stage_long["지표"].map({"error_rate": "pickup error rate", "stop_time_sec": "stop time"}).fillna(stage_long["지표"])
            fig_stage = px.bar(
                stage_long.sort_values(["지표", "값"], ascending=[True, False]),
                x="stage_label",
                y="값",
                color="대표 공정/스테이지",
                facet_row="지표",
                title="공정/스테이지 비교",
            )
            fig_stage.update_layout(xaxis_title="Line / Stage", yaxis_title="")
            st.plotly_chart(_plot_style(fig_stage, "공정/스테이지 비교", 320), use_container_width=True)

    with evidence_cols[1]:
        if not machine_benchmark.empty:
            pickup_plot = machine_benchmark.copy()
            pickup_plot["대표 설비"] = np.where(pickup_plot["machine_id"].astype(str).eq(rep_machine), "대표 설비", "비교 설비")
            fig_pickup = px.bar(
                pickup_plot.sort_values("error_rate", ascending=False),
                x="machine_id",
                y="error_rate",
                color="대표 설비",
                text="error_rate",
                title="동급 설비 대비 Pickup Error",
            )
            fig_pickup.update_layout(xaxis_title="설비", yaxis_title="pickup error rate")
            st.plotly_chart(_plot_style(fig_pickup, "동급 설비 대비 Pickup Error", 320), use_container_width=True)

    with evidence_cols[2]:
        if not machine_benchmark.empty:
            stop_plot = machine_benchmark.copy()
            stop_plot["대표 설비"] = np.where(stop_plot["machine_id"].astype(str).eq(rep_machine), "대표 설비", "비교 설비")
            fig_stop = px.bar(
                stop_plot.sort_values("stop_time_sec", ascending=False),
                x="machine_id",
                y="stop_time_sec",
                color="대표 설비",
                text="stop_time_sec",
                title="동급 설비 대비 Stop Time",
            )
            fig_stop.update_layout(xaxis_title="설비", yaxis_title="stop time (sec)")
            st.plotly_chart(_plot_style(fig_stop, "동급 설비 대비 Stop Time", 320), use_container_width=True)

    timeline = pd.DataFrame()
    if not comp_machine.empty:
        comp_t = comp_machine.groupby("time_bucket", as_index=False).agg(pickup_error_count=("pickup_error_count", "sum"))
        timeline = comp_t.copy()
    if not stop_machine.empty:
        stop_t = stop_machine.groupby("time_bucket", as_index=False).agg(stop_time_sec=("duration_sec", "sum"))
        timeline = stop_t if timeline.empty else timeline.merge(stop_t, on="time_bucket", how="outer")
    if not quality_lot.empty:
        qual_t = quality_lot.groupby("time_bucket", as_index=False).agg(aoi_fail_count=("is_fail", "sum"))
        timeline = qual_t if timeline.empty else timeline.merge(qual_t, on="time_bucket", how="outer")
    if not timeline.empty:
        timeline = timeline.sort_values("time_bucket").fillna(0)

    st.markdown("#### C. 시나리오 검토 및 설명")
    scenario_title = f"{primary_machine}에서 시작된 pickup 계열 이상이 정지 손실을 만들고, 이후 {primary_lot} 품질 영향으로 이어졌는가?"
    st.markdown(f"##### 이번에 검토할 시나리오")
    st.markdown(f"- {scenario_title}")

    machine_judgement = "가설 적합" if benchmark_rows else "추가 확인"
    timing_judgement = "가설 적합" if not timeline.empty else "추가 확인"
    material_judgement = "가설 적합" if not comp.empty else "추가 확인"
    lot_judgement = "가설 적합" if not lot_compare.empty else "추가 확인"
    scenario_fit_count = sum(v == "가설 적합" for v in [machine_judgement, timing_judgement, material_judgement, lot_judgement])
    scenario_fit = "높음" if scenario_fit_count >= 4 else "보통" if scenario_fit_count >= 3 else "낮음"

    c_cards = st.columns(4)
    card_items = [
        ("설비 정합성", machine_judgement, "M05가 동급 설비보다 확실히 나쁜가"),
        ("시간 정합성", timing_judgement, "pickup 이후 stop이 같은 구간에서 커지는가"),
        ("자재 정합성", material_judgement, f"{primary_feeder}/{primary_part}/{primary_nozzle} 조합이 집중되는가"),
        ("LOT 정합성", lot_judgement, f"{primary_lot}에 하류 품질 영향이 붙는가"),
    ]
    for col, (label, value, foot) in zip(c_cards, card_items):
        with col:
            st.markdown(_card(label, value, foot), unsafe_allow_html=True)
    st.caption(f"가설 적합도: {scenario_fit}. 4개 질문 중 {scenario_fit_count}개가 현재 데이터에서 지지됩니다.")

    review_df = pd.DataFrame(
        [
            {
                "확인 질문": "왜 M05를 먼저 봐야 하는가?",
                "현재 판단": machine_judgement,
                "확인 기준": "동급 설비 대비 pickup error rate / stop time 비교",
                "핵심 해석": "M05가 동급 설비보다 확실히 나쁘면 machine-local 시나리오가 맞습니다.",
            },
            {
                "확인 질문": "pickup 이상이 stop보다 앞서거나 함께 움직이는가?",
                "현재 판단": timing_judgement,
                "확인 기준": "문제 구간 전후 및 시간 전이 확인",
                "핵심 해석": "pickup이 먼저 올라오고 stop이 뒤따르면 경로 정합성이 높습니다.",
            },
            {
                "확인 질문": "특정 자재 조합이 경로를 설명하는가?",
                "현재 판단": material_judgement,
                "확인 기준": "feeder / part / nozzle 조합 집중도 비교",
                "핵심 해석": "한 조합에 집중되면 원인분석과 같은 축으로 연결됩니다.",
            },
            {
                "확인 질문": "하류 품질 영향까지 이어졌는가?",
                "현재 판단": lot_judgement,
                "확인 기준": "LOT별 AOI fail rate 비교",
                "핵심 해석": "대표 LOT가 더 나쁘면 생산 손실이 품질로 전이된 것입니다.",
            },
        ]
    )
    st.dataframe(review_df, use_container_width=True, hide_index=True)

    scenario_lines = [
        f"한 줄 설명: {primary_machine}에서 pickup 계열 이상이 시작됐고, 같은 구간의 정지 손실을 키웠으며, 이후 {primary_lot}의 품질 리스크로 이어졌다는 시나리오입니다.",
        f"공장 의미: line-wide 문제가 아니라 특정 설비와 특정 자재 조건을 먼저 확인해야 하는 국소 이슈라는 뜻입니다.",
        f"현재 해석: {primary_feeder}/{primary_part}/{primary_nozzle} 조합이 대표 경로의 가장 강한 자재 축입니다.",
        "남은 확인: feeder 교체 이력, nozzle 마모 기록, 셋업 변경 시점이 붙으면 시나리오 확정도가 더 올라갑니다.",
    ]
    st.markdown("\n".join([f"- {line}" for line in scenario_lines]))

    evidence_cols_top = st.columns(2)
    with evidence_cols_top[0]:
        if not machine_benchmark.empty:
            machine_plot = machine_benchmark.copy().fillna(0)
            fig_machine_focus = px.bar(
                machine_plot.sort_values(["pickup_error_count", "stop_time_sec"], ascending=False),
                x="machine_id",
                y=["pickup_error_count", "stop_time_sec"],
                barmode="group",
                title="1. 동급 설비 비교",
            )
            st.plotly_chart(_plot_style(fig_machine_focus, "1. 동급 설비 비교", 320), use_container_width=True)
            st.caption("무엇을 보여주나: M05가 다른 설비보다 얼마나 나쁜지 보여줍니다.")
            st.caption("왜 중요한가: 임원이 가장 먼저 묻는 '왜 하필 M05인가'에 대한 직접 증거입니다.")
            st.caption("다음 확인: M05의 최근 maintenance / setup 변경 이력을 겹쳐 보십시오.")

    with evidence_cols_top[1]:
        if not timeline.empty:
            timeline_plot = timeline.copy()
            for col in ["pickup_error_count", "stop_time_sec", "aoi_fail_count"]:
                if col not in timeline_plot.columns:
                    timeline_plot[col] = 0
            timeline_plot["time_label"] = pd.to_datetime(timeline_plot["time_bucket"], errors="coerce").dt.strftime("%H:%M")
            timeline_long = timeline_plot.melt(
                id_vars=["time_bucket", "time_label"],
                value_vars=["pickup_error_count", "stop_time_sec", "aoi_fail_count"],
                var_name="지표",
                value_name="값",
            )
            fig_timeline = px.line(timeline_long, x="time_label", y="값", color="지표", markers=True, title="2. 시간 흐름 증거")
            st.plotly_chart(_plot_style(fig_timeline, "2. 시간 흐름 증거", 320), use_container_width=True)
            st.caption("무엇을 보여주나: pickup, stop, AOI가 어떤 순서로 움직였는지 보여줍니다.")
            st.caption("왜 중요한가: 시나리오가 말뿐이 아니라 시간 순서로 맞는지 검토할 수 있습니다.")
            st.caption("다음 확인: 교대 전환이나 자재 교체 직후에 패턴이 강화되는지 확인합니다.")

    evidence_cols_bottom = st.columns(2)
    with evidence_cols_bottom[0]:
        part_focus = pd.DataFrame()
        if not comp.empty:
            part_focus = comp.groupby(["part_number", "feeder_id", "nozzle_serial"], as_index=False).agg(error_count=("error_count", "sum"), pickup_count=("pickup_count", "sum"))
            part_focus["error_rate"] = part_focus.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
            part_focus = part_focus.sort_values(["error_rate", "error_count"], ascending=False).head(8)
        if not part_focus.empty:
            part_focus["조합"] = part_focus.apply(lambda r: f"{r.get('feeder_id', '-')}\n{r.get('part_number', '-')}\n{r.get('nozzle_serial', '-')}", axis=1)
            fig_part_focus = px.bar(part_focus, x="조합", y="error_rate", text="error_count", title="3. 자재 조합 집중")
            st.plotly_chart(_plot_style(fig_part_focus, "3. 자재 조합 집중", 320), use_container_width=True)
            st.caption("무엇을 보여주나: 어떤 자재 조합이 대표 경로를 가장 강하게 설명하는지 보여줍니다.")
            st.caption("왜 중요한가: 원인분석의 최종 후보와 경로분석을 같은 축으로 묶어줍니다.")
            st.caption("다음 확인: 상위 조합의 교체 이력과 오염 여부를 확인합니다.")

    with evidence_cols_bottom[1]:
        if not lot_compare.empty:
            lot_plot = lot_compare.copy()
            if "pickup_success_rate" in lot_plot.columns:
                avg_success = float(lot_plot["pickup_success_rate"].mean())
                lot_plot = lot_plot.sort_values(["pickup_success_rate", "stop_time_sec"], ascending=[True, False])
                fig_lot = px.bar(lot_plot, x="lot_id", y="pickup_success_rate", text="stop_time_sec", title="4. 대표 구간 LOT 성과 비교")
                fig_lot.add_hline(y=avg_success, line_dash="dash", line_color="#ff8a3d", annotation_text="전체 평균")
                st.plotly_chart(_plot_style(fig_lot, "4. 대표 구간 LOT 성과 비교", 320), use_container_width=True)
                rep_lot_row = lot_plot[lot_plot["lot_id"].astype(str).eq(primary_lot)].head(1)
                rep_success = float(rep_lot_row.iloc[0]["pickup_success_rate"]) if not rep_lot_row.empty else 0.0
                st.caption(f"무엇을 보여주나: 대표 구간에서 진행된 LOT들의 성과를 비교하고, {primary_lot}의 성공률이 전체 평균보다 낮았는지 보여줍니다.")
                st.caption(f"왜 중요한가: 대표 경로 시간대의 LOT가 평균보다 낮다면, 이슈가 실제 생산 성과 저하로 이어졌다는 해석이 가능합니다. 현재 {primary_lot} 성공률은 {rep_success:.2%}, 전체 평균은 {avg_success:.2%}입니다.")
                st.caption("다음 확인: 대표 구간 LOT와 직전 LOT의 자재 셋업, 정지 이력, AOI 결과를 함께 비교합니다.")

    st.markdown("#### D. 조치 가이드")
    action_df = pd.DataFrame(
        [
            {"우선순위": 1, "조치": "M05 feeder/nozzle 상태 점검", "왜": "대표 경로와 가장 직접 연결됨", "다음 확인": "nozzle wear, feeder alignment, reel tension"},
            {"우선순위": 2, "조치": "LOT002 영향 범위 확인", "왜": "pickup instability window 이후 AOI fail이 나타남", "다음 확인": "영향 board/lot 격리 필요 여부"},
            {"우선순위": 3, "조치": "같은 조건 재발 여부 확인", "왜": "repeat_count로 구조적 문제인지 일회성인지 판단", "다음 확인": "same machine / same feeder / same part recurrence"},
        ]
    )
    st.dataframe(action_df, use_container_width=True, hide_index=True)

    with st.expander("고급 Route Dataset", expanded=False):
        advanced_cols = [
            "path_id", "line_id", "stage_no", "machine_id", "lot_id", "metric_family", "impact_value", "impact_unit",
            "event_group", "cause_family", "cause_detail", "first_seen_ts", "last_seen_ts", "repeat_count",
            "affected_time_bucket_count", "affected_lot_count", "path_confidence", "selection_reason", "recommended_check"
        ]
        st.dataframe(route_df[[c for c in advanced_cols if c in route_df.columns]], use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_rca_product(raw: Dict[str, pd.DataFrame], clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame], sample_mode: bool):
    filters = _build_filter_panel(clean, "rca4")
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame())
    stop = clean.get("vw_stop_event_fact", pd.DataFrame())
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame())
    tag = clean.get("vw_tag_event_fact", pd.DataFrame())
    comp = clean.get("vw_component_error_fact", pd.DataFrame())
    filtered_clean = {
        **clean,
        "vw_shopfloor_event_fact": _apply_selection(shop, filters),
        "vw_stop_event_fact": _apply_selection(stop, filters),
        "vw_inspection_event_fact": _apply_selection(insp, filters),
        "vw_tag_event_fact": _apply_selection(tag, filters),
        "vw_component_error_fact": _apply_selection(comp, filters),
    }

    scope = build_analysis_scope_summary(filtered_clean, marts)
    capability = build_analysis_capability_summary(filtered_clean)
    focus = build_analysis_focus_summary(filtered_clean, marts)
    rca_capability = build_rca_capability_summary(raw)
    loss_paths = build_rca_loss_path_view(filtered_clean)
    hotspot = build_rca_hotspot_view(filtered_clean)
    repeat = build_rca_repeat_pattern_view(filtered_clean)
    drilldown = build_rca_drilldown_view(filtered_clean)
    candidate = build_rca_candidate_view(filtered_clean)

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("### 4. RCA 제품")
    st.caption("이 탭은 제조 책임자가 바로 의사결정할 수 있도록, RCA를 제품 형태로 정리한 설계 화면입니다.")
    st.markdown(
        """
        <div style="padding:.8rem 1rem;border-radius:14px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);margin:.4rem 0 1rem 0;">
            <div style="font-size:.8rem;color:#9fb0c4;margin-bottom:.25rem;">제품 목적</div>
            <div style="font-size:.95rem;color:#e5edf7;font-weight:700;line-height:1.6;">
                이상 감지 → 범위 축소 → 원인 후보 생성 → 조치 우선순위 → 바로 실행
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_cards = st.columns(4)
    summary_cards = [
        ("RCA 가능성", "가능" if not rca_capability.empty and rca_capability["상태"].astype(str).eq("가능").any() else "제한", "설비 / 공정 / LOT 추적"),
        ("분석 축", str(capability["분석축"].nunique()) if not capability.empty else "0", "현재 데이터 범위"),
        ("핵심 이슈", str(focus.shape[0] if not focus.empty else 0), "대표 hotspot / 이상 후보"),
        ("후보 조치", str(loss_paths.shape[0] if not loss_paths.empty else 0), "실행 가능한 조치"),
    ]
    for col, (label, value, foot) in zip(top_cards, summary_cards):
        with col:
            st.markdown(_card(label, value, foot), unsafe_allow_html=True)

    st.markdown(_section_header("1. 데이터 모델", "원천 데이터 → 표준화 → RCA 후보 생성으로 이어지는 구조입니다.", PRIMARY), unsafe_allow_html=True)
    data_model = pd.DataFrame([
        {"레이어": "RAW", "핵심 항목": "timestamp, line_id, stage_id, machine_id, lot_id, part_no, feeder_id, nozzle_id", "역할": "원천 이벤트 보관", "산출물": "event log"},
        {"레이어": "표준화", "핵심 항목": "line_id / stage_id / machine_id / model_id / lot_id / time_bucket", "역할": "비교 가능한 공통 축", "산출물": "normalized fact"},
        {"레이어": "이벤트 분류", "핵심 항목": "STOP / PERFORMANCE / QUALITY", "역할": "증상과 현상 분리", "산출물": "event group"},
        {"레이어": "원인 분류", "핵심 항목": "FEEDER / NOZZLE / TRANSFER / VISION / MATERIAL / PROCESS / UPSTREAM / DOWNSTREAM / UNKNOWN", "역할": "원인 후보군 구분", "산출물": "cause group"},
        {"레이어": "핵심 KPI", "핵심 항목": "stop_time_sec, stop_count, pickup_error_count, defect_count, production_qty, cycle_time, loss_qty", "역할": "손실 규모 측정", "산출물": "metrics"},
        {"레이어": "영향도", "핵심 항목": "impact_score = stop_time_sec*W1 + defect_count*W2 + production_loss*W3", "역할": "우선순위 계산", "산출물": "ranked impact"},
    ])
    st.dataframe(data_model, use_container_width=True, hide_index=True)

    st.markdown(_section_header("2. 변환 로직", "원천 이벤트를 RCA가 읽을 수 있는 표준 형태로 바꿉니다.", SECONDARY), unsafe_allow_html=True)
    transform_logic = pd.DataFrame([
        {"순서": 1, "단계": "정규화", "입력": "raw event log", "출력": "표준 dimension + fact", "메모": "event_ts / time_key 보정"},
        {"순서": 2, "단계": "증상 분류", "입력": "stop / defect / slowdown", "출력": "STOP / PERFORMANCE / QUALITY", "메모": "symptom first"},
        {"순서": 3, "단계": "원인 도메인 분리", "입력": "error_code / stop_reason / part / feeder / nozzle", "출력": "cause group", "메모": "single-cause bias 방지"},
        {"순서": 4, "단계": "집중도 계산", "입력": "model / lot / part별 분포", "출력": "concentration ratio", "메모": "dominant LOT / PN 탐지"},
        {"순서": 5, "단계": "반복성 확인", "입력": "hour / shift / machine / lot / part", "출력": "repeat ratio", "메모": "재발성 여부 판단"},
        {"순서": 6, "단계": "영향도 산정", "입력": "stop_time / defect_count / production_loss", "출력": "impact_score", "메모": "순위 계산과 직접 연결"},
        {"순서": 7, "단계": "조치 후보", "입력": "cause group + evidence", "출력": "recommended action", "메모": "바로 실행 가능한 조치"},
        {"순서": 8, "단계": "품질 경고", "입력": "sample size / proxy metric", "출력": "confidence label", "메모": "small sample은 강한 결론 금지"},
    ])
    st.dataframe(transform_logic, use_container_width=True, hide_index=True)

    st.markdown(_section_header("3. RCA 엔진", "이 흐름은 원인을 단정하지 않고 후보를 좁히는 방식입니다.", PRIMARY), unsafe_allow_html=True)
    rca_engine = pd.DataFrame([
        {"단계": "1. 이상 탐지", "입력": "stop_time / defect_rate / cycle deviation", "출력": "abnormal entities", "핵심 함수": "build_rca_loss_path_view"},
        {"단계": "2. 영향 범위", "입력": "line / stage / machine / model / lot / part", "출력": "impact scoping", "핵심 함수": "build_rca_card_summary"},
        {"단계": "3. 증상 분해", "입력": "STOP / PERFORMANCE / QUALITY", "출력": "symptom split", "핵심 함수": "build_rca_hotspot_view"},
        {"단계": "4. 교차 분석", "입력": "feeder / nozzle / transfer / lot / model", "출력": "co-occurrence", "핵심 함수": "build_rca_repeat_pattern_view"},
        {"단계": "5. 재발성", "입력": "hour / shift / machine / lot", "출력": "repeatability", "핵심 함수": "build_rca_drilldown_view"},
        {"단계": "6. 후보 생성", "입력": "evidence bundle", "출력": "top cause candidates", "핵심 함수": "build_rca_candidate_view"},
        {"단계": "7. 조치 제안", "입력": "cause + evidence + owner", "출력": "action plan", "핵심 함수": "_reason_action_hint"},
    ])
    st.dataframe(rca_engine, use_container_width=True, hide_index=True)

    st.markdown(_section_header("4. UI 구조", "고객은 숫자보다 흐름을 읽고 조치를 결정해야 합니다.", SECONDARY), unsafe_allow_html=True)
    ui_structure = pd.DataFrame([
        {"화면": "Executive Summary", "사용자": "임원 / 관리감독", "목적": "상위 손실과 우선순위 확인", "구성": "Top loss, Top anomaly, One-line insight"},
        {"화면": "Drill-down Flow", "사용자": "공정 / 설비 엔지니어", "목적": "Line → Stage → Machine → Part → Feeder/Nozzle", "구성": "filter + ranked tables + scoped chart"},
        {"화면": "RCA Flow Visualization", "사용자": "RCA 담당", "목적": "anomaly → breakdown → cause → action", "구성": "timeline, hotspot, repeat, action"},
        {"화면": "Action Panel", "사용자": "현장 / 설비 / 품질", "목적": "지금 무엇을 할지 결정", "구성": "recommended action + owner + priority"},
    ])
    st.dataframe(ui_structure, use_container_width=True, hide_index=True)

    st.markdown(_section_header("5. 실제 데이터 예시", "현재 데이터로 어떤 형태의 결과가 나오는지 보여줍니다.", PRIMARY), unsafe_allow_html=True)
    left, right = st.columns([0.56, 0.44])
    with left:
        example_rows = []
        if not loss_paths.empty:
            top = loss_paths.iloc[0]
            example_rows.append({
                "문제 위치": f"{top.get('machine_id', '-')}",
                "문제 유형": f"{top.get('what', '-')}",
                "원인 후보": f"{top.get('cause_group', '-')} / {top.get('cause_detail', '-')}",
                "추천 조치": _reason_action_hint(" ".join(str(top.get(k, "")) for k in ["what", "where", "cause_group", "cause_detail"])),
            })
        if not repeat.empty:
            top = repeat.iloc[0]
            example_rows.append({
                "문제 위치": f"{top.get('machine_id', '-')}",
                "문제 유형": "반복 패턴",
                "원인 후보": f"{top.get('cause_group', '-')} / {top.get('cause_detail', '-')}",
                "추천 조치": _reason_action_hint(" ".join(str(top.get(k, "")) for k in ["pattern", "cause_group", "cause_detail"])),
            })
        if not hotspot.empty:
            top = hotspot.iloc[0]
            example_rows.append({
                "문제 위치": f"{top.get('machine_id', '-')}",
                "문제 유형": "핫스팟",
                "원인 후보": f"{top.get('cause_group', '-')} / {top.get('cause_detail', '-')}",
                "추천 조치": _reason_action_hint(" ".join(str(top.get(k, "")) for k in ["cause_group", "cause_detail"])),
            })
        if example_rows:
            st.dataframe(pd.DataFrame(example_rows), use_container_width=True, hide_index=True)
        else:
            st.info("실제 데이터 예시를 만들 수 있는 RCA 후보가 부족합니다.")
    with right:
        if not loss_paths.empty:
            top_loss = loss_paths.head(8).copy()
            st.plotly_chart(_plot_style(px.bar(top_loss, x="path_key", y="impact", text="impact", title="Top RCA 후보"), "Top RCA 후보", 320), use_container_width=True)
        elif not candidate.empty:
            cand = candidate.head(8).copy()
            display_cols = [c for c in ["event_ts", "line_id", "stage_no", "machine_id", "lot_id", "model_label", "part_number", "feeder_id", "nozzle_id", "error_code", "event_class"] if c in cand.columns]
            st.dataframe(cand[display_cols] if display_cols else cand, use_container_width=True, hide_index=True)
        else:
            st.info("예시 차트를 만들 데이터가 부족합니다.")

    st.markdown(_section_header("6. 조치 우선순위", "가장 영향이 큰 후보부터 실행 조치로 바꿉니다.", SECONDARY), unsafe_allow_html=True)
    action_rows = []
    if not loss_paths.empty:
        for _, row in loss_paths.head(5).iterrows():
            action_rows.append({
                "대상": row.get("path_key", "-"),
                "해석": "우선 확인 대상",
                "추천 조치": _reason_action_hint(" ".join(str(row.get(k, "")) for k in ["what", "where", "cause_group", "cause_detail"])),
                "책임팀": "설비 + 공정",
            })
    elif not focus.empty:
        for _, row in focus.head(5).iterrows():
            action_rows.append({
                "대상": row.get("구분", "-"),
                "해석": "대표 hotspot",
                "추천 조치": "대표 설비와 대표 LOT부터 교차 확인",
                "책임팀": "RCA 담당",
            })
    if action_rows:
        st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)
    else:
        st.info("조치 우선순위를 만들 데이터가 부족합니다.")

    if not scope.empty:
        st.markdown("#### 현재 데이터 범위")
        st.dataframe(scope, use_container_width=True, hide_index=True)
    if not capability.empty:
        st.markdown("#### 분석 가능 축")
        st.dataframe(capability, use_container_width=True, hide_index=True)

    if sample_mode:
        st.info("현재 탭은 샘플 또는 proxy 데이터를 포함해 RCA 제품 구조를 설명합니다.")
    st.markdown("</div>", unsafe_allow_html=True)


def _render_equipment_screen_legacy(clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame]):
    shop = clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    stop = clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    insp = clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    tag = clean.get("vw_tag_event_fact", pd.DataFrame()).copy()
    comp = clean.get("vw_component_error_fact", pd.DataFrame()).copy()

    filters = _build_filter_panel(clean, "problem")
    filtered_clean = {
        **clean,
        "vw_shopfloor_event_fact": _apply_selection(shop, filters),
        "vw_stop_event_fact": _apply_selection(stop, filters),
        "vw_inspection_event_fact": _apply_selection(insp, filters),
        "vw_tag_event_fact": _apply_selection(tag, filters),
        "vw_component_error_fact": _apply_selection(comp, filters),
    }

    filtered_shop = filtered_clean.get("vw_shopfloor_event_fact", pd.DataFrame()).copy()
    filtered_stop = filtered_clean.get("vw_stop_event_fact", pd.DataFrame()).copy()
    filtered_insp = filtered_clean.get("vw_inspection_event_fact", pd.DataFrame()).copy()
    filtered_tag = filtered_clean.get("vw_tag_event_fact", pd.DataFrame()).copy()
    filtered_comp = filtered_clean.get("vw_component_error_fact", pd.DataFrame()).copy()

    equipment = build_equipment_overview(filtered_clean)
    process = build_process_overview(filtered_clean)
    lot = build_lot_analysis_view(filtered_clean)
    time_view = build_time_pattern_view(filtered_clean)
    quality_overview = build_quality_overview(filtered_clean)

    def _safe_float(v, default: float = 0.0) -> float:
        try:
            if pd.isna(v):
                return default
            return float(v)
        except Exception:
            return default

    def _fmt_pct(v) -> str:
        return f"{_safe_float(v) * 100:.1f}%"

    def _fmt_sec_value(v) -> str:
        return _fmt_sec(_safe_float(v))

    def _fmt_sec_base(v: float) -> str:
        return _fmt_sec(v)

    def _norm(series: pd.Series) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce").fillna(0)
        if s.empty:
            return s
        mn = s.min()
        mx = s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - mn) / (mx - mn)

    def _top_reason(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
        if df.empty or group_col not in df.columns:
            return pd.DataFrame(columns=[group_col, value_col])
        out = df.groupby(group_col, as_index=False).agg(**{value_col: (value_col, "sum")})
        return out.sort_values(value_col, ascending=False)

    def _first_value(series: pd.Series, default: str = "-") -> str:
        if series is None or series.empty:
            return default
        vals = series.dropna().astype(str)
        vals = vals[vals.ne("") & vals.ne("nan") & vals.ne("None")]
        return str(vals.iloc[0]) if not vals.empty else default

    def _machine_ct(source: pd.DataFrame) -> pd.DataFrame:
        if source.empty or "machine_id" not in source.columns or "event_ts" not in source.columns:
            return pd.DataFrame(columns=["machine_id", "avg_ct_sec", "ct_std_sec"])
        tmp = source[["machine_id", "event_ts"]].copy()
        tmp["event_ts"] = pd.to_datetime(tmp["event_ts"], errors="coerce")
        tmp = tmp.dropna(subset=["machine_id", "event_ts"]).sort_values(["machine_id", "event_ts"])
        if tmp.empty:
            return pd.DataFrame(columns=["machine_id", "avg_ct_sec", "ct_std_sec"])
        tmp["ct_sec"] = tmp.groupby("machine_id")["event_ts"].diff().dt.total_seconds()
        out = tmp.groupby("machine_id", as_index=False).agg(avg_ct_sec=("ct_sec", "mean"), ct_std_sec=("ct_sec", "std"))
        out["avg_ct_sec"] = pd.to_numeric(out["avg_ct_sec"], errors="coerce").fillna(0)
        out["ct_std_sec"] = pd.to_numeric(out["ct_std_sec"], errors="coerce").fillna(0)
        return out

    def _process_ct(source: pd.DataFrame) -> pd.DataFrame:
        if source.empty or "process_name" not in source.columns or "event_ts" not in source.columns:
            return pd.DataFrame(columns=["process_name", "avg_ct_sec", "ct_std_sec"])
        tmp = source[["process_name", "event_ts"]].copy()
        tmp["event_ts"] = pd.to_datetime(tmp["event_ts"], errors="coerce")
        tmp = tmp.dropna(subset=["process_name", "event_ts"]).sort_values(["process_name", "event_ts"])
        if tmp.empty:
            return pd.DataFrame(columns=["process_name", "avg_ct_sec", "ct_std_sec"])
        tmp["ct_sec"] = tmp.groupby("process_name")["event_ts"].diff().dt.total_seconds()
        out = tmp.groupby("process_name", as_index=False).agg(avg_ct_sec=("ct_sec", "mean"), ct_std_sec=("ct_sec", "std"))
        out["avg_ct_sec"] = pd.to_numeric(out["avg_ct_sec"], errors="coerce").fillna(0)
        out["ct_std_sec"] = pd.to_numeric(out["ct_std_sec"], errors="coerce").fillna(0)
        return out

    def _retry_proxy_by_machine(source: pd.DataFrame) -> pd.DataFrame:
        if source.empty or "machine_id" not in source.columns:
            return pd.DataFrame(columns=["machine_id", "retry_rate"])
        key_cols = [c for c in ["machine_id", "lot_id", "model_label", "process_name", "result_primary"] if c in source.columns]
        if len(key_cols) < 2:
            return pd.DataFrame(columns=["machine_id", "retry_rate"])
        tmp = source[key_cols].copy()
        tmp["retry_flag"] = tmp.duplicated(subset=key_cols, keep=False).astype(int)
        out = tmp.groupby("machine_id", as_index=False).agg(retry_flag=("retry_flag", "sum"), total=("retry_flag", "size"))
        out["retry_rate"] = out.apply(lambda r: _safe_div(r.get("retry_flag", 0), r.get("total", 0)), axis=1)
        return out[["machine_id", "retry_rate"]]

    def _retry_proxy_by_lot(source: pd.DataFrame) -> pd.DataFrame:
        if source.empty or "lot_id" not in source.columns:
            return pd.DataFrame(columns=["lot_id", "retry_rate"])
        key_cols = [c for c in ["lot_id", "machine_id", "model_label", "process_name", "result_primary"] if c in source.columns]
        if len(key_cols) < 2:
            return pd.DataFrame(columns=["lot_id", "retry_rate"])
        tmp = source[key_cols].copy()
        tmp["retry_flag"] = tmp.duplicated(subset=key_cols, keep=False).astype(int)
        out = tmp.groupby("lot_id", as_index=False).agg(retry_flag=("retry_flag", "sum"), total=("retry_flag", "size"))
        out["retry_rate"] = out.apply(lambda r: _safe_div(r.get("retry_flag", 0), r.get("total", 0)), axis=1)
        return out[["lot_id", "retry_rate"]]

    def _issue_type_machine(row: pd.Series) -> str:
        stop_time = _safe_float(row.get("stop_time", 0))
        defect_rate = _safe_float(row.get("defect_rate", 0))
        retry_rate = _safe_float(row.get("retry_rate", 0))
        wait_count = _safe_float(row.get("wait_count", 0))
        if stop_time > 0.7 and defect_rate > 0.7:
            return "복합 문제형"
        if stop_time > 0.7 and defect_rate <= 0.7:
            return "생산성 손실형"
        if defect_rate > 0.7 and stop_time <= 0.4:
            return "품질 집중형"
        if wait_count > 0.6:
            return "공정 연계형"
        if retry_rate > 0.6:
            return "재작업 집중형"
        return "주의"

    def _issue_comment_machine(row: pd.Series) -> str:
        stop_reason = str(row.get("top_stop_reason", "-"))
        wait_reason = str(row.get("top_wait_reason", "-"))
        if "WAIT" in wait_reason.upper() or row.get("wait_count", 0) > 0:
            return "WAIT 비중이 높아 전후공정 연계와 자재 흐름을 우선 확인해야 합니다."
        if any(tok in stop_reason.upper() for tok in ["PICK", "FEED", "NOZZ", "REEL"]):
            return "PICKUP/FEEDER 계열 정지가 보여 자재 공급과 흡착 조건을 먼저 봐야 합니다."
        if any(tok in stop_reason.upper() for tok in ["RECOG", "VISION", "MARK", "CAM"]):
            return "RECOG/비전 계열 정지가 보여 인식 조건과 조명/카메라 상태 점검이 필요합니다."
        if any(tok in stop_reason.upper() for tok in ["PLACE", "OFFSET", "ALIGN"]):
            return "PLACE 계열 이슈 가능성이 높아 좌표 보정과 head 상태를 확인해야 합니다."
        if _safe_float(row.get("defect_rate", 0)) > _safe_float(row.get("stop_time", 0)) and _safe_float(row.get("defect_rate", 0)) > 0:
            return "정지보다 불량이 더 높아 품질 조건 중심으로 접근하는 것이 효율적입니다."
        return "정지와 불량의 조합을 기준으로 원인 우선순위를 정해야 합니다."

    def _action_machine(row: pd.Series) -> str:
        wait_reason = str(row.get("top_wait_reason", "-")).upper()
        stop_reason = str(row.get("top_stop_reason", "-")).upper()
        if "WAIT" in wait_reason:
            return "전후공정 buffer, feeder 보급 타이밍, line balance 점검"
        if any(tok in stop_reason for tok in ["PICK", "FEED", "NOZZ", "REEL"]):
            return "nozzle 점검, feeder 정렬, reel 품질 확인"
        if any(tok in stop_reason for tok in ["RECOG", "VISION", "MARK", "CAM"]):
            return "camera/vision tuning, mark 인식 조건, 조명 calibration 점검"
        if any(tok in stop_reason for tok in ["PLACE", "OFFSET", "ALIGN"]):
            return "placement offset 보정, head calibration, 흡착 안정성 점검"
        return "상위 reason Pareto 기준으로 설비 조건과 알람 코드 정비"

    def _issue_type_process(row: pd.Series) -> str:
        stop_time = _safe_float(row.get("stop_time", 0))
        defect_rate = _safe_float(row.get("fail_rate", row.get("defect_rate", 0)))
        wait_count = _safe_float(row.get("wait_count", 0))
        ct_std = _safe_float(row.get("ct_std_sec", 0))
        if stop_time > 0.7 and defect_rate > 0.7:
            return "복합 병목"
        if stop_time > 0.7:
            return "정지 병목"
        if defect_rate > 0.7:
            return "품질 병목"
        if wait_count > 0.6:
            return "흐름 병목"
        if ct_std > 0.7:
            return "안정성 문제"
        return "주의"

    def _action_process(row: pd.Series) -> str:
        process_name = str(row.get("process_display", row.get("process_name", "-"))).upper()
        if "AOI" in process_name:
            return "검사 조건, false call, review capacity 점검"
        if "SPI" in process_name:
            return "인쇄 품질, threshold, 검사 조건 재점검"
        if "MOUNTER" in process_name:
            return "feeder, nozzle, line balance, 자재 보급 타이밍 점검"
        if "PRINTER" in process_name:
            return "printing recipe, stencil, raw event 보강"
        return "공정별 stop/wait Pareto를 기준으로 병목 원인 정리"

    def _issue_type_lot(row: pd.Series) -> str:
        stop_time = _safe_float(row.get("stop_time", 0))
        machine_count = _safe_float(row.get("machine_count", 0))
        process_count = _safe_float(row.get("process_count", 0))
        impact = _safe_float(row.get("impact_score", 0))
        if stop_time > 0.7 and machine_count <= 0.4:
            return "국소 LOT 문제"
        if stop_time > 0.7 and process_count > 0.7:
            return "전파 LOT 문제"
        if impact > 0.8:
            return "우선 점검 LOT"
        return "주의"

    def _action_lot(row: pd.Series) -> str:
        machine_count = _safe_float(row.get("machine_count", 0))
        process_count = _safe_float(row.get("process_count", 0))
        if process_count > machine_count:
            return "trace 재확인, upstream/downstream 확산 여부 확인"
        if machine_count <= 1:
            return "대표 설비와 자재 lot 매핑 우선 확인"
        return "LOT별 자재, 교대, 셋업 이력과 반복 발생 패턴 확인"

    def _priority_comment(row: pd.Series) -> str:
        kind = str(row.get("대상유형", "-"))
        issue = str(row.get("문제유형", "-"))
        if kind == "설비":
            return "설비 조건을 손보면 바로 stop/defect 둘 다 줄일 가능성이 큽니다."
        if kind == "공정":
            return "공정 흐름을 막고 있어 throughput 회복 효과가 큽니다."
        if kind == "LOT":
            return "특정 LOT 확산을 막으면 고객 영향 범위를 빠르게 줄일 수 있습니다."
        return issue

    def _priority_action(row: pd.Series) -> str:
        kind = str(row.get("대상유형", "-"))
        issue = str(row.get("문제유형", "-"))
        if kind == "설비":
            return "nozzle / feeder / vision / interlock 중 상위 reason부터 조치"
        if kind == "공정":
            return "line balance, buffer, recipe, review capacity 순서로 개선"
        if kind == "LOT":
            return "대표 설비와 공정 trace를 먼저 확인하고 원인 확산을 차단"
        return issue

    machine_ct = _machine_ct(filtered_shop)
    process_ct = _process_ct(filtered_shop)
    retry_machine = _retry_proxy_by_machine(filtered_insp if not filtered_insp.empty else filtered_shop)
    retry_lot = _retry_proxy_by_lot(filtered_insp if not filtered_insp.empty else filtered_shop)

    if not equipment.empty and not machine_ct.empty:
        equipment = equipment.merge(machine_ct, on="machine_id", how="left")
    if not equipment.empty and not retry_machine.empty:
        equipment = equipment.merge(retry_machine, on="machine_id", how="left")
    if not process.empty and not process_ct.empty:
        process = process.merge(process_ct, on="process_name", how="left")
    if not lot.empty and not retry_lot.empty:
        lot = lot.merge(retry_lot, on="lot_id", how="left")

    if not filtered_tag.empty and "event_class" in filtered_tag.columns and "machine_id" in filtered_tag.columns:
        wait_machine = filtered_tag[filtered_tag["event_class"].eq("WAIT")].groupby("machine_id", as_index=False).size().rename(columns={"size": "wait_count"})
        if not equipment.empty:
            equipment = equipment.merge(wait_machine, on="machine_id", how="left")
    elif not equipment.empty:
        equipment["wait_count"] = 0

    if not filtered_tag.empty and "event_class" in filtered_tag.columns and "machine_id" in filtered_tag.columns:
        wait_reason = (
            filtered_tag[filtered_tag["event_class"].eq("WAIT")]
            .groupby(["machine_id", "cause_detail"], as_index=False)
            .agg(wait_count=("event_class", "size"))
            .sort_values(["machine_id", "wait_count"], ascending=[True, False])
        )
        wait_reason = wait_reason.groupby("machine_id").head(1).rename(columns={"cause_detail": "top_wait_reason"})
        if not equipment.empty:
            equipment = equipment.merge(wait_reason[["machine_id", "top_wait_reason"]], on="machine_id", how="left")

    if not filtered_stop.empty and "machine_id" in filtered_stop.columns:
        stop_reason = (
            filtered_stop.groupby(["machine_id", "stop_like_reason"], as_index=False)
            .agg(stop_time=("duration_sec", "sum"))
            .sort_values(["machine_id", "stop_time"], ascending=[True, False])
        )
        stop_reason = stop_reason.groupby("machine_id").head(1).rename(columns={"stop_like_reason": "top_stop_reason"})
        if not equipment.empty:
            equipment = equipment.merge(stop_reason[["machine_id", "top_stop_reason"]], on="machine_id", how="left")

    if not filtered_shop.empty and "process_name" in filtered_shop.columns and "machine_id" in filtered_shop.columns:
        process_map = (
            filtered_shop.groupby("machine_id", as_index=False)
            .agg(process_name=("process_name", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]))
        )
        if not filtered_tag.empty and "event_class" in filtered_tag.columns:
            tag_wait = filtered_tag[filtered_tag["event_class"].eq("WAIT")][["machine_id"]].copy()
            if not tag_wait.empty:
                tag_wait = tag_wait.merge(process_map, on="machine_id", how="left")
                wait_by_process = tag_wait.groupby("process_name", as_index=False).size().rename(columns={"size": "wait_count"})
                if not process.empty:
                    process = process.merge(wait_by_process, on="process_name", how="left")
    if not process.empty and "wait_count" not in process.columns:
        process["wait_count"] = 0

    def _series_or_zeros(frame: pd.DataFrame, col: str) -> pd.Series:
        if col in frame.columns:
            return pd.to_numeric(frame[col], errors="coerce")
        return pd.Series([0] * len(frame), index=frame.index, dtype="float64")

    if not lot.empty and "retry_rate" not in lot.columns:
        lot["retry_rate"] = 0
    if not equipment.empty and "retry_rate" not in equipment.columns:
        equipment["retry_rate"] = 0
    if not process.empty and "avg_ct_sec" not in process.columns:
        process["avg_ct_sec"] = 0
        process["ct_std_sec"] = 0

    if not equipment.empty:
        equipment["wait_count"] = _series_or_zeros(equipment, "wait_count").fillna(0)
        equipment["retry_rate"] = _series_or_zeros(equipment, "retry_rate").fillna(0)
        equipment["avg_ct_sec"] = _series_or_zeros(equipment, "avg_ct_sec").fillna(0)
        equipment["ct_std_sec"] = _series_or_zeros(equipment, "ct_std_sec").fillna(0)
        equipment["stop_time_norm"] = _norm(equipment["stop_time"] if "stop_time" in equipment.columns else pd.Series([0] * len(equipment)))
        equipment["defect_norm"] = _norm(equipment["defect_rate"] if "defect_rate" in equipment.columns else pd.Series([0] * len(equipment)))
        equipment["retry_norm"] = _norm(equipment["retry_rate"])
        equipment["wait_norm"] = _norm(equipment["wait_count"])
        equipment["ct_norm"] = _norm(equipment["ct_std_sec"])
        equipment["severity_score"] = (
            equipment["stop_time_norm"] * 0.35
            + equipment["defect_norm"] * 0.30
            + equipment["retry_norm"] * 0.15
            + equipment["wait_norm"] * 0.10
            + equipment["ct_norm"] * 0.10
        )
        q1 = equipment["severity_score"].quantile(0.50) if len(equipment) else 0
        q2 = equipment["severity_score"].quantile(0.80) if len(equipment) else 0
        equipment["상태 등급"] = np.select(
            [equipment["severity_score"].ge(q2), equipment["severity_score"].ge(q1)],
            ["개선필요", "주의"],
            default="정상",
        )
        equipment["문제유형"] = equipment.apply(_issue_type_machine, axis=1)
        equipment["해석"] = equipment.apply(_issue_comment_machine, axis=1)
        equipment["개선액션"] = equipment.apply(_action_machine, axis=1)

    if not process.empty:
        process["avg_ct_sec"] = _series_or_zeros(process, "avg_ct_sec").fillna(0)
        process["ct_std_sec"] = _series_or_zeros(process, "ct_std_sec").fillna(0)
        process["wait_count"] = _series_or_zeros(process, "wait_count").fillna(0)
        process["stop_norm"] = _norm(process["stop_time"] if "stop_time" in process.columns else pd.Series([0] * len(process)))
        process["defect_norm"] = _norm(process["fail_rate"] if "fail_rate" in process.columns else pd.Series([0] * len(process)))
        process["wait_norm"] = _norm(process["wait_count"])
        process["ct_norm"] = _norm(process["ct_std_sec"])
        process["throughput_norm"] = 1 - _norm(process["output_qty"] if "output_qty" in process.columns else pd.Series([0] * len(process)))
        process["bottleneck_norm"] = (
            process["stop_norm"] * 0.35
            + process["defect_norm"] * 0.25
            + process["wait_norm"] * 0.20
            + process["ct_norm"] * 0.20
        )
        process["문제유형"] = process.apply(_issue_type_process, axis=1)
        process["해석"] = process.apply(
            lambda r: "output 저하와 stop 증가가 함께 보여 전형적 병목입니다."
            if _safe_float(r.get("stop_time", 0)) > 0.7 and _safe_float(r.get("output_qty", 0)) <= _safe_float(process["output_qty"].median() if not process.empty else 0)
            else ("wait 증가로 전후공정 연계 문제를 먼저 봐야 합니다." if _safe_float(r.get("wait_count", 0)) > 0.6 else "CT 편차와 stop reason을 함께 봐야 합니다."),
            axis=1,
        )
        process["개선액션"] = process.apply(_action_process, axis=1)

    if not lot.empty:
        lot["retry_rate"] = _series_or_zeros(lot, "retry_rate").fillna(0)
        lot["impact_norm"] = _norm(lot["impact_score"] if "impact_score" in lot.columns else pd.Series([0] * len(lot)))
        lot["stop_norm"] = _norm(lot["stop_time"] if "stop_time" in lot.columns else pd.Series([0] * len(lot)))
        lot["defect_norm"] = _norm(lot["fail_rate"] if "fail_rate" in lot.columns else pd.Series([0] * len(lot)))
        lot["spread_norm"] = _norm(_series_or_zeros(lot, "machine_count").fillna(0) + _series_or_zeros(lot, "process_count").fillna(0))
        lot["retry_norm"] = _norm(lot["retry_rate"])
        lot["priority_score"] = (
            lot["impact_norm"] * 0.40
            + lot["stop_norm"] * 0.25
            + lot["defect_norm"] * 0.20
            + lot["spread_norm"] * 0.15
        )
        lot["문제유형"] = lot.apply(_issue_type_lot, axis=1)
        lot["해석"] = lot.apply(
            lambda r: "특정 설비에 국한된 국소 문제 가능성이 높습니다."
            if _safe_float(r.get("machine_count", 0)) <= 1.0
            else ("다수 공정/설비로 번지는 전파형 문제 가능성이 높습니다." if _safe_float(r.get("process_count", 0)) > _safe_float(r.get("machine_count", 0)) else "LOT 영향과 산출 손실을 함께 봐야 합니다."),
            axis=1,
        )
        lot["개선액션"] = lot.apply(_action_lot, axis=1)

    if not quality_overview.empty and "scope" in quality_overview.columns:
        quality_overview = quality_overview.copy()

    total_stop_time = _safe_float(equipment["stop_time"].sum() if not equipment.empty and "stop_time" in equipment.columns else 0)
    total_defect_count = _safe_float(filtered_insp[filtered_insp.get("quality_flag", pd.Series(dtype=str)).astype(str).eq("FAIL")].shape[0] if not filtered_insp.empty and "quality_flag" in filtered_insp.columns else 0)
    total_insp = _safe_float(len(filtered_insp))
    total_defect_rate = _safe_div(total_defect_count, total_insp)
    total_retry_rate = _safe_float(lot["retry_rate"].mean() if not lot.empty and "retry_rate" in lot.columns else 0)
    total_wait_count = _safe_float(filtered_tag[filtered_tag.get("event_class", pd.Series(dtype=str)).astype(str).eq("WAIT")].shape[0] if not filtered_tag.empty and "event_class" in filtered_tag.columns else 0)
    bottleneck_stage = process.sort_values("bottleneck_norm", ascending=False).iloc[0] if not process.empty else pd.Series(dtype="object")
    impact_lot_count = _safe_float((lot["impact_score"] > lot["impact_score"].median()).sum() if not lot.empty and "impact_score" in lot.columns else 0)
    top_machine = equipment.sort_values("severity_score", ascending=False).iloc[0] if not equipment.empty else pd.Series(dtype="object")
    top_process = process.sort_values("bottleneck_norm", ascending=False).iloc[0] if not process.empty else pd.Series(dtype="object")
    top_lot = lot.sort_values("priority_score", ascending=False).iloc[0] if not lot.empty else pd.Series(dtype="object")

    _panel(
        "탭 읽는 법",
        [
            "먼저 요약 카드로 전체 손실 규모를 봅니다.",
            "그 다음 설비 → 공정 → 정지/대기 → 불량/미스 → LOT → 우선순위 순서로 내려갑니다.",
            "각 표는 상위 값부터 보고, 해석 박스는 왜 그렇게 보는지 설명합니다.",
        ],
        [
            "이 탭은 단순 현황표가 아니라 개선 순서를 정하기 위한 판단 화면입니다.",
            "상위 값만 보지 말고 분류 근거와 추천 액션까지 같이 봐야 합니다.",
        ],
        tone="accent",
    )

    machine_stop_th = pd.to_numeric(equipment["stop_time"], errors="coerce").quantile(0.75) if not equipment.empty and "stop_time" in equipment.columns else 0
    machine_defect_th = pd.to_numeric(equipment["defect_rate"], errors="coerce").quantile(0.75) if not equipment.empty and "defect_rate" in equipment.columns else 0
    machine_wait_th = pd.to_numeric(equipment["wait_count"], errors="coerce").quantile(0.75) if not equipment.empty and "wait_count" in equipment.columns else 0
    process_output_th = pd.to_numeric(process["output_qty"], errors="coerce").quantile(0.25) if not process.empty and "output_qty" in process.columns else 0
    process_stop_th = pd.to_numeric(process["stop_time"], errors="coerce").quantile(0.75) if not process.empty and "stop_time" in process.columns else 0
    lot_spread_th = pd.to_numeric((lot.get("machine_count", 0) + lot.get("process_count", 0)), errors="coerce").quantile(0.75) if not lot.empty else 0
    lot_machine_th = pd.to_numeric(lot.get("machine_count", pd.Series([0])), errors="coerce").quantile(0.25) if not lot.empty else 0
    lot_process_th = pd.to_numeric(lot.get("process_count", pd.Series([0])), errors="coerce").quantile(0.75) if not lot.empty else 0
    lot_impact_th = pd.to_numeric(lot.get("impact_score", pd.Series([0])), errors="coerce").quantile(0.75) if not lot.empty else 0

    if not equipment.empty:
        equipment["problem_type"] = equipment.apply(
            lambda r: _problem_type_from_signals(
                _safe_float(r.get("stop_time", 0)),
                _safe_float(r.get("defect_rate", 0)),
                _safe_float(r.get("wait_count", 0)),
                _safe_float(r.get("retry_rate", 0)),
                machine_stop_th,
                machine_defect_th,
                machine_wait_th,
            ),
            axis=1,
        )
        equipment["quadrant"] = equipment.apply(
            lambda r: _quadrant_label(_safe_float(r.get("stop_time", 0)), _safe_float(r.get("defect_rate", 0)), machine_stop_th, machine_defect_th),
            axis=1,
        )
        equipment["confidence"] = equipment.apply(lambda r: _confidence_label(True, bool(_safe_float(r.get("wait_count", 0)) > 0 or _safe_float(r.get("retry_rate", 0)) > 0)), axis=1)
        equipment["status_label"] = equipment["quadrant"].map(_status_badge)
    if not process.empty:
        process["problem_type"] = process.apply(
            lambda r: _problem_type_from_signals(
                _safe_float(r.get("stop_time", 0)),
                _safe_float(r.get("fail_rate", r.get("defect_rate", 0))),
                _safe_float(r.get("wait_count", 0)),
                _safe_float(r.get("ct_std_sec", 0)),
                process_stop_th,
                pd.to_numeric(process["fail_rate"], errors="coerce").quantile(0.75) if "fail_rate" in process.columns else 0,
                pd.to_numeric(process["wait_count"], errors="coerce").quantile(0.75) if "wait_count" in process.columns else 0,
            ),
            axis=1,
        )
        process["confidence"] = process.apply(lambda r: _confidence_label(True, bool(_safe_float(r.get("wait_count", 0)) > 0)), axis=1)
    if not lot.empty:
        lot["problem_type"] = lot.apply(
            lambda r: "전파 LOT 문제"
            if _safe_float(r.get("process_count", 0)) >= lot_spread_th
            else ("국소 LOT 문제" if _safe_float(r.get("machine_count", 0)) <= 1 else ("우선 점검 LOT" if _safe_float(r.get("impact_score", 0)) >= 0.75 else "주의")),
            axis=1,
        )
        lot["confidence"] = lot.apply(lambda r: _confidence_label(True, bool(_safe_float(r.get("retry_rate", 0)) > 0)), axis=1)

    if not equipment.empty:
        equipment["분류 근거"] = equipment.apply(lambda r: _classification_basis_machine(r, machine_stop_th, machine_defect_th, machine_wait_th), axis=1)
    if not process.empty:
        process["분류 근거"] = process.apply(
            lambda r: _classification_basis_process(
                r,
                process_output_th,
                process_stop_th,
                pd.to_numeric(process["wait_count"], errors="coerce").quantile(0.75) if "wait_count" in process.columns else 0,
                pd.to_numeric(process["ct_std_sec"], errors="coerce").quantile(0.75) if "ct_std_sec" in process.columns else 0,
            ),
            axis=1,
        )
    if not lot.empty:
        lot["분류 근거"] = lot.apply(lambda r: _classification_basis_lot(r, lot_machine_th, lot_process_th, lot_impact_th), axis=1)

    alert_rows = []
    if not equipment.empty and _safe_float(top_machine.get("severity_score", 0)) >= 0.7:
        alert_rows.append({
            "대상": str(top_machine.get("machine_id", "-")),
            "경고": f"{top_machine.get('problem_type', '정상')} / {top_machine.get('quadrant', '정상')}",
            "액션": top_machine.get("개선액션", "-"),
        })
    if not process.empty and _safe_float(top_process.get("bottleneck_norm", 0)) >= 0.7:
        alert_rows.append({
            "대상": str(top_process.get("process_display", top_process.get("process_name", "-"))),
            "경고": str(top_process.get("problem_type", "정상")),
            "액션": top_process.get("개선액션", "-"),
        })
    if not lot.empty and _safe_float(top_lot.get("priority_score", 0)) >= 0.7:
        alert_rows.append({
            "대상": str(top_lot.get("lot_id", "-")),
            "경고": str(top_lot.get("problem_type", "정상")),
            "액션": top_lot.get("개선액션", "-"),
        })

    stop_reason = (
        filtered_stop.groupby("stop_like_reason", as_index=False).agg(loss_time=("duration_sec", "sum"), event_count=("stop_count", "sum")).sort_values("loss_time", ascending=False)
        if not filtered_stop.empty and "stop_like_reason" in filtered_stop.columns
        else pd.DataFrame(columns=["stop_like_reason", "loss_time", "event_count"])
    )
    wait_reason = (
        filtered_tag[filtered_tag["event_class"].eq("WAIT")].groupby("cause_detail", as_index=False).agg(wait_count=("event_class", "size")).sort_values("wait_count", ascending=False)
        if not filtered_tag.empty and "event_class" in filtered_tag.columns and "cause_detail" in filtered_tag.columns
        else pd.DataFrame(columns=["cause_detail", "wait_count"])
    )
    defect_reason = (
        filtered_insp.groupby("quality_flag", as_index=False).agg(defect_count=("quality_flag", "size")).sort_values("defect_count", ascending=False)
        if not filtered_insp.empty and "quality_flag" in filtered_insp.columns
        else pd.DataFrame(columns=["quality_flag", "defect_count"])
    )

    hour_view = time_view[time_view["grain"].eq("hour")].copy() if not time_view.empty and "grain" in time_view.columns else pd.DataFrame()
    shift_view = time_view[time_view["grain"].eq("shift")].copy() if not time_view.empty and "grain" in time_view.columns else pd.DataFrame()

    priority_parts = []
    if not equipment.empty:
        tmp = equipment.copy()
        tmp["대상유형"] = "설비"
        tmp["대상"] = tmp["machine_id"].astype(str)
        tmp["문제유형"] = tmp["문제유형"] if "문제유형" in tmp.columns else "주의"
        tmp["priority_score"] = tmp["severity_score"]
        tmp["근거 KPI"] = tmp.apply(lambda r: f"stop { _fmt_sec_base(_safe_float(r.get('stop_time', 0))) } / defect {_fmt_pct(r.get('defect_rate', 0)) }", axis=1)
        tmp["예상 영향"] = tmp.apply(lambda r: "정지와 불량 동시 개선" if r.get("문제유형") == "복합 문제형" else "정지 또는 품질 개선", axis=1)
        tmp["추천 액션"] = tmp["개선액션"] if "개선액션" in tmp.columns else "설비 조건 점검"
        tmp["기대 효과"] = tmp.apply(lambda r: "stop/defect 동시 개선 가능" if r.get("상태 등급") == "개선필요" else "일부 손실 회수", axis=1)
        priority_parts.append(tmp[["대상유형", "대상", "문제유형", "priority_score", "근거 KPI", "예상 영향", "추천 액션", "기대 효과"]])
    if not process.empty:
        tmp = process.copy()
        tmp["대상유형"] = "공정"
        tmp["대상"] = tmp["process_display"].astype(str)
        tmp["문제유형"] = tmp["문제유형"] if "문제유형" in tmp.columns else "주의"
        tmp["priority_score"] = tmp["bottleneck_norm"]
        tmp["근거 KPI"] = tmp.apply(lambda r: f"output {_safe_float(r.get('output_qty', 0)):.0f} / stop {_fmt_sec_base(_safe_float(r.get('stop_time', 0)))}", axis=1)
        tmp["예상 영향"] = tmp.apply(lambda r: "throughput 회복" if "병목" in str(r.get("문제유형", "")) else "품질 안정화", axis=1)
        tmp["추천 액션"] = tmp["개선액션"] if "개선액션" in tmp.columns else "공정 조건 점검"
        tmp["기대 효과"] = tmp.apply(lambda r: "병목 완화" if r.get("priority_score", 0) >= 0.7 else "공정 안정성 향상", axis=1)
        priority_parts.append(tmp[["대상유형", "대상", "문제유형", "priority_score", "근거 KPI", "예상 영향", "추천 액션", "기대 효과"]])
    if not lot.empty:
        tmp = lot.copy()
        tmp["대상유형"] = "LOT"
        tmp["대상"] = tmp["lot_id"].astype(str)
        tmp["문제유형"] = tmp["문제유형"] if "문제유형" in tmp.columns else "주의"
        tmp["priority_score"] = tmp["priority_score"] if "priority_score" in tmp.columns else tmp["impact_score"]
        tmp["근거 KPI"] = tmp.apply(lambda r: f"impact {_safe_float(r.get('impact_score', 0)):.1f} / stop {_fmt_sec_base(_safe_float(r.get('stop_time', 0)))}", axis=1)
        tmp["예상 영향"] = tmp.apply(lambda r: "다중 설비/공정 전파 차단" if _safe_float(r.get("process_count", 0)) > _safe_float(r.get("machine_count", 0)) else "국소 LOT 문제 차단", axis=1)
        tmp["추천 액션"] = tmp["개선액션"] if "개선액션" in tmp.columns else "LOT trace 확인"
        tmp["기대 효과"] = tmp.apply(lambda r: "재발 범위 축소" if r.get("priority_score", 0) >= 0.7 else "영향 LOT 분리", axis=1)
        priority_parts.append(tmp[["대상유형", "대상", "문제유형", "priority_score", "근거 KPI", "예상 영향", "추천 액션", "기대 효과"]])
    priority = pd.concat(priority_parts, ignore_index=True, sort=False) if priority_parts else pd.DataFrame()
    if not priority.empty:
        priority = priority.sort_values("priority_score", ascending=False).head(10).copy()
        priority["순위"] = np.arange(1, len(priority) + 1)
        priority = priority[["순위", "대상유형", "대상", "문제유형", "근거 KPI", "예상 영향", "추천 액션", "기대 효과", "priority_score"]]

    st.markdown('<div class="box">', unsafe_allow_html=True)
    st.markdown("### 2. 문제 분석 및 개선")
    st.caption("이 탭은 고객이 설비 / 공정 / LOT / 정지 / 불량을 한 화면에서 연결해 보고, 바로 개선 우선순위를 정하는 화면입니다.")
    st.markdown(
        """
        <div style="margin:.4rem 0 1rem 0;padding:.75rem .9rem;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);color:#d7dee8;">
            <b>읽는 순서</b> : 전체 문제 요약 → 설비 문제 → 공정 병목 → 정지/대기 원인 → 불량/미스 → LOT 영향 → 개선 우선순위
        </div>
        """,
        unsafe_allow_html=True,
    )
    _panel(
        "전체 문제 요약 해석",
        [
            "총 정지 시간, 불량률, 재작업률, 대기 건수, 병목 공정, 영향 LOT을 한 번에 봅니다.",
            "문제 대상 상위 3개와 분류 기준 표는 문제 위치를 빠르게 좁히기 위한 카드입니다.",
            "오른쪽 차트는 정지 원인, 대기 원인, 불량 유형, 정지 시간대를 함께 보여줍니다.",
        ],
        [
            "여기서는 현재 라인에서 무엇이 가장 무거운 손실인지 먼저 봅니다.",
            "정지와 불량이 함께 높으면 복합 문제, 특정 LOT가 높으면 전파 범위부터 봐야 합니다.",
        ],
        tone="info",
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    cards = [
        ("총 정지 시간", _fmt_sec_base(total_stop_time), "절대 정지 손실"),
        ("총 불량률", _fmt_pct(total_defect_rate), "불량 비율"),
        ("총 재작업률", _fmt_pct(total_retry_rate), "재작업 추정치"),
        ("총 대기 건수", f"{int(total_wait_count):,}", "대기 이벤트 수"),
        ("병목 공정", str(bottleneck_stage.get("process_display", bottleneck_stage.get("process_name", "-"))), "bottleneck_score 최고"),
        ("영향 LOT", f"{int(impact_lot_count):,}", "중간 이상 영향 LOT"),
    ]
    for col, (label, value, foot) in zip([c1, c2, c3, c4, c5, c6], cards):
        with col:
            st.markdown(_card(label, value, foot), unsafe_allow_html=True)

    if alert_rows:
        st.markdown("#### 경보 현황")
        alert_df = pd.DataFrame(alert_rows)
        st.dataframe(alert_df, use_container_width=True, hide_index=True)
    else:
        st.info("현재 필터 조건에서 즉시 경고 수준의 이상은 제한적입니다.")

    st.markdown("#### 행동형 인사이트 카드")
    insight_cols = st.columns(3)
    insight_items = [
        {
            "target": str(top_machine.get("machine_id", "-")),
            "type": str(top_machine.get("problem_type", "정상")),
            "evidence": f"정지 { _fmt_sec_base(_safe_float(top_machine.get('stop_time', 0))) } / 불량 { _fmt_pct(top_machine.get('defect_rate', 0)) } / 대기 { int(_safe_float(top_machine.get('wait_count', 0))) }",
            "cause": str(top_machine.get("top_wait_reason", top_machine.get("top_stop_reason", "-"))),
            "action": str(top_machine.get("개선액션", "설비 조건 점검")),
            "confidence": str(top_machine.get("confidence", "Actual")),
        },
        {
            "target": str(top_process.get("process_display", top_process.get("process_name", "-"))),
            "type": str(top_process.get("problem_type", "정상")),
            "evidence": f"출력 {_safe_float(top_process.get('output_qty', 0)):.0f} / 정지 {_fmt_sec_base(_safe_float(top_process.get('stop_time', 0)))} / 대기 {int(_safe_float(top_process.get('wait_count', 0)))}",
            "cause": str(top_process.get("bottleneck_hint", top_process.get("문제유형", "-"))),
            "action": str(top_process.get("개선액션", "공정 조건 점검")),
            "confidence": str(top_process.get("confidence", "Actual")),
        },
        {
            "target": str(top_lot.get("lot_id", "-")),
            "type": str(top_lot.get("problem_type", "정상")),
            "evidence": f"영향도 {_safe_float(top_lot.get('impact_score', 0)):.1f} / 확산 {int(_safe_float(top_lot.get('machine_count', 0)) + _safe_float(top_lot.get('process_count', 0)))}",
            "cause": f"확산 = 설비 {int(_safe_float(top_lot.get('machine_count', 0)))}개 / 공정 {int(_safe_float(top_lot.get('process_count', 0)))}개",
            "action": str(top_lot.get("개선액션", "LOT trace 확인")),
            "confidence": str(top_lot.get("confidence", "Actual")),
        },
    ]
    for col, item in zip(insight_cols, insight_items):
        with col:
            st.markdown(
                f"""
                <div style="padding:.95rem 1rem;border-radius:16px;background:linear-gradient(180deg,#263244,#1a2330);border:1px solid rgba(148,163,184,.18);min-height:180px;box-shadow:0 10px 24px rgba(15,23,42,.12)">
                    <div style="font-size:.78rem;color:#cbd5e1;">대상</div>
                    <div style="font-size:1.15rem;font-weight:800;color:#f8fafc;margin:.15rem 0 .35rem 0;">{item['target']}</div>
                    <div style="margin-bottom:.3rem;color:#f8c77d;font-weight:700;">분류: {item['type']}</div>
                    <div style="font-size:.82rem;color:#d8dee8;">{item['evidence']}</div>
                    <div style="font-size:.82rem;color:#d8dee8;margin-top:.25rem;">원인: {item['cause']}</div>
                    <div style="font-size:.82rem;color:#9fd0ff;margin-top:.35rem;">권장 조치: {item['action']}</div>
                    <div style="font-size:.75rem;color:#9fb0c4;margin-top:.35rem;">신뢰도: {item['confidence']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(_section_header("전체 문제 요약", "고객이 첫 화면에서 어디가 문제인지 바로 보게 하는 구역", PRIMARY), unsafe_allow_html=True)
    left, right = st.columns([0.5, 0.5])
    with left:
        st.markdown("#### 문제 대상 상위 3개")
        summary_rows = []
        if not equipment.empty:
            for _, row in equipment.sort_values("severity_score", ascending=False).head(3).iterrows():
                summary_rows.append({
                    "대상": str(row.get("machine_id", "-")),
                    "분류": row.get("문제유형", "-"),
                    "분류 근거": row.get("분류 근거", "-"),
                    "개선 아이디어": row.get("개선액션", "-"),
                })
        if not process.empty:
            for _, row in process.sort_values("bottleneck_norm", ascending=False).head(3).iterrows():
                summary_rows.append({
                    "대상": f"{row.get('process_display', row.get('process_name', '-'))}",
                    "분류": row.get("문제유형", "-"),
                    "분류 근거": row.get("분류 근거", "-"),
                    "개선 아이디어": row.get("개선액션", "-"),
                })
        if not lot.empty:
            for _, row in lot.sort_values("priority_score", ascending=False).head(3).iterrows():
                summary_rows.append({
                    "대상": str(row.get("lot_id", "-")),
                    "분류": row.get("문제유형", "-"),
                    "분류 근거": row.get("분류 근거", "-"),
                    "개선 아이디어": row.get("개선액션", "-"),
                })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        st.markdown("#### 분류 기준 표")
        basis_rows = []
        if not equipment.empty:
            for _, row in equipment.sort_values("severity_score", ascending=False).head(2).iterrows():
                basis_rows.append({"구분": "설비", "대상": str(row.get("machine_id", "-")), "분류": row.get("문제유형", "-"), "기준": row.get("분류 근거", "-")})
        if not process.empty:
            for _, row in process.sort_values("bottleneck_norm", ascending=False).head(2).iterrows():
                basis_rows.append({"구분": "공정", "대상": str(row.get("process_display", row.get("process_name", "-"))), "분류": row.get("문제유형", "-"), "기준": row.get("분류 근거", "-")})
        if not lot.empty:
            for _, row in lot.sort_values("priority_score", ascending=False).head(2).iterrows():
                basis_rows.append({"구분": "LOT", "대상": str(row.get("lot_id", "-")), "분류": row.get("문제유형", "-"), "기준": row.get("분류 근거", "-")})
        st.dataframe(pd.DataFrame(basis_rows), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### 원인 및 시간대")
        rc1, rc2 = st.columns(2)
        rc3, rc4 = st.columns(2)
        with rc1:
            if not stop_reason.empty:
                fig = px.bar(stop_reason.head(5), x="stop_like_reason", y="loss_time", text="loss_time", title="상위 정지 원인")
                st.plotly_chart(_plot_style(fig, "상위 정지 원인", 260), use_container_width=True)
        with rc2:
            if not wait_reason.empty:
                fig = px.bar(wait_reason.head(5), x="cause_detail", y="wait_count", text="wait_count", title="상위 대기 원인")
                st.plotly_chart(_plot_style(fig, "상위 대기 원인", 260), use_container_width=True)
        with rc3:
            if not defect_reason.empty:
                fig = px.bar(defect_reason.head(5), x="quality_flag", y="defect_count", text="defect_count", title="상위 불량 유형")
                st.plotly_chart(_plot_style(fig, "상위 불량 유형", 260), use_container_width=True)
        with rc4:
            if not hour_view.empty and "stop_time" in hour_view.columns:
                fig = px.line(hour_view, x="bucket", y="stop_time", markers=True, title="정지 집중 시간대")
                st.plotly_chart(_plot_style(fig, "정지 집중 시간대", 260), use_container_width=True)

    st.markdown(_section_header("설비별 문제 분석", "설비별 stop / defect / wait / CT 편차를 한 번에 보는 구역", SECONDARY), unsafe_allow_html=True)
    _panel(
        "설비별 문제 분석 해석",
        [
            "설비별 stop_time, defect_rate, retry_rate, wait_count, CT 편차를 같이 봅니다.",
            "산점도는 정지-불량 사분면으로 설비를 나누고, 막대는 상위 정지 원인을 보여줍니다.",
            "표는 분류 근거와 개선액션까지 같이 보여줍니다.",
        ],
        [
            "정지와 불량이 같이 높은 설비는 최우선 점검 대상입니다.",
            "정지는 높은데 불량이 낮으면 생산성 손실형, 불량이 높고 정지가 낮으면 품질 집중형입니다.",
            "기대보다 흔들림이 크면 설비 안정성 문제로 봐야 합니다.",
        ],
        tone="accent",
    )
    left, right = st.columns([0.58, 0.42])
    with left:
        if not equipment.empty:
            machine_table = equipment.sort_values("severity_score", ascending=False).head(10).copy()
            display_cols = [c for c in ["machine_id", "line_id", "stage_no", "stop_time", "stop_count", "defect_rate", "retry_rate", "avg_ct_sec", "ct_std_sec", "wait_count", "top_stop_reason", "top_wait_reason", "status_label", "problem_type", "분류 근거", "confidence", "해석", "개선액션"] if c in machine_table.columns]
            st.dataframe(machine_table[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("설비 분석 데이터가 부족합니다.")
    with right:
        if not equipment.empty:
            x_th = machine_stop_th if pd.notna(machine_stop_th) else pd.to_numeric(equipment["stop_time"], errors="coerce").median()
            y_th = machine_defect_th if pd.notna(machine_defect_th) else pd.to_numeric(equipment["defect_rate"], errors="coerce").median()
            machine_plot = equipment.copy()
            machine_plot["quadrant"] = machine_plot.apply(
                lambda r: _quadrant_label(_safe_float(r.get("stop_time", 0)), _safe_float(r.get("defect_rate", 0)), x_th, y_th),
                axis=1,
            )
            fig = px.scatter(
                machine_plot,
                x="stop_time",
                y="defect_rate",
                size="retry_rate" if "retry_rate" in machine_plot.columns else None,
                color="quadrant",
                hover_name="machine_id",
                text="machine_id",
                color_discrete_map={
                    "심각": "#ef4444",
                    "생산성 손실형": "#f59e0b",
                    "품질 집중형": "#3b82f6",
                    "정상": "#64748b",
                },
            )
            fig.add_vline(x=x_th, line_dash="dash", line_color="#94a3b8")
            fig.add_hline(y=y_th, line_dash="dash", line_color="#94a3b8")
            fig.update_layout(legend_title_text="분류 구역")
            st.plotly_chart(_plot_style(fig, "설비별 정지-불량 분류도", 320), use_container_width=True)
            if "top_stop_reason" in equipment.columns:
                reason_fig = px.bar(equipment.sort_values("severity_score", ascending=False).head(8), x="machine_id", y="stop_time", color="top_stop_reason", text="stop_time")
                st.plotly_chart(_plot_style(reason_fig, "설비별 정지 원인 순위", 280), use_container_width=True)
            st.markdown("##### 분류 해석 기준")
            st.markdown("- 심각: 정지와 불량이 동시에 높아 즉시 점검이 필요합니다.")
            st.markdown("- 생산성 손실형: 정지가 높고 불량은 낮아 설비/연계 손실이 핵심입니다.")
            st.markdown("- 품질 집중형: 불량이 높고 정지는 낮아 조건/자재/비전 점검이 우선입니다.")
            st.markdown("- 다음 확인: 상위 정지 원인과 대기 원인입니다.")

    st.markdown(_section_header("공정별 병목 분석", "output / stop / defect / wait / CT 편차로 병목을 판정하는 구역", PRIMARY), unsafe_allow_html=True)
    _panel(
        "공정별 병목 해석",
        [
            "공정별 output, stop_time, fail_rate, wait_count, CT 편차를 비교합니다.",
            "산점도는 출력과 정지의 관계를 보여주고, 막대는 대기량을 보여줍니다.",
            "표는 병목 점수와 해석, 개선액션을 함께 보여줍니다.",
        ],
        [
            "출력이 낮고 정지가 높으면 전형적인 병목입니다.",
            "대기가 높으면 전후공정 흐름 문제를 먼저 봐야 합니다.",
            "CT 편차가 크면 takt drift나 작업 조건 변화를 의심합니다.",
        ],
        tone="info",
    )
    left, right = st.columns([0.58, 0.42])
    with left:
        if not process.empty:
            process_table = process.sort_values("bottleneck_norm", ascending=False).head(10).copy()
            display_cols = [c for c in ["process_display", "line_id", "stage_no", "output_qty", "stop_time", "fail_rate", "wait_count", "avg_ct_sec", "ct_std_sec", "bottleneck_norm", "problem_type", "분류 근거", "confidence", "해석", "개선액션"] if c in process_table.columns]
            st.dataframe(process_table[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("공정 분석 데이터가 부족합니다.")
    with right:
        if not process.empty:
            fig = px.scatter(
                process,
                x="output_qty",
                y="stop_time",
                size="bottleneck_norm",
                color="문제유형" if "문제유형" in process.columns else None,
                hover_name="process_display",
                text="process_display",
            )
            st.plotly_chart(_plot_style(fig, "공정별 출력-정지 분류도", 320), use_container_width=True)
            if "wait_count" in process.columns:
                fig = px.bar(process.sort_values("bottleneck_norm", ascending=False), x="process_display", y="wait_count", text="wait_count")
                st.plotly_chart(_plot_style(fig, "공정별 대기 분포", 280), use_container_width=True)
            st.markdown("##### 분류 해석 기준")
            st.markdown("- 정지 병목: 출력이 낮고 정지가 높아 흐름을 직접 막는 공정입니다.")
            st.markdown("- 품질 병목: 정지는 낮지만 불량이 높아 검사/조건 재점검이 필요합니다.")
            st.markdown("- 흐름 병목: 대기가 높아 전후공정 밸런스부터 봐야 합니다.")
            st.markdown("- 안정성 문제: CT 편차가 커서 takt drift와 recipe 변화를 확인해야 합니다.")

    st.markdown(_section_header("정지·대기 원인 분석", "정지와 대기를 분리해서 손실 구조를 보는 구역", SECONDARY), unsafe_allow_html=True)
    _panel(
        "정지·대기 원인 해석",
        [
            "정지 원인과 대기 원인을 분리해서 봅니다.",
            "원인별 개선 방향 표는 어떤 조건을 먼저 조치할지 보여줍니다.",
            "시간대 / Shift 차트는 특정 시점에 몰리는지를 확인합니다.",
        ],
        [
            "Bwait / McFwait는 전공정 불균형, Rwait / McRwait는 후공정 적체로 해석합니다.",
            "Cwait는 검사/비전 지연, Pwait는 pickup/공급 문제, CnvStop는 전달/인터락 문제로 봅니다.",
            "특정 시간대에 몰리면 교대, 자재보급, 셋업 시점을 의심합니다.",
        ],
        tone="warn",
    )
    left, right = st.columns([0.56, 0.44])
    with left:
        if not stop_reason.empty:
            st.markdown("#### 정지 원인 상위 10개")
            st.dataframe(stop_reason.head(10), use_container_width=True, hide_index=True)
        if not wait_reason.empty:
            st.markdown("#### 대기 원인 상위 10개")
            st.dataframe(wait_reason.head(10), use_container_width=True, hide_index=True)
        st.markdown("#### 원인별 개선 방향")
        reason_map = pd.DataFrame(
            [
                {"원인": "Bwait / McFwait", "분류": "전공정 불균형", "개선 방향": "전공정 공급, feeder 보급, 버퍼 조정"},
                {"원인": "Rwait / McRwait", "분류": "후공정 적체", "개선 방향": "후공정 처리능력, 대기 버퍼, 인원 배치"},
                {"원인": "Cwait", "분류": "검사/비전 지연", "개선 방향": "camera, vision, review capacity 점검"},
                {"원인": "Pwait", "분류": "pickup/공급 문제", "개선 방향": "nozzle, feeder, reel, 보급 타이밍 점검"},
                {"원인": "CnvStop", "분류": "전달/인터락 문제", "개선 방향": "conveyor sensor, interlock, transfer 확인"},
                {"원인": "SCStop / SCEStop", "분류": "알람/시스템 불안정", "개선 방향": "알람 코드, 시스템 안정성, 재발 조건 확인"},
                {"원인": "OthrStop", "분류": "미분류", "개선 방향": "stop code 세분화 및 기록 표준화"},
            ]
        )
        st.dataframe(reason_map, use_container_width=True, hide_index=True)
    with right:
        right_top, right_bottom = st.columns(2)
        with right_top:
            if not hour_view.empty:
                metric_col = "stop_time" if "stop_time" in hour_view.columns and hour_view["stop_time"].sum() > 0 else "error_count"
                fig = px.bar(hour_view, x="bucket", y=metric_col, text=metric_col)
                st.plotly_chart(_plot_style(fig, "시간대별 정지 집중", 280), use_container_width=True)
        with right_bottom:
            if not shift_view.empty:
                metric_col = "stop_time" if "stop_time" in shift_view.columns else "error_count"
                fig = px.bar(shift_view, x="bucket", y=metric_col, text=metric_col)
                st.plotly_chart(_plot_style(fig, "Shift별 정지 비교", 280), use_container_width=True)

    st.markdown(_section_header("불량 / 미스 분석", "미스 유형과 불량이 어디서 발생하는지 보는 구역", PRIMARY), unsafe_allow_html=True)
    _panel(
        "불량 / 미스 해석",
        [
            "PMiss, DMiss, HMiss, MMiss, RMiss를 분리해서 봅니다.",
            "미스 분포, 시간대별 불량 추이, 공정별 불량률, 설비별 불량률을 같이 봅니다.",
            "미스 유형별 개선 방향 표는 어떤 조건을 먼저 바꿔야 하는지 보여줍니다.",
        ],
        [
            "PMiss는 흡착/공급, DMiss는 인식/조명, HMiss는 헤드, MMiss는 위치 보정, RMiss는 리젝트 로직 문제로 봅니다.",
            "Retry가 증가하면 첫 통과 품질이 흔들리고 있다는 뜻입니다.",
        ],
        tone="accent",
    )
    left, right = st.columns([0.56, 0.44])
    miss_view = pd.DataFrame()
    if not filtered_insp.empty and "quality_flag" in filtered_insp.columns:
        miss_view = filtered_insp["quality_flag"].astype(str).value_counts().reset_index()
        miss_view.columns = ["quality_flag", "count"]
    with left:
        if not miss_view.empty:
            st.markdown("#### 미스 유형 분포")
            st.dataframe(miss_view, use_container_width=True, hide_index=True)
        if not lot.empty:
            st.markdown("#### LOT 영향 상위 10개")
            lot_table = lot.sort_values("priority_score", ascending=False).head(10).copy()
            display_cols = [c for c in ["lot_id", "model_label", "output_qty", "stop_time", "fail_rate", "retry_rate", "impact_score", "machine_count", "process_count", "stop_machine_count", "stop_process_count", "representative_machine", "representative_process", "problem_type", "분류 근거", "confidence", "해석", "개선액션"] if c in lot_table.columns]
            st.dataframe(lot_table[display_cols], use_container_width=True, hide_index=True)
        st.markdown("#### 미스 유형별 개선 방향")
        miss_map = pd.DataFrame(
            [
                {"미스": "PMiss", "해석": "흡착 실패", "개선 방향": "nozzle, feeder, reel 공급 안정성 점검"},
                {"미스": "DMiss", "해석": "인식 실패", "개선 방향": "camera, lighting, vision threshold 점검"},
                {"미스": "HMiss", "해석": "헤드 실패", "개선 방향": "head calibration, 흡착 조건 점검"},
                {"미스": "MMiss", "해석": "배치 오프셋", "개선 방향": "placement offset, board alignment 보정"},
                {"미스": "RMiss", "해석": "리젝트 판정", "개선 방향": "reject 로직과 판정 기준 재점검"},
            ]
        )
        st.dataframe(miss_map, use_container_width=True, hide_index=True)
    with right:
        right_top, right_bottom = st.columns(2)
        with right_top:
            if not filtered_insp.empty and "quality_flag" in filtered_insp.columns:
                fig = px.bar(miss_view.head(5), x="quality_flag", y="count", text="count")
                st.plotly_chart(_plot_style(fig, "미스 분포", 260), use_container_width=True)
            if not hour_view.empty and "error_count" in hour_view.columns:
                fig = px.line(hour_view, x="bucket", y="error_count", markers=True)
                st.plotly_chart(_plot_style(fig, "시간대별 불량 추이", 260), use_container_width=True)
        with right_bottom:
            if not process.empty and "fail_rate" in process.columns:
                fig = px.bar(process.sort_values("fail_rate", ascending=False), x="process_display", y="fail_rate", text="fail_rate")
                st.plotly_chart(_plot_style(fig, "공정별 불량률", 260), use_container_width=True)
            if not equipment.empty and "defect_rate" in equipment.columns:
                fig = px.bar(equipment.sort_values("defect_rate", ascending=False).head(8), x="machine_id", y="defect_rate", text="defect_rate")
                st.plotly_chart(_plot_style(fig, "설비별 불량률", 260), use_container_width=True)
        st.markdown("##### 분류 해석 기준")
        st.markdown("- PMiss는 흡착/공급, DMiss는 인식/조명, HMiss는 헤드, MMiss는 위치 보정, RMiss는 리젝트 로직을 먼저 봅니다.")
        st.markdown("- Retry가 증가하면 첫 통과 품질이 흔들리고 있다는 뜻입니다.")

    st.markdown(_section_header("LOT 영향 분석", "문제가 특정 LOT에 국한되는지, 다수 공정으로 번지는지 보는 구역", SECONDARY), unsafe_allow_html=True)
    _panel(
        "LOT 영향 해석",
        [
            "LOT별 output, stop_time, fail_rate, retry_rate, impact_score, machine_count, process_count를 봅니다.",
            "산점도는 LOT가 국소인지 전파인지 보여주고, 막대는 영향도 상위를 보여줍니다.",
            "표는 대표 설비 / 대표 공정과 분류 근거를 같이 보여줍니다.",
        ],
        [
            "machine_count가 낮고 impact_score가 높으면 국소 LOT 문제입니다.",
            "process_count가 높으면 다수 공정으로 번지는 전파 LOT 문제입니다.",
            "영향도 상위 LOT부터 먼저 확인하면 고객 영향 범위를 빠르게 줄일 수 있습니다.",
        ],
        tone="info",
    )
    left, right = st.columns([0.58, 0.42])
    with left:
        if not lot.empty:
            lot_cols = [c for c in ["lot_id", "model_label", "output_qty", "stop_time", "fail_rate", "retry_rate", "impact_score", "machine_count", "process_count", "stop_machine_count", "stop_process_count", "representative_machine", "representative_process", "problem_type", "분류 근거", "confidence", "해석", "개선액션"] if c in lot.columns]
            st.dataframe(lot.sort_values("priority_score", ascending=False).head(10)[lot_cols], use_container_width=True, hide_index=True)
        else:
            st.info("LOT 분석 데이터가 부족합니다.")
    with right:
        right_top, right_bottom = st.columns(2)
        with right_top:
            if not lot.empty:
                fig = px.scatter(
                    lot,
                    x="machine_count" if "machine_count" in lot.columns else "impact_score",
                    y="process_count" if "process_count" in lot.columns else "impact_score",
                    size="impact_score" if "impact_score" in lot.columns else None,
                    color="문제유형" if "문제유형" in lot.columns else None,
                    hover_name="lot_id",
                    text="lot_id",
                )
                st.plotly_chart(_plot_style(fig, "LOT 확산-영향 분류도", 280), use_container_width=True)
        with right_bottom:
            if not lot.empty:
                fig = px.bar(lot.sort_values("impact_score", ascending=False).head(8), x="lot_id", y="impact_score", text="impact_score")
                st.plotly_chart(_plot_style(fig, "LOT 영향도 상위 8개", 280), use_container_width=True)
        st.markdown("##### 분류 해석 기준")
        st.markdown("- machine_count가 낮고 impact_score가 높으면 국소 LOT 문제입니다.")
        st.markdown("- process_count가 높으면 다수 공정으로 퍼지는 전파 LOT 문제입니다.")
        st.markdown("- 다음 확인: 대표 설비와 대표 공정입니다.")

    st.markdown(_section_header("개선 우선순위", "고객이 바로 무엇부터 할지 정하는 구역", PRIMARY), unsafe_allow_html=True)
    _panel(
        "개선 우선순위 해석",
        [
            "정지 영향도, 불량 영향도, LOT 확산도, 반복 발생도를 합산해 우선순위를 정합니다.",
            "우선순위 표는 무엇을 먼저 할지, 누구에게 맡길지, 어떤 데이터를 더 봐야 하는지를 정리합니다.",
            "자동 코멘트 예시는 고객 설명용 문장으로 바로 쓸 수 있습니다.",
        ],
        [
            "정지 + 불량 + 확산 + 반복이 동시에 높은 대상부터 먼저 개선해야 효과가 빠릅니다.",
            "우선순위는 단순 점수가 아니라 현장 실행 순서입니다.",
        ],
        tone="accent",
    )
    if not priority.empty:
        st.dataframe(priority, use_container_width=True, hide_index=True)
        st.markdown("#### 우선순위 산정 방식")
        st.markdown("- 정지 영향도, 불량 영향도, LOT 확산도, 반복 발생도를 합산합니다.")
        st.markdown("- `정지 + 불량 + 확산 + 반복`이 동시에 높은 대상을 먼저 개선합니다.")
    else:
        st.info("개선 우선순위를 만들 데이터가 부족합니다.")

    st.markdown("#### 자동 코멘트 예시")
    _panel(
        "자동 코멘트 해석",
        [
            "설비 / 공정 / LOT의 대표 해석을 한 줄씩 보여줍니다.",
            "고객 보고용으로 바로 읽을 수 있게 짧게 요약합니다.",
        ],
        [
            "이 문장은 각 표의 상위 항목을 사람이 읽기 쉬운 문장으로 바꾼 것입니다.",
            "발표할 때는 이 문장을 먼저 말하고, 아래 표를 근거로 붙이면 이해가 쉽습니다.",
        ],
        tone="info",
    )
    if not equipment.empty:
        st.markdown(f"- `{top_machine.get('machine_id', '-')}`: {top_machine.get('해석', '설비 조건을 점검해야 합니다.')} {top_machine.get('개선액션', '')}")
    if not process.empty:
        st.markdown(f"- `{top_process.get('process_display', top_process.get('process_name', '-'))}`: {top_process.get('해석', '공정 흐름을 점검해야 합니다.')} {top_process.get('개선액션', '')}")
    if not lot.empty:
        st.markdown(f"- `{top_lot.get('lot_id', '-')}`: {top_lot.get('해석', 'LOT 영향 범위를 확인해야 합니다.')} {top_lot.get('개선액션', '')}")
    st.markdown("#### 개선 액션 분류")
    _panel(
        "개선 액션 해석",
        [
            "설비, 공정, 자재/운영, 데이터 개선으로 나눠서 봅니다.",
            "각 액션은 현장에서 바로 실행할 수 있는 행동으로 적습니다.",
        ],
        [
            "어떤 팀이 먼저 움직여야 하는지, 무엇을 먼저 점검해야 하는지 이 구간에서 정리합니다.",
            "고객 설명에서는 '왜 이 조치를 해야 하는지'를 위의 해석과 연결해서 말하면 됩니다.",
        ],
        tone="accent",
    )
    st.markdown("- 설비 조건 개선: nozzle, feeder, head calibration, vision, conveyor/interlock")
    st.markdown("- 공정 조건 개선: line balance, buffer, recipe, review capacity, takt 안정화")
    st.markdown("- 자재/운영 개선: reel 품질, vendor lot, 교대시간, 보급 타이밍, reason 코드 정비")
    st.markdown("- 데이터 개선: stop/wait 구분, reason 코드 세분화, raw event 보강")
    st.markdown("#### 해석 주의")
    _panel(
        "해석 주의",
        [
            "Retry는 원천 필드가 없으면 검사 중복 proxy로 계산될 수 있습니다.",
            "CT는 event timestamp 간격 proxy이므로 설비 내부 cycle time과 다를 수 있습니다.",
        ],
        [
            "이 탭의 숫자는 실제 원천 + proxy가 섞여 있을 수 있으므로, 비교 시 정의를 같이 봐야 합니다.",
            "고객에게는 숫자보다 분류 근거와 개선액션을 같이 설명하는 것이 안전합니다.",
        ],
        tone="warn",
    )
    st.markdown("- Retry는 현재 데이터에서 원천 필드가 없으면 검사 중복 proxy로 계산합니다.")
    st.markdown("- CT는 event timestamp 간격 proxy이므로 설비 내부 cycle time과 다를 수 있습니다.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_equipment_screen(clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame], mode: str = "full"):
    item = clean.get("vw_mounter_item_fact", pd.DataFrame()).copy()
    is_full_flow = mode == "full"
    st.markdown('<div class="box">', unsafe_allow_html=True)
    if mode == "overview":
        st.markdown("### 2. 문제 진단")
        st.caption("이 탭은 고객이 가장 궁금해하는 '어디가 먼저 문제인지'를 빠르게 찾기 위한 화면입니다. 설비, 공정, LOT 중 어디를 먼저 봐야 하는지 우선순위를 제시합니다.")
        _story_box(
            "이 화면의 의사결정 포인트",
            [
                "생산량이 떨어지거나 흐름이 흔들리는 구간을 먼저 찾습니다.",
                "문제가 한 설비의 국소 이슈인지, 여러 공정과 LOT로 번지는 이슈인지 구분합니다.",
                "최종적으로 현장에서 먼저 점검할 대상을 좁혀주는 것이 목적입니다.",
            ],
            tone="dark",
        )
    elif is_full_flow:
        st.markdown("### 2. 이상치 분석 기반 문제 진단")
        st.caption("데이터 설명 다음에 이상치를 먼저 찾고, 그 근거를 그대로 이어서 고객이 이해할 수 있는 문제 진단 시나리오로 정리합니다.")
        _story_box(
            "이 화면의 진행 순서",
            [
                "먼저 설비, 공정, 피더/파트에서 기준을 벗어난 이상치를 찾습니다.",
                "그 다음 이상치가 같은 날짜·설비·공정 축에서 서로 연결되는지 확인합니다.",
                "마지막으로 연결된 근거만 남겨 고객이 이해할 수 있는 문제 진단 시나리오로 설명합니다.",
            ],
            tone="dark",
        )

    def _panel(title: str, method_lines: List[str], interpretation_lines: List[str], tone: str = "info"):
        icon = {"info": "ℹ️", "warn": "⚠️", "accent": "🔎"}.get(tone, "ℹ️")
        with st.expander(f"{icon} {title}", expanded=False):
            st.markdown("**분석 방법**")
            st.markdown("\n".join([f"- {line}" for line in method_lines]))
            st.markdown("**해석**")
            st.markdown("\n".join([f"- {line}" for line in interpretation_lines]))

    if item.empty:
        st.info("mounter 데이터가 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

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
        tmp = df[group_cols + ["event_ts"]].copy().dropna(subset=["event_ts"]).sort_values(group_cols + ["event_ts"])
        if tmp.empty:
            return pd.DataFrame(columns=group_cols + ["avg_cycle_sec", "cycle_std_sec"])
        tmp["cycle_sec"] = tmp.groupby(group_cols)["event_ts"].diff().dt.total_seconds()
        out = tmp.groupby(group_cols, as_index=False).agg(avg_cycle_sec=("cycle_sec", "mean"), cycle_std_sec=("cycle_sec", "std"))
        out["avg_cycle_sec"] = pd.to_numeric(out["avg_cycle_sec"], errors="coerce").fillna(0)
        out["cycle_std_sec"] = pd.to_numeric(out["cycle_std_sec"], errors="coerce").fillna(0)
        return out

    def _machine_view(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "machine_id" not in df.columns:
            return pd.DataFrame()
        out = df.groupby(["machine_id", "line_id", "stage_no"], as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            lot_count=("lot_id", "nunique"),
            model_count=("model_label", "nunique"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        cycle = _cycle_frame(df, ["machine_id"])
        out = out.merge(cycle, on="machine_id", how="left")
        out["observed_span_sec"] = (pd.to_datetime(out["last_event_ts"], errors="coerce") - pd.to_datetime(out["first_event_ts"], errors="coerce")).dt.total_seconds().fillna(0)
        out["output_per_hour"] = out.apply(lambda r: _safe_div(r.get("output_qty", 0), max(r.get("observed_span_sec", 0), 1) / 3600), axis=1)
        total_output = float(df["output_qty"].sum()) or 1.0
        out["output_share"] = out["output_qty"] / total_output
        q1_out = pd.to_numeric(out["output_qty"], errors="coerce").quantile(0.25) if not out.empty else 0
        q3_cycle = pd.to_numeric(out["cycle_std_sec"], errors="coerce").quantile(0.75) if not out.empty else 0
        q3_share = pd.to_numeric(out["output_share"], errors="coerce").quantile(0.75) if not out.empty else 0

        def _ptype(r):
            if r.get("output_qty", 0) <= q1_out and r.get("cycle_std_sec", 0) >= q3_cycle:
                return "생산성 손실형"
            if r.get("output_share", 0) >= q3_share and r.get("output_qty", 0) > 0:
                return "집중형"
            if r.get("cycle_std_sec", 0) >= q3_cycle:
                return "안정성 문제"
            return "주의"

        out["problem_type"] = out.apply(_ptype, axis=1)
        out["bottleneck_score"] = (
            (1 - pd.to_numeric(out["output_qty"], errors="coerce").rank(pct=True, ascending=True).fillna(0)) * 0.45
            + pd.to_numeric(out["cycle_std_sec"], errors="coerce").rank(pct=True, ascending=True).fillna(0) * 0.35
            + pd.to_numeric(out["observed_span_sec"], errors="coerce").rank(pct=True, ascending=True).fillna(0) * 0.20
        )
        out["reasoning"] = out.apply(lambda r: f"출력 {r.get('output_qty', 0):.0f}, cycle 편차 {r.get('cycle_std_sec', 0):.1f}s, 점유율 {r.get('output_share', 0) * 100:.1f}%", axis=1)
        out["recommended_action"] = out.apply(
            lambda r: "라인 밸런스 및 자재 보급 타이밍 점검"
            if r.get("problem_type") == "생산성 손실형"
            else "부하 집중도와 전환 구간 점검"
            if r.get("problem_type") == "집중형"
            else "작업 표준화 및 cycle 변동 원인 확인"
            if r.get("problem_type") == "안정성 문제"
            else "상위 설비와 비교해 변동성 확인",
            axis=1,
        )
        q75 = out["bottleneck_score"].quantile(0.75) if not out.empty else 0
        q40 = out["bottleneck_score"].quantile(0.40) if not out.empty else 0
        out["status_label"] = out["bottleneck_score"].apply(lambda v: "개선필요" if v >= q75 else "주의" if v >= q40 else "정상")
        out["confidence"] = np.where(out["production_rows"].ge(10), "Actual", "Estimated")
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    def _process_view(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not {"line_id", "stage_no"}.issubset(df.columns):
            return pd.DataFrame()
        out = df.groupby(["line_id", "stage_no"], as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            machine_count=("machine_id", "nunique"),
            lot_count=("lot_id", "nunique"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        cycle = _cycle_frame(df, ["line_id", "stage_no"])
        out = out.merge(cycle, on=["line_id", "stage_no"], how="left")
        out["observed_span_sec"] = (pd.to_datetime(out["last_event_ts"], errors="coerce") - pd.to_datetime(out["first_event_ts"], errors="coerce")).dt.total_seconds().fillna(0)
        out["output_per_hour"] = out.apply(lambda r: _safe_div(r.get("output_qty", 0), max(r.get("observed_span_sec", 0), 1) / 3600), axis=1)
        total_output = float(df["output_qty"].sum()) or 1.0
        out["output_share"] = out["output_qty"] / total_output
        q1_out = pd.to_numeric(out["output_qty"], errors="coerce").quantile(0.25) if not out.empty else 0
        q3_cycle = pd.to_numeric(out["cycle_std_sec"], errors="coerce").quantile(0.75) if not out.empty else 0

        def _ptype(r):
            if r.get("output_qty", 0) <= q1_out and r.get("cycle_std_sec", 0) >= q3_cycle:
                return "정지 병목"
            if r.get("cycle_std_sec", 0) >= q3_cycle:
                return "흐름 병목"
            if r.get("output_qty", 0) <= q1_out:
                return "주의"
            return "정상"

        out["process_display"] = out.apply(
            lambda r: f"Line-{r.get('line_id', '-')} / Stage-{int(r.get('stage_no', 0)) if pd.notna(r.get('stage_no', np.nan)) else '-'}",
            axis=1,
        )
        out["problem_type"] = out.apply(_ptype, axis=1)
        out["bottleneck_score"] = (
            (1 - pd.to_numeric(out["output_qty"], errors="coerce").rank(pct=True, ascending=True).fillna(0)) * 0.5
            + pd.to_numeric(out["cycle_std_sec"], errors="coerce").rank(pct=True, ascending=True).fillna(0) * 0.3
            + pd.to_numeric(out["observed_span_sec"], errors="coerce").rank(pct=True, ascending=True).fillna(0) * 0.2
        )
        out["reasoning"] = out.apply(lambda r: f"출력 {r.get('output_qty', 0):.0f}, cycle 편차 {r.get('cycle_std_sec', 0):.1f}s, machine {r.get('machine_count', 0):.0f}개", axis=1)
        out["recommended_action"] = out.apply(
            lambda r: "라인 밸런스와 보급 타이밍 조정"
            if r.get("problem_type") == "정지 병목"
            else "전후 stage 연결과 전환 시간 확인"
            if r.get("problem_type") == "흐름 병목"
            else "생산량 변동 원인과 기준선 재설정",
            axis=1,
        )
        out["confidence"] = np.where(out["production_rows"].ge(10), "Actual", "Estimated")
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    def _lot_view(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "lot_id" not in df.columns:
            return pd.DataFrame()
        out = df.groupby(["lot_id", "model_label"], as_index=False).agg(
            production_rows=("event_ts", "size"),
            output_qty=("output_qty", "sum"),
            machine_count=("machine_id", "nunique"),
            stage_count=("stage_no", "nunique"),
            line_count=("line_id", "nunique"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        cycle = _cycle_frame(df, ["lot_id"])
        out = out.merge(cycle, on=["lot_id"], how="left")
        out["observed_span_sec"] = (pd.to_datetime(out["last_event_ts"], errors="coerce") - pd.to_datetime(out["first_event_ts"], errors="coerce")).dt.total_seconds().fillna(0)
        out["output_per_hour"] = out.apply(lambda r: _safe_div(r.get("output_qty", 0), max(r.get("observed_span_sec", 0), 1) / 3600), axis=1)
        out["spread_score"] = out[["machine_count", "stage_count", "line_count"]].sum(axis=1)
        out["concentration_score"] = 1 / (1 + out["spread_score"])
        out["priority_score"] = out["spread_score"] * 0.6 + out["concentration_score"] * 0.4
        machine_q = out["machine_count"].quantile(0.25) if not out.empty else 0
        stage_q = out["stage_count"].quantile(0.75) if not out.empty else 0
        density_q = out["output_per_hour"].quantile(0.25) if not out.empty else 0

        def _ptype(r):
            if r.get("machine_count", 0) <= machine_q and r.get("stage_count", 0) <= 1:
                return "국소 LOT"
            if r.get("stage_count", 0) >= stage_q or r.get("machine_count", 0) >= stage_q:
                return "전파 LOT"
            if r.get("output_per_hour", 0) <= density_q:
                return "저생산 LOT"
            return "주의"

        out["problem_type"] = out.apply(_ptype, axis=1)
        out["reasoning"] = out.apply(lambda r: f"설비 {r.get('machine_count', 0):.0f}개 / 공정 {r.get('stage_count', 0):.0f}개 / 집중도 {r.get('concentration_score', 0) * 100:.1f}%", axis=1)
        out["recommended_action"] = out.apply(
            lambda r: "대표 설비와 stage를 먼저 확인"
            if r.get("problem_type") == "국소 LOT"
            else "전파 범위와 연속 lot 비교"
            if r.get("problem_type") == "전파 LOT"
            else "생산 밀도와 스케줄 확인",
            axis=1,
        )
        out["confidence"] = np.where(out["production_rows"].ge(10), "Actual", "Estimated")
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    def _machine_day_view(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not {"machine_id", "day"}.issubset(df.columns):
            return pd.DataFrame()
        output_baseline = 2200.0
        out = df.groupby(["machine_id", "day"], as_index=False).agg(
            actual_output=("output_qty", "sum"),
            production_rows=("event_ts", "size"),
            lot_count=("lot_id", "nunique"),
            stage_count=("stage_no", "nunique"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        cycle = _cycle_frame(df, ["machine_id", "day"])
        out = out.merge(cycle, on=["machine_id", "day"], how="left")
        out["span_sec"] = (
            pd.to_datetime(out["last_event_ts"], errors="coerce") - pd.to_datetime(out["first_event_ts"], errors="coerce")
        ).dt.total_seconds().fillna(0)
        out["actual_takt_sec"] = out.apply(lambda r: _safe_div(r.get("span_sec", 0), max(r.get("actual_output", 0), 1)), axis=1)
        ref = out.groupby("machine_id", as_index=False).agg(
            expected_takt_sec=("actual_takt_sec", "median"),
            takt_std_sec=("actual_takt_sec", "std"),
        )
        out = out.merge(ref, on="machine_id", how="left")
        out["expected_output"] = output_baseline
        out["expected_takt_sec"] = pd.to_numeric(out["expected_takt_sec"], errors="coerce").fillna(out["actual_takt_sec"])
        # 발표용 비교가 명확하게 보이도록 설비별 가상 기준선 택타임을 고정 패턴으로 부여합니다.
        takt_offset_map = {0: 0.96, 1: 0.99, 2: 1.00, 3: 1.05, 4: 1.12}
        out["expected_takt_sec"] = out.apply(
            lambda r: _safe_float(r.get("expected_takt_sec", 0))
            * takt_offset_map[sum(ord(ch) for ch in str(r.get("machine_id", ""))) % len(takt_offset_map)],
            axis=1,
        )
        out["takt_std_sec"] = pd.to_numeric(out["takt_std_sec"], errors="coerce").fillna(0)
        out["output_gap"] = out["actual_output"] - out["expected_output"]
        out["output_gap_pct"] = out.apply(lambda r: _safe_div(r.get("output_gap", 0), max(r.get("expected_output", 0), 1)), axis=1)
        out["takt_gap_sec"] = out["actual_takt_sec"] - out["expected_takt_sec"]
        q1_gap = out["output_gap_pct"].quantile(0.25) if not out.empty else 0
        q3_gap = out["output_gap_pct"].quantile(0.75) if not out.empty else 0
        q3_takt = out["takt_gap_sec"].quantile(0.75) if not out.empty else 0

        def _ptype(r):
            if r.get("output_gap_pct", 0) <= -0.05 and r.get("takt_gap_sec", 0) >= q3_takt:
                return "생산성 손실형"
            if r.get("output_gap_pct", 0) <= q1_gap:
                return "정지 병목"
            if r.get("takt_gap_sec", 0) >= q3_takt:
                return "흐름 병목"
            if r.get("output_gap_pct", 0) >= q3_gap:
                return "주의"
            return "정상"

        def _hint(r):
            gap = r.get("output_gap", 0)
            takt_gap = r.get("takt_gap_sec", 0)
            if r.get("problem_type") == "생산성 손실형":
                return f"기준 대비 출력 {gap:.0f} 감소, 실측 택타임 {takt_gap:.1f}s 증가"
            if r.get("problem_type") == "정지 병목":
                return f"출력 편차 {gap:.0f}, 장시간 정체 가능성"
            if r.get("problem_type") == "흐름 병목":
                return f"기준 택타임 대비 {takt_gap:.1f}s 지연"
            return f"기준 대비 출력 {gap:.0f}, 택타임 변화 {takt_gap:.1f}s"

        def _action(r):
            if r.get("problem_type") == "생산성 손실형":
                return "라인 밸런스, 투입 타이밍, 전환 손실을 먼저 점검"
            if r.get("problem_type") == "정지 병목":
                return "정체 구간과 작업 전환 표준을 우선 조정"
            if r.get("problem_type") == "흐름 병목":
                return "전후 공정 연결과 buffer를 먼저 조정"
            if r.get("problem_type") == "주의":
                return "기준 대비 변동 구간을 재확인"
            return "현재 상태 유지, 추세만 모니터링"

        out["problem_type"] = out.apply(_ptype, axis=1)
        out["reasoning"] = out.apply(_hint, axis=1)
        out["recommended_action"] = out.apply(_action, axis=1)
        out["confidence"] = np.where(out["production_rows"].ge(10), "Actual", "Estimated")
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    def _stage_day_view(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or not {"line_id", "stage_no", "day"}.issubset(df.columns):
            return pd.DataFrame()
        output_baseline = 11000.0
        out = df.groupby(["line_id", "stage_no", "day"], as_index=False).agg(
            actual_output=("output_qty", "sum"),
            production_rows=("event_ts", "size"),
            machine_count=("machine_id", "nunique"),
            lot_count=("lot_id", "nunique"),
            first_event_ts=("event_ts", "min"),
            last_event_ts=("event_ts", "max"),
        )
        cycle = _cycle_frame(df, ["line_id", "stage_no", "day"])
        out = out.merge(cycle, on=["line_id", "stage_no", "day"], how="left")
        out["span_sec"] = (
            pd.to_datetime(out["last_event_ts"], errors="coerce") - pd.to_datetime(out["first_event_ts"], errors="coerce")
        ).dt.total_seconds().fillna(0)
        out["actual_takt_sec"] = out.apply(lambda r: _safe_div(r.get("span_sec", 0), max(r.get("actual_output", 0), 1)), axis=1)
        ref = out.groupby(["line_id", "stage_no"], as_index=False).agg(
            expected_takt_sec=("actual_takt_sec", "median"),
            takt_std_sec=("actual_takt_sec", "std"),
        )
        out = out.merge(ref, on=["line_id", "stage_no"], how="left")
        out["expected_output"] = output_baseline
        out["expected_takt_sec"] = pd.to_numeric(out["expected_takt_sec"], errors="coerce").fillna(out["actual_takt_sec"])
        out["takt_std_sec"] = pd.to_numeric(out["takt_std_sec"], errors="coerce").fillna(0)
        out["output_gap"] = out["actual_output"] - out["expected_output"]
        out["output_gap_pct"] = out.apply(lambda r: _safe_div(r.get("output_gap", 0), max(r.get("expected_output", 0), 1)), axis=1)
        out["takt_gap_sec"] = out["actual_takt_sec"] - out["expected_takt_sec"]
        q1_gap = out["output_gap_pct"].quantile(0.25) if not out.empty else 0
        q3_gap = out["output_gap_pct"].quantile(0.75) if not out.empty else 0
        q3_takt = out["takt_gap_sec"].quantile(0.75) if not out.empty else 0

        def _ptype(r):
            if r.get("output_gap_pct", 0) <= -0.05 and r.get("takt_gap_sec", 0) >= q3_takt:
                return "정지 병목"
            if r.get("takt_gap_sec", 0) >= q3_takt:
                return "흐름 병목"
            if r.get("output_gap_pct", 0) >= q3_gap:
                return "주의"
            return "정상"

        def _hint(r):
            gap = r.get("output_gap", 0)
            takt_gap = r.get("takt_gap_sec", 0)
            if r.get("problem_type") == "정지 병목":
                return f"Stage 출력 {gap:.0f} 감소, 택타임 {takt_gap:.1f}s 증가"
            if r.get("problem_type") == "흐름 병목":
                return f"Stage 전환 지연 {takt_gap:.1f}s"
            return f"기준 대비 출력 {gap:.0f}, 택타임 변화 {takt_gap:.1f}s"

        def _action(r):
            if r.get("problem_type") == "정지 병목":
                return "stage 간 handoff, 대기, 전환 시간을 먼저 점검"
            if r.get("problem_type") == "흐름 병목":
                return "전후 공정 buffer와 stage 연결을 먼저 조정"
            if r.get("problem_type") == "주의":
                return "기준 대비 변동 구간을 재확인"
            return "현재 상태 유지, 추세만 모니터링"

        out["process_display"] = out.apply(
            lambda r: f"Line-{r.get('line_id', '-')} / Stage-{int(r.get('stage_no', 0)) if pd.notna(r.get('stage_no', np.nan)) else '-'}",
            axis=1,
        )
        out["problem_type"] = out.apply(_ptype, axis=1)
        out["reasoning"] = out.apply(_hint, axis=1)
        out["recommended_action"] = out.apply(_action, axis=1)
        out["confidence"] = np.where(out["production_rows"].ge(10), "Actual", "Estimated")
        out["rank"] = np.arange(1, len(out) + 1)
        return out

    def _alarm_view(machine_df: pd.DataFrame, process_df: pd.DataFrame, lot_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        def _alarm_level(output_gap_pct: float, takt_gap_sec: float, spread: float = 0.0, fail_rows: int = 0) -> str:
            if fail_rows > 0:
                return "심각"
            if output_gap_pct <= -0.10 and takt_gap_sec >= 30:
                return "심각"
            if output_gap_pct <= -0.05 and takt_gap_sec >= 15:
                return "경고"
            if output_gap_pct <= -0.03 or takt_gap_sec >= 10:
                return "주의"
            if spread >= 5:
                return "주의"
            return "정상"

        if not machine_df.empty:
            m_top = machine_df.sort_values("bottleneck_score", ascending=False).iloc[0]
            out_gap = _safe_float(m_top.get("output_gap_pct", 0))
            takt_gap = _safe_float(m_top.get("takt_gap_sec", 0))
            rows.append({
                "알람유형": "생산성 경고",
                "기준": "출력 편차 -5% 이하 또는 택타임 +15초 이상",
                "판정": _alarm_level(out_gap, takt_gap),
                "현재상태": m_top.get("problem_type", "-"),
                "우선순위": "1. 즉시조치",
                "조치": m_top.get("recommended_action", "-"),
            })
        if not process_df.empty:
            p_top = process_df.sort_values("bottleneck_score", ascending=False).iloc[0]
            out_gap = _safe_float(p_top.get("output_gap_pct", 0))
            takt_gap = _safe_float(p_top.get("takt_gap_sec", 0))
            rows.append({
                "알람유형": "흐름 경고",
                "기준": "stage 기준 출력 -5% 이하 또는 택타임 +15초 이상",
                "판정": _alarm_level(out_gap, takt_gap),
                "현재상태": p_top.get("problem_type", "-"),
                "우선순위": "1. 즉시조치",
                "조치": p_top.get("recommended_action", "-"),
            })
        if not lot_df.empty:
            l_top = lot_df.sort_values("priority_score", ascending=False).iloc[0]
            spread = _safe_float(l_top.get("spread_score", 0))
            machine_count = _safe_float(l_top.get("machine_count", 0))
            stage_count = _safe_float(l_top.get("stage_count", 0))
            rows.append({
                "알람유형": "LOT 확산 경고",
                "기준": "machine 2개 이상 또는 stage 2개 이상 확산",
                "판정": "심각" if machine_count >= 2 or stage_count >= 2 else _alarm_level(0, 0, spread=spread),
                "현재상태": l_top.get("problem_type", "-"),
                "우선순위": "2. 자동 해소",
                "조치": l_top.get("recommended_action", "-"),
            })
        result_fail = 0
        if "result" in filtered.columns:
            result_fail = int((filtered["result"].astype(str).str.upper() != "PASS").sum())
        rows.append({
            "알람유형": "품질 경고",
            "기준": "PASS 이외 결과 1건 이상",
            "판정": "심각" if result_fail > 0 else "정상",
            "현재상태": "발생 없음" if result_fail == 0 else f"비정상 {result_fail:,}",
            "우선순위": "1. 즉시조치",
            "조치": "현재 백업에는 불량 결과가 없어 원천 품질 알람은 비활성",
        })
        return pd.DataFrame(rows)

    def _priority_rule_view() -> pd.DataFrame:
        rows = [
            {
                "순위": "1",
                "구분": "즉시조치 우선",
                "설명": "output_gap_pct -5% 이하 또는 택타임 +15초 이상",
                "예시": "기준 대비 출력 감소, 택타임 지연",
                "조치": "설비 / 공정 / 투입 타이밍 우선 점검",
            },
            {
                "순위": "2",
                "구분": "자동 해소",
                "설명": "output_gap_pct -3% ~ -5% 또는 택타임 +10~15초",
                "예시": "일시적 cycle 흔들림, 단발성 지연",
                "조치": "재시도 / 경보 / 임계치 조정",
            },
            {
                "순위": "3",
                "구분": "세부 원인 확인",
                "설명": "단일 LOT 또는 단일 stage에만 국한된 반복 패턴",
                "예시": "특정 LOT, 특정 stage, 반복 편차",
                "조치": "원인 구간과 대표 lot 비교",
            },
        ]
        return pd.DataFrame(rows)

    def _component_error_view(base_comp: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        comp_df = base_comp.copy()
        source_label = "실데이터"
        if comp_df.empty:
            try:
                comp_df = build_component_fact(generate_sample_data())
                source_label = "설명용 샘플"
            except Exception:
                return pd.DataFrame(), "데이터 없음"
        if comp_df.empty:
            return pd.DataFrame(), "데이터 없음"
        if filters.get("machine") != "전체" and "machine_id" in comp_df.columns:
            comp_df = comp_df[comp_df["machine_id"].astype(str).eq(filters["machine"])]
        if filters.get("lot") != "전체" and "lot_id" in comp_df.columns:
            comp_df = comp_df[comp_df["lot_id"].astype(str).eq(filters["lot"])]
        if filters.get("model") != "전체" and "model_label" in comp_df.columns:
            comp_df = comp_df[comp_df["model_label"].astype(str).eq(filters["model"])]
        if comp_df.empty:
            return pd.DataFrame(), source_label
        for col in ["machine_id", "lot_id", "part_number", "feeder_id", "feeder_serial", "nozzle_serial", "defect_type"]:
            if col not in comp_df.columns:
                comp_df[col] = "-"
            comp_df[col] = comp_df[col].fillna("-")
        for col in ["pickup_count", "error_count", "pickup_error_count", "recognition_error_count"]:
            if col not in comp_df.columns:
                comp_df[col] = 0
            comp_df[col] = pd.to_numeric(comp_df[col], errors="coerce").fillna(0)
        comp_df["error_rate"] = comp_df.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
        comp_df["error_domain"] = np.where(
            comp_df["recognition_error_count"].gt(comp_df["pickup_error_count"]),
            "인식/비전",
            np.where(comp_df["pickup_error_count"].gt(0), "픽업/흡착", "기타/복합"),
        )
        comp_df["error_message"] = comp_df.get("defect_type", pd.Series(["에러"] * len(comp_df), index=comp_df.index)).fillna("에러").astype(str)
        comp_df["target_key"] = (
            comp_df.get("machine_id", pd.Series(["-"] * len(comp_df), index=comp_df.index)).astype(str)
            + " / "
            + comp_df.get("part_number", pd.Series(["-"] * len(comp_df), index=comp_df.index)).astype(str)
            + " / "
            + comp_df.get("feeder_id", pd.Series(["-"] * len(comp_df), index=comp_df.index)).astype(str)
        )
        grouped = comp_df.groupby(
            ["machine_id", "lot_id", "part_number", "feeder_id", "feeder_serial", "nozzle_serial", "error_domain", "error_message"],
            as_index=False,
        ).agg(
            pickup_count=("pickup_count", "sum"),
            error_count=("error_count", "sum"),
            pickup_error_count=("pickup_error_count", "sum"),
            recognition_error_count=("recognition_error_count", "sum"),
        )
        grouped["error_rate"] = grouped.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
        grouped["생산성영향"] = grouped["machine_id"].astype(str).map(
            daily_view.groupby("machine_id")["output_gap_pct"].min().to_dict() if not daily_view.empty else {}
        ).fillna(0)
        q90_rate = grouped["error_rate"].quantile(0.90) if not grouped.empty else 0
        q75_rate = grouped["error_rate"].quantile(0.75) if not grouped.empty else 0
        grouped["우선순위"] = grouped.apply(
            lambda r: "1. 즉시조치"
            if r.get("error_rate", 0) >= q90_rate or r.get("생산성영향", 0) <= -0.05
            else "2. 자동 해소"
            if r.get("error_rate", 0) >= q75_rate
            else "3. 세부 원인 확인",
            axis=1,
        )
        grouped["즉시조치"] = grouped.apply(
            lambda r: "feeder 정렬, reel 장력, 흡착 상태 확인"
            if r.get("error_domain") == "픽업/흡착"
            else "vision 조건, camera, mark 인식 조건 확인"
            if r.get("error_domain") == "인식/비전"
            else "부품 조건과 반복 LOT를 함께 확인",
            axis=1,
        )
        grouped = grouped.sort_values(["error_rate", "error_count", "pickup_count"], ascending=[False, False, False]).reset_index(drop=True)
        grouped["순위"] = np.arange(1, len(grouped) + 1)
        return grouped, source_label

    def _error_alert_view(error_df: pd.DataFrame, source_label: str) -> pd.DataFrame:
        if error_df.empty:
            return pd.DataFrame([
                {"알람유형": "에러 증가", "기준": "error_rate 상위 조합 감시", "현재": "데이터 부족", "판정": "보류", "조치": "component_pickup_summary 연결 필요", "데이터기준": source_label}
            ])
        top_row = error_df.iloc[0]
        rows = [
            {
                "알람유형": "에러 증가",
                "기준": "error_rate 상위 10% 또는 생산성 저하 동반",
                "현재": f"{top_row.get('machine_id', '-')} / {top_row.get('feeder_id', '-')}, error_rate {top_row.get('error_rate', 0) * 100:.2f}%",
                "판정": top_row.get("우선순위", "-"),
                "조치": top_row.get("즉시조치", "-"),
                "데이터기준": source_label,
            }
        ]
        quality_hit = error_df[error_df["error_message"].astype(str).str.contains("Defect|NG|Fail", case=False, na=False)]
        if not quality_hit.empty:
            q_top = quality_hit.iloc[0]
            rows.append(
                {
                    "알람유형": "품질 영향 경고",
                    "기준": "error 메시지가 defect 계열이고 error_rate가 높음",
                    "현재": f"{q_top.get('error_message', '-')} / {q_top.get('part_number', '-')}",
                    "판정": q_top.get("우선순위", "-"),
                    "조치": "품질 판정 조건과 부품 LOT를 함께 확인",
                    "데이터기준": source_label,
                }
            )
        prod_hit = error_df[error_df["생산성영향"] <= -0.05]
        if not prod_hit.empty:
            p_top = prod_hit.iloc[0]
            rows.append(
                {
                    "알람유형": "생산성 영향 경고",
                    "기준": "에러 증가와 output_gap_pct -5% 이하 동시 발생",
                    "현재": f"{p_top.get('machine_id', '-')} / 출력편차 {p_top.get('생산성영향', 0) * 100:.1f}%",
                    "판정": "1. 즉시조치",
                    "조치": p_top.get("즉시조치", "-"),
                    "데이터기준": source_label,
                }
            )
        return pd.DataFrame(rows)

    def _error_priority_view(error_df: pd.DataFrame) -> pd.DataFrame:
        if error_df.empty:
            return pd.DataFrame()
        priority = error_df.groupby(["error_message", "error_domain"], as_index=False).agg(
            error_count=("error_count", "sum"),
            pickup_count=("pickup_count", "sum"),
            max_error_rate=("error_rate", "max"),
            machine_count=("machine_id", "nunique"),
        )
        priority["error_rate"] = priority.apply(lambda r: _safe_div(r.get("error_count", 0), max(r.get("pickup_count", 0), 1)), axis=1)
        priority["우선순위"] = priority.apply(
            lambda r: "1. 즉시조치" if r.get("error_rate", 0) >= priority["error_rate"].quantile(0.75) else "2. 자동 해소" if r.get("error_count", 0) >= 2 else "3. 세부 원인 확인",
            axis=1,
        )
        priority["확인포인트"] = priority.apply(
            lambda r: "피더 정렬, reel 장력, nozzle 흡착 상태"
            if r.get("error_domain") == "픽업/흡착"
            else "vision 조건, camera, mark 인식"
            if r.get("error_domain") == "인식/비전"
            else "부품 조건과 반복 LOT",
            axis=1,
        )
        return priority.sort_values(["error_rate", "error_count"], ascending=[False, False]).reset_index(drop=True)

    machine = _machine_view(item)
    process = _process_view(item)
    lot = _lot_view(item)
    daily_view = _machine_day_view(item)
    stage_daily_view = _stage_day_view(item)
    time_view = item.groupby(["hour", "shift"], as_index=False).agg(
        production_rows=("event_ts", "size"),
        output_qty=("output_qty", "sum"),
        machine_count=("machine_id", "nunique"),
        lot_count=("lot_id", "nunique"),
    )
    if not time_view.empty:
        shift_order = {"주간(06-14)": 0, "석간(14-22)": 1, "야간(22-06)": 2, "미상": 3}
        time_view["bucket"] = time_view["hour"].astype("Int64").astype(str).str.zfill(2)
        time_view["bucket_order"] = time_view["hour"].fillna(99)
        time_view["grain"] = "hour"
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

    filters = {}
    cols = st.columns(5)
    options = {
        "line": ["전체"] + sorted(item["line_id"].dropna().astype(str).unique().tolist()) if "line_id" in item.columns else ["전체"],
        "stage": ["전체"] + sorted(item["stage_no"].dropna().astype(int).astype(str).unique().tolist()) if "stage_no" in item.columns else ["전체"],
        "machine": ["전체"] + sorted(item["machine_id"].dropna().astype(str).unique().tolist()) if "machine_id" in item.columns else ["전체"],
        "lot": ["전체"] + sorted(item["lot_id"].dropna().astype(str).unique().tolist()) if "lot_id" in item.columns else ["전체"],
        "model": ["전체"] + sorted(item["model_label"].dropna().astype(str).unique().tolist()) if "model_label" in item.columns else ["전체"],
    }
    labels = ["line", "stage", "machine", "lot", "model"]
    filter_prefix = f"mounter_{mode}"
    for col, key in zip(cols, labels):
        with col:
            filters[key] = st.selectbox(key, options[key], key=f"{filter_prefix}_{key}_filter")

    filtered = item.copy()
    if filters.get("line") != "전체" and "line_id" in filtered.columns:
        filtered = filtered[filtered["line_id"].astype(str).eq(filters["line"])]
    if filters.get("stage") != "전체" and "stage_no" in filtered.columns:
        filtered = filtered[filtered["stage_no"].astype("Int64").astype(str).eq(filters["stage"])]
    if filters.get("machine") != "전체" and "machine_id" in filtered.columns:
        filtered = filtered[filtered["machine_id"].astype(str).eq(filters["machine"])]
    if filters.get("lot") != "전체" and "lot_id" in filtered.columns:
        filtered = filtered[filtered["lot_id"].astype(str).eq(filters["lot"])]
    if filters.get("model") != "전체" and "model_label" in filtered.columns:
        filtered = filtered[filtered["model_label"].astype(str).eq(filters["model"])]

    filtered_comp = clean.get("vw_component_error_fact", pd.DataFrame()).copy()

    if filtered.empty:
        st.warning("선택 조건에 해당하는 mounter 데이터가 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    machine = _machine_view(filtered)
    process = _process_view(filtered)
    lot = _lot_view(filtered)
    daily_view = _machine_day_view(filtered)
    stage_daily_view = _stage_day_view(filtered)
    error_view, error_source_label = _component_error_view(filtered_comp)
    error_alerts = _error_alert_view(error_view, error_source_label)
    error_priority = _error_priority_view(error_view)
    time_view = filtered.groupby(["hour", "shift"], as_index=False).agg(
        production_rows=("event_ts", "size"),
        output_qty=("output_qty", "sum"),
        machine_count=("machine_id", "nunique"),
        lot_count=("lot_id", "nunique"),
    )
    if not time_view.empty:
        shift_order = {"주간(06-14)": 0, "석간(14-22)": 1, "야간(22-06)": 2, "미상": 3}
        time_view["bucket"] = time_view["hour"].astype("Int64").astype(str).str.zfill(2)
        time_view["bucket_order"] = time_view["hour"].fillna(99)
        time_view["grain"] = "hour"
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

    top_machine = machine.sort_values("bottleneck_score", ascending=False).head(1)
    top_process = process.sort_values("bottleneck_score", ascending=False).head(1)
    top_lot = lot.sort_values("priority_score", ascending=False).head(1)

    if mode == "overview":
        summary_cards = [
            ("총 출력", f"{int(filtered['output_qty'].sum()):,}", "mounter output 합계"),
            ("설비 수", f"{filtered['machine_id'].nunique():,}" if "machine_id" in filtered.columns else "0", "활성 설비"),
            ("공정 수", f"{filtered[['line_id', 'stage_no']].drop_duplicates().shape[0]:,}" if {"line_id", "stage_no"}.issubset(filtered.columns) else "0", "line/stage 조합"),
            ("LOT 수", f"{filtered['lot_id'].nunique():,}" if "lot_id" in filtered.columns else "0", "활성 LOT"),
            ("평균 cycle", _fmt_sec(_safe_float(machine["avg_cycle_sec"].mean() if not machine.empty and "avg_cycle_sec" in machine.columns else 0)), "설비 평균 cycle"),
            ("우선 점검", str(machine.sort_values("bottleneck_score", ascending=False).iloc[0].get("machine_id", "-") if not machine.empty else "-"), "bottleneck 최고"),
        ]
        cards = st.columns(6)
        for col, (label, value, foot) in zip(cards, summary_cards):
            with col:
                st.markdown(_card(label, value, foot), unsafe_allow_html=True)

        st.markdown("#### 행동형 인사이트")
        ic1, ic2, ic3 = st.columns(3)
        insight_data = [
            (
                top_machine.iloc[0] if not top_machine.empty else pd.Series(dtype=object),
                "설비",
                "출력 집중도와 cycle 변동을 동시에 봅니다.",
                "상위 설비의 부하 집중을 줄이고 변동성이 큰 구간을 먼저 잡아야 합니다.",
            ),
            (
                top_process.iloc[0] if not top_process.empty else pd.Series(dtype=object),
                "공정",
                "출력 낮음 + cycle 편차 큼이면 병목입니다.",
                "line balance, 전환 시간, 보급 타이밍을 먼저 확인해야 합니다.",
            ),
            (
                top_lot.iloc[0] if not top_lot.empty else pd.Series(dtype=object),
                "LOT",
                "다수 설비/공정으로 퍼지면 전파 LOT입니다.",
                "대표 설비와 stage를 먼저 잡고 확산을 차단해야 합니다.",
            ),
        ]
        for col, (row, kind, hint, action) in zip([ic1, ic2, ic3], insight_data):
            with col:
                if row is not None and not row.empty:
                    target = row.get("machine_id", row.get("process_display", row.get("lot_id", "-")))
                    problem = row.get("problem_type", "주의")
                    evidence = row.get("reasoning", "-")
                    rec = row.get("recommended_action", action)
                    conf = row.get("confidence", "Estimated")
                else:
                    target, problem, evidence, rec, conf = "-", "주의", "-", action, "Estimated"
                st.markdown(
                    f"""
                    <div style="padding:.95rem 1rem;border-radius:16px;background:linear-gradient(180deg,#263244,#1a2330);border:1px solid rgba(148,163,184,.18);min-height:180px;box-shadow:0 10px 24px rgba(15,23,42,.12)">
                        <div style="font-size:.78rem;color:#cbd5e1;">대상</div>
                        <div style="font-size:1.15rem;font-weight:800;color:#f8fafc;margin:.15rem 0 .35rem 0;">{target}</div>
                        <div style="margin-bottom:.3rem;color:#f8c77d;font-weight:700;">분류: {problem}</div>
                        <div style="font-size:.82rem;color:#d8dee8;">{hint}</div>
                        <div style="font-size:.82rem;color:#d8dee8;margin-top:.25rem;">근거: {evidence}</div>
                        <div style="font-size:.82rem;color:#9fd0ff;margin-top:.35rem;">권장 조치: {rec}</div>
                        <div style="font-size:.75rem;color:#9fb0c4;margin-top:.35rem;">신뢰도: {conf}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if mode == "overview":
        st.markdown(_section_header("전체 문제 요약", "설비 / 공정 / LOT 중 어디가 먼저 문제인지 한눈에 보는 구역", PRIMARY), unsafe_allow_html=True)
        left, right = st.columns([0.55, 0.45])
        with left:
            summary = pd.DataFrame([
                {"구분": "설비", "대상": row.get("machine_id", "-"), "문제유형": row.get("problem_type", "-"), "근거": row.get("reasoning", "-"), "조치": row.get("recommended_action", "-")}
                for _, row in top_machine.iterrows()
            ] + [
                {"구분": "공정", "대상": row.get("process_display", "-"), "문제유형": row.get("problem_type", "-"), "근거": row.get("reasoning", "-"), "조치": row.get("recommended_action", "-")}
                for _, row in top_process.iterrows()
            ] + [
                {"구분": "LOT", "대상": row.get("lot_id", "-"), "문제유형": row.get("problem_type", "-"), "근거": row.get("reasoning", "-"), "조치": row.get("recommended_action", "-")}
                for _, row in top_lot.iterrows()
            ])
            st.dataframe(summary, use_container_width=True, hide_index=True)
        with right:
            if not time_view.empty and "output_qty" in time_view.columns:
                if "grain" in time_view.columns:
                    hour_view = time_view[time_view["grain"].eq("hour")].copy()
                elif "hour" in time_view.columns:
                    hour_view = time_view.copy()
                    hour_view["bucket"] = pd.to_numeric(hour_view["hour"], errors="coerce").astype("Int64").astype(str).str.zfill(2)
                    hour_view["bucket_order"] = pd.to_numeric(hour_view["hour"], errors="coerce").fillna(99)
                else:
                    hour_view = pd.DataFrame()
                if not hour_view.empty:
                    fig = px.bar(hour_view.sort_values("bucket_order" if "bucket_order" in hour_view.columns else "hour"), x="bucket" if "bucket" in hour_view.columns else "hour", y="output_qty", text="output_qty")
                    st.plotly_chart(_plot_style(fig, "시간대별 출력", 330), use_container_width=True)

    if mode == "overview":
        st.markdown(_section_header("설비별 문제 분석", "출력량, cycle 변동, 점유율로 설비를 분류합니다.", SECONDARY), unsafe_allow_html=True)
        left, right = st.columns([0.56, 0.44])
        with left:
            if not machine.empty:
                display_cols = [c for c in ["rank", "machine_id", "line_id", "stage_no", "production_rows", "output_qty", "output_per_hour", "avg_cycle_sec", "cycle_std_sec", "lot_count", "problem_type", "status_label", "reasoning", "recommended_action", "confidence"] if c in machine.columns]
                st.dataframe(machine.sort_values("bottleneck_score", ascending=False)[display_cols].head(10), use_container_width=True, hide_index=True)
            else:
                st.info("설비 분석 데이터가 없습니다.")
        with right:
            if not machine.empty:
                x_th = machine["output_qty"].quantile(0.25)
                y_th = machine["cycle_std_sec"].quantile(0.75)
                fig = px.scatter(
                    machine,
                    x="output_qty",
                    y="cycle_std_sec",
                    size="production_rows",
                    color="problem_type",
                    hover_name="machine_id",
                    text="machine_id",
                    color_discrete_map={"생산성 손실형": "#ef4444", "집중형": "#f59e0b", "안정성 문제": "#3b82f6", "주의": "#64748b"},
                )
                fig.add_vline(x=x_th, line_dash="dash", line_color="#94a3b8")
                fig.add_hline(y=y_th, line_dash="dash", line_color="#94a3b8")
                st.plotly_chart(_plot_style(fig, "설비별 출력-변동 분류도", 340), use_container_width=True)
            st.markdown("##### 해석 기준")
            st.markdown("- 출력이 낮고 cycle 편차가 크면 생산성 손실형입니다.")
            st.markdown("- 출력 점유율이 크면 특정 설비에 부하가 집중된 것입니다.")
            st.markdown("- 다음 확인: 상위 설비의 line/stage 조합과 동일 LOT 분포입니다.")

    if mode == "overview":
        st.markdown(_section_header("공정별 병목 분석", "라인/Stage 단위의 출력과 변동성을 봅니다.", PRIMARY), unsafe_allow_html=True)
        left, right = st.columns([0.56, 0.44])
        with left:
            if not process.empty:
                display_cols = [c for c in ["rank", "process_display", "line_id", "stage_no", "production_rows", "output_qty", "output_per_hour", "machine_count", "lot_count", "avg_cycle_sec", "cycle_std_sec", "problem_type", "reasoning", "recommended_action", "confidence"] if c in process.columns]
                st.dataframe(process.sort_values("bottleneck_score", ascending=False)[display_cols].head(10), use_container_width=True, hide_index=True)
            else:
                st.info("공정 분석 데이터가 없습니다.")
        with right:
            if not process.empty:
                fig = px.scatter(
                    process,
                    x="output_qty",
                    y="cycle_std_sec",
                    size="production_rows",
                    color="problem_type",
                    hover_name="process_display",
                    text="process_display",
                    color_discrete_map={"정지 병목": "#ef4444", "흐름 병목": "#f59e0b", "주의": "#64748b", "정상": "#3b82f6"},
                )
                st.plotly_chart(_plot_style(fig, "공정별 출력-변동 분류도", 340), use_container_width=True)
            st.markdown("##### 해석 기준")
            st.markdown("- 출력이 낮고 cycle 편차가 크면 병목 stage입니다.")
            st.markdown("- machine_count가 크면 해당 공정에 장비가 몰려 있는 것입니다.")
            st.markdown("- 다음 확인: stage별 전환 구간과 같은 lot의 분산입니다.")

    if mode == "overview":
        st.markdown(_section_header("LOT 영향 분석", "한 LOT이 얼마나 넓게 퍼지는지 봅니다.", SECONDARY), unsafe_allow_html=True)
        left, right = st.columns([0.56, 0.44])
        with left:
            if not lot.empty:
                display_cols = [c for c in ["rank", "lot_id", "model_label", "production_rows", "output_qty", "output_per_hour", "machine_count", "stage_count", "line_count", "spread_score", "priority_score", "problem_type", "reasoning", "recommended_action", "confidence"] if c in lot.columns]
                st.dataframe(lot.sort_values("priority_score", ascending=False)[display_cols].head(10), use_container_width=True, hide_index=True)
            else:
                st.info("LOT 분석 데이터가 없습니다.")
        with right:
            if not lot.empty:
                fig = px.scatter(
                    lot,
                    x="machine_count",
                    y="stage_count",
                    size="spread_score",
                    color="problem_type",
                    hover_name="lot_id",
                    text="lot_id",
                    color_discrete_map={"국소 LOT": "#ef4444", "전파 LOT": "#f59e0b", "저생산 LOT": "#3b82f6", "주의": "#64748b"},
                )
                st.plotly_chart(_plot_style(fig, "LOT 확산-영향 분류도", 340), use_container_width=True)
            st.markdown("##### 해석 기준")
            st.markdown("- machine_count와 stage_count가 낮으면 국소 LOT입니다.")
            st.markdown("- machine_count 또는 stage_count가 넓으면 전파 LOT입니다.")
            st.markdown("- 다음 확인: 대표 설비와 대표 stage입니다.")

    if mode == "overview":
        st.markdown(_section_header("개선 우선순위", "먼저 개선해야 할 대상 순서입니다.", PRIMARY), unsafe_allow_html=True)
        priority_rows = []
        if not machine.empty:
            for _, row in machine.sort_values("bottleneck_score", ascending=False).head(4).iterrows():
                priority_rows.append({"대상유형": "설비", "대상": row.get("machine_id", "-"), "문제유형": row.get("problem_type", "-"), "근거 KPI": row.get("reasoning", "-"), "예상 영향": f"line {row.get('line_id', '-')}, stage {row.get('stage_no', '-')}", "추천 액션": row.get("recommended_action", "-"), "기대 효과": "기준 변동 축소", "priority_score": row.get("bottleneck_score", 0)})
        if not process.empty:
            for _, row in process.sort_values("bottleneck_score", ascending=False).head(4).iterrows():
                priority_rows.append({"대상유형": "공정", "대상": row.get("process_display", "-"), "문제유형": row.get("problem_type", "-"), "근거 KPI": row.get("reasoning", "-"), "예상 영향": f"machine {row.get('machine_count', 0):.0f}개", "추천 액션": row.get("recommended_action", "-"), "기대 효과": "흐름 안정화", "priority_score": row.get("bottleneck_score", 0)})
        if not lot.empty:
            for _, row in lot.sort_values("priority_score", ascending=False).head(4).iterrows():
                priority_rows.append({"대상유형": "LOT", "대상": row.get("lot_id", "-"), "문제유형": row.get("problem_type", "-"), "근거 KPI": row.get("reasoning", "-"), "예상 영향": f"machine {row.get('machine_count', 0):.0f}개 / stage {row.get('stage_count', 0):.0f}개", "추천 액션": row.get("recommended_action", "-"), "기대 효과": "확산 차단", "priority_score": row.get("priority_score", 0)})
        priority = pd.DataFrame(priority_rows)
        if not priority.empty:
            priority = priority.sort_values("priority_score", ascending=False).reset_index(drop=True)
            priority["순위"] = np.arange(1, len(priority) + 1)
            st.dataframe(priority[["순위", "대상유형", "대상", "문제유형", "근거 KPI", "예상 영향", "추천 액션", "기대 효과"]], use_container_width=True, hide_index=True)
        else:
            st.info("개선 우선순위를 만들 데이터가 부족합니다.")

        st.markdown("#### 자동 코멘트")
        if not machine.empty:
            top = machine.sort_values("bottleneck_score", ascending=False).iloc[0]
            st.markdown(f"- `{top.get('machine_id', '-')}`: {top.get('reasoning', '-')} → {top.get('recommended_action', '-')}")
        if not process.empty:
            top = process.sort_values("bottleneck_score", ascending=False).iloc[0]
            st.markdown(f"- `{top.get('process_display', '-')}`: {top.get('reasoning', '-')} → {top.get('recommended_action', '-')}")
        if not lot.empty:
            top = lot.sort_values("priority_score", ascending=False).iloc[0]
            st.markdown(f"- `{top.get('lot_id', '-')}`: {top.get('reasoning', '-')} → {top.get('recommended_action', '-')}")

        st.markdown(_section_header("이슈 유형별 조치 방향", "문제 유형을 보면 어디를 먼저 손대야 하는지 정합니다.", SECONDARY), unsafe_allow_html=True)
        action_map = pd.DataFrame([
            {"문제유형": "생산성 손실형", "왜 그렇게 보이나": "출력 감소 + 택타임 지연", "먼저 볼 것": "라인 밸런스 / 투입 타이밍 / 전환 손실", "해소점": "기준 출력 회복"},
            {"문제유형": "정지 병목", "왜 그렇게 보이나": "출력 저하 + 정체 구간 집중", "먼저 볼 것": "정지 구간 / buffer / 전환 표준", "해소점": "정체 구간 축소"},
            {"문제유형": "흐름 병목", "왜 그렇게 보이나": "stage 간 편차와 지연 증가", "먼저 볼 것": "전후 공정 연결 / stage handoff", "해소점": "연결 시간 축소"},
            {"문제유형": "주의", "왜 그렇게 보이나": "기준 대비 변동이 있으나 임계 미만", "먼저 볼 것": "같은 machine의 다른 day", "해소점": "추세 모니터링"},
            {"문제유형": "국소 LOT", "왜 그렇게 보이나": "특정 lot에만 집중", "먼저 볼 것": "대표 설비 / 대표 stage", "해소점": "확산 차단"},
            {"문제유형": "전파 LOT", "왜 그렇게 보이나": "여러 machine / stage로 퍼짐", "먼저 볼 것": "원인 lot의 경계", "해소점": "연속 lot 비교"},
            {"문제유형": "픽업/흡착 에러", "왜 그렇게 보이나": "pickup_error_count 비중이 높고 error_rate 상위", "먼저 볼 것": "feeder 정렬 / reel 장력 / nozzle 흡착", "해소점": "픽업 에러율 하락"},
            {"문제유형": "인식/비전 에러", "왜 그렇게 보이나": "recognition_error_count 비중이 높음", "먼저 볼 것": "vision 조건 / camera / 조명 / mark 인식", "해소점": "인식 에러율 하락"},
        ])
        st.dataframe(action_map, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.markdown(_section_header("이상치 기준 설정", "어떤 수준부터 이상으로 볼지 먼저 정합니다.", PRIMARY), unsafe_allow_html=True)
    _story_box(
        "이 탭의 분석 시나리오",
        [
            "먼저 이상치 기준을 정하고, 생산성과 공정 흐름이 흔들리는 설비를 찾습니다.",
            "그 다음 피더와 파트 조합을 연결해 같은 축에서 반복되는 이상 후보만 남깁니다.",
            "마지막으로 이 이상치를 근거로 문제 진단 시나리오를 만들고 즉시 조치와 예방 조치를 구분합니다.",
        ],
        tone="accent",
    )
    st.markdown("#### 우선순위 의미와 기준")
    left, right = st.columns([0.55, 0.45])
    with left:
        priority_meaning = pd.DataFrame([
            {"우선순위": "1. 즉시조치", "의미": "사람이 먼저 보고 바로 조치해야 하는 수준", "언제 쓰나": "생산 저하와 택타임 지연이 크거나 에러 증가가 함께 나타날 때"},
            {"우선순위": "2. 자동 해소", "의미": "경미하거나 단발성이라 경보·재시도·임계치 관리로 먼저 흡수 가능한 수준", "언제 쓰나": "짧은 흔들림이나 일시적 편차일 때"},
            {"우선순위": "3. 세부 원인 확인", "의미": "즉시 멈출 수준은 아니지만 반복 패턴이 있어 추가 분석이 필요한 수준", "언제 쓰나": "단일 LOT·단일 stage 반복처럼 범위는 좁지만 재발할 때"},
        ])
        st.dataframe(priority_meaning, use_container_width=True, hide_index=True)
    with right:
        priority_rule = pd.DataFrame([
            {"구분": "1. 즉시조치", "정의 기준": "output_gap_pct -5% 이하 또는 택타임 +15초 이상", "해석": "라인 손실이 커서 사람이 먼저 조치"},
            {"구분": "2. 자동 해소", "정의 기준": "output_gap_pct -3%~-5% 또는 택타임 +10~15초", "해석": "경미한 흔들림으로 자동 대응 우선"},
            {"구분": "3. 세부 원인 확인", "정의 기준": "단일 LOT·단일 stage 위주 반복", "해석": "즉시 조치보다 원인 비교가 먼저"},
        ])
        st.dataframe(priority_rule, use_container_width=True, hide_index=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        machine_gap_threshold_pct = st.slider("설비 생산 차이 기준(%)", 1, 20, 5, key=f"{mode}_machine_gap_threshold")
    with c2:
        stage_gap_threshold_pct = st.slider("공정 생산 차이 기준(%)", 1, 20, 5, key=f"{mode}_stage_gap_threshold")
    with c3:
        takt_gap_threshold_sec = st.slider("택타임 차이 기준(초)", 1, 30, 5, key=f"{mode}_takt_gap_threshold")
    with c4:
        error_rate_threshold_pct = st.slider("피더/파트 에러율 기준(%)", 0.1, 20.0, 0.5, 0.1, key=f"{mode}_error_rate_threshold")

    machine_anomaly = daily_view.copy()
    stage_anomaly = stage_daily_view.copy()
    feeder_anomaly = error_view.copy()
    if not machine_anomaly.empty:
        machine_anomaly["is_anomaly"] = (
            machine_anomaly["output_gap_pct"].le(-(machine_gap_threshold_pct / 100.0))
            | machine_anomaly["takt_gap_sec"].ge(takt_gap_threshold_sec)
        )
        machine_anomaly = machine_anomaly[machine_anomaly["is_anomaly"]].copy()
        if machine_anomaly.empty:
            machine_anomaly = daily_view.copy()
        machine_anomaly["abs_gap"] = machine_anomaly["output_gap"].abs()
        machine_anomaly = machine_anomaly.sort_values(["output_gap_pct", "takt_gap_sec", "abs_gap"], ascending=[True, False, False])
    if not stage_anomaly.empty:
        stage_anomaly["is_anomaly"] = (
            stage_anomaly["output_gap_pct"].le(-(stage_gap_threshold_pct / 100.0))
            | stage_anomaly["takt_gap_sec"].ge(takt_gap_threshold_sec)
        )
        stage_anomaly = stage_anomaly[stage_anomaly["is_anomaly"]].copy()
        if stage_anomaly.empty:
            stage_anomaly = stage_daily_view.copy()
        stage_anomaly["abs_gap"] = stage_anomaly["output_gap"].abs()
        stage_anomaly = stage_anomaly.sort_values(["output_gap_pct", "takt_gap_sec", "abs_gap"], ascending=[True, False, False])
    if not feeder_anomaly.empty:
        feeder_abs_threshold = error_rate_threshold_pct / 100.0
        feeder_top10_threshold = feeder_anomaly["error_rate"].quantile(0.90) if not feeder_anomaly.empty else 0
        feeder_anomaly["meets_absolute_threshold"] = feeder_anomaly["error_rate"].ge(feeder_abs_threshold)
        feeder_anomaly["meets_relative_threshold"] = feeder_anomaly["error_rate"].ge(feeder_top10_threshold)
        feeder_rank_view = feeder_anomaly.copy()
        feeder_rank_view["판정"] = feeder_rank_view.apply(
            lambda r: "절대 기준 초과"
            if r.get("meets_absolute_threshold", False)
            else "상대 비교 상위"
            if r.get("meets_relative_threshold", False)
            else "관찰",
            axis=1,
        )
        feeder_anomaly = feeder_anomaly[
            feeder_anomaly["meets_absolute_threshold"] | feeder_anomaly["meets_relative_threshold"]
        ].copy()
        if feeder_anomaly.empty:
            feeder_anomaly = error_view.copy()
            feeder_anomaly["meets_absolute_threshold"] = feeder_anomaly["error_rate"].ge(feeder_abs_threshold)
            feeder_anomaly["meets_relative_threshold"] = feeder_anomaly["error_rate"].ge(feeder_top10_threshold)
            feeder_rank_view = feeder_anomaly.copy()
            feeder_rank_view["판정"] = feeder_rank_view.apply(
                lambda r: "절대 기준 초과"
                if r.get("meets_absolute_threshold", False)
                else "상대 비교 상위"
                if r.get("meets_relative_threshold", False)
                else "관찰",
                axis=1,
            )
        feeder_anomaly = feeder_anomaly.sort_values(["error_rate", "error_count"], ascending=[False, False])
        feeder_rank_view = feeder_rank_view.sort_values(["error_rate", "error_count"], ascending=[False, False]).reset_index(drop=True)
    else:
        feeder_rank_view = pd.DataFrame()

    if not feeder_rank_view.empty:
        machine_benchmark = (
            feeder_rank_view.groupby("machine_id", as_index=False)
            .agg(machine_error_count=("error_count", "sum"), machine_pickup_count=("pickup_count", "sum"))
        )
        machine_benchmark["same_machine_error_rate"] = machine_benchmark.apply(
            lambda r: _safe_div(r.get("machine_error_count", 0), max(r.get("machine_pickup_count", 0), 1)),
            axis=1,
        )
        overall_error_rate = _safe_div(
            float(pd.to_numeric(feeder_rank_view.get("error_count", 0), errors="coerce").fillna(0).sum()),
            max(float(pd.to_numeric(feeder_rank_view.get("pickup_count", 0), errors="coerce").fillna(0).sum()), 1.0),
        )
        feeder_rank_view = feeder_rank_view.merge(
            machine_benchmark[["machine_id", "same_machine_error_rate"]],
            on="machine_id",
            how="left",
        )
        feeder_rank_view["all_error_rate"] = overall_error_rate
        feeder_rank_view["vs_same_machine_gap_pct"] = feeder_rank_view["error_rate"] - feeder_rank_view["same_machine_error_rate"]
        feeder_rank_view["vs_all_gap_pct"] = feeder_rank_view["error_rate"] - feeder_rank_view["all_error_rate"]
        feeder_rank_view["vs_same_machine_x"] = feeder_rank_view.apply(
            lambda r: _safe_div(r.get("error_rate", 0), max(r.get("same_machine_error_rate", 0), 1e-9)),
            axis=1,
        )
        feeder_rank_view["vs_all_x"] = feeder_rank_view.apply(
            lambda r: _safe_div(r.get("error_rate", 0), max(r.get("all_error_rate", 0), 1e-9)),
            axis=1,
        )
        feeder_rank_view["비교요약"] = feeder_rank_view.apply(
            lambda r: (
                f"설비평균 대비 {r.get('vs_same_machine_x', 0):.1f}배 "
                f"({r.get('vs_same_machine_gap_pct', 0) * 100:+.2f}%p), "
                f"전체평균 대비 {r.get('vs_all_x', 0):.1f}배 "
                f"({r.get('vs_all_gap_pct', 0) * 100:+.2f}%p)"
            ),
            axis=1,
        )
        feeder_rank_view["현재 에러율"] = feeder_rank_view["error_rate"].map(lambda v: f"{float(v) * 100:.2f}%")
        feeder_rank_view["동일 설비 평균"] = feeder_rank_view["same_machine_error_rate"].map(lambda v: f"{float(v) * 100:.2f}%")
        feeder_rank_view["전체 평균"] = feeder_rank_view["all_error_rate"].map(lambda v: f"{float(v) * 100:.2f}%")
        feeder_rank_view["설비 대비"] = feeder_rank_view["vs_same_machine_x"].map(lambda v: f"{float(v):.1f}배")
        feeder_rank_view["전체 대비"] = feeder_rank_view["vs_all_x"].map(lambda v: f"{float(v):.1f}배")

    top_machine_row = machine_anomaly.iloc[0] if not machine_anomaly.empty else pd.Series(dtype=object)
    top_stage_row = stage_anomaly.iloc[0] if not stage_anomaly.empty else pd.Series(dtype=object)
    top_feeder_row = feeder_anomaly.iloc[0] if not feeder_anomaly.empty else pd.Series(dtype=object)
    top_feeder_absolute_hit = bool(top_feeder_row.get("meets_absolute_threshold", False)) if not top_feeder_row.empty else False
    top_feeder_relative_hit = bool(top_feeder_row.get("meets_relative_threshold", False)) if not top_feeder_row.empty else False
    top_feeder_label = "우선 의심 피더/파트" if top_feeder_absolute_hit else "상대 비교상 최상위 피더/파트"
    top_feeder_foot = (
        f"에러율 {top_feeder_row.get('error_rate', 0) * 100:.2f}% / 설비 {top_feeder_row.get('machine_id', '-')}"
        if not top_feeder_row.empty and top_feeder_absolute_hit
        else f"에러율 {top_feeder_row.get('error_rate', 0) * 100:.2f}% / 절대 기준 미만이지만 상대 비교상 최상위"
        if not top_feeder_row.empty and top_feeder_relative_hit
        else "기준에 맞는 조합 없음"
    )

    st.markdown(_section_header("1. 어떤 설비가 가장 먼저 흔들리는가", "생산성과 택타임 기준으로 이상 설비를 먼저 좁힙니다.", SECONDARY), unsafe_allow_html=True)
    cards = st.columns(3)
    summary_cards = [
        (
            "우선 의심 설비",
            str(top_machine_row.get("machine_id", "-")) if not top_machine_row.empty else "-",
            f"생산 차이 {top_machine_row.get('output_gap_pct', 0) * 100:.1f}% / 택타임 차이 {top_machine_row.get('takt_gap_sec', 0):.1f}초" if not top_machine_row.empty else "기준에 맞는 설비 없음",
        ),
        (
            "우선 의심 공정",
            str(top_stage_row.get("process_display", "-")) if not top_stage_row.empty else "-",
            f"공정 생산 차이 {top_stage_row.get('output_gap_pct', 0) * 100:.1f}% / 택타임 차이 {top_stage_row.get('takt_gap_sec', 0):.1f}초" if not top_stage_row.empty else "기준에 맞는 공정 없음",
        ),
        (
            top_feeder_label,
            f"{top_feeder_row.get('feeder_id', '-')} / {top_feeder_row.get('part_number', '-')}" if not top_feeder_row.empty else "-",
            top_feeder_foot,
        ),
    ]
    for col, (label, value, foot) in zip(cards, summary_cards):
        with col:
            st.markdown(_card(label, value, foot), unsafe_allow_html=True)

    left, right = st.columns([0.58, 0.42])
    with left:
        if not machine_anomaly.empty:
            display_cols = [c for c in ["machine_id", "day", "actual_output", "expected_output", "output_gap", "output_gap_pct", "actual_takt_sec", "expected_takt_sec", "takt_gap_sec", "problem_type", "reasoning", "recommended_action"] if c in machine_anomaly.columns]
            st.dataframe(machine_anomaly[display_cols].head(8), use_container_width=True, hide_index=True)
        else:
            st.info("설비 이상치가 없습니다.")
    with right:
        if not machine_anomaly.empty:
            plot_df = machine_anomaly.copy().head(8)
            fig = px.bar(plot_df, x="machine_id", y="output_gap", text="output_gap", color="output_gap", color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"])
            st.plotly_chart(_plot_style(fig, "설비별 생산 차이", 320), use_container_width=True)
        st.markdown("##### 해석")
        st.markdown(f"- 현재 기준은 설비 생산 차이 {machine_gap_threshold_pct}% 이상 또는 택타임 차이 {takt_gap_threshold_sec}초 이상입니다.")
        st.markdown(f"- 가장 먼저 의심되는 설비는 `{top_machine_row.get('machine_id', '-')}`이며, 생산성과 택타임이 동시에 흔들립니다." if not top_machine_row.empty else "- 현재 기준에서는 강한 설비 이상이 없습니다.")
        st.markdown("- 다음 확인: 같은 설비에서 피더와 파트 에러가 같이 올라오는지 확인합니다.")

    st.markdown(_section_header("2. 생산성과 공정/Stage, 택타임을 함께 보면 어디가 원인 후보인가", "설비만의 문제인지, 공정 흐름까지 같이 흔들리는지 비교합니다.", PRIMARY), unsafe_allow_html=True)
    left, right = st.columns([0.58, 0.42])
    with left:
        if not stage_anomaly.empty:
            display_cols = [c for c in ["process_display", "day", "actual_output", "expected_output", "output_gap", "output_gap_pct", "actual_takt_sec", "expected_takt_sec", "takt_gap_sec", "problem_type", "reasoning", "recommended_action"] if c in stage_anomaly.columns]
            st.dataframe(stage_anomaly[display_cols].head(8), use_container_width=True, hide_index=True)
        else:
            st.info("공정/Stage 이상치가 없습니다.")
        if not daily_view.empty:
            takt_view = daily_view.groupby("machine_id", as_index=False).agg(
                기준선택타임_sec=("expected_takt_sec", "median"),
                실측택타임_sec=("actual_takt_sec", "median"),
                택타임편차_sec=("takt_gap_sec", "median"),
            )
            takt_view["기준선택타임_sec"] = pd.to_numeric(takt_view["기준선택타임_sec"], errors="coerce").round(0)
            takt_view = takt_view.sort_values("택타임편차_sec", ascending=False)
            st.dataframe(takt_view.head(8), use_container_width=True, hide_index=True)
    with right:
        if not stage_anomaly.empty:
            plot_df = stage_anomaly.copy().head(8)
            fig = px.bar(plot_df, x=plot_df["process_display"] + " / " + plot_df["day"].astype(str), y="output_gap", text="output_gap", color="output_gap", color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"])
            st.plotly_chart(_plot_style(fig, "공정별 생산 차이", 260), use_container_width=True)
        if not daily_view.empty:
            takt_view = daily_view.groupby("machine_id", as_index=False).agg(
                기준선택타임_sec=("expected_takt_sec", "median"),
                실측택타임_sec=("actual_takt_sec", "median"),
            )
            takt_view["기준선택타임_sec"] = pd.to_numeric(takt_view["기준선택타임_sec"], errors="coerce").round(0)
            fig = px.scatter(takt_view, x="기준선택타임_sec", y="실측택타임_sec", text="machine_id", hover_name="machine_id")
            min_v = min(float(takt_view["기준선택타임_sec"].min()), float(takt_view["실측택타임_sec"].min()))
            max_v = max(float(takt_view["기준선택타임_sec"].max()), float(takt_view["실측택타임_sec"].max()))
            fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="#94a3b8", dash="dash"))
            st.plotly_chart(_plot_style(fig, "기준선 택타임 vs 실측 택타임", 260), use_container_width=True)
        st.markdown("##### 해석")
        st.markdown(f"- 공정 기준은 생산 차이 {stage_gap_threshold_pct}% 이상 또는 택타임 차이 {takt_gap_threshold_sec}초 이상입니다.")
        st.markdown(f"- `{top_stage_row.get('process_display', '-')}` 공정이 가장 먼저 흔들리며, 설비 단위 이상이 공정 흐름으로 이어질 가능성이 있습니다." if not top_stage_row.empty else "- 현재 기준에서는 공정 병목이 강하게 보이지 않습니다.")
        st.markdown("- 대각선 위 설비일수록 실측 택타임이 기준선보다 느려 생산성 저하와 연결될 가능성이 큽니다.")

    st.markdown(_section_header("3. 어떤 피더와 파트가 원인인 것 같은가", "생산성 저하 설비와 연결된 피더/파트 조합을 확인합니다.", SECONDARY), unsafe_allow_html=True)
    left, right = st.columns([0.58, 0.42])
    with left:
        if not feeder_rank_view.empty:
            st.caption(f"현재 조건에서 비교 가능한 피더/파트 조합은 {len(feeder_rank_view)}건이며, 표는 상위 10건을 보여줍니다.")
            error_cols = [c for c in ["machine_id", "lot_id", "part_number", "feeder_id", "nozzle_serial", "error_domain", "현재 에러율", "동일 설비 평균", "전체 평균", "설비 대비", "전체 대비", "error_count", "pickup_count", "판정", "비교요약", "우선순위", "즉시조치"] if c in feeder_rank_view.columns]
            st.dataframe(feeder_rank_view[error_cols].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("피더/파트 원인 후보 데이터가 없습니다.")
    with right:
        if not feeder_rank_view.empty:
            plot_df = feeder_rank_view.head(10).copy()
            plot_df["조합"] = plot_df["feeder_id"].astype(str) + " / " + plot_df["part_number"].astype(str)
            fig = px.bar(plot_df, x="조합", y="error_rate", text="error_rate", color="error_domain")
            st.plotly_chart(_plot_style(fig, "피더/파트 에러율", 300), use_container_width=True)
        st.markdown("##### 해석")
        st.markdown(f"- 현재 피더/파트 기준은 에러율 {error_rate_threshold_pct}% 이상입니다.")
        if not top_feeder_row.empty and top_feeder_absolute_hit:
            st.markdown(f"- 가장 먼저 볼 조합은 `{top_feeder_row.get('machine_id', '-')} / {top_feeder_row.get('feeder_id', '-')} / {top_feeder_row.get('part_number', '-')}`이며 절대 기준을 넘었습니다.")
        elif not top_feeder_row.empty and top_feeder_relative_hit:
            st.markdown(f"- `{top_feeder_row.get('machine_id', '-')} / {top_feeder_row.get('feeder_id', '-')} / {top_feeder_row.get('part_number', '-')}` 조합은 절대 기준 미만이지만 상대 비교상 최상위입니다.")
        else:
            st.markdown("- 현재 기준에서는 강한 피더/파트 이상 조합이 없습니다.")
        st.markdown(f"- 이 조합은 `{top_feeder_row.get('error_domain', '-')}` 계열로 보이며, 생산성 저하 설비와 함께 보면 원인 후보로 해석할 수 있습니다." if not top_feeder_row.empty else f"- 현재 연결된 에러 데이터 기준은 `{error_source_label}`입니다.")
        if not error_priority.empty:
            msg_cols = [c for c in ["error_message", "error_domain", "error_rate", "error_count", "machine_count", "우선순위", "확인포인트"] if c in error_priority.columns]
            st.dataframe(error_priority[msg_cols].head(6), use_container_width=True, hide_index=True)

    st.markdown(_section_header("4. 원인 조치와 예방 조치", "발견된 이상을 바로 점검하고, 같은 문제가 다시 생기지 않게 정리합니다.", PRIMARY), unsafe_allow_html=True)
    action_rows = []
    if not top_machine_row.empty:
        action_rows.append({"구분": "즉시 조치", "대상": top_machine_row.get("machine_id", "-"), "왜 필요한가": top_machine_row.get("reasoning", "-"), "조치 내용": top_machine_row.get("recommended_action", "상위 설비 점검"), "기대 효과": "설비 생산성 회복"})
    if not top_feeder_row.empty:
        action_rows.append({"구분": "즉시 조치", "대상": f"{top_feeder_row.get('feeder_id', '-')} / {top_feeder_row.get('part_number', '-')}", "왜 필요한가": f"에러율 {top_feeder_row.get('error_rate', 0) * 100:.2f}% / {top_feeder_row.get('error_domain', '-')}", "조치 내용": top_feeder_row.get("즉시조치", "피더와 파트 조건 확인"), "기대 효과": "픽업/인식 에러 감소"})
    if not top_stage_row.empty:
        action_rows.append({"구분": "예방 조치", "대상": top_stage_row.get("process_display", "-"), "왜 필요한가": top_stage_row.get("reasoning", "-"), "조치 내용": top_stage_row.get("recommended_action", "공정 연결 확인"), "기대 효과": "공정 흐름 안정화"})
    action_rows.append({"구분": "예방 조치", "대상": "기준 관리", "왜 필요한가": "같은 기준으로 이상을 반복 감시해야 합니다.", "조치 내용": "설비 생산 차이, 공정 생산 차이, 택타임 차이, 피더/파트 에러율 기준을 월별로 점검", "기대 효과": "재발 조기 감지"})
    st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)

    st.markdown("##### 요약")
    if not top_machine_row.empty and not top_feeder_row.empty:
        st.markdown(
            f"- 현재 기준에서는 `{top_machine_row.get('machine_id', '-')}` 설비가 가장 먼저 흔들리고, `{top_feeder_row.get('feeder_id', '-')} / {top_feeder_row.get('part_number', '-')}` 조합이 가장 강한 원인 후보로 보입니다."
        )
    elif not top_machine_row.empty:
        st.markdown(f"- 현재 기준에서는 `{top_machine_row.get('machine_id', '-')}` 설비를 먼저 점검해야 합니다.")
    else:
        st.markdown("- 현재 기준에서는 강한 이상 설비가 제한적입니다.")
    st.markdown("- 이 탭의 목적은 이상치를 보여주는 것이 아니라, 어떤 설비와 피더/파트를 먼저 점검할지 결정하는 것입니다.")

    if is_full_flow:
        st.markdown(_section_header("5. AI 학습 데이터셋 구성", "이상치 분석 결과를 문제유형, 원인, 조치 학습 데이터로 정리하는 단계입니다.", PRIMARY), unsafe_allow_html=True)
        scenario_machine = str(top_machine_row.get("machine_id", "-")) if not top_machine_row.empty else "-"
        scenario_stage = str(top_stage_row.get("process_display", "-")) if not top_stage_row.empty else "-"
        scenario_lot = str(top_feeder_row.get("lot_id", top_lot.iloc[0].get("lot_id", "-") if not top_lot.empty else "-"))
        scenario_part = (
            f"{top_feeder_row.get('feeder_id', '-')} / {top_feeder_row.get('part_number', '-')}"
            if not top_feeder_row.empty
            else "-"
        )
        scenario_day = (
            str(top_machine_row.get("day", top_stage_row.get("day", "-")))
            if (not top_machine_row.empty or not top_stage_row.empty)
            else "-"
        )
        scenario_summary = [
            "1. 이상치 탐지: 생산 차이, 택타임, 에러율 기준으로 이상 이벤트를 찾습니다.",
            "2. 라벨링: 각 이상 이벤트를 문제유형, 원인 후보, 조치안으로 정리합니다.",
            "3. 학습셋 적재: 같은 형식의 사례를 계속 누적해 AI가 비교 학습할 수 있게 만듭니다.",
            "4. 고객 설명: 새 이상이 나오면 과거 학습 사례와 연결해 가장 이해하기 쉬운 시나리오로 설명합니다.",
        ]
        _story_box("AI 학습용 구성 시나리오", scenario_summary, tone="accent")

        evidence_rows = [
            {
                "단계": "이상치 탐지",
                "설명": f"{scenario_day} 기준 `{scenario_machine}` 설비가 생산성과 택타임에서 가장 먼저 기준을 벗어났습니다.",
                "학습 필드": "이상치 근거",
                "저장 값": f"생산 차이 {top_machine_row.get('output_gap_pct', 0) * 100:.1f}% / 택타임 차이 {top_machine_row.get('takt_gap_sec', 0):.1f}초" if not top_machine_row.empty else "강한 설비 이상 없음",
                "고객 해석": "AI가 어떤 조건을 이상으로 볼지 학습하는 시작점입니다.",
            },
            {
                "단계": "문제유형 라벨",
                "설명": f"`{scenario_stage}` 공정이 같은 기준에서 흔들리면 설비 국소 이슈인지 공정 흐름 이슈인지 분류합니다.",
                "학습 필드": "문제유형",
                "저장 값": top_stage_row.get("problem_type", top_machine_row.get("problem_type", "분류 대기")) if (not top_stage_row.empty or not top_machine_row.empty) else "분류 대기",
                "고객 해석": "AI가 이번 건을 어떤 유형의 문제로 볼지 학습하는 단계입니다.",
            },
            {
                "단계": "원인 라벨",
                "설명": f"`{scenario_part}` 조합이 `{scenario_machine}` 설비와 연결되면 원인 후보로 기록합니다.",
                "학습 필드": "원인",
                "저장 값": top_feeder_row.get("error_domain", top_feeder_foot if top_feeder_foot else "원인 후보 없음") if not top_feeder_row.empty else "원인 후보 없음",
                "고객 해석": "AI가 비슷한 이상치에서 어떤 원인을 먼저 의심할지 학습하는 단계입니다.",
            },
            {
                "단계": "조치 라벨",
                "설명": f"대표 LOT `{scenario_lot}`까지 영향 범위를 본 뒤 현장에서 실행할 조치를 정리합니다.",
                "학습 필드": "조치",
                "저장 값": top_feeder_row.get("즉시조치", top_machine_row.get("recommended_action", top_stage_row.get("recommended_action", "조치 정의 필요"))) if (not top_feeder_row.empty or not top_machine_row.empty or not top_stage_row.empty) else "조치 정의 필요",
                "고객 해석": "AI가 문제유형과 원인에 맞는 조치를 추천하도록 학습하는 단계입니다.",
            },
        ]
        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

        learning_rows = []
        if not top_machine.empty:
            for _, row in top_machine.iterrows():
                learning_rows.append({"구분": "설비", "대상": row.get("machine_id", "-"), "문제유형": row.get("problem_type", "-"), "원인": row.get("reasoning", "-"), "조치": row.get("recommended_action", "-"), "학습점수": row.get("bottleneck_score", 0)})
        if not top_process.empty:
            for _, row in top_process.iterrows():
                learning_rows.append({"구분": "공정", "대상": row.get("process_display", "-"), "문제유형": row.get("problem_type", "-"), "원인": row.get("reasoning", "-"), "조치": row.get("recommended_action", "-"), "학습점수": row.get("bottleneck_score", 0)})
        if not top_lot.empty:
            for _, row in top_lot.iterrows():
                learning_rows.append({"구분": "LOT", "대상": row.get("lot_id", "-"), "문제유형": row.get("problem_type", "-"), "원인": row.get("reasoning", "-"), "조치": row.get("recommended_action", "-"), "학습점수": row.get("priority_score", 0)})
        learning_df = pd.DataFrame(learning_rows)
        if not learning_df.empty:
            learning_df = learning_df.sort_values("학습점수", ascending=False).reset_index(drop=True)
            learning_df["샘플순번"] = np.arange(1, len(learning_df) + 1)
            st.markdown("#### AI 학습 데이터셋 예시")
            st.dataframe(learning_df[["샘플순번", "구분", "대상", "문제유형", "원인", "조치"]], use_container_width=True, hide_index=True)

            st.markdown("#### AI 이상치 사전 알림 흐름")
            st.caption("고객 설명 포인트: 현재 예시 데이터셋이 학습으로 끝나는 것이 아니라, 실시간 공정 중 위험 점수를 계산해 사전 알림을 주는 운영 흐름으로 이어집니다.")

            learning_count = int(len(learning_df))
            machine_target = scenario_machine
            process_target = scenario_stage
            lot_target = scenario_lot
            cause_target = scenario_part
            action_target = top_feeder_row.get("즉시조치", top_machine_row.get("recommended_action", top_stage_row.get("recommended_action", "현장 점검"))) if (not top_feeder_row.empty or not top_machine_row.empty or not top_stage_row.empty) else "현장 점검"

            flow_nodes = [
                f"1. 학습 데이터셋 적재<br>{learning_count}개 사례 / 문제유형·원인·조치",
                "2. 특징 추출·모델 학습<br>설비, 공정, LOT, takt, error rate",
                f"3. 실시간 공정 점수화<br>{machine_target} / {process_target} / {lot_target}",
                f"4. 사전 알림 발행<br>{cause_target} 이상 징후 감지",
                f"5. 작업자 조치·재학습<br>{action_target}",
            ]
            flow_links = pd.DataFrame(
                [
                    {"source": 0, "target": 1, "value": max(learning_count, 1), "label": "라벨링 사례 누적"},
                    {"source": 1, "target": 2, "value": max(learning_count, 1), "label": "모델 기준 배포"},
                    {"source": 2, "target": 3, "value": max(1, int(top_machine_row.get("bottleneck_score", 1) if not top_machine_row.empty else 1)), "label": "실시간 anomaly score"},
                    {"source": 3, "target": 4, "value": max(1, int(top_feeder_row.get("error_count", 1) if not top_feeder_row.empty else 1)), "label": "알림 후 조치"},
                ]
            )
            fig_flow = go.Figure(
                data=[
                    go.Sankey(
                        arrangement="snap",
                        node=dict(
                            pad=22,
                            thickness=26,
                            line=dict(color="rgba(15,23,42,0.18)", width=1),
                            label=flow_nodes,
                            color=["#1d4ed8", "#0f766e", "#b45309", "#dc2626", "#475569"],
                            hovertemplate="%{label}<extra></extra>",
                        ),
                        link=dict(
                            source=flow_links["source"],
                            target=flow_links["target"],
                            value=flow_links["value"],
                            label=flow_links["label"],
                            color=["rgba(37,99,235,0.22)", "rgba(15,118,110,0.22)", "rgba(180,83,9,0.22)", "rgba(220,38,38,0.24)"],
                            hovertemplate="%{label}<extra></extra>",
                        ),
                    )
                ]
            )
            st.plotly_chart(_plot_style(fig_flow, "학습 데이터셋 기반 이상치 사전 알림 운영 흐름", 380), use_container_width=True)

            flow_explain = pd.DataFrame(
                [
                    {"운영 단계": "학습 데이터셋", "현재 예시": f"{learning_count}개 라벨 사례", "고객이 이해할 포인트": "과거 이상치와 현장 조치를 구조화해 AI가 기준을 배웁니다."},
                    {"운영 단계": "모델 학습", "현재 예시": "설비/공정/LOT/택타임/에러율 특징", "고객이 이해할 포인트": "단순 임계치가 아니라 복합 패턴으로 이상 가능성을 계산합니다."},
                    {"운영 단계": "실시간 추론", "현재 예시": f"{machine_target} / {process_target} / {lot_target} 스트림 점수화", "고객이 이해할 포인트": "공정이 진행되는 중간에도 위험 점수를 계속 계산합니다."},
                    {"운영 단계": "사전 알림", "현재 예시": f"{cause_target} 이상 징후 발생 시 알림", "고객이 이해할 포인트": "불량 확정 후가 아니라 병목과 품질 악화 전에 작업자에게 먼저 알려줍니다."},
                    {"운영 단계": "조치 및 재학습", "현재 예시": action_target, "고객이 이해할 포인트": "현장 조치 결과를 다시 저장해 다음 알림 정확도를 높입니다."},
                ]
            )
            st.dataframe(flow_explain, use_container_width=True, hide_index=True)
            st.markdown(
                "\n".join(
                    [
                        f"- 고객 메시지 1: 지금 보는 `AI 학습 데이터셋 예시`는 보고서용 테이블이 아니라, 실시간 사전 알림 모델의 학습 원본입니다.",
                        f"- 고객 메시지 2: 예를 들어 `{machine_target}` 설비에서 `{cause_target}` 패턴이 다시 올라오면, 불량 확정 전에 작업자에게 먼저 알릴 수 있습니다.",
                        "- 고객 메시지 3: 알림 이후 실제 조치 결과까지 다시 쌓으면, 시간이 갈수록 현장에 맞는 조치 추천 정확도가 올라갑니다.",
                    ]
                )
            )

    st.markdown("</div>", unsafe_allow_html=True)


def render_rca_workflow(clean: Dict[str, pd.DataFrame], marts: Dict[str, pd.DataFrame], sample_mode: bool):
    render_prototype_tab()
