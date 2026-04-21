from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from rca_prototype.analysis.detection import analyze_problem_detection
from rca_prototype.analysis.diagnosis import analyze_stage_machine
from rca_prototype.analysis.interpretation import build_executive_messages
from rca_prototype.analysis.lot_analysis import analyze_lot_impact
from rca_prototype.analysis.quality_impact import analyze_quality_impact
from rca_prototype.analysis.rca_rules import rank_cause_candidates
from rca_prototype.data.synthetic_generator import generate_synthetic_data
from rca_prototype.ui import conclusion_page, detection_page, error_page, executive, machine_page, quality_page
from rca_prototype.utils.constants import RANDOM_SEED, SCENARIO_TITLE


@st.cache_data(show_spinner=False)
def _load_bundle(seed: int = RANDOM_SEED) -> Dict:
    data = generate_synthetic_data(seed=seed)
    detection = analyze_problem_detection(data)
    diagnosis = analyze_stage_machine(data, detection)
    causes = rank_cause_candidates(data, detection, diagnosis)
    quality = analyze_quality_impact(data)
    lots = analyze_lot_impact(data)
    messages = build_executive_messages(
        {
            "detection": detection,
            "causes": causes,
            "quality": quality,
            "lots": lots,
        }
    )
    return {
        "data": data,
        "detection": detection,
        "diagnosis": diagnosis,
        "causes": causes,
        "quality": quality,
        "lots": lots,
        "messages": messages,
    }


def render_prototype_tab() -> None:
    bundle = _load_bundle()
    st.markdown("### RCA 프로토타입")
    st.caption("이 탭은 실제 공장 데이터가 없다는 가정하에, 일관된 synthetic data만으로 구성한 RCA 데모입니다. 나중에 실제 데이터로 교체해도 같은 구조를 유지할 수 있도록 설계했습니다.")
    st.info(f"시나리오: {SCENARIO_TITLE}")
    pages = st.tabs(
        [
            "1. 요약",
            "2. 문제 감지",
            "3. 공정·설비 진단",
            "4. 에러·자재 RCA",
            "5. 품질·LOT 영향",
            "6. 결론·조치 가이드",
        ]
    )
    with pages[0]:
        executive.render_page(bundle)
    with pages[1]:
        detection_page.render_page(bundle)
    with pages[2]:
        machine_page.render_page(bundle)
    with pages[3]:
        error_page.render_page(bundle)
    with pages[4]:
        quality_page.render_page(bundle)
    with pages[5]:
        conclusion_page.render_page(bundle)


def render_ai_demo_tab() -> None:
    bundle = _load_bundle()
    data = bundle["data"]
    detection = bundle["detection"]
    diagnosis = bundle["diagnosis"]
    causes = bundle["causes"]["candidates"]
    quality = bundle["quality"]["quality_effect"]
    lots = bundle["lots"]["lot_finding"]

    top_machine = diagnosis["machine_rank"].iloc[0] if not diagnosis["machine_rank"].empty else None
    top_stage = detection["stage_summary"].iloc[0] if not detection["stage_summary"].empty else None
    top_cause = causes.iloc[0] if not causes.empty else None
    top_quality = quality.iloc[0] if not quality.empty else None
    top_lot = lots.iloc[0] if not lots.empty else None

    st.markdown("### 생성형AI 데모")
    st.caption("이 탭은 실제 생성형 AI 연동 전 단계의 데모입니다. 고객이 질문하면, 현재 샘플 RCA 결과를 근거로 답변하는 방식입니다.")

    def _get_openai_config() -> tuple[str | None, str]:
        api_key = None
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = None
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        return api_key, model

    def _build_llm_context() -> str:
        machine_note = (
            f"상위 설비: {top_machine['machine_id']}, output_gap={top_machine['output_gap']:.0f}, cycle_delay_sec={top_machine['cycle_delay_sec']:.2f}, pickup_error_count={top_machine['pickup_error_count']:.0f}"
            if top_machine is not None
            else "상위 설비 정보 없음"
        )
        stage_note = (
            f"상위 공정: {top_stage['stage_name']}, output_gap={top_stage['output_gap']:.0f}, avg_cycle_delay_sec={top_stage['avg_cycle_delay_sec']:.2f}"
            if top_stage is not None
            else "상위 공정 정보 없음"
        )
        cause_note = (
            f"상위 원인 후보: {top_cause['cause_category']} / {top_cause['hypothesis']} / 근거: {top_cause['evidence']}"
            if top_cause is not None
            else "상위 원인 후보 정보 없음"
        )
        quality_note = (
            f"품질 영향: {top_quality['finding']} / peak_timestamp={top_quality['peak_timestamp']}"
            if top_quality is not None
            else "품질 영향 정보 없음"
        )
        lot_note = (
            f"LOT 영향: {top_lot['finding']} / impact_type={top_lot['impact_type']}"
            if top_lot is not None
            else "LOT 영향 정보 없음"
        )
        return "\n".join(
            [
                "이 시스템은 SMT RCA 샘플 시나리오를 설명하는 데모입니다.",
                "시나리오: Mounter의 pickup 불안정이 생산 손실과 AOI 하류 품질 영향으로 이어짐",
                machine_note,
                stage_note,
                cause_note,
                quality_note,
                lot_note,
                "답변 규칙:",
                "- 한국어로 답변",
                "- 발표자가 고객에게 설명하듯 간결하게 답변",
                "- 추정과 사실을 구분",
                "- 가능하면 설비, 공정, LOT, 품질 영향, 조치를 함께 연결",
                "- 데이터에 없는 내용은 단정하지 말고 추가 확인이 필요하다고 말할 것",
            ]
        )

    def _call_openai_llm(user_question: str) -> tuple[str | None, str | None]:
        api_key, model = _get_openai_config()
        if not api_key:
            return None, None
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _build_llm_context()},
                {"role": "user", "content": user_question},
            ],
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return str(content).strip(), model
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            return f"OpenAI API 호출에 실패했습니다. 상태코드 {exc.code}. {detail[:240]}", model
        except Exception as exc:
            return f"OpenAI API 호출에 실패했습니다. {exc}", model

    tab_questions = [
        {
            "tab_key": "detection",
            "tab_label": "문제 감지",
            "question": "라인 생산성은 어디에서 무너지기 시작했는가?",
            "hint": "시간대와 공정 기준으로 문제 시작점을 설명",
        },
        {
            "tab_key": "machine",
            "tab_label": "공정·설비 진단",
            "question": "어느 공정과 어느 설비가 책임 구간인가?",
            "hint": "문제 공정과 핵심 설비를 설명",
        },
        {
            "tab_key": "cause",
            "tab_label": "에러·자재 RCA",
            "question": "가장 가능성이 높은 원인 후보는 무엇인가?",
            "hint": "피더·노즐·부품 조합 중심 원인 설명",
        },
        {
            "tab_key": "quality",
            "tab_label": "품질·LOT 영향",
            "question": "이 설비 이슈가 하류 품질과 LOT에 영향을 주었는가?",
            "hint": "AOI와 LOT 영향 관점으로 설명",
        },
        {
            "tab_key": "action",
            "tab_label": "결론·조치 가이드",
            "question": "공장은 무엇부터 점검해야 하는가?",
            "hint": "즉시 조치와 우선순위를 설명",
        },
    ]
    st.markdown("#### 원인분석 탭 질문")
    question_labels = [f"{item['tab_label']} · {item['question']}" for item in tab_questions]
    selected_label = st.selectbox("질문 선택", question_labels, key="ai_demo_question_select")
    selected_item = next(item for item in tab_questions if f"{item['tab_label']} · {item['question']}" == selected_label)
    st.caption(f"선택된 질문 관점: {selected_item['hint']}")
    custom_question = st.text_input("직접 질문", value="", placeholder="예: 왜 M05 설비를 먼저 봐야 하나요?", key="ai_demo_custom_question")
    execute_cols = st.columns([0.2, 0.8])
    with execute_cols[0]:
        run_clicked = st.button("질문 실행", key="ai_demo_run")
    with execute_cols[1]:
        st.caption("직접 입력한 질문이 있으면 그 질문을 우선 실행하고, 없으면 선택한 추천 질문을 실행합니다.")

    question = custom_question.strip() or selected_item["question"]
    question_key = selected_item["tab_key"] if not custom_question.strip() else "custom"

    def _answer_for_question(user_question: str, question_key: str) -> tuple[str, list[str], list[str], str]:
        q = user_question.lower()
        if question_key == "quality" or any(token in q for token in ["품질", "lot", "aoi", "영향", "하류", "불량"]):
            answer = (
                f"이번 이슈는 하류 품질과 LOT에도 연결된 것으로 보입니다. "
                f"{top_quality['finding'] if top_quality is not None else '하류 품질 영향은 추가 확인이 필요합니다.'} "
                f"{top_lot['finding'] if top_lot is not None else 'LOT 영향은 추가 확인이 필요합니다.'}"
            )
            evidence = [
                f"품질 영향: {top_quality['finding']}" if top_quality is not None else "품질 영향 근거가 부족합니다.",
                f"LOT 영향: {top_lot['finding']}" if top_lot is not None else "LOT 영향 근거가 부족합니다.",
                f"대표 원인 후보: {top_cause['cause_category']} / {top_cause['hypothesis']}" if top_cause is not None else "원인 후보 근거가 부족합니다.",
            ]
            next_checks = [
                "AOI 불량 증가 시점이 실장기 이상 시간대와 맞물리는지 확인",
                "영향이 큰 LOT와 직전 LOT를 비교",
                "같은 LOT가 여러 설비에 퍼졌는지 확인",
            ]
            return answer, evidence, next_checks, "quality_lot"

        if question_key == "machine" or any(token in q for token in ["왜", "설비", "m05", "mounter", "실장기"]):
            answer = (
                f"현재 샘플 시나리오에서는 {top_machine['machine_id'] if top_machine is not None else '-'} 설비를 가장 먼저 보는 것이 맞습니다. "
                f"이 설비가 동급 설비 대비 생산 차이와 사이클 지연이 가장 크고, "
                f"{top_cause['cause_category'] if top_cause is not None else '원인 후보'} 근거가 함께 연결되기 때문입니다."
            )
            evidence = [
                f"설비 근거: {top_machine['machine_id']} 설비가 가장 높은 이상 점수를 보입니다." if top_machine is not None else "설비 근거가 부족합니다.",
                f"공정 근거: {top_stage['stage_name']} 공정 손실이 가장 큽니다." if top_stage is not None else "공정 근거가 부족합니다.",
                f"원인 근거: {top_cause['hypothesis']}" if top_cause is not None else "원인 후보 근거가 부족합니다.",
            ]
            next_checks = [
                "해당 설비의 피더 정렬과 노즐 상태 확인",
                "흡착 관련 에러가 특정 부품에서 반복되는지 확인",
                "같은 시간대 AOI와 LOT 영향 연결 여부 확인",
            ]
            return answer, evidence, next_checks, "machine_focus"

        if question_key == "cause" or any(token in q for token in ["원인", "피더", "노즐", "부품", "흡착"]):
            answer = (
                "현재 샘플 시나리오에서는 흡착 계열 원인 후보가 가장 강합니다. "
                f"{top_cause['hypothesis'] if top_cause is not None else '핵심 원인 후보는 추가 확인이 필요합니다.'}"
            )
            evidence = [
                f"원인 범주: {top_cause['cause_category']}" if top_cause is not None else "원인 범주 근거가 부족합니다.",
                f"가설: {top_cause['hypothesis']}" if top_cause is not None else "가설 근거가 부족합니다.",
                "자재 조합 근거: FDR-05 / NZ-03 / PN-004 집중" if top_cause is not None else "자재 조합 근거가 부족합니다.",
            ]
            next_checks = [
                "FDR-05 피더 정렬 상태 확인",
                "NZ-03 노즐 마모 및 오염 확인",
                "PN-004 공급 안정성과 흡착 조건 확인",
            ]
            return answer, evidence, next_checks, "cause_material"

        if question_key == "action" or any(token in q for token in ["조치", "예방", "무엇을", "어떻게", "해야", "대응"]):
            answer = (
                "우선은 이상 설비와 자재 조건을 바로 점검하고, 이후에는 같은 기준으로 재발을 감시하는 예방 조치까지 같이 가져가는 것이 적절합니다."
            )
            evidence = [
                f"즉시 조치 대상: {top_machine['machine_id']} 설비" if top_machine is not None else "즉시 조치 대상 근거가 부족합니다.",
                f"핵심 원인 후보: {top_cause['cause_category']}" if top_cause is not None else "핵심 원인 후보 근거가 부족합니다.",
                f"LOT 영향: {top_lot['finding']}" if top_lot is not None else "LOT 영향 근거가 부족합니다.",
            ]
            next_checks = [
                "즉시 조치: 설비, 피더, 노즐, 부품 흡착 조건 점검",
                "예방 조치: 기준 생산량과 택타임 기준 월별 재점검",
                "확산 방지: 영향 LOT와 인접 LOT 비교",
            ]
            return answer, evidence, next_checks, "action_plan"

        if question_key == "detection":
            answer = (
                f"문제는 넓은 구간이 아니라 특정 시간대에서 시작된 것으로 보입니다. "
                f"{top_stage['stage_name'] if top_stage is not None else '핵심 공정'} 공정이 가장 큰 생산 손실을 만들었고, 그 안에서 추가 설비 드릴다운이 필요합니다."
            )
            evidence = [
                f"상위 공정: {top_stage['stage_name']}" if top_stage is not None else "공정 근거가 부족합니다.",
                f"공정 손실: output_gap={top_stage['output_gap']:.0f}" if top_stage is not None else "공정 손실 근거가 부족합니다.",
                "시간대별 실제 생산량이 계획 생산량 아래로 내려가는 구간이 있습니다.",
            ]
            next_checks = [
                "이상 시간대를 설비별 변화와 비교",
                "실장기 공정 내부 설비 비교로 드릴다운",
                "동시간대 품질 영향 여부 확인",
            ]
            return answer, evidence, next_checks, "detection_flow"

        answer = (
            f"가장 먼저 점검해야 할 대상은 "
            f"{top_machine['machine_id'] if top_machine is not None else '-'} 설비로 보입니다. "
            f"{top_stage['stage_name'] if top_stage is not None else '핵심 공정'} 공정에서 손실이 가장 크고, "
            f"가장 강한 원인 후보는 {top_cause['cause_category'] if top_cause is not None else '추가 확인 필요'}입니다."
        )
        evidence = [
            f"설비 근거: {top_machine['machine_id']} 설비의 생산 차이가 가장 크고 사이클 지연도 큽니다." if top_machine is not None else "설비 근거가 부족합니다.",
            f"공정 근거: {top_stage['stage_name']} 공정에서 생산 손실이 가장 큽니다." if top_stage is not None else "공정 근거가 부족합니다.",
            f"원인 근거: {top_cause['hypothesis']}" if top_cause is not None else "원인 후보 근거가 부족합니다.",
        ]
        next_checks = [
            f"{top_machine['machine_id']} 설비를 먼저 점검" if top_machine is not None else "상위 설비를 먼저 점검",
            "피더 정렬, 노즐 상태, 부품 흡착 조건 확인",
            "같은 시간대 품질과 LOT 영향이 같이 나타나는지 확인",
        ]
        return answer, evidence, next_checks, "machine_focus"

    llm_cache = st.session_state.setdefault("ai_demo_llm_cache", {})
    llm_cache_key = f"{question_key}::{question}"
    fallback_answer, evidence, next_checks, answer_type = _answer_for_question(question, question_key)
    if llm_cache_key not in llm_cache or run_clicked:
        llm_cache[llm_cache_key] = _call_openai_llm(question)
    llm_answer, llm_model = llm_cache.get(llm_cache_key, (None, None))
    answer = llm_answer or fallback_answer

    def _render_supporting_view(view_type: str) -> None:
        st.markdown("#### 관련 데이터")
        if view_type == "detection_flow":
            line_time = detection["line_time"].copy()
            stage_summary = detection["stage_summary"].copy()
            left, right = st.columns(2)
            with left:
                if not line_time.empty:
                    line_view = line_time.rename(
                        columns={
                            "timestamp": "시간",
                            "line_actual_output": "실제 생산량",
                            "line_expected_output": "계획 생산량",
                        }
                    )
                    fig = px.line(line_view, x="시간", y=["실제 생산량", "계획 생산량"], markers=True, title="시간대별 실제 생산량 vs 계획 생산량")
                    st.plotly_chart(fig, use_container_width=True, key="ai_demo_detection_flow_line")
                else:
                    st.info("라인 시계열 데이터가 없습니다.")
            with right:
                if not stage_summary.empty:
                    stage_view = stage_summary.rename(
                        columns={
                            "stage_name": "공정",
                            "actual_output": "실제 생산량",
                            "expected_output": "계획 생산량",
                            "output_gap": "생산량 차이",
                            "avg_cycle_delay_sec": "평균 지연 시간(초)",
                        }
                    )
                    st.dataframe(stage_view[["공정", "실제 생산량", "계획 생산량", "생산량 차이", "평균 지연 시간(초)"]], use_container_width=True, hide_index=True)
                else:
                    st.info("공정 비교 데이터가 없습니다.")
            return

        if view_type == "machine_focus":
            machine_rank = diagnosis["machine_rank"].copy()
            material_context = data["material_context"].copy()
            left, right = st.columns(2)
            with left:
                if not machine_rank.empty:
                    machine_view = machine_rank.rename(
                        columns={
                            "machine_id": "설비",
                            "output_gap": "생산량 차이",
                            "pickup_error_count": "흡착 에러 수",
                            "cycle_delay_sec": "Cycle 지연(초)",
                            "actual_output": "실제 생산량",
                            "expected_output": "계획 생산량",
                            "confidence": "신뢰도",
                            "diagnosis_note": "해석",
                        }
                    )
                    fig = px.bar(machine_view, x="설비", y="생산량 차이", color="흡착 에러 수", text="Cycle 지연(초)", title="실장기 설비별 비교")
                    st.plotly_chart(fig, use_container_width=True, key="ai_demo_machine_focus_bar")
                else:
                    st.info("설비 비교 데이터가 없습니다.")
            with right:
                if not machine_rank.empty:
                    machine_view = machine_rank.rename(
                        columns={
                            "machine_id": "설비",
                            "actual_output": "실제 생산량",
                            "expected_output": "계획 생산량",
                            "output_gap": "생산량 차이",
                            "cycle_delay_sec": "Cycle 지연(초)",
                            "pickup_error_count": "흡착 에러 수",
                            "confidence": "신뢰도",
                            "diagnosis_note": "해석",
                        }
                    )
                    st.dataframe(machine_view[["설비", "실제 생산량", "계획 생산량", "생산량 차이", "Cycle 지연(초)", "흡착 에러 수", "신뢰도", "해석"]], use_container_width=True, hide_index=True)
                elif not material_context.empty:
                    st.dataframe(material_context.head(8), use_container_width=True, hide_index=True)
                else:
                    st.info("설비 상세 데이터가 없습니다.")
            return

        if view_type == "cause_material":
            event_mix = diagnosis["event_mix"].copy()
            material_context = data["material_context"].copy()
            left, right = st.columns(2)
            with left:
                if not event_mix.empty:
                    event_view = event_mix.melt(
                        id_vars="machine_id",
                        value_vars=["pickup_error_count", "recognition_error_count", "mechanical_error_count"],
                        var_name="error_type",
                        value_name="error_count",
                    )
                    event_view["error_type"] = event_view["error_type"].map(
                        {
                            "pickup_error_count": "흡착 에러",
                            "recognition_error_count": "인식 에러",
                            "mechanical_error_count": "기계 에러",
                        }
                    )
                    event_view = event_view.rename(columns={"machine_id": "설비", "error_count": "에러 수"})
                    fig = px.bar(event_view, x="설비", y="에러 수", color="error_type", barmode="group", title="설비별 에러 유형 비교")
                    st.plotly_chart(fig, use_container_width=True, key="ai_demo_cause_material_event_mix")
                else:
                    st.info("에러 유형 비교 데이터가 없습니다.")
            with right:
                if not material_context.empty:
                    material_view = (
                        material_context.groupby(["machine_id", "feeder_id", "nozzle_id", "part_number"], as_index=False)
                        .agg(pickup_error_count=("pickup_error_count", "sum"))
                        .sort_values("pickup_error_count", ascending=False)
                        .head(8)
                        .rename(
                            columns={
                                "machine_id": "설비",
                                "feeder_id": "피더",
                                "nozzle_id": "노즐",
                                "part_number": "부품",
                                "pickup_error_count": "흡착 에러 수",
                            }
                        )
                    )
                    st.dataframe(material_view, use_container_width=True, hide_index=True)
                elif not causes.empty:
                    cause_view = causes[["rank", "cause_category", "hypothesis", "confidence", "evidence"]].rename(
                        columns={"rank": "순위", "cause_category": "원인 범주", "hypothesis": "가설", "confidence": "신뢰도", "evidence": "근거"}
                    )
                    st.dataframe(cause_view, use_container_width=True, hide_index=True)
                else:
                    st.info("원인 후보 데이터가 없습니다.")
            return

        if view_type == "quality_lot":
            quality_time = bundle["quality"]["quality_time"].copy()
            defect_mix = bundle["quality"]["defect_mix"].copy()
            lot_summary = bundle["lots"]["lot_summary"].copy()
            left, right = st.columns(2)
            with left:
                if not quality_time.empty:
                    fig = px.line(quality_time, x="timestamp", y="aoi_fail_count", markers=True, title="AOI 불량 추이")
                    st.plotly_chart(fig, use_container_width=True, key="ai_demo_quality_lot_quality_time")
                    if not defect_mix.empty:
                        defect_view = defect_mix.rename(columns={"defect_type": "불량 유형", "defect_count": "불량 수"})
                        st.dataframe(defect_view, use_container_width=True, hide_index=True)
                else:
                    st.info("품질 추이 데이터가 없습니다.")
            with right:
                if not lot_summary.empty:
                    lot_view = lot_summary[["lot_id", "lot_output_gap", "lot_defect_rate", "affected_machine_count", "affected_stage_count", "impact_type"]].copy()
                    st.dataframe(lot_view, use_container_width=True, hide_index=True)
                else:
                    st.info("LOT 비교 데이터가 없습니다.")
            return

        if view_type == "action_plan":
            action_rows = [
                {"우선순위": 1, "조치 항목": f"실장기 {top_machine['machine_id']} 설비를 가장 먼저 점검" if top_machine is not None else "상위 설비를 가장 먼저 점검", "목적": "설비 국소 이상 여부 확인"},
                {"우선순위": 2, "조치 항목": "핵심 피더 정렬 상태 확인", "목적": "흡착 관련 집중 여부 검증"},
                {"우선순위": 3, "조치 항목": "핵심 노즐 상태 점검", "목적": "노즐 마모 또는 오염 배제"},
                {"우선순위": 4, "조치 항목": "핵심 부품 공급 안정성 확인", "목적": "자재 조건이 손실을 만들었는지 확인"},
                {"우선순위": 5, "조치 항목": "영향 LOT와 직전 LOT 비교", "목적": "영향이 국소적인지 확산형인지 확인"},
            ]
            left, right = st.columns(2)
            with left:
                st.dataframe(pd.DataFrame(action_rows), use_container_width=True, hide_index=True)
            with right:
                if not causes.empty:
                    cause_view = causes[["rank", "cause_category", "hypothesis", "confidence"]].rename(
                        columns={"rank": "순위", "cause_category": "원인 범주", "hypothesis": "가설", "confidence": "신뢰도"}
                    )
                    st.dataframe(cause_view, use_container_width=True, hide_index=True)
                else:
                    st.info("조치 우선순위 근거 데이터가 없습니다.")
            return

        st.info("관련 데이터를 구성할 수 없습니다.")

    st.markdown("#### 질문")
    st.markdown(f"- {question}")
    if question_key != "custom":
        source_tab = next((item["tab_label"] for item in tab_questions if item["tab_key"] == question_key), "-")
        st.caption(f"질문 출처: 원인분석(샘플 시나리오) > {source_tab}")
    elif custom_question.strip():
        st.caption("질문 출처: 직접 질문")
    st.markdown("#### 답변")
    st.markdown(answer)
    if llm_answer and llm_model:
        st.caption(f"실제 LLM 응답 사용: {llm_model}")
    else:
        st.caption("OpenAI API 키가 없거나 호출에 실패해 규칙 기반 답변으로 표시했습니다.")
    st.markdown("#### 답변 근거")
    st.markdown("\n".join([f"- {line}" for line in evidence]))
    st.markdown("#### 다음 확인")
    st.markdown("\n".join([f"- {line}" for line in next_checks]))
    _render_supporting_view(answer_type)
