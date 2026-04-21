from __future__ import annotations

from typing import Iterable

import streamlit as st


def section_title(question: str, finding: str) -> None:
    st.markdown("#### 질문")
    st.markdown(question)
    st.markdown("#### 핵심 발견")
    st.info(finding)


def takeaway(text: str, next_check: str) -> None:
    st.caption(f"해석: {text}")
    st.caption(f"다음 확인: {next_check}")


def bullet_block(lines: Iterable[str], title: str) -> None:
    st.markdown(f"#### {title}")
    for line in lines:
        st.markdown(f"- {line}")
