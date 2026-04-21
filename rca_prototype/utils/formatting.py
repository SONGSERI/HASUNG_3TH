import pandas as pd


def fmt_pct(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def fmt_num(value: float) -> str:
    return f"{float(value):,.0f}"


def fmt_sec(value: float) -> str:
    return f"{float(value):.1f}s"


def confidence_label(score: float) -> str:
    if score >= 3.6:
        return "높음"
    if score >= 2.2:
        return "중간"
    return "낮음"


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default
