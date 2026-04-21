from datetime import timedelta

import numpy as np
import pandas as pd


def _safe_div(a, b):
    try:
        if b is None or pd.isna(b) or float(b) == 0:
            return 0.0
        return float(a) / float(b)
    except Exception:
        return 0.0


def _fmt_sec(sec: float) -> str:
    try:
        sec = int(sec)
        return str(timedelta(seconds=sec))
    except Exception:
        return "0:00:00"


def _pick(df: pd.DataFrame, cols: list[str], default=np.nan):
    out = None
    for c in cols:
        if c in df.columns:
            out = df[c] if out is None else out.combine_first(df[c])
    return out if out is not None else pd.Series([default] * len(df), index=df.index)


def _pick_dt(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = None
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            out = s if out is None else out.combine_first(s)
    return out if out is not None else pd.Series(pd.NaT, index=df.index)


def _pick_txt(df: pd.DataFrame, cols: list[str], default="미상") -> pd.Series:
    return _pick(df, cols, default=default).astype("object").fillna(default)


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    return out
