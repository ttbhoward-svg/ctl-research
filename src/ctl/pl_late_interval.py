"""Focused late-interval PL basis deep-dive utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class AlignStats:
    rows: int
    median_signed_diff: float
    mean_abs_diff: float
    p95_abs_diff: float

    def to_dict(self) -> dict:
        return asdict(self)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last")
    return out


def compute_align_stats(can_df: pd.DataFrame, ref_df: pd.DataFrame, date_shift_days: int = 0) -> AlignStats:
    """Compute signed/absolute diff summary for aligned dates."""
    can = _prep(can_df)
    ref = _prep(ref_df)
    ref = ref[["Date", "Close"]].copy()
    if date_shift_days != 0:
        ref["Date"] = ref["Date"] + pd.Timedelta(days=date_shift_days)

    m = can[["Date", "Close"]].merge(ref, on="Date", suffixes=("_can", "_ref"))
    if m.empty:
        return AlignStats(rows=0, median_signed_diff=0.0, mean_abs_diff=0.0, p95_abs_diff=0.0)

    sdiff = m["Close_can"] - m["Close_ref"]
    absd = sdiff.abs()
    return AlignStats(
        rows=int(len(m)),
        median_signed_diff=float(sdiff.median()),
        mean_abs_diff=float(absd.mean()),
        p95_abs_diff=float(absd.quantile(0.95)),
    )


def date_overlap_breakdown(can_df: pd.DataFrame, ref_df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return overlap / can_only / ref_only date lists (YYYY-MM-DD)."""
    can = _prep(can_df)
    ref = _prep(ref_df)

    can_dates = set(can["Date"])
    ref_dates = set(ref["Date"])

    overlap = sorted(can_dates & ref_dates)
    can_only = sorted(can_dates - ref_dates)
    ref_only = sorted(ref_dates - can_dates)

    return {
        "overlap": [d.strftime("%Y-%m-%d") for d in overlap],
        "can_only": [d.strftime("%Y-%m-%d") for d in can_only],
        "ref_only": [d.strftime("%Y-%m-%d") for d in ref_only],
    }
