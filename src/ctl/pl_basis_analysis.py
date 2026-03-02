"""PL interval-level basis analysis helpers.

Turns L2/L4 diagnostic outputs into a ranked interval report with signed-basis
statistics and roll-proximity mismatch counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _to_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _extract_roll_dates(l2_detail_df: pd.DataFrame) -> pd.Series:
    """Extract usable dates from L2 detail rows (canonical + TS where present)."""
    if l2_detail_df.empty:
        return pd.Series(dtype="datetime64[ns]")
    dates = []
    if "canonical_date" in l2_detail_df.columns:
        dates.append(_to_ts(l2_detail_df["canonical_date"]))
    if "ts_date" in l2_detail_df.columns:
        dates.append(_to_ts(l2_detail_df["ts_date"]))
    if not dates:
        return pd.Series(dtype="datetime64[ns]")
    out = pd.concat(dates, ignore_index=True).dropna()
    return out


def build_interval_basis_report(
    drift_df: pd.DataFrame,
    l2_detail_df: pd.DataFrame,
    explanation: Dict,
    top_n: int = 5,
    roll_window_days: int = 3,
) -> pd.DataFrame:
    """Build a ranked PL basis report for top drift-contributing intervals.

    Parameters
    ----------
    drift_df : pd.DataFrame
        L4 drift dataframe containing Date, close_can, close_ts.
    l2_detail_df : pd.DataFrame
        L2 detail dataframe containing status/canonical_date/ts_date.
    explanation : dict
        L4 explanation dict (e.g., ``diag.l4.explanation.to_dict()``).
    top_n : int
        Number of highest-contribution intervals to include.
    roll_window_days : int
        Window around interval bounds for counting nearby FAIL roll rows.

    Returns
    -------
    pd.DataFrame
        Interval metrics sorted by drift contribution descending.
    """
    if drift_df.empty or not explanation:
        return pd.DataFrame()

    work = drift_df.copy()
    work["Date"] = _to_ts(work["Date"])
    work = work.dropna(subset=["Date", "close_can", "close_ts"]).sort_values("Date")
    if work.empty:
        return pd.DataFrame()

    work["signed_diff"] = work["close_can"] - work["close_ts"]
    work["abs_diff"] = work["signed_diff"].abs()

    intervals: List[dict] = list(explanation.get("intervals", []))
    if not intervals:
        return pd.DataFrame()

    top = sorted(intervals, key=lambda x: x.get("drift_contribution_pct", 0.0), reverse=True)[:top_n]

    fail_rows = l2_detail_df[l2_detail_df.get("status") == "FAIL"] if not l2_detail_df.empty else pd.DataFrame()
    fail_dates = _extract_roll_dates(fail_rows)

    rows = []
    for item in top:
        start = pd.to_datetime(item.get("interval_start"), errors="coerce")
        end = pd.to_datetime(item.get("interval_end"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue

        seg = work[(work["Date"] >= start) & (work["Date"] <= end)]
        if seg.empty:
            continue

        n_bars = int(len(seg))
        can_above = float((seg["signed_diff"] > 0).mean())
        near_fail = 0
        if not fail_dates.empty:
            lo = start - pd.Timedelta(days=roll_window_days)
            hi = end + pd.Timedelta(days=roll_window_days)
            near_fail = int(((fail_dates >= lo) & (fail_dates <= hi)).sum())

        rows.append(
            {
                "interval_start": str(start.date()),
                "interval_end": str(end.date()),
                "roll_status": item.get("roll_status", ""),
                "drift_contribution_pct": round(float(item.get("drift_contribution_pct", 0.0)), 4),
                "n_bars": n_bars,
                "mean_abs_diff": round(float(seg["abs_diff"].mean()), 6),
                "p95_abs_diff": round(float(np.nanpercentile(seg["abs_diff"], 95)), 6),
                "median_signed_diff": round(float(seg["signed_diff"].median()), 6),
                "pct_can_above_ts": round(can_above, 4),
                "nearby_fail_roll_rows": near_fail,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("drift_contribution_pct", ascending=False).reset_index(drop=True)


def save_interval_basis_report(df: pd.DataFrame, out_path: Path) -> Optional[Path]:
    """Persist report to CSV when non-empty."""
    if df.empty:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
