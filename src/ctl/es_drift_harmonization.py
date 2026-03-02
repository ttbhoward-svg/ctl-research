"""ES drift-focused harmonization helpers (offline diagnostics)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import pandas as pd

from ctl.roll_reconciliation import DriftExplanationResult


@dataclass(frozen=True)
class DriftIntervalSummary:
    interval_start: str
    interval_end: str
    roll_status: str
    mean_drift: float
    max_drift: float
    drift_contribution_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RegimeOffset:
    label: str
    start: str
    end: str
    median_signed_diff: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_top_drift_intervals(
    explanation: DriftExplanationResult,
    top_n: int = 3,
) -> List[DriftIntervalSummary]:
    """Return top drift-contribution intervals from L4 explanation."""
    if explanation is None or not explanation.intervals or top_n <= 0:
        return []
    ranked = sorted(explanation.intervals, key=lambda x: x.drift_contribution_pct, reverse=True)[:top_n]
    return [
        DriftIntervalSummary(
            interval_start=x.interval_start,
            interval_end=x.interval_end,
            roll_status=x.roll_status,
            mean_drift=float(x.mean_drift),
            max_drift=float(x.max_drift),
            drift_contribution_pct=float(x.drift_contribution_pct),
        )
        for x in ranked
    ]


def derive_regime_offsets(
    can_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    regimes: Sequence[Tuple[str, str, str]],
) -> List[RegimeOffset]:
    """Derive median signed diff (canonical - TS) for each date regime."""
    can = can_df.copy()
    can["Date"] = pd.to_datetime(can["Date"], errors="coerce")
    can = can.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    ts = ts_df.copy()
    ts["Date"] = pd.to_datetime(ts["Date"], errors="coerce")
    ts = ts.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    m = can[["Date", "Close"]].merge(ts[["Date", "Close"]], on="Date", suffixes=("_can", "_ts"))
    if m.empty:
        return []

    m["signed_diff"] = m["Close_can"] - m["Close_ts"]
    out: List[RegimeOffset] = []
    for label, start, end in regimes:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        seg = m[(m["Date"] >= s) & (m["Date"] <= e)]
        med = float(seg["signed_diff"].median()) if not seg.empty else 0.0
        out.append(RegimeOffset(label=label, start=str(s.date()), end=str(e.date()), median_signed_diff=med))
    return out


def apply_regime_offsets(
    can_df: pd.DataFrame,
    offsets: Sequence[RegimeOffset],
) -> pd.DataFrame:
    """Apply regime offsets to canonical close (offline harmonization)."""
    out = can_df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.sort_values("Date").reset_index(drop=True)
    if "Close" not in out.columns:
        raise ValueError("canonical dataframe must contain Close")

    corrected = out["Close"].astype(float).copy()
    for off in offsets:
        s = pd.Timestamp(off.start)
        e = pd.Timestamp(off.end)
        mask = (out["Date"] >= s) & (out["Date"] <= e)
        corrected.loc[mask] = corrected.loc[mask] - float(off.median_signed_diff)
    out["Close"] = corrected
    return out
