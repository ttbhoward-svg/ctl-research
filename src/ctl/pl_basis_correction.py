"""Offline PL regime-aware basis correction prototype utilities.

These helpers are diagnostic-only and are not used in production gating.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class RegimeOffset:
    label: str
    start: str
    end: str
    median_signed_diff: float

    def to_dict(self) -> dict:
        return asdict(self)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last")
    return out


def derive_regime_offsets(
    can_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    regimes: List[Tuple[str, str, str]],
) -> List[RegimeOffset]:
    """Derive median signed-diff offsets by regime.

    Signed diff is ``close_can - close_ts`` on same-date overlap.
    """
    can = _prep(can_df)
    ts = _prep(ts_df)
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


def apply_regime_offsets(can_df: pd.DataFrame, offsets: List[RegimeOffset]) -> pd.DataFrame:
    """Apply per-regime offsets to canonical Close (diagnostic correction).

    For each regime row in offset window, corrected close is:
    ``Close_corrected = Close - median_signed_diff``.
    """
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
        corrected.loc[mask] = corrected.loc[mask] - off.median_signed_diff

    out["Close"] = corrected
    return out
