"""ES drift walk-forward harmonization helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import pandas as pd


@dataclass(frozen=True)
class WalkforwardOffset:
    train_start: str
    train_end: str
    apply_start: str
    apply_end: str
    median_signed_diff: float
    n_train_rows: int

    def to_dict(self) -> dict:
        return asdict(self)


def _aligned_signed_diff(can_df: pd.DataFrame, ts_df: pd.DataFrame) -> pd.DataFrame:
    can = can_df.copy()
    can["Date"] = pd.to_datetime(can["Date"], errors="coerce")
    can = can.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    ts = ts_df.copy()
    ts["Date"] = pd.to_datetime(ts["Date"], errors="coerce")
    ts = ts.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    m = can[["Date", "Close"]].merge(ts[["Date", "Close"]], on="Date", suffixes=("_can", "_ts"))
    if m.empty:
        return m
    m["signed_diff"] = m["Close_can"] - m["Close_ts"]
    m["abs_diff"] = m["signed_diff"].abs()
    return m


def derive_walkforward_offset(
    can_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    train_start: str,
    train_end: str,
    apply_start: str,
    apply_end: str,
) -> WalkforwardOffset:
    """Derive one walk-forward median signed-diff offset from train window."""
    m = _aligned_signed_diff(can_df, ts_df)
    if m.empty:
        return WalkforwardOffset(
            train_start=train_start,
            train_end=train_end,
            apply_start=apply_start,
            apply_end=apply_end,
            median_signed_diff=0.0,
            n_train_rows=0,
        )

    s = pd.Timestamp(train_start)
    e = pd.Timestamp(train_end)
    train = m[(m["Date"] >= s) & (m["Date"] <= e)]
    med = float(train["signed_diff"].median()) if not train.empty else 0.0
    return WalkforwardOffset(
        train_start=str(s.date()),
        train_end=str(e.date()),
        apply_start=str(pd.Timestamp(apply_start).date()),
        apply_end=str(pd.Timestamp(apply_end).date()),
        median_signed_diff=med,
        n_train_rows=int(len(train)),
    )


def apply_walkforward_offset(
    can_df: pd.DataFrame,
    offset: WalkforwardOffset,
) -> pd.DataFrame:
    """Apply walk-forward offset only on apply window (offline)."""
    out = can_df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.sort_values("Date").reset_index(drop=True)
    if "Close" not in out.columns:
        raise ValueError("canonical dataframe must contain Close")

    s = pd.Timestamp(offset.apply_start)
    e = pd.Timestamp(offset.apply_end)
    mask = (out["Date"] >= s) & (out["Date"] <= e)
    out.loc[mask, "Close"] = out.loc[mask, "Close"].astype(float) - float(offset.median_signed_diff)
    return out


def window_abs_drift_mean(
    can_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    start: str,
    end: str,
) -> Tuple[float, int]:
    """Mean absolute drift over one date window on date overlap."""
    m = _aligned_signed_diff(can_df, ts_df)
    if m.empty:
        return 0.0, 0
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    seg = m[(m["Date"] >= s) & (m["Date"] <= e)]
    if seg.empty:
        return 0.0, 0
    return float(seg["abs_diff"].mean()), int(len(seg))
