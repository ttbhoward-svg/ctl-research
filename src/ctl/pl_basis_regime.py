"""PL signed-basis regime split analysis utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class RegimeStats:
    label: str
    start: str
    end: str
    n_rows: int
    median_signed_diff: float
    mean_signed_diff: float
    mean_abs_diff: float
    p95_abs_diff: float
    pct_can_above_ts: float

    def to_dict(self) -> dict:
        return asdict(self)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date")
    out = out.drop_duplicates(subset=["Date"], keep="last")
    return out


def _aligned(can_df: pd.DataFrame, ts_df: pd.DataFrame) -> pd.DataFrame:
    can = _prep(can_df)
    ts = _prep(ts_df)
    m = can[["Date", "Close"]].merge(ts[["Date", "Close"]], on="Date", suffixes=("_can", "_ts"))
    if m.empty:
        return m
    m["signed_diff"] = m["Close_can"] - m["Close_ts"]
    m["abs_diff"] = m["signed_diff"].abs()
    return m


def split_regime_stats(
    can_df: pd.DataFrame,
    ts_df: pd.DataFrame,
    splits: List[tuple[str, str, str]],
) -> List[RegimeStats]:
    """Compute signed-basis stats for provided date regimes.

    Parameters
    ----------
    can_df : pd.DataFrame
        Canonical series with Date/Close.
    ts_df : pd.DataFrame
        Reference series with Date/Close.
    splits : list of (label, start, end)
        Date strings parsable by pandas.

    Returns
    -------
    list[RegimeStats]
    """
    m = _aligned(can_df, ts_df)
    if m.empty:
        return []

    out: List[RegimeStats] = []
    for label, start, end in splits:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        seg = m[(m["Date"] >= s) & (m["Date"] <= e)]
        if seg.empty:
            out.append(
                RegimeStats(
                    label=label,
                    start=str(s.date()),
                    end=str(e.date()),
                    n_rows=0,
                    median_signed_diff=0.0,
                    mean_signed_diff=0.0,
                    mean_abs_diff=0.0,
                    p95_abs_diff=0.0,
                    pct_can_above_ts=0.0,
                )
            )
            continue

        out.append(
            RegimeStats(
                label=label,
                start=str(s.date()),
                end=str(e.date()),
                n_rows=int(len(seg)),
                median_signed_diff=float(seg["signed_diff"].median()),
                mean_signed_diff=float(seg["signed_diff"].mean()),
                mean_abs_diff=float(seg["abs_diff"].mean()),
                p95_abs_diff=float(seg["abs_diff"].quantile(0.95)),
                pct_can_above_ts=float((seg["signed_diff"] > 0).mean()),
            )
        )

    return out
