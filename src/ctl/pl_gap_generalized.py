"""Generalized (forward-safe) PL gap correction helpers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ctl.roll_reconciliation import RollManifestEntry

_CONTRACT_RE = re.compile(r"^[A-Z]+([FGHJKMNQUVXZ])\d+$")


@dataclass(frozen=True)
class MonthGapBias:
    month_code: str
    n_rows: int
    median_signed_gap_delta: float
    mean_signed_gap_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_month_code(contract: str) -> Optional[str]:
    m = _CONTRACT_RE.match(str(contract))
    if not m:
        return None
    return m.group(1)


def estimate_month_gap_biases(
    l2_detail_df: pd.DataFrame,
    train_end_date: str,
) -> List[MonthGapBias]:
    """Estimate signed gap deltas by roll month from training-window L2 rows."""
    if l2_detail_df.empty:
        return []

    work = l2_detail_df.copy()
    mask = (
        work["status"].isin(["PASS", "WATCH"])
        & work["canonical_gap"].notna()
        & work["ts_gap"].notna()
        & work["canonical_date"].notna()
        & work["to_contract"].notna()
    )
    work = work.loc[mask].copy()
    if work.empty:
        return []

    work["canonical_date"] = pd.to_datetime(work["canonical_date"], errors="coerce")
    work = work.dropna(subset=["canonical_date"])
    work = work[work["canonical_date"] <= pd.Timestamp(train_end_date)]
    if work.empty:
        return []

    work["month_code"] = work["to_contract"].map(_extract_month_code)
    work = work.dropna(subset=["month_code"])
    work["signed_gap_delta"] = work["canonical_gap"] - work["ts_gap"]
    if work.empty:
        return []

    out: List[MonthGapBias] = []
    grouped = work.groupby("month_code", sort=True)
    for month_code, g in grouped:
        mon = month_code[0] if isinstance(month_code, tuple) else month_code
        out.append(
            MonthGapBias(
                month_code=str(mon),
                n_rows=int(len(g)),
                median_signed_gap_delta=float(g["signed_gap_delta"].median()),
                mean_signed_gap_delta=float(g["signed_gap_delta"].mean()),
            )
        )
    return out


def apply_month_gap_biases(
    manifest_entries: Iterable[RollManifestEntry],
    month_biases: List[MonthGapBias],
    apply_start_date: Optional[str] = None,
) -> List[RollManifestEntry]:
    """Apply month-based signed gap corrections to manifest entries."""
    bias_by_month: Dict[str, float] = {
        b.month_code: float(b.median_signed_gap_delta) for b in month_biases
    }
    start_ts = pd.Timestamp(apply_start_date) if apply_start_date else None

    out: List[RollManifestEntry] = []
    cumulative = 0.0
    for e in manifest_entries:
        d = pd.Timestamp(e.roll_date)
        month_code = _extract_month_code(e.to_contract)
        bias = 0.0
        if month_code is not None and month_code in bias_by_month:
            if start_ts is None or d >= start_ts:
                bias = bias_by_month[month_code]

        new_gap = float(e.gap) - bias
        cumulative += new_gap
        out.append(
            RollManifestEntry(
                roll_date=e.roll_date,
                from_contract=e.from_contract,
                to_contract=e.to_contract,
                from_close=float(e.from_close),
                to_close=float(e.from_close + new_gap),
                gap=float(new_gap),
                cumulative_adj=float(cumulative),
                trigger_reason=e.trigger_reason,
                confirmation_days=e.confirmation_days,
                convention=e.convention,
                session_template=e.session_template,
                close_type=e.close_type,
            )
        )
    return out
