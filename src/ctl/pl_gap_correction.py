"""Offline PL roll-gap bias correction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List

import pandas as pd

from ctl.roll_reconciliation import RollManifestEntry


@dataclass(frozen=True)
class GapBiasEstimate:
    n_rows: int
    median_signed_gap_delta: float
    mean_signed_gap_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_gap_bias(l2_detail_df: pd.DataFrame) -> GapBiasEstimate:
    """Estimate signed gap bias from matched/watch L2 rows.

    Signed gap delta is ``canonical_gap - ts_gap`` where both values exist.
    """
    if l2_detail_df.empty:
        return GapBiasEstimate(0, 0.0, 0.0)

    m = l2_detail_df.copy()
    mask = m["status"].isin(["PASS", "WATCH"]) & m["canonical_gap"].notna() & m["ts_gap"].notna()
    seg = m.loc[mask].copy()
    if seg.empty:
        return GapBiasEstimate(0, 0.0, 0.0)

    seg["signed_gap_delta"] = seg["canonical_gap"] - seg["ts_gap"]
    return GapBiasEstimate(
        n_rows=int(len(seg)),
        median_signed_gap_delta=float(seg["signed_gap_delta"].median()),
        mean_signed_gap_delta=float(seg["signed_gap_delta"].mean()),
    )


def apply_gap_bias(
    manifest_entries: List[RollManifestEntry],
    signed_gap_bias: float,
) -> List[RollManifestEntry]:
    """Apply signed gap-bias correction to manifest entries.

    New gap is ``gap - signed_gap_bias``. ``to_close`` and cumulative
    adjustment are updated for consistency.
    """
    out: List[RollManifestEntry] = []
    cumulative = 0.0
    for e in manifest_entries:
        new_gap = float(e.gap) - float(signed_gap_bias)
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
