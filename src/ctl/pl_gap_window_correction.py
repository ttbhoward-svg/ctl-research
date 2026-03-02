"""Offline PL roll-window-specific gap correction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import pandas as pd

from ctl.roll_reconciliation import RollManifestEntry


@dataclass(frozen=True)
class WindowBias:
    roll_date: str
    from_contract: str
    to_contract: str
    signed_gap_delta: float  # canonical_gap - ts_gap

    def to_dict(self) -> dict:
        return asdict(self)


def select_top_window_biases(l2_detail_df: pd.DataFrame, top_k: int = 5) -> List[WindowBias]:
    """Select top-K window biases from L2 detail by absolute gap diff.

    Uses PASS/WATCH rows where canonical/ts gaps are both present.
    """
    if l2_detail_df.empty or top_k <= 0:
        return []

    m = l2_detail_df.copy()
    mask = (
        m["status"].isin(["PASS", "WATCH"])
        & m["canonical_gap"].notna()
        & m["ts_gap"].notna()
        & m["canonical_date"].notna()
        & m["from_contract"].notna()
        & m["to_contract"].notna()
    )
    m = m.loc[mask].copy()
    if m.empty:
        return []

    m["signed_gap_delta"] = m["canonical_gap"] - m["ts_gap"]
    m["abs_gap_delta"] = m["signed_gap_delta"].abs()
    top = m.sort_values("abs_gap_delta", ascending=False).head(top_k)

    out: List[WindowBias] = []
    for _, r in top.iterrows():
        out.append(
            WindowBias(
                roll_date=str(pd.Timestamp(r["canonical_date"]).date()),
                from_contract=str(r["from_contract"]),
                to_contract=str(r["to_contract"]),
                signed_gap_delta=float(r["signed_gap_delta"]),
            )
        )
    return out


def apply_window_biases(
    manifest_entries: List[RollManifestEntry],
    window_biases: List[WindowBias],
) -> List[RollManifestEntry]:
    """Apply explicit per-window signed gap corrections to manifest entries.

    For a matched window, corrected gap is ``gap - signed_gap_delta``.
    Matching key: (roll_date, from_contract, to_contract).
    """
    key_to_bias: Dict[tuple, float] = {
        (b.roll_date, b.from_contract, b.to_contract): float(b.signed_gap_delta)
        for b in window_biases
    }

    out: List[RollManifestEntry] = []
    cumulative = 0.0
    for e in manifest_entries:
        key = (str(pd.Timestamp(e.roll_date).date()), e.from_contract, e.to_contract)
        bias = key_to_bias.get(key, 0.0)
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
