"""Offline PL segment-based gap-bias correction helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence, Tuple

import pandas as pd

from ctl.roll_reconciliation import RollManifestEntry


@dataclass(frozen=True)
class SegmentGapBias:
    label: str
    start: str
    end: str
    n_rows: int
    median_signed_gap_delta: float
    mean_signed_gap_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_segment_gap_bias(
    l2_detail_df: pd.DataFrame,
    segments: Sequence[Tuple[str, str, str]],
) -> List[SegmentGapBias]:
    """Estimate signed gap bias by segment from L2 detail rows.

    Uses matched/watch rows with both canonical_gap and ts_gap present.
    Signed gap delta is ``canonical_gap - ts_gap``.
    """
    if l2_detail_df.empty:
        return [
            SegmentGapBias(label=lab, start=s, end=e, n_rows=0,
                           median_signed_gap_delta=0.0, mean_signed_gap_delta=0.0)
            for lab, s, e in segments
        ]

    work = l2_detail_df.copy()
    mask = (
        work["status"].isin(["PASS", "WATCH"])
        & work["canonical_gap"].notna()
        & work["ts_gap"].notna()
        & work["canonical_date"].notna()
    )
    work = work.loc[mask].copy()
    if work.empty:
        return [
            SegmentGapBias(label=lab, start=s, end=e, n_rows=0,
                           median_signed_gap_delta=0.0, mean_signed_gap_delta=0.0)
            for lab, s, e in segments
        ]

    work["canonical_date"] = pd.to_datetime(work["canonical_date"], errors="coerce")
    work = work.dropna(subset=["canonical_date"])
    work["signed_gap_delta"] = work["canonical_gap"] - work["ts_gap"]

    out: List[SegmentGapBias] = []
    for label, start, end in segments:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        seg = work[(work["canonical_date"] >= s) & (work["canonical_date"] <= e)]
        if seg.empty:
            out.append(SegmentGapBias(label=label, start=str(s.date()), end=str(e.date()),
                                      n_rows=0, median_signed_gap_delta=0.0, mean_signed_gap_delta=0.0))
            continue
        out.append(
            SegmentGapBias(
                label=label,
                start=str(s.date()),
                end=str(e.date()),
                n_rows=int(len(seg)),
                median_signed_gap_delta=float(seg["signed_gap_delta"].median()),
                mean_signed_gap_delta=float(seg["signed_gap_delta"].mean()),
            )
        )
    return out


def apply_segment_gap_bias(
    manifest_entries: List[RollManifestEntry],
    segment_biases: Sequence[SegmentGapBias],
) -> List[RollManifestEntry]:
    """Apply segment-specific signed gap bias to each manifest roll entry.

    For roll date within a segment:
    ``gap_corrected = gap - segment.median_signed_gap_delta``.
    """
    segs = [
        (pd.Timestamp(s.start), pd.Timestamp(s.end), float(s.median_signed_gap_delta))
        for s in segment_biases
    ]

    out: List[RollManifestEntry] = []
    cumulative = 0.0
    for e in manifest_entries:
        d = pd.Timestamp(e.roll_date)
        bias = 0.0
        for s, t, b in segs:
            if s <= d <= t:
                bias = b
                break
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
