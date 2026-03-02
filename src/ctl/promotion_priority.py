"""Promotion-priority diagnostics for gated symbols.

Builds a concise, comparable snapshot per symbol using:
- canonical acceptance status and blocker reasons
- L2/L3/L4 severity metrics
- optional MTFA audit metrics from latest run summary
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from ctl.canonical_acceptance import AcceptanceThresholds, FuturesAcceptanceResult


@dataclass(frozen=True)
class PromotionPriorityRow:
    """Comparable promotion priority snapshot for one symbol."""

    symbol: str
    decision: str
    accepted: bool
    reasons: List[str]
    n_paired: int
    n_fail: int
    unmatched_frac: float
    mean_gap_diff: float
    mean_drift: float
    drift_excess: float
    gap_excess: float
    fail_excess: float
    unmatched_excess: float
    priority_score: float
    priority_band: str
    mtfa_weekly_rate: Optional[float] = None
    mtfa_monthly_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def load_latest_run_summary(summary_dir: Path) -> Optional[dict]:
    """Load latest ``*_portfolio_run.json`` summary if available."""
    summary_dir = Path(summary_dir)
    files = sorted(summary_dir.glob("*_portfolio_run.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def extract_mtfa_rates(summary: Optional[dict]) -> Dict[str, dict]:
    """Extract MTFA rates keyed by symbol from run summary JSON."""
    if not summary:
        return {}
    out: Dict[str, dict] = {}
    for row in summary.get("symbol_run_results", []):
        sym = row.get("symbol")
        if not sym:
            continue
        out[sym] = {
            "mtfa_weekly_rate": row.get("mtfa_weekly_rate"),
            "mtfa_monthly_rate": row.get("mtfa_monthly_rate"),
        }
    return out


def _priority_band(score: float) -> str:
    if score >= 1.0:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "LOW"


def build_priority_row(
    symbol: str,
    acceptance: FuturesAcceptanceResult,
    thresholds: Optional[AcceptanceThresholds] = None,
    mtfa: Optional[dict] = None,
) -> PromotionPriorityRow:
    """Build one comparable priority row from acceptance + optional MTFA."""
    if thresholds is None:
        thresholds = AcceptanceThresholds()
    if mtfa is None:
        mtfa = {}

    s = acceptance.input_summary
    n_canonical = int(s["n_canonical"])
    n_ts = int(s["n_ts"])
    unmatched_total = int(s["unmatched_canonical"]) + int(s["unmatched_ts"])
    total = n_canonical + n_ts
    unmatched_frac = (unmatched_total / total) if total else 0.0

    n_paired = int(s["n_paired"])
    n_fail = int(s["n_fail"])
    fail_frac = (n_fail / n_paired) if n_paired else 0.0

    mean_gap = float(s["mean_gap_diff"])
    mean_drift = float(s["mean_drift"])

    drift_excess = max(0.0, mean_drift - thresholds.max_mean_drift) / thresholds.max_mean_drift
    gap_excess = max(0.0, mean_gap - thresholds.max_mean_gap_diff) / thresholds.max_mean_gap_diff
    fail_excess = max(0.0, fail_frac - thresholds.max_fail_frac) / thresholds.max_fail_frac
    unmatched_excess = max(0.0, unmatched_frac - thresholds.max_unmatched_frac) / thresholds.max_unmatched_frac

    # Weighted for current bottlenecks: drift first, then gap, then pair-quality issues.
    score = (
        0.55 * drift_excess
        + 0.25 * gap_excess
        + 0.10 * fail_excess
        + 0.10 * unmatched_excess
    )

    return PromotionPriorityRow(
        symbol=symbol,
        decision=acceptance.decision,
        accepted=acceptance.accepted,
        reasons=list(acceptance.reasons),
        n_paired=n_paired,
        n_fail=n_fail,
        unmatched_frac=round(unmatched_frac, 6),
        mean_gap_diff=round(mean_gap, 6),
        mean_drift=round(mean_drift, 6),
        drift_excess=round(drift_excess, 6),
        gap_excess=round(gap_excess, 6),
        fail_excess=round(fail_excess, 6),
        unmatched_excess=round(unmatched_excess, 6),
        priority_score=round(score, 6),
        priority_band=_priority_band(score),
        mtfa_weekly_rate=mtfa.get("mtfa_weekly_rate"),
        mtfa_monthly_rate=mtfa.get("mtfa_monthly_rate"),
    )


def rank_priority(rows: List[PromotionPriorityRow]) -> List[PromotionPriorityRow]:
    """Sort rows highest urgency first."""
    return sorted(rows, key=lambda r: (r.priority_score, r.mean_drift, r.mean_gap_diff), reverse=True)
