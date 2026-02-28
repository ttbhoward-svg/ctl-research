"""CL roll-policy calibration runner (Data Cutover Task H.3).

Sweeps roll policy variants (consecutive_days, eligible_months, roll_timing)
and scores each against TradeStation reference data using L2/L3/L4 diagnostics.

Outputs:
- Ranked calibration table (CSV)
- Recommended policy (JSON)

See docs/notes/TaskH_assumptions.md for design rationale.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ctl.continuous_builder import (
    ContinuousResult,
    ContractSpec,
    RollEvent,
    _build_roll_log,
    apply_panama_adjustment,
    detect_rolls,
    load_contract_data,
    parse_contracts,
)
from ctl.cutover_diagnostics import run_diagnostics
from ctl.roll_reconciliation import RollManifestEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CL contract month presets
# ---------------------------------------------------------------------------

#: All CL delivery months (monthly contracts).
CL_ALL_MONTHS: Tuple[str, ...] = (
    "F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z",
)

#: Quarterly subset (standard IMM months).
CL_QUARTERLY_MONTHS: Tuple[str, ...] = ("H", "M", "U", "Z")

# ---------------------------------------------------------------------------
# Plausibility guardrail
# ---------------------------------------------------------------------------

#: Minimum plausible CL rolls over ~8 years of data.
MIN_PLAUSIBLE_ROLLS: int = 40

#: Maximum plausible CL rolls over ~8 years of data.
MAX_PLAUSIBLE_ROLLS: int = 140


# ---------------------------------------------------------------------------
# Composite scoring weights
# ---------------------------------------------------------------------------

#: Weight for total unmatched rolls (canonical + TS) in composite score.
SCORE_W_UNMATCHED: float = 10.0

#: Weight for FAIL count in composite score.
SCORE_W_FAIL: float = 5.0

#: Weight for mean gap difference in composite score.
SCORE_W_GAP: float = 100.0

#: Weight for mean drift in composite score.
SCORE_W_DRIFT: float = 1.0


# ---------------------------------------------------------------------------
# Variant definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RollPolicyVariant:
    """One roll-policy configuration to test."""

    consecutive_days: int
    eligible_months: Optional[Tuple[str, ...]]  # None = all months
    roll_timing: str  # "same_day" or "next_session"

    @property
    def label(self) -> str:
        return (
            f"cd={self.consecutive_days}"
            f"_months={self.months_label}"
            f"_timing={self.roll_timing}"
        )

    @property
    def months_label(self) -> str:
        if self.eligible_months is None:
            return "all"
        if self.eligible_months == CL_QUARTERLY_MONTHS:
            return "quarterly"
        return "_".join(self.eligible_months)


def generate_cl_variants() -> List[RollPolicyVariant]:
    """Generate the standard CL calibration sweep variants.

    Sweep dimensions:
    - consecutive_days: [1, 2, 3]
    - eligible_months: [all, quarterly]
    - roll_timing: ["same_day", "next_session"]

    Returns 3 × 2 × 2 = 12 variants.
    """
    variants: List[RollPolicyVariant] = []
    for cd in [1, 2, 3]:
        for months in [None, CL_QUARTERLY_MONTHS]:
            for timing in ["same_day", "next_session"]:
                variants.append(RollPolicyVariant(
                    consecutive_days=cd,
                    eligible_months=months,
                    roll_timing=timing,
                ))
    return variants


# ---------------------------------------------------------------------------
# Calibration score
# ---------------------------------------------------------------------------

@dataclass
class CalibrationScore:
    """Diagnostic scores for one variant."""

    variant: RollPolicyVariant
    n_rolls: int = 0
    n_canonical: int = 0
    n_ts: int = 0
    n_paired: int = 0
    n_matched: int = 0
    n_watch: int = 0
    n_fail: int = 0
    unmatched_canonical: int = 0
    unmatched_ts: int = 0
    day_delta_histogram: Dict[int, int] = field(default_factory=dict)
    cumulative_signed_day_shift: int = 0
    mean_gap_diff: float = 0.0
    max_gap_diff: float = 0.0
    mean_drift: float = 0.0
    max_drift: float = 0.0
    composite_score: float = float("inf")
    strict_status: str = "FAIL"
    policy_status: str = "FAIL"
    rank: int = 0
    valid: bool = True

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "label": self.variant.label,
            "valid": self.valid,
            "consecutive_days": self.variant.consecutive_days,
            "eligible_months": self.variant.months_label,
            "roll_timing": self.variant.roll_timing,
            "n_rolls": self.n_rolls,
            "n_canonical": self.n_canonical,
            "n_ts": self.n_ts,
            "n_paired": self.n_paired,
            "n_matched": self.n_matched,
            "n_watch": self.n_watch,
            "n_fail": self.n_fail,
            "unmatched_canonical": self.unmatched_canonical,
            "unmatched_ts": self.unmatched_ts,
            "mean_gap_diff": round(self.mean_gap_diff, 6),
            "max_gap_diff": round(self.max_gap_diff, 6),
            "mean_drift": round(self.mean_drift, 6),
            "max_drift": round(self.max_drift, 6),
            "composite_score": round(self.composite_score, 4),
            "strict_status": self.strict_status,
            "policy_status": self.policy_status,
        }


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def compute_composite_score(score: CalibrationScore) -> float:
    """Compute the composite calibration score (lower is better)."""
    return (
        SCORE_W_UNMATCHED * (score.unmatched_canonical + score.unmatched_ts)
        + SCORE_W_FAIL * score.n_fail
        + SCORE_W_GAP * score.mean_gap_diff
        + SCORE_W_DRIFT * score.mean_drift
    )


def rank_variants(scores: List[CalibrationScore]) -> List[CalibrationScore]:
    """Rank calibration scores by composite (lower is better).

    Invalid variants (implausible roll counts) are sorted last.
    Tie-break order: fewer FAILs → fewer unmatched → lower drift.
    """
    for s in scores:
        if s.valid:
            s.composite_score = compute_composite_score(s)
        else:
            s.composite_score = float("inf")

    scores.sort(key=lambda s: (
        not s.valid,  # valid first
        s.composite_score,
        s.n_fail,
        s.unmatched_canonical + s.unmatched_ts,
        s.mean_drift,
    ))

    for i, s in enumerate(scores):
        s.rank = i + 1
    return scores


# ---------------------------------------------------------------------------
# Manifest builder (internal)
# ---------------------------------------------------------------------------

def _manifest_from_roll_log(
    roll_log: pd.DataFrame,
    convention: str,
) -> List[RollManifestEntry]:
    """Convert a roll log DataFrame to RollManifestEntry list."""
    entries: List[RollManifestEntry] = []
    if roll_log.empty:
        return entries
    for _, row in roll_log.iterrows():
        gap = float(row["to_close"]) - float(row["from_close"])
        entries.append(RollManifestEntry(
            roll_date=str(row["date"]),
            from_contract=str(row["from_contract"]),
            to_contract=str(row["to_contract"]),
            from_close=float(row["from_close"]),
            to_close=float(row["to_close"]),
            gap=gap,
            cumulative_adj=float(row["cumulative_adjustment"]),
            convention=convention,
        ))
    return entries


# ---------------------------------------------------------------------------
# Variant evaluation
# ---------------------------------------------------------------------------

def _evaluate_variant(
    contracts: Dict[str, pd.DataFrame],
    contract_order: List[ContractSpec],
    ts_adj_df: pd.DataFrame,
    ts_unadj_df: pd.DataFrame,
    root: str,
    tick_size: float,
    convention: str,
    variant: RollPolicyVariant,
    max_day_delta: int,
    min_gap_ticks: int,
    dedup_window: int,
    min_rolls: int,
    max_rolls: int,
) -> CalibrationScore:
    """Evaluate a single roll-policy variant."""
    # Build continuous series with variant params.
    rolls, active_series = detect_rolls(
        contracts, contract_order,
        consecutive_days=variant.consecutive_days,
        eligible_months=variant.eligible_months,
        roll_timing=variant.roll_timing,
    )

    # Plausibility guardrail.
    if len(rolls) < min_rolls or len(rolls) > max_rolls:
        logger.info(
            "Variant %s: %d rolls outside plausible range [%d, %d]; marked invalid",
            variant.label, len(rolls), min_rolls, max_rolls,
        )
        return CalibrationScore(
            variant=variant, n_rolls=len(rolls), valid=False,
        )

    if active_series.empty:
        return CalibrationScore(variant=variant, valid=False)

    # Fresh roll copies for cumulative adjustment computation.
    rolls_copy = [
        RollEvent(
            date=r.date,
            from_contract=r.from_contract,
            to_contract=r.to_contract,
            from_close=r.from_close,
            to_close=r.to_close,
            adjustment=r.adjustment,
        )
        for r in rolls
    ]
    continuous = apply_panama_adjustment(
        contracts, active_series, rolls_copy, convention=convention,
    )

    if continuous.empty:
        return CalibrationScore(variant=variant, n_rolls=len(rolls), valid=False)

    # Build manifest entries from rolls.
    roll_log = _build_roll_log(rolls_copy, len(contracts))
    manifest_entries = _manifest_from_roll_log(roll_log, convention)

    # Run diagnostics.
    diag = run_diagnostics(
        canonical_adj_df=continuous,
        ts_adj_df=ts_adj_df,
        manifest_entries=manifest_entries,
        ts_unadj_df=ts_unadj_df,
        symbol=root,
        tick_size=tick_size,
        min_gap_ticks=min_gap_ticks,
        max_day_delta=max_day_delta,
        dedup_window=dedup_window,
    )

    # Extract scores.
    l2 = diag.l2.comparison
    return CalibrationScore(
        variant=variant,
        n_rolls=len(rolls),
        n_canonical=l2.n_canonical,
        n_ts=l2.n_ts,
        n_paired=l2.n_paired,
        n_matched=l2.n_matched,
        n_watch=l2.n_watch,
        n_fail=l2.n_fail,
        unmatched_canonical=l2.unmatched_canonical,
        unmatched_ts=l2.unmatched_ts,
        day_delta_histogram=dict(l2.day_delta_histogram),
        cumulative_signed_day_shift=l2.cumulative_signed_day_shift,
        mean_gap_diff=diag.l3.mean_gap_diff,
        max_gap_diff=diag.l3.max_gap_diff,
        mean_drift=diag.l4.mean_drift,
        max_drift=diag.l4.max_drift,
        strict_status=diag.strict_status,
        policy_status=diag.policy_status,
        valid=True,
    )


# ---------------------------------------------------------------------------
# Calibration runner
# ---------------------------------------------------------------------------

def calibrate_roll_policy(
    contracts: Dict[str, pd.DataFrame],
    contract_order: List[ContractSpec],
    ts_adj_df: pd.DataFrame,
    ts_unadj_df: pd.DataFrame,
    root: str = "CL",
    tick_size: float = 0.01,
    convention: str = "add",
    variants: Optional[List[RollPolicyVariant]] = None,
    max_day_delta: int = 2,
    min_gap_ticks: int = 2,
    dedup_window: int = 3,
    min_rolls: int = MIN_PLAUSIBLE_ROLLS,
    max_rolls: int = MAX_PLAUSIBLE_ROLLS,
) -> List[CalibrationScore]:
    """Run roll-policy calibration sweep.

    For each variant, builds the continuous series, exports a manifest,
    and runs L2/L3/L4 diagnostics against the TradeStation reference.

    Parameters
    ----------
    contracts : dict
        Pre-loaded contract data from ``load_contract_data``.
    contract_order : list of ContractSpec
        Sorted contract chain.
    ts_adj_df : pd.DataFrame
        TradeStation adjusted reference (Date, Close).
    ts_unadj_df : pd.DataFrame
        TradeStation unadjusted reference (Date, Close).
    root : str
        Root symbol (e.g. "CL").
    tick_size : float
        Instrument tick size.
    convention : str
        Adjustment convention for parity comparison.
    variants : list of RollPolicyVariant, optional
        Variants to test.  Defaults to ``generate_cl_variants()``.
    max_day_delta : int
        Maximum calendar-day tolerance for roll matching.
    min_gap_ticks : int
        Minimum spread-step in ticks for TS roll detection.
    dedup_window : int
        Trading-day dedup window for TS roll inference.
    min_rolls : int
        Minimum plausible roll count (below → invalid).
    max_rolls : int
        Maximum plausible roll count (above → invalid).

    Returns
    -------
    List of CalibrationScore, ranked by composite score.
    """
    if variants is None:
        variants = generate_cl_variants()

    scores: List[CalibrationScore] = []

    for var in variants:
        try:
            score = _evaluate_variant(
                contracts, contract_order, ts_adj_df, ts_unadj_df,
                root, tick_size, convention, var, max_day_delta,
                min_gap_ticks, dedup_window, min_rolls, max_rolls,
            )
        except Exception as exc:
            logger.warning("Variant %s failed: %s", var.label, exc)
            score = CalibrationScore(variant=var, valid=False)
        scores.append(score)

    return rank_variants(scores)


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def save_calibration_artifacts(
    scores: List[CalibrationScore],
    out_dir: Path,
    symbol: str = "CL",
) -> Dict[str, Path]:
    """Save calibration results as CSV and recommendation JSON.

    Files written:
    - ``{symbol}_policy_calibration.csv``
    - ``{symbol}_policy_recommendation.json``

    Parameters
    ----------
    scores : list of CalibrationScore
        Ranked calibration scores (from ``calibrate_roll_policy``).
    out_dir : Path
        Output directory.
    symbol : str
        Symbol label for file names.

    Returns
    -------
    dict mapping artifact name to saved file path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    # CSV table.
    csv_path = out_dir / f"{symbol}_policy_calibration.csv"
    rows = [s.to_dict() for s in scores]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    paths["calibration_csv"] = csv_path

    # Recommendation JSON.
    json_path = out_dir / f"{symbol}_policy_recommendation.json"
    best = scores[0] if scores else None
    recommendation = {
        "symbol": symbol,
        "recommended_variant": best.variant.label if best else None,
        "recommended_params": {
            "consecutive_days": best.variant.consecutive_days,
            "eligible_months": best.variant.months_label,
            "roll_timing": best.variant.roll_timing,
        } if best else None,
        "composite_score": round(best.composite_score, 4) if best else None,
        "strict_status": best.strict_status if best else None,
        "policy_status": best.policy_status if best else None,
        "top_3": [s.to_dict() for s in scores[:3]],
        "total_variants_tested": len(scores),
    }
    with open(json_path, "w") as f:
        json.dump(recommendation, f, indent=2)
    paths["recommendation_json"] = json_path

    return paths
