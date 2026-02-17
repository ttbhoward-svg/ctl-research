"""OOS evaluation and Gate 1 checking per Phase Gate Checklist v2.

Provides:
  - Tercile scoring evaluation (spread, monotonicity)
  - Quintile calibration check
  - Kill criteria checks
  - Gate 1 result aggregation with pass/iterate/reject logic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Tercile evaluation
# ---------------------------------------------------------------------------

@dataclass
class TercileResult:
    """OOS tercile analysis on TheoreticalR."""

    n_trades: int
    top_avg_r: float
    mid_avg_r: float
    bottom_avg_r: float
    top_count: int
    mid_count: int
    bottom_count: int
    spread: float           # top_avg_r - bottom_avg_r
    is_monotonic: bool      # strict: top > mid > bottom


def evaluate_terciles(
    scores: np.ndarray,
    outcomes: np.ndarray,
) -> TercileResult:
    """Split OOS trades into terciles by score; compute avg TheoreticalR.

    Parameters
    ----------
    scores : array of float
        Model-assigned scores per trade.
    outcomes : array of float
        TheoreticalR per trade.

    Returns
    -------
    TercileResult
    """
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(scores)

    if n < 3:
        return TercileResult(
            n_trades=n,
            top_avg_r=float("nan"),
            mid_avg_r=float("nan"),
            bottom_avg_r=float("nan"),
            top_count=0, mid_count=0, bottom_count=0,
            spread=float("nan"),
            is_monotonic=False,
        )

    # Rank-based tercile assignment.
    order = np.argsort(scores)
    third = n // 3

    bottom_idx = order[:third]
    mid_idx = order[third:2 * third]
    top_idx = order[2 * third:]

    top_avg = float(np.mean(outcomes[top_idx]))
    mid_avg = float(np.mean(outcomes[mid_idx]))
    bottom_avg = float(np.mean(outcomes[bottom_idx]))

    return TercileResult(
        n_trades=n,
        top_avg_r=top_avg,
        mid_avg_r=mid_avg,
        bottom_avg_r=bottom_avg,
        top_count=len(top_idx),
        mid_count=len(mid_idx),
        bottom_count=len(bottom_idx),
        spread=top_avg - bottom_avg,
        is_monotonic=(top_avg > mid_avg > bottom_avg),
    )


# ---------------------------------------------------------------------------
# Quintile calibration
# ---------------------------------------------------------------------------

@dataclass
class QuintileResult:
    """OOS quintile calibration (Gate 1 item 9)."""

    bin_avg_r: List[float]      # 5 elements, bin 0 = lowest scores
    bin_counts: List[int]
    is_monotonic: bool          # non-decreasing avg R from bottom to top


def evaluate_quintiles(
    scores: np.ndarray,
    outcomes: np.ndarray,
) -> QuintileResult:
    """Split OOS trades into quintiles by score; check calibration."""
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    n = len(scores)

    if n < 5:
        return QuintileResult(
            bin_avg_r=[float("nan")] * 5,
            bin_counts=[0] * 5,
            is_monotonic=False,
        )

    order = np.argsort(scores)
    fifth = n // 5

    bin_avg_r: List[float] = []
    bin_counts: List[int] = []

    for q in range(5):
        start = q * fifth
        end = (q + 1) * fifth if q < 4 else n
        idx = order[start:end]
        bin_avg_r.append(float(np.mean(outcomes[idx])))
        bin_counts.append(len(idx))

    # Non-decreasing: each bin avg R <= next bin avg R.
    is_mono = all(bin_avg_r[i] <= bin_avg_r[i + 1] for i in range(4))

    return QuintileResult(
        bin_avg_r=bin_avg_r,
        bin_counts=bin_counts,
        is_monotonic=is_mono,
    )


# ---------------------------------------------------------------------------
# Kill criteria
# ---------------------------------------------------------------------------

@dataclass
class KillCheck:
    """Kill / pause criteria results from Phase Gate Checklist v2."""

    # REJECT conditions
    top_r_below_threshold: bool = False    # top avg R < 0.5
    no_predictive_power: bool = False      # score-R correlation < 0.05

    # PAUSE conditions
    monotonicity_failure: bool = False     # mid > top on OOS

    details: List[str] = field(default_factory=list)

    @property
    def any_reject(self) -> bool:
        return self.top_r_below_threshold or self.no_predictive_power

    @property
    def any_pause(self) -> bool:
        return self.monotonicity_failure


def check_kill_criteria(
    tercile: TercileResult,
    scores: np.ndarray,
    outcomes: np.ndarray,
    top_r_threshold: float = 0.5,
    corr_threshold: float = 0.05,
) -> KillCheck:
    """Evaluate kill criteria from Phase Gate Checklist v2.

    Parameters
    ----------
    tercile : TercileResult
        Pre-computed tercile evaluation.
    scores, outcomes : arrays
        Raw score and TheoreticalR arrays for correlation check.
    top_r_threshold : float
        Minimum acceptable top-tercile avg R (default 0.5).
    corr_threshold : float
        Minimum score-outcome Pearson correlation (default 0.05).
    """
    details: List[str] = []
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)

    # REJECT: top avg R < threshold.
    top_r_low = bool(tercile.top_avg_r < top_r_threshold)
    if top_r_low:
        details.append(
            f"REJECT: top-tercile avg R ({tercile.top_avg_r:.3f}) "
            f"< {top_r_threshold}"
        )

    # REJECT: score-outcome correlation < threshold.
    if len(scores) >= 3 and np.std(scores) > 0 and np.std(outcomes) > 0:
        corr = float(np.corrcoef(scores, outcomes)[0, 1])
    else:
        corr = 0.0
    no_power = bool(corr < corr_threshold)
    if no_power:
        details.append(
            f"REJECT: score-outcome correlation ({corr:.4f}) "
            f"< {corr_threshold}"
        )

    # PAUSE: monotonicity failure (mid > top).
    mono_fail = bool(tercile.mid_avg_r > tercile.top_avg_r)
    if mono_fail:
        details.append(
            f"PAUSE: mid-tercile avg R ({tercile.mid_avg_r:.3f}) "
            f"> top ({tercile.top_avg_r:.3f})"
        )

    return KillCheck(
        top_r_below_threshold=top_r_low,
        no_predictive_power=no_power,
        monotonicity_failure=mono_fail,
        details=details,
    )


# ---------------------------------------------------------------------------
# Gate 1 result
# ---------------------------------------------------------------------------

@dataclass
class Gate1Result:
    """Gate 1 (Phase 1a -> 1b) evaluation per Phase Gate Checklist v2.

    Nine required items.  Items not yet evaluated are None; overall is
    INCOMPLETE until all items are populated.
    """

    # --- Items evaluated by this module ---
    oos_trade_count: int = 0
    oos_trade_count_pass: bool = False              # item 1: >= 30

    tercile_spread: float = 0.0
    tercile_spread_pass: bool = False               # item 2: >= 1.0R

    monotonicity_pass: bool = False                 # item 3

    quintile_calibration_pass: Optional[bool] = None  # item 9

    # --- Items set by caller (later tasks) ---
    feature_cap_respected: Optional[bool] = None       # item 4
    model_card_complete: Optional[bool] = None          # item 5
    negative_controls_passed: Optional[bool] = None     # item 6
    entry_degradation_pass: Optional[bool] = None       # item 7
    slippage_stress_pass: Optional[bool] = None         # item 8

    # --- Kill criteria ---
    kill_check: Optional[KillCheck] = None

    # --- Overall ---
    overall: str = "INCOMPLETE"

    @property
    def items_summary(self) -> List[Dict]:
        """Return a list of dicts describing each gate item."""
        return [
            {"item": 1, "name": "OOS trade count >= 30",
             "result": self.oos_trade_count_pass,
             "value": self.oos_trade_count},
            {"item": 2, "name": "Tercile spread >= 1.0R",
             "result": self.tercile_spread_pass,
             "value": round(self.tercile_spread, 3)},
            {"item": 3, "name": "Score-to-outcome monotonicity",
             "result": self.monotonicity_pass},
            {"item": 4, "name": "Feature cap respected",
             "result": self.feature_cap_respected},
            {"item": 5, "name": "Model card complete",
             "result": self.model_card_complete},
            {"item": 6, "name": "Negative controls passed",
             "result": self.negative_controls_passed},
            {"item": 7, "name": "Entry degradation within tolerance",
             "result": self.entry_degradation_pass},
            {"item": 8, "name": "Slippage stress profitable at 2 ticks",
             "result": self.slippage_stress_pass},
            {"item": 9, "name": "Quintile calibration monotonic",
             "result": self.quintile_calibration_pass},
        ]


def check_gate1(
    tercile: TercileResult,
    quintile: QuintileResult | None = None,
    kill: KillCheck | None = None,
    *,
    feature_cap_respected: bool | None = None,
    model_card_complete: bool | None = None,
    negative_controls_passed: bool | None = None,
    entry_degradation_pass: bool | None = None,
    slippage_stress_pass: bool | None = None,
    min_oos_trades: int = 30,
    min_spread: float = 1.0,
) -> Gate1Result:
    """Evaluate Gate 1 criteria per Phase Gate Checklist v2.

    Parameters
    ----------
    tercile : TercileResult
    quintile : QuintileResult, optional
    kill : KillCheck, optional
    feature_cap_respected ... slippage_stress_pass : optional bools
        External gate items populated by later tasks.
    min_oos_trades : int
        Minimum OOS trades for item 1 (default 30).
    min_spread : float
        Minimum top-bottom tercile spread for item 2 (default 1.0R).
    """
    result = Gate1Result()

    # Item 1: trade count.
    result.oos_trade_count = tercile.n_trades
    result.oos_trade_count_pass = tercile.n_trades >= min_oos_trades

    # Item 2: tercile spread.
    result.tercile_spread = tercile.spread
    result.tercile_spread_pass = bool(tercile.spread >= min_spread)

    # Item 3: monotonicity.
    result.monotonicity_pass = tercile.is_monotonic

    # Item 9: quintile calibration.
    if quintile is not None:
        result.quintile_calibration_pass = quintile.is_monotonic

    # External items.
    result.feature_cap_respected = feature_cap_respected
    result.model_card_complete = model_card_complete
    result.negative_controls_passed = negative_controls_passed
    result.entry_degradation_pass = entry_degradation_pass
    result.slippage_stress_pass = slippage_stress_pass

    # Kill check.
    result.kill_check = kill

    # --- Overall decision ---
    # Severity: REJECT > PAUSE > ITERATE > INCOMPLETE > PASS
    if kill is not None and kill.any_reject:
        result.overall = "REJECT"
    elif kill is not None and kill.any_pause:
        result.overall = "PAUSE"
    else:
        all_items = [
            result.oos_trade_count_pass,
            result.tercile_spread_pass,
            result.monotonicity_pass,
            result.quintile_calibration_pass,
            result.feature_cap_respected,
            result.model_card_complete,
            result.negative_controls_passed,
            result.entry_degradation_pass,
            result.slippage_stress_pass,
        ]
        has_fail = any(item is False for item in all_items)
        has_none = any(item is None for item in all_items)

        if has_fail:
            result.overall = "ITERATE"
        elif has_none:
            result.overall = "INCOMPLETE"
        else:
            result.overall = "PASS"

    return result
