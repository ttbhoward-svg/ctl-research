"""Canonical futures acceptance evaluation framework (H.5).

Evaluates whether a futures symbol's canonical series should be accepted
based on L2/L3/L4 explainability diagnostics — NOT external exact price
match.  Cross-provider strict parity is structurally infeasible due to
roll schedule, session convention, and adjustment-basis differences.

Acceptance criteria measure internal consistency and explainability of
divergence rather than magnitude of provider disagreement.

See docs/governance/cutover_h2_h3_decision_log.md for policy rationale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ctl.cutover_diagnostics import DiagnosticResult


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Configurable thresholds for futures acceptance."""

    max_unmatched_frac: float = 0.10
    max_fail_frac: float = 0.15
    max_mean_gap_diff: float = 1.0
    max_mean_drift: float = 5.0
    min_paired_rolls: int = 20


@dataclass
class FuturesAcceptanceInput:
    """Aggregated diagnostic inputs for acceptance evaluation."""

    symbol: str
    n_canonical: int
    n_ts: int
    n_paired: int
    n_matched: int
    n_watch: int
    n_fail: int
    unmatched_canonical: int
    unmatched_ts: int
    mean_gap_diff: float
    max_gap_diff: float
    mean_drift: float
    max_drift: float
    strict_status: str   # from DiagnosticResult
    policy_status: str   # from DiagnosticResult


@dataclass
class FuturesAcceptanceResult:
    """Acceptance evaluation outcome."""

    symbol: str
    accepted: bool
    decision: str        # "ACCEPT" | "WATCH" | "REJECT"
    reasons: List[str]   # human-readable explanations for each check
    thresholds_used: AcceptanceThresholds
    input_summary: dict  # FuturesAcceptanceInput as dict

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "accepted": self.accepted,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "thresholds_used": asdict(self.thresholds_used),
            "input_summary": dict(self.input_summary),
        }


def evaluate_futures_acceptance(
    inp: FuturesAcceptanceInput,
    thresholds: Optional[AcceptanceThresholds] = None,
) -> FuturesAcceptanceResult:
    """Evaluate whether a futures symbol's canonical series should be accepted.

    Parameters
    ----------
    inp : FuturesAcceptanceInput
        Aggregated diagnostic inputs.
    thresholds : AcceptanceThresholds, optional
        Custom thresholds.  Defaults to ``AcceptanceThresholds()``.

    Returns
    -------
    FuturesAcceptanceResult
    """
    if thresholds is None:
        thresholds = AcceptanceThresholds()

    reasons: List[str] = []
    hard_fail = False
    soft_fail = False

    # 1. Minimum paired rolls.
    if inp.n_paired < thresholds.min_paired_rolls:
        reasons.append(
            f"Too few paired rolls: {inp.n_paired} < {thresholds.min_paired_rolls}"
        )
        hard_fail = True

    # 2. Unmatched fraction.
    total = inp.n_canonical + inp.n_ts
    if total > 0:
        unmatched_frac = (inp.unmatched_canonical + inp.unmatched_ts) / total
    else:
        unmatched_frac = 0.0

    if unmatched_frac > thresholds.max_unmatched_frac:
        reasons.append(
            f"Too many unmatched rolls: {unmatched_frac:.2%} > "
            f"{thresholds.max_unmatched_frac:.2%}"
        )
        hard_fail = True

    # 3. Fail fraction.
    if inp.n_paired > 0:
        fail_frac = inp.n_fail / inp.n_paired
    else:
        fail_frac = 0.0

    if fail_frac > thresholds.max_fail_frac:
        reasons.append(
            f"Too many FAIL matches: {fail_frac:.2%} > "
            f"{thresholds.max_fail_frac:.2%}"
        )
        hard_fail = True

    # 4. Mean gap diff (soft).
    if inp.mean_gap_diff > thresholds.max_mean_gap_diff:
        reasons.append(
            f"Mean gap diff too high: {inp.mean_gap_diff:.4f} > "
            f"{thresholds.max_mean_gap_diff:.4f}"
        )
        soft_fail = True

    # 5. Mean drift (soft).
    if inp.mean_drift > thresholds.max_mean_drift:
        reasons.append(
            f"Mean drift too high: {inp.mean_drift:.4f} > "
            f"{thresholds.max_mean_drift:.4f}"
        )
        soft_fail = True

    # Decision logic.
    if hard_fail:
        decision = "REJECT"
    elif soft_fail:
        decision = "WATCH"
    else:
        decision = "ACCEPT"

    return FuturesAcceptanceResult(
        symbol=inp.symbol,
        accepted=decision == "ACCEPT",
        decision=decision,
        reasons=reasons,
        thresholds_used=thresholds,
        input_summary=asdict(inp),
    )


def acceptance_from_diagnostics(
    diag: "DiagnosticResult",
    thresholds: Optional[AcceptanceThresholds] = None,
) -> FuturesAcceptanceResult:
    """Convenience: evaluate acceptance directly from a DiagnosticResult.

    Extracts ``FuturesAcceptanceInput`` fields from the diagnostic result's
    L2/L3/L4 layers and delegates to ``evaluate_futures_acceptance``.

    Parameters
    ----------
    diag : DiagnosticResult
        Combined L2 + L3 + L4 diagnostic result.
    thresholds : AcceptanceThresholds, optional
        Custom thresholds.

    Returns
    -------
    FuturesAcceptanceResult
    """
    comp = diag.l2.comparison

    inp = FuturesAcceptanceInput(
        symbol=diag.symbol,
        n_canonical=comp.n_canonical,
        n_ts=comp.n_ts,
        n_paired=comp.n_paired,
        n_matched=comp.n_matched,
        n_watch=comp.n_watch,
        n_fail=comp.n_fail,
        unmatched_canonical=comp.unmatched_canonical,
        unmatched_ts=comp.unmatched_ts,
        mean_gap_diff=diag.l3.mean_gap_diff,
        max_gap_diff=diag.l3.max_gap_diff,
        mean_drift=diag.l4.mean_drift,
        max_drift=diag.l4.max_drift,
        strict_status=diag.strict_status,
        policy_status=diag.policy_status,
    )

    return evaluate_futures_acceptance(inp, thresholds)
