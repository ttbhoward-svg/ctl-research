"""Gate 1 Pass/Fail Decision Package (Task 14).

Produces the formal Gate 1 decision artifact by evaluating all 9
Phase Gate Checklist criteria against locked thresholds and upstream
task outputs.

See docs/notes/Task14_assumptions.md for design rationale.

Usage
-----
>>> from ctl.gate_decision import evaluate_gate1, GateInput
>>> inp = GateInput(
...     oos_result=oos_result,          # Task 13
...     stress_report=stress_report,    # Task 4b
...     nc_report=nc_report,            # Task 11b
...     degradation_report=deg_report,  # Task 11c
...     feature_cap_respected=True,
...     model_card_complete=True,
...     dataset_hash="abc123",
... )
>>> decision = evaluate_gate1(inp)
>>> print(decision.summary())
>>> decision.save_json(Path("outputs/gate1_decision.json"))
>>> decision.save_markdown(Path("outputs/gate1_decision.md"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional

from ctl.entry_degradation import DegradationReport
from ctl.negative_controls import NegativeControlReport
from ctl.oos_evaluation import CriterionResult as OOSCriterion
from ctl.oos_evaluation import OOSResult
from ctl.slippage_stress import StressReport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Verdict = Literal["PASS", "ITERATE", "PAUSE", "REJECT", "INCOMPLETE"]

#: Severity ordering (higher index = more severe).
_SEVERITY = {"PASS": 0, "INCOMPLETE": 1, "ITERATE": 2, "PAUSE": 3, "REJECT": 4}

#: Pre-written remediation actions per criterion.
_REMEDIATION: Dict[int, str] = {
    1: "Collect more OOS trades; do not lower the 30-trade minimum.",
    2: "Iterate features/ranges per pre-registered fallback only.",
    3: "Investigate feature decay or regime shift; do not override.",
    4: "Revert to locked feature set; no ad-hoc additions.",
    5: "Complete model card v2 template with all required sections.",
    6: "Investigate data leakage or overfitting; re-run negative controls.",
    7: "Review entry assumptions; tighten or document edge sensitivity.",
    8: "Review execution assumptions; edge may be execution-dependent.",
    9: "Investigate calibration breakdown; check for score distribution shift.",
}


# ---------------------------------------------------------------------------
# Gate criterion
# ---------------------------------------------------------------------------

@dataclass
class GateCriterion:
    """Evaluation result for one Gate 1 criterion."""

    item: int
    name: str
    status: str  # "PASS", "FAIL", "INCOMPLETE"
    value: Optional[str] = None  # human-readable evidence
    reason: str = ""
    remediation: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


# ---------------------------------------------------------------------------
# Kill check
# ---------------------------------------------------------------------------

@dataclass
class KillCheckResult:
    """Kill/pause criteria aggregated from upstream checks."""

    top_tercile_kill: bool = False  # K.1: top avg R < 0.5
    correlation_kill: bool = False  # K.2: score-R corr < 0.05
    monotonicity_pause: bool = False  # mid > top on OOS
    details: List[str] = field(default_factory=list)

    @property
    def any_reject(self) -> bool:
        return self.top_tercile_kill or self.correlation_kill

    @property
    def any_pause(self) -> bool:
        return self.monotonicity_pause and not self.any_reject


# ---------------------------------------------------------------------------
# Input / output
# ---------------------------------------------------------------------------

@dataclass
class GateInput:
    """All inputs needed for Gate 1 evaluation."""

    # Upstream task results (None = not yet computed).
    oos_result: Optional[OOSResult] = None
    stress_report: Optional[StressReport] = None
    nc_report: Optional[NegativeControlReport] = None
    degradation_report: Optional[DegradationReport] = None

    # Manual attestations.
    feature_cap_respected: Optional[bool] = None
    model_card_complete: Optional[bool] = None

    # Provenance metadata.
    dataset_hash: str = ""
    config_hash: str = ""
    code_commit_hash: str = ""


@dataclass
class GateDecision:
    """Complete Gate 1 decision artifact."""

    verdict: Verdict
    criteria: List[GateCriterion]
    kill_check: KillCheckResult
    timestamp: str
    dataset_hash: str
    config_hash: str
    code_commit_hash: str

    @property
    def n_passed(self) -> int:
        return sum(1 for c in self.criteria if c.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.criteria if c.status == "FAIL")

    @property
    def n_incomplete(self) -> int:
        return sum(1 for c in self.criteria if c.status == "INCOMPLETE")

    @property
    def failed_items(self) -> List[GateCriterion]:
        return [c for c in self.criteria if c.status == "FAIL"]

    @property
    def incomplete_items(self) -> List[GateCriterion]:
        return [c for c in self.criteria if c.status == "INCOMPLETE"]

    # --- Human-readable ---

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "GATE 1 DECISION — Phase 1a → Phase 1b",
            "=" * 65,
            f"Timestamp : {self.timestamp}",
            f"Dataset   : {self.dataset_hash or '(not provided)'}",
            f"Config    : {self.config_hash or '(not provided)'}",
            f"Commit    : {self.code_commit_hash or '(not provided)'}",
            "",
            f"VERDICT: {self.verdict}",
            f"  Passed: {self.n_passed}/9  Failed: {self.n_failed}  "
            f"Incomplete: {self.n_incomplete}",
            "",
            "--- Criteria ---",
        ]
        for c in self.criteria:
            icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "INCOMPLETE": "[----]"}[c.status]
            lines.append(f"  {icon} Item {c.item}: {c.name}")
            if c.value:
                lines.append(f"         Value: {c.value}")
            if c.reason:
                lines.append(f"         Reason: {c.reason}")
            if c.remediation:
                lines.append(f"         Action: {c.remediation}")
        if self.kill_check.any_reject or self.kill_check.any_pause:
            lines.append("")
            lines.append("--- Kill / Pause ---")
            for d in self.kill_check.details:
                lines.append(f"  {d}")
        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)

    # --- Export ---

    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict,
            "timestamp": self.timestamp,
            "dataset_hash": self.dataset_hash,
            "config_hash": self.config_hash,
            "code_commit_hash": self.code_commit_hash,
            "n_passed": self.n_passed,
            "n_failed": self.n_failed,
            "n_incomplete": self.n_incomplete,
            "criteria": [
                {
                    "item": c.item, "name": c.name, "status": c.status,
                    "value": c.value, "reason": c.reason,
                    "remediation": c.remediation,
                }
                for c in self.criteria
            ],
            "kill_check": {
                "top_tercile_kill": self.kill_check.top_tercile_kill,
                "correlation_kill": self.kill_check.correlation_kill,
                "monotonicity_pause": self.kill_check.monotonicity_pause,
                "details": self.kill_check.details,
            },
        }

    def save_json(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    def save_markdown(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Gate 1 Decision — Phase 1a to Phase 1b",
            "",
            f"**Verdict: {self.verdict}**",
            "",
            "## Provenance",
            "",
            f"- Timestamp: `{self.timestamp}`",
            f"- Dataset hash: `{self.dataset_hash or 'N/A'}`",
            f"- Config hash: `{self.config_hash or 'N/A'}`",
            f"- Code commit: `{self.code_commit_hash or 'N/A'}`",
            "",
            "## Criteria Table",
            "",
            "| # | Criterion | Status | Value | Reason |",
            "|---|-----------|--------|-------|--------|",
        ]
        for c in self.criteria:
            val = c.value or "—"
            reason = c.reason or "—"
            lines.append(f"| {c.item} | {c.name} | {c.status} | {val} | {reason} |")

        if self.failed_items:
            lines.append("")
            lines.append("## Remediation Actions")
            lines.append("")
            for c in self.failed_items:
                if c.remediation:
                    lines.append(f"- **Item {c.item} ({c.name}):** {c.remediation}")

        if self.kill_check.any_reject or self.kill_check.any_pause:
            lines.append("")
            lines.append("## Kill / Pause Flags")
            lines.append("")
            for d in self.kill_check.details:
                lines.append(f"- {d}")

        lines.append("")
        lines.append("---")
        lines.append(
            "*Decision per Phase Gate Checklist v2. "
            "Gates are binary: PASS or ITERATE. No overrides by discretion.*"
        )
        lines.append("")

        path.write_text("\n".join(lines))
        return path


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _find_oos_criterion(oos: OOSResult, criterion_id: str) -> Optional[OOSCriterion]:
    """Find a specific criterion from OOSResult by ID."""
    for c in oos.criteria:
        if c.criterion_id == criterion_id:
            return c
    return None


def _eval_item_from_oos(
    oos: Optional[OOSResult],
    item: int,
    name: str,
    criterion_id: str,
) -> GateCriterion:
    """Evaluate a gate item backed by an OOSResult criterion."""
    if oos is None:
        return GateCriterion(
            item=item, name=name, status="INCOMPLETE",
            reason="OOS evaluation not provided",
            remediation=_REMEDIATION.get(item, ""),
        )
    c = _find_oos_criterion(oos, criterion_id)
    if c is None:
        return GateCriterion(
            item=item, name=name, status="INCOMPLETE",
            reason=f"Criterion {criterion_id} not found in OOS result",
            remediation=_REMEDIATION.get(item, ""),
        )
    return GateCriterion(
        item=item,
        name=name,
        status="PASS" if c.passed else "FAIL",
        value=f"{c.value}" if c.value != 0.0 else None,
        reason=c.reason,
        remediation=_REMEDIATION.get(item, "") if not c.passed else "",
    )


def _eval_bool_item(
    item: int,
    name: str,
    value: Optional[bool],
    source_name: str,
) -> GateCriterion:
    """Evaluate a gate item backed by a simple boolean."""
    if value is None:
        return GateCriterion(
            item=item, name=name, status="INCOMPLETE",
            reason=f"{source_name} not provided",
            remediation=_REMEDIATION.get(item, ""),
        )
    return GateCriterion(
        item=item,
        name=name,
        status="PASS" if value else "FAIL",
        value=str(value),
        reason=f"{source_name}: {'passed' if value else 'failed'}",
        remediation=_REMEDIATION.get(item, "") if not value else "",
    )


def _build_kill_check(oos: Optional[OOSResult]) -> KillCheckResult:
    """Extract kill/pause flags from OOS criteria."""
    kc = KillCheckResult()
    if oos is None:
        return kc

    k1 = _find_oos_criterion(oos, "K.1")
    if k1 is not None and not k1.passed:
        kc.top_tercile_kill = True
        kc.details.append(f"REJECT — {k1.reason}")

    k2 = _find_oos_criterion(oos, "K.2")
    if k2 is not None and not k2.passed:
        kc.correlation_kill = True
        kc.details.append(f"REJECT — {k2.reason}")

    g3 = _find_oos_criterion(oos, "G1.3")
    if g3 is not None and not g3.passed:
        kc.monotonicity_pause = True
        kc.details.append(f"PAUSE — {g3.reason}")

    return kc


def _resolve_verdict(
    criteria: List[GateCriterion],
    kill_check: KillCheckResult,
) -> Verdict:
    """Determine overall verdict from criteria and kill checks.

    Severity: REJECT > PAUSE > ITERATE > INCOMPLETE > PASS.
    """
    if kill_check.any_reject:
        return "REJECT"
    if kill_check.any_pause:
        return "PAUSE"
    if any(c.status == "FAIL" for c in criteria):
        return "ITERATE"
    if any(c.status == "INCOMPLETE" for c in criteria):
        return "INCOMPLETE"
    return "PASS"


def evaluate_gate1(inp: GateInput) -> GateDecision:
    """Evaluate all 9 Gate 1 criteria and produce the decision artifact.

    Parameters
    ----------
    inp : GateInput
        Upstream results and metadata.

    Returns
    -------
    GateDecision
        Complete decision with verdict, criteria table, and export methods.
    """
    criteria: List[GateCriterion] = []

    # Item 1: OOS trade count >= 30.
    criteria.append(_eval_item_from_oos(inp.oos_result, 1, "oos_trade_count", "G1.1"))

    # Item 2: OOS tercile spread >= 1.0R.
    criteria.append(_eval_item_from_oos(inp.oos_result, 2, "oos_tercile_spread", "G1.2"))

    # Item 3: Score monotonicity (top > mid > bottom).
    criteria.append(_eval_item_from_oos(inp.oos_result, 3, "score_monotonicity", "G1.3"))

    # Item 4: Feature cap respected.
    criteria.append(_eval_bool_item(
        4, "feature_cap_respected",
        inp.feature_cap_respected, "Feature cap attestation",
    ))

    # Item 5: Model card complete.
    criteria.append(_eval_bool_item(
        5, "model_card_complete",
        inp.model_card_complete, "Model card attestation",
    ))

    # Item 6: Negative controls passed.
    nc_passed = inp.nc_report.all_passed if inp.nc_report is not None else None
    criteria.append(_eval_bool_item(
        6, "negative_controls_passed", nc_passed, "Negative controls",
    ))

    # Item 7: Entry degradation within tolerances.
    deg_passed = inp.degradation_report.all_passed if inp.degradation_report is not None else None
    criteria.append(_eval_bool_item(
        7, "entry_degradation_pass", deg_passed, "Entry degradation",
    ))

    # Item 8: Slippage stress profitable at 2 ticks.
    slip_passed = inp.stress_report.gate_pass if inp.stress_report is not None else None
    criteria.append(_eval_bool_item(
        8, "slippage_stress_pass", slip_passed, "Slippage stress",
    ))

    # Item 9: Quintile calibration.
    criteria.append(_eval_item_from_oos(inp.oos_result, 9, "quintile_calibration", "G1.9"))

    # Kill / pause checks.
    kill_check = _build_kill_check(inp.oos_result)

    # Overall verdict.
    verdict = _resolve_verdict(criteria, kill_check)

    return GateDecision(
        verdict=verdict,
        criteria=criteria,
        kill_check=kill_check,
        timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_hash=inp.dataset_hash,
        config_hash=inp.config_hash,
        code_commit_hash=inp.code_commit_hash,
    )
