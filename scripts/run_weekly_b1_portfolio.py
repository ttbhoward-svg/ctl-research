#!/usr/bin/env python3
"""Gate-first weekly B1 portfolio runner (H.8/H.9).

Enforces the operating-profile gate check before any strategy execution.
If the gate fails (acceptance mismatch), the run aborts with exit code 2.
If the gate passes, the runner executes B1 detection + simulation for all
gating symbols (and optionally non-gating symbols like PA).

Usage
-----
    python scripts/run_weekly_b1_portfolio.py
    python scripts/run_weekly_b1_portfolio.py --dry-run
    python scripts/run_weekly_b1_portfolio.py --include-non-gating --json
    python scripts/run_weekly_b1_portfolio.py --profile configs/cutover/operating_profile_v1.yaml

Exit codes: 0 = gate passed + run completed, 2 = gate mismatch.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.operating_profile import load_operating_profile  # noqa: E402
from ctl.run_orchestrator import (  # noqa: E402
    build_run_plan,
    execute_run_plan,
    make_b1_executor,
    run_profile_gate,
    save_run_summary,
    summarize_run,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_weekly_b1_portfolio")

DEFAULT_PROFILE = REPO_ROOT / "configs" / "cutover" / "operating_profile_v1.yaml"


def _format_gate_text(gate_result) -> str:
    """Format gate result as a human-readable table."""
    lines = [
        f"Gate: {'PASS' if gate_result.passed else 'MISMATCH'}",
        f"Recommendation: {gate_result.recommendation}",
    ]
    for r in gate_result.symbol_results:
        status = "OK" if r.passed else "FAIL"
        lines.append(f"  {r.symbol:<6} {r.expected:<10} -> {r.actual:<10} [{status}] {r.detail}")
    return "\n".join(lines)


def _format_run_text(summary) -> str:
    """Format run summary as human-readable output."""
    lines = [
        f"Portfolio Run Summary",
        f"  Timestamp: {summary.timestamp}",
        f"  Cycle:     {summary.cycle_id}",
        f"  Gate:      {'PASS' if summary.gate_passed else 'FAIL'}",
        f"  Dry run:   {summary.dry_run}",
        f"  Symbols:   {', '.join(r['symbol'] for r in summary.symbol_run_results)}",
        "",
        f"{'Symbol':<8} {'Status':<12} Detail",
        f"{'------':<8} {'------':<12} ------",
    ]
    for r in summary.symbol_run_results:
        lines.append(f"{r['symbol']:<8} {r['status']:<12} {r['detail']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate-first weekly B1 portfolio runner (H.8).",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
        help="Path to operating profile YAML.",
    )
    parser.add_argument(
        "--include-non-gating",
        action="store_true",
        default=False,
        help="Include non-gating symbols (e.g. PA) in the run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Gate + plan only, no strategy execution.",
    )
    args = parser.parse_args()

    # --- Step 1: Gate check ---
    logger.info("Running operating profile gate check...")
    gate_result = run_profile_gate(args.profile)

    if not gate_result.passed:
        if args.json_output:
            print(json.dumps({
                "gate_passed": False,
                "gate_result": gate_result.to_dict(),
                "aborted": True,
                "reason": "Gate mismatch — acceptance status does not match locked profile.",
            }, indent=2))
        else:
            print("ABORTED: Gate mismatch — run cannot proceed.\n")
            print(_format_gate_text(gate_result))
        sys.exit(2)

    logger.info("Gate passed. Building run plan...")

    # --- Step 2: Build plan ---
    profile = load_operating_profile(args.profile)
    plan = build_run_plan(
        profile,
        include_non_gating=args.include_non_gating,
        profile_path=str(args.profile),
    )

    # --- Step 3: Execute ---
    executor = make_b1_executor() if not args.dry_run else None
    symbol_results = execute_run_plan(plan, executor=executor, dry_run=args.dry_run)

    # --- Step 4: Summarize ---
    summary = summarize_run(plan, gate_result, symbol_results, dry_run=args.dry_run)

    # --- Step 5: Save (skip on dry-run) ---
    if not args.dry_run:
        save_run_summary(summary)

    # --- Output ---
    if args.json_output:
        print(json.dumps(summary.to_dict(), indent=2))
    else:
        print(_format_gate_text(gate_result))
        print()
        print(_format_run_text(summary))

    sys.exit(0)


if __name__ == "__main__":
    main()
