#!/usr/bin/env python3
"""Run research-tier batch backtests from a registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.research_batch import run_research_batch, save_research_batch_summary  # noqa: E402
from ctl.research_registry import DEFAULT_RESEARCH_REGISTRY, load_research_registry  # noqa: E402
from ctl.run_orchestrator import run_profile_gate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Run research-tier B1 backtests in batch")
    p.add_argument("--registry", type=Path, default=DEFAULT_RESEARCH_REGISTRY)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true", dest="json_output")
    p.add_argument(
        "--skip-gate-check",
        action="store_true",
        help="Skip operating-profile gate check before research batch.",
    )
    args = p.parse_args()

    registry = load_research_registry(args.registry)
    gate_result = None
    if not args.skip_gate_check:
        profile_path = (REPO_ROOT / registry.profile_path).resolve()
        gate_result = run_profile_gate(profile_path)
        if not gate_result.passed:
            payload = {
                "aborted": True,
                "reason": "operating-profile gate mismatch",
                "gate_result": gate_result.to_dict(),
            }
            if args.json_output:
                print(json.dumps(payload, indent=2))
            else:
                print("Research batch aborted: operating-profile gate mismatch")
                print(json.dumps(gate_result.to_dict(), indent=2))
            raise SystemExit(2)

    summary = run_research_batch(registry, dry_run=args.dry_run)
    out_path = None
    if not args.dry_run:
        out_path = save_research_batch_summary(summary)

    payload = {
        "aborted": False,
        "gate_checked": not args.skip_gate_check,
        "gate_passed": (gate_result.passed if gate_result is not None else None),
        "summary_path": str(out_path) if out_path else None,
        "summary": summary.to_dict(),
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print(f"Research batch complete: {len(summary.symbols)} symbols")
    for row in summary.symbol_results:
        print(f"- {row['symbol']}: {row['status']} ({row['detail']})")
    if out_path is not None:
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
