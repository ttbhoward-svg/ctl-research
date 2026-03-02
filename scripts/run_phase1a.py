#!/usr/bin/env python3
"""Phase 1a legacy entrypoint.

This script is intentionally non-operational to avoid accidental drift into an
outdated execution path. Use the cutover/governed runners listed below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Legacy Phase 1a runner (deprecated)")
    p.add_argument(
        "--ack-deprecated",
        action="store_true",
        help="Acknowledge deprecation and print current canonical entrypoints.",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    tracker = root / "docs" / "governance" / "CTL_Phase1a_Project_Tracker_v3.md"

    lines = [
        "run_phase1a.py is deprecated and intentionally disabled.",
        "",
        "Canonical references:",
        f"- Tracker: {tracker}",
        "",
        "Use these execution entrypoints instead:",
        "- .venv/bin/python scripts/check_operating_profile.py",
        "- .venv/bin/python scripts/run_weekly_b1_portfolio.py --dry-run",
        "- .venv/bin/python scripts/run_weekly_ops.py --dry-run --notify stdout",
    ]

    for line in lines:
        print(line)

    if args.ack_deprecated:
        return
    raise SystemExit(2)


if __name__ == "__main__":
    main()
