#!/usr/bin/env python3
"""Provider connectivity check CLI (Data Cutover Task E).

Validates local provider configuration without making network calls:
  - Environment variables (DATABENTO_API_KEY, NORGATE_DATABASE_PATH)
  - Provider stub instantiation + metadata validation
  - Symbol map loading + validation
  - Sample .csv.zst schema smoke check

Usage
-----
    python scripts/check_provider_connectivity.py
    python scripts/check_provider_connectivity.py --verbose
    python scripts/check_provider_connectivity.py --json

Exit codes: 0 = all PASS/SKIP, 1 = any FAIL.
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

from ctl.provider_connectivity import run_connectivity_checks  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("check_provider_connectivity")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check provider connectivity and configuration.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed check output.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON.",
    )
    args = parser.parse_args()

    report = run_connectivity_checks()

    if args.json_output:
        print(json.dumps(report.summary_dict(), indent=2))
    else:
        print(report.summary_text())

    if args.verbose and not args.json_output:
        print()
        for c in report.checks:
            if c.status != "PASS":
                logger.info("%s: %s — %s", c.status, c.name, c.detail)

    sys.exit(1 if report.any_fail else 0)


if __name__ == "__main__":
    main()
