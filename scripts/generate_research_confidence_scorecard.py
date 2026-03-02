#!/usr/bin/env python3
"""Generate research-tier confidence scorecard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.research_registry import DEFAULT_RESEARCH_REGISTRY, load_research_registry  # noqa: E402
from ctl.research_scorecard import (  # noqa: E402
    build_research_scorecard,
    load_latest_research_batch,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Build research confidence scorecard from latest batch run")
    p.add_argument("--registry", type=Path, default=DEFAULT_RESEARCH_REGISTRY)
    p.add_argument("--batch-summary", type=Path, default=None)
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    registry = load_research_registry(args.registry)
    if args.batch_summary is not None:
        with open(args.batch_summary) as f:
            batch = json.load(f)
    else:
        batch = load_latest_research_batch()
    if not batch:
        raise SystemExit("No research batch summary found")

    rows = build_research_scorecard(registry, batch)
    rows = sorted(rows, key=lambda r: r.confidence_score, reverse=True)
    payload = {
        "registry_id": registry.registry_id,
        "cycle_id": registry.cycle_id,
        "rows": [r.to_dict() for r in rows],
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("Research Confidence Scorecard")
    print("Symbol  RunStatus  Conf  Score   DiagAvail  DiagStatus   Decision  Trades  TotalR")
    print("------  ---------  ----  -----   ---------  ----------   --------  ------  ------")
    for r in rows:
        print(
            f"{r.symbol:<6}  {r.run_status:<9}  {r.confidence_band:<4}  {r.confidence_score:<6.4f}  "
            f"{str(r.diagnostics_available):<9}  {r.diagnostics_status:<10}  {str(r.decision or '-'):<8}  "
            f"{str(r.trade_count if r.trade_count is not None else '-'):>6}  "
            f"{str(r.total_r if r.total_r is not None else '-'):>6}"
        )


if __name__ == "__main__":
    main()
