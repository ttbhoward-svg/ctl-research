#!/usr/bin/env python3
"""Run PL signed-basis regime split analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.operating_profile import discover_ts_custom_file  # noqa: E402
from ctl.parity_prep import load_and_validate  # noqa: E402
from ctl.pl_basis_regime import split_regime_stats  # noqa: E402

DEFAULT_DB = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous" / "PL_continuous.csv"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"


def main() -> None:
    p = argparse.ArgumentParser(description="PL basis regime split analyzer")
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    can_df, e0 = load_and_validate(DEFAULT_DB, "DB PL")
    ts_adj_path = discover_ts_custom_file("PL", DEFAULT_TS_DIR, "ADJ")
    ts_df, e1 = load_and_validate(ts_adj_path, "TS PL ADJ")
    if e0 or e1:
        raise SystemExit("; ".join(e0 + e1))

    # Fixed splits for current investigation.
    splits = [
        ("pre_2020", "2018-01-01", "2019-12-31"),
        ("post_2024", "2024-01-01", "2026-02-17"),
    ]
    stats = split_regime_stats(can_df, ts_df, splits)

    payload = {
        "symbol": "PL",
        "splits": [s.to_dict() for s in stats],
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("PL Basis Regime Split")
    print("Label      Rows  MedianSigned  MeanSigned  MeanAbs  P95Abs  PctCanAbove")
    print("-----      ----  ------------  ----------  -------  ------  -----------")
    for s in stats:
        print(
            f"{s.label:<10} {s.n_rows:<4}  {s.median_signed_diff:>12.4f}"
            f"  {s.mean_signed_diff:>10.4f}  {s.mean_abs_diff:>7.4f}"
            f"  {s.p95_abs_diff:>6.4f}  {s.pct_can_above_ts:>11.4f}"
        )

    if len(stats) == 2:
        delta = stats[1].median_signed_diff - stats[0].median_signed_diff
        print(f"\nMedian signed-diff shift (post_2024 - pre_2020): {delta:.4f}")


if __name__ == "__main__":
    main()
