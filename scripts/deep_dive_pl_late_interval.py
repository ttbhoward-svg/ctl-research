#!/usr/bin/env python3
"""Deep-dive PL late interval for session/close-type artifact checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.operating_profile import discover_ts_custom_file  # noqa: E402
from ctl.parity_prep import load_and_validate  # noqa: E402
from ctl.pl_late_interval import compute_align_stats, date_overlap_breakdown  # noqa: E402


DEFAULT_DB = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous" / "PL_continuous.csv"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out[(out["Date"] >= start) & (out["Date"] <= end)]


def main() -> None:
    p = argparse.ArgumentParser(description="PL late-interval basis deep-dive")
    p.add_argument("--start", default="2025-12-28")
    p.add_argument("--end", default="2026-02-17")
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    can_df, e0 = load_and_validate(DEFAULT_DB, "DB PL")
    ts_adj_path = discover_ts_custom_file("PL", DEFAULT_TS_DIR, "ADJ")
    ts_unadj_path = discover_ts_custom_file("PL", DEFAULT_TS_DIR, "UNADJ")
    adj_df, e1 = load_and_validate(ts_adj_path, "TS PL ADJ")
    unadj_df, e2 = load_and_validate(ts_unadj_path, "TS PL UNADJ")
    if e0 or e1 or e2:
        raise SystemExit("; ".join(e0 + e1 + e2))

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    can_i = _slice(can_df, start, end)
    adj_i = _slice(adj_df, start, end)
    un_i = _slice(unadj_df, start, end)

    same = compute_align_stats(can_i, adj_i, 0)
    shift_m1 = compute_align_stats(can_i, adj_i, -1)
    shift_p1 = compute_align_stats(can_i, adj_i, 1)
    can_vs_un = compute_align_stats(can_i, un_i, 0)

    overlap = date_overlap_breakdown(can_i, adj_i)

    result = {
        "interval": {"start": args.start, "end": args.end},
        "rows": {"canonical": len(can_i), "ts_adj": len(adj_i), "ts_unadj": len(un_i)},
        "same_day": same.to_dict(),
        "shift_minus_1_day": shift_m1.to_dict(),
        "shift_plus_1_day": shift_p1.to_dict(),
        "canonical_vs_unadjusted": can_vs_un.to_dict(),
        "date_overlap": {
            "overlap_count": len(overlap["overlap"]),
            "can_only_count": len(overlap["can_only"]),
            "adj_only_count": len(overlap["ref_only"]),
            "can_only_dates": overlap["can_only"],
        },
    }

    if args.json_output:
        print(json.dumps(result, indent=2))
        return

    print("PL Late Interval Deep-Dive")
    print(f"Interval: {args.start} -> {args.end}")
    print(f"Rows (can/adj/unadj): {len(can_i)}/{len(adj_i)}/{len(un_i)}")
    print("\nAlignment Stats (canonical - TS ADJ):")
    print(f"same-day    rows={same.rows} median={same.median_signed_diff:.4f} mean_abs={same.mean_abs_diff:.4f} p95_abs={same.p95_abs_diff:.4f}")
    print(f"shift -1d   rows={shift_m1.rows} median={shift_m1.median_signed_diff:.4f} mean_abs={shift_m1.mean_abs_diff:.4f} p95_abs={shift_m1.p95_abs_diff:.4f}")
    print(f"shift +1d   rows={shift_p1.rows} median={shift_p1.median_signed_diff:.4f} mean_abs={shift_p1.mean_abs_diff:.4f} p95_abs={shift_p1.p95_abs_diff:.4f}")
    print(f"can-unadj   rows={can_vs_un.rows} median={can_vs_un.median_signed_diff:.4f} mean_abs={can_vs_un.mean_abs_diff:.4f} p95_abs={can_vs_un.p95_abs_diff:.4f}")
    print(f"\nDate overlap: overlap={len(overlap['overlap'])} can_only={len(overlap['can_only'])} adj_only={len(overlap['ref_only'])}")
    if overlap["can_only"]:
        print("can_only dates:", ", ".join(overlap["can_only"]))


if __name__ == "__main__":
    main()
