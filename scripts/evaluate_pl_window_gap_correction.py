#!/usr/bin/env python3
"""Evaluate offline PL top-window gap correction candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.canonical_acceptance import acceptance_from_diagnostics  # noqa: E402
from ctl.cutover_diagnostics import run_diagnostics  # noqa: E402
from ctl.operating_profile import discover_ts_custom_file, load_operating_profile  # noqa: E402
from ctl.parity_prep import load_and_validate  # noqa: E402
from ctl.pl_basis_correction import apply_regime_offsets, derive_regime_offsets  # noqa: E402
from ctl.pl_gap_window_correction import apply_window_biases, select_top_window_biases  # noqa: E402
from ctl.roll_reconciliation import load_roll_manifest  # noqa: E402

DEFAULT_PROFILE = REPO_ROOT / "configs" / "cutover" / "operating_profile_v1.yaml"
DEFAULT_DB_DIR = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"


def _diag(can_df, ts_adj_df, ts_unadj_df, manifest_entries, tick_size, mdd):
    return run_diagnostics(
        canonical_adj_df=can_df,
        ts_adj_df=ts_adj_df,
        manifest_entries=manifest_entries,
        ts_unadj_df=ts_unadj_df,
        symbol="PL",
        tick_size=tick_size,
        max_day_delta=mdd,
    )


def _snap(diag):
    acc = acceptance_from_diagnostics(diag)
    return {
        "strict": diag.strict_status,
        "policy": diag.policy_status,
        "decision": acc.decision,
        "mean_gap_diff": diag.l3.mean_gap_diff,
        "mean_drift": diag.l4.mean_drift,
        "reasons": acc.reasons,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Offline PL window-specific gap correction evaluator")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    profile = load_operating_profile(args.profile)
    cfg = profile.symbol_settings["PL"]

    can_df, e0 = load_and_validate(DEFAULT_DB_DIR / "PL_continuous.csv", "DB PL")
    ts_adj, e1 = load_and_validate(discover_ts_custom_file("PL", DEFAULT_TS_DIR, "ADJ"), "TS PL ADJ")
    ts_unadj, e2 = load_and_validate(discover_ts_custom_file("PL", DEFAULT_TS_DIR, "UNADJ"), "TS PL UNADJ")
    if e0 or e1 or e2:
        raise SystemExit("; ".join(e0 + e1 + e2))

    manifest = load_roll_manifest(DEFAULT_DB_DIR / "PL_roll_manifest.json")

    base = _diag(can_df, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)

    biases = select_top_window_biases(base.l2.detail_df, top_k=args.top_k)
    manifest_w = apply_window_biases(manifest, biases)
    win_only = _diag(can_df, ts_adj, ts_unadj, manifest_w, cfg.tick_size, cfg.max_day_delta)

    # Combine with H.23 drift-regime correction.
    regimes = [
        ("pre_2020", "2018-01-01", "2019-12-31"),
        ("post_2024", "2024-01-01", "2026-02-17"),
    ]
    offsets = derive_regime_offsets(can_df, ts_adj, regimes)
    can_regime = apply_regime_offsets(can_df, offsets)
    combined = _diag(can_regime, ts_adj, ts_unadj, manifest_w, cfg.tick_size, cfg.max_day_delta)

    payload = {
        "symbol": "PL",
        "top_k": args.top_k,
        "window_biases": [b.to_dict() for b in biases],
        "baseline": _snap(base),
        "window_gap_only": _snap(win_only),
        "combined": _snap(combined),
        "delta_window_gap_vs_baseline": {
            "mean_gap_diff": win_only.l3.mean_gap_diff - base.l3.mean_gap_diff,
            "mean_drift": win_only.l4.mean_drift - base.l4.mean_drift,
        },
        "delta_combined_vs_baseline": {
            "mean_gap_diff": combined.l3.mean_gap_diff - base.l3.mean_gap_diff,
            "mean_drift": combined.l4.mean_drift - base.l4.mean_drift,
        },
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print(f"PL Window Gap Correction (Offline, top_k={args.top_k})")
    print("Corrected windows:")
    for b in biases:
        print(f"- {b.roll_date} {b.from_contract}->{b.to_contract}: signed_gap_delta={b.signed_gap_delta:.4f}")

    for name in ["baseline", "window_gap_only", "combined"]:
        s = payload[name]
        print(f"\n{name}:")
        print(f"  decision={s['decision']} strict/policy={s['strict']}/{s['policy']} gap={s['mean_gap_diff']:.4f} drift={s['mean_drift']:.4f}")

    d1 = payload["delta_window_gap_vs_baseline"]
    d2 = payload["delta_combined_vs_baseline"]
    print("\nDelta window_gap_only - baseline:")
    print(f"  mean_gap={d1['mean_gap_diff']:.4f} mean_drift={d1['mean_drift']:.4f}")
    print("\nDelta combined - baseline:")
    print(f"  mean_gap={d2['mean_gap_diff']:.4f} mean_drift={d2['mean_drift']:.4f}")


if __name__ == "__main__":
    main()
