#!/usr/bin/env python3
"""Evaluate offline PL regime-aware basis correction candidate."""

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
from ctl.roll_reconciliation import load_roll_manifest  # noqa: E402

DEFAULT_PROFILE = REPO_ROOT / "configs" / "cutover" / "operating_profile_v1.yaml"
DEFAULT_DB_DIR = REPO_ROOT / "data" / "processed" / "databento" / "cutover_v1" / "continuous"
DEFAULT_TS_DIR = REPO_ROOT / "data" / "raw" / "tradestation" / "cutover_v1"


def _diag(can_df, ts_adj_df, ts_unadj_df, manifest, tick_size, mdd):
    return run_diagnostics(
        canonical_adj_df=can_df,
        ts_adj_df=ts_adj_df,
        manifest_entries=manifest,
        ts_unadj_df=ts_unadj_df,
        symbol="PL",
        tick_size=tick_size,
        max_day_delta=mdd,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Offline PL regime correction evaluator")
    p.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
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

    regimes = [
        ("pre_2020", "2018-01-01", "2019-12-31"),
        ("post_2024", "2024-01-01", "2026-02-17"),
    ]
    offsets = derive_regime_offsets(can_df, ts_adj, regimes)
    can_corr = apply_regime_offsets(can_df, offsets)

    base = _diag(can_df, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)
    corr = _diag(can_corr, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)

    base_acc = acceptance_from_diagnostics(base)
    corr_acc = acceptance_from_diagnostics(corr)

    payload = {
        "symbol": "PL",
        "regime_offsets": [o.to_dict() for o in offsets],
        "baseline": {
            "strict": base.strict_status,
            "policy": base.policy_status,
            "decision": base_acc.decision,
            "mean_gap_diff": base.l3.mean_gap_diff,
            "mean_drift": base.l4.mean_drift,
            "reasons": base_acc.reasons,
        },
        "corrected": {
            "strict": corr.strict_status,
            "policy": corr.policy_status,
            "decision": corr_acc.decision,
            "mean_gap_diff": corr.l3.mean_gap_diff,
            "mean_drift": corr.l4.mean_drift,
            "reasons": corr_acc.reasons,
        },
        "delta": {
            "mean_gap_diff": corr.l3.mean_gap_diff - base.l3.mean_gap_diff,
            "mean_drift": corr.l4.mean_drift - base.l4.mean_drift,
        },
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("PL Regime-Aware Basis Correction (Offline)")
    print("Offsets:")
    for o in offsets:
        print(f"- {o.label}: {o.start}..{o.end} median_signed_diff={o.median_signed_diff:.4f}")

    print("\nBaseline:")
    print(f"  decision={payload['baseline']['decision']} strict/policy={payload['baseline']['strict']}/{payload['baseline']['policy']}")
    print(f"  mean_gap={payload['baseline']['mean_gap_diff']:.4f} mean_drift={payload['baseline']['mean_drift']:.4f}")

    print("\nCorrected (offline diagnostic):")
    print(f"  decision={payload['corrected']['decision']} strict/policy={payload['corrected']['strict']}/{payload['corrected']['policy']}")
    print(f"  mean_gap={payload['corrected']['mean_gap_diff']:.4f} mean_drift={payload['corrected']['mean_drift']:.4f}")

    print("\nDelta (corrected - baseline):")
    print(f"  mean_gap={payload['delta']['mean_gap_diff']:.4f}")
    print(f"  mean_drift={payload['delta']['mean_drift']:.4f}")


if __name__ == "__main__":
    main()
