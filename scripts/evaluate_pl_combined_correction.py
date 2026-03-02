#!/usr/bin/env python3
"""Evaluate offline combined PL correction: regime-drift + gap-bias."""

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
from ctl.pl_gap_correction import apply_gap_bias, estimate_gap_bias  # noqa: E402
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


def _snapshot(diag):
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
    p = argparse.ArgumentParser(description="Offline combined PL correction evaluator")
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

    # Baseline.
    base = _diag(can_df, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)

    # Drift-regime correction (H.23).
    regimes = [
        ("pre_2020", "2018-01-01", "2019-12-31"),
        ("post_2024", "2024-01-01", "2026-02-17"),
    ]
    offsets = derive_regime_offsets(can_df, ts_adj, regimes)
    can_regime = apply_regime_offsets(can_df, offsets)
    drift_only = _diag(can_regime, ts_adj, ts_unadj, manifest, cfg.tick_size, cfg.max_day_delta)

    # Gap-bias from baseline matched/watch rows.
    gap_bias = estimate_gap_bias(base.l2.detail_df)
    manifest_gap = apply_gap_bias(manifest, gap_bias.median_signed_gap_delta)

    gap_only = _diag(can_df, ts_adj, ts_unadj, manifest_gap, cfg.tick_size, cfg.max_day_delta)
    combined = _diag(can_regime, ts_adj, ts_unadj, manifest_gap, cfg.tick_size, cfg.max_day_delta)

    payload = {
        "symbol": "PL",
        "regime_offsets": [o.to_dict() for o in offsets],
        "gap_bias": gap_bias.to_dict(),
        "baseline": _snapshot(base),
        "drift_only": _snapshot(drift_only),
        "gap_only": _snapshot(gap_only),
        "combined": _snapshot(combined),
        "delta_combined_vs_baseline": {
            "mean_gap_diff": combined.l3.mean_gap_diff - base.l3.mean_gap_diff,
            "mean_drift": combined.l4.mean_drift - base.l4.mean_drift,
        },
    }

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return

    print("PL Combined Offline Correction")
    print(f"Gap bias median signed delta: {gap_bias.median_signed_gap_delta:.4f} (n={gap_bias.n_rows})")
    print("\nBaseline:")
    b = payload["baseline"]
    print(f"  decision={b['decision']} strict/policy={b['strict']}/{b['policy']} gap={b['mean_gap_diff']:.4f} drift={b['mean_drift']:.4f}")
    print("\nDrift-only:")
    d = payload["drift_only"]
    print(f"  decision={d['decision']} strict/policy={d['strict']}/{d['policy']} gap={d['mean_gap_diff']:.4f} drift={d['mean_drift']:.4f}")
    print("\nGap-only:")
    g = payload["gap_only"]
    print(f"  decision={g['decision']} strict/policy={g['strict']}/{g['policy']} gap={g['mean_gap_diff']:.4f} drift={g['mean_drift']:.4f}")
    print("\nCombined:")
    c = payload["combined"]
    print(f"  decision={c['decision']} strict/policy={c['strict']}/{c['policy']} gap={c['mean_gap_diff']:.4f} drift={c['mean_drift']:.4f}")
    dd = payload["delta_combined_vs_baseline"]
    print("\nDelta (combined - baseline):")
    print(f"  mean_gap={dd['mean_gap_diff']:.4f} mean_drift={dd['mean_drift']:.4f}")


if __name__ == "__main__":
    main()
