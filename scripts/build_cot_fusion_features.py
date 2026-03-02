#!/usr/bin/env python3
"""Build trigger-aligned COT fusion features from legacy/disagg/tff overlays.

Outputs a CSV keyed by (Date, Ticker) with strict-lag lookups per COT source
and derived fusion candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT / "src"))

from ctl.cot_loader import load_and_compute  # noqa: E402
from ctl.external_merge import _build_cot_lookup, lookup_cot  # noqa: E402


def _latest_dataset(dataset_dir: Path) -> Path:
    files = sorted(dataset_dir.glob("phase1a_triggers_v1_full_universe_real*.csv"))
    if not files:
        raise FileNotFoundError(f"No strict dataset found under: {dataset_dir}")
    return files[-1]


def _safe_sign(x: float) -> float:
    if np.isnan(x):
        return np.nan
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0


def _row_fusion_features(row: pd.Series) -> Dict[str, float]:
    deltas = np.array(
        [row["legacy_delta"], row["disagg_delta"], row["tff_delta"]], dtype=float
    )
    zscores = np.array(
        [row["legacy_z"], row["disagg_z"], row["tff_z"]], dtype=float
    )

    valid_d = deltas[~np.isnan(deltas)]
    valid_z = zscores[~np.isnan(zscores)]
    source_count = int(len(valid_d))

    out: Dict[str, float] = {
        "cot_source_count": float(source_count),
        "cot_mean_delta": float(np.nanmean(deltas)) if source_count else np.nan,
        "cot_mean_z": float(np.nanmean(zscores)) if len(valid_z) else np.nan,
        "cot_median_delta": float(np.nanmedian(deltas)) if source_count else np.nan,
        "cot_median_z": float(np.nanmedian(zscores)) if len(valid_z) else np.nan,
        "cot_dispersion_delta": float(np.nanstd(deltas)) if source_count else np.nan,
        "cot_dispersion_z": float(np.nanstd(zscores)) if len(valid_z) else np.nan,
    }

    signs = np.array([_safe_sign(v) for v in deltas], dtype=float)
    signs = signs[~np.isnan(signs)]
    nonzero = signs[signs != 0]
    if len(nonzero) >= 2:
        agree = float(abs(nonzero.sum()) == len(nonzero))
        disagreement_frac = float(1.0 - (abs(nonzero.sum()) / len(nonzero)))
    else:
        agree = np.nan
        disagreement_frac = np.nan
    out["cot_agree_sign"] = agree
    out["cot_disagreement_frac"] = disagreement_frac

    # Fusion option A: all-source mean.
    out["fusion_mean_delta"] = out["cot_mean_delta"]
    out["fusion_mean_z"] = out["cot_mean_z"]

    # Fusion option B: robust median.
    out["fusion_median_delta"] = out["cot_median_delta"]
    out["fusion_median_z"] = out["cot_median_z"]

    # Fusion option C: legacy shrunk when sources disagree.
    legacy_delta = row["legacy_delta"]
    legacy_z = row["legacy_z"]
    if np.isnan(disagreement_frac):
        out["fusion_shrink_delta"] = legacy_delta
        out["fusion_shrink_z"] = legacy_z
    else:
        shrink = max(0.0, 1.0 - disagreement_frac)
        out["fusion_shrink_delta"] = legacy_delta * shrink
        out["fusion_shrink_z"] = legacy_z * shrink

    # Fusion option D: consensus-only mean, fallback to legacy when low agreement.
    if source_count >= 2 and (np.isnan(agree) or agree >= 0.5):
        out["fusion_consensus_delta"] = out["cot_mean_delta"]
        out["fusion_consensus_z"] = out["cot_mean_z"]
    else:
        out["fusion_consensus_delta"] = legacy_delta
        out["fusion_consensus_z"] = legacy_z

    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build COT fusion features aligned to strict dataset rows.")
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Strict dataset path. Defaults to latest under data/processed/cutover_v1/datasets.",
    )
    p.add_argument(
        "--legacy-cot",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "external" / "cot_phase1a_legacy.csv",
    )
    p.add_argument(
        "--disagg-cot",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "external" / "cot_phase1a_disagg.csv",
    )
    p.add_argument(
        "--tff-cot",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "external" / "cot_phase1a_tff.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_fusion_features_latest.csv",
    )
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    dataset_path = args.dataset or _latest_dataset(
        REPO_ROOT / "data" / "processed" / "cutover_v1" / "datasets"
    )
    df = pd.read_csv(dataset_path)
    if "Date" not in df.columns or "Ticker" not in df.columns:
        raise ValueError("Dataset must contain Date and Ticker columns")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    legacy_lookup = _build_cot_lookup(load_and_compute(args.legacy_cot))
    disagg_lookup = _build_cot_lookup(load_and_compute(args.disagg_cot))
    tff_lookup = _build_cot_lookup(load_and_compute(args.tff_cot))

    rows = []
    for _, r in df.iterrows():
        sym = str(r["Ticker"])
        date = r["Date"]
        # COT applies to futures roots only.
        if not sym.startswith("/") or pd.isna(date):
            base = {
                "Date": r["Date"],
                "Ticker": sym,
                "legacy_delta": np.nan,
                "legacy_z": np.nan,
                "disagg_delta": np.nan,
                "disagg_z": np.nan,
                "tff_delta": np.nan,
                "tff_z": np.nan,
            }
        else:
            ld, lz = lookup_cot(sym, date, legacy_lookup)
            dd, dz = lookup_cot(sym, date, disagg_lookup)
            td, tz = lookup_cot(sym, date, tff_lookup)
            base = {
                "Date": r["Date"],
                "Ticker": sym,
                "legacy_delta": np.nan if ld is None else float(ld),
                "legacy_z": np.nan if lz is None else float(lz),
                "disagg_delta": np.nan if dd is None else float(dd),
                "disagg_z": np.nan if dz is None else float(dz),
                "tff_delta": np.nan if td is None else float(td),
                "tff_z": np.nan if tz is None else float(tz),
            }
        base.update(_row_fusion_features(pd.Series(base)))
        rows.append(base)

    out = pd.DataFrame(rows)
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    payload = {
        "dataset_path": str(dataset_path),
        "out_path": str(args.out),
        "rows": int(len(out)),
        "futures_rows": int(out["Ticker"].astype(str).str.startswith("/").sum()),
        "legacy_non_null": int(out["legacy_delta"].notna().sum()),
        "disagg_non_null": int(out["disagg_delta"].notna().sum()),
        "tff_non_null": int(out["tff_delta"].notna().sum()),
    }
    if args.json_output:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
