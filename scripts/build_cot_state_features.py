#!/usr/bin/env python3
"""Build first-pass COT state-model features from fusion feature table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rolling_pctile_last(s: pd.Series) -> float:
    last = s.iloc[-1]
    return float((s <= last).mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Build COT state-model features.")
    p.add_argument(
        "--fusion-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_fusion_features_latest.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis" / "cot_state_features_latest.csv",
    )
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    df = pd.read_csv(args.fusion_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # State components derived per symbol over trigger timeline.
    g = df.groupby("Ticker", sort=False)
    df["state_level_pctile"] = g["fusion_mean_z"].transform(
        lambda s: s.rolling(window=52, min_periods=8).apply(_rolling_pctile_last, raw=False)
    )
    df["state_velocity"] = g["fusion_mean_delta"].diff(1)
    df["state_accel"] = g["state_velocity"].diff(1)
    df["state_crowding"] = df["fusion_mean_z"].abs()
    df["state_disagreement"] = df["cot_disagreement_frac"]
    df["state_dispersion_z"] = df["cot_dispersion_z"]

    # Composite state scores to replace legacy COT fields in ablation.
    # delta-score: direction + velocity damped by disagreement/dispersion.
    damp = 1.0 - df["state_disagreement"].fillna(0.0).clip(lower=0.0, upper=1.0)
    disp_penalty = 1.0 / (1.0 + df["state_dispersion_z"].abs().fillna(0.0))
    df["state_score_delta"] = (
        (0.65 * df["fusion_mean_delta"].fillna(0.0))
        + (0.35 * df["state_velocity"].fillna(0.0))
    ) * damp * disp_penalty
    df["state_score_z"] = (
        (0.60 * df["fusion_mean_z"].fillna(0.0))
        + (0.20 * df["state_level_pctile"].fillna(0.5))
        + (0.20 * (-df["state_crowding"].fillna(0.0)))
    ) * damp * disp_penalty

    cols = [
        "Date",
        "Ticker",
        "state_level_pctile",
        "state_velocity",
        "state_accel",
        "state_crowding",
        "state_disagreement",
        "state_dispersion_z",
        "state_score_delta",
        "state_score_z",
    ]
    out = df[cols].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    payload = {
        "fusion_csv": str(args.fusion_csv),
        "out_path": str(args.out),
        "rows": int(len(out)),
        "non_null_state_score_delta": int(out["state_score_delta"].notna().sum()),
        "non_null_state_score_z": int(out["state_score_z"].notna().sum()),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
