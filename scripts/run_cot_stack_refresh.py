#!/usr/bin/env python3
"""One-command COT stack refresh.

Runs:
1) COT source overlays
2) COT source benchmark
3) COT fusion feature build
4) COT fusion ablation
5) COT regime report
6) Secondary drift guardrail update
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"

ANALYSIS_DIR = REPO_ROOT / "data" / "processed" / "cutover_v1" / "analysis"
TRACKING_PATH = REPO_ROOT / "configs" / "cutover" / "cot_tracking_v1.yaml"
HISTORY_PATH = ANALYSIS_DIR / "cot_tracking_history.jsonl"


def _run_json(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return json.loads(proc.stdout)


def _append_history(record: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _apply_guardrails(tracking_path: Path, current: dict[str, Any]) -> dict[str, Any]:
    tracking = yaml.safe_load(tracking_path.read_text(encoding="utf-8")) or {}
    secondary = current.get("secondary_variant")
    disabled = False
    disable_reason = None

    hist = _load_history()
    if secondary:
        recent = [h for h in hist if h.get("secondary_variant") == secondary][-2:]
        if len(recent) == 2:
            spread_bad = all(
                (r.get("lift_spread_vs_baseline") is not None and r["lift_spread_vs_baseline"] < 0)
                for r in recent
            )
            corr_bad = all(
                (r.get("lift_corr_vs_baseline") is not None and r["lift_corr_vs_baseline"] < -0.02)
                for r in recent
            )
            if spread_bad or corr_bad:
                disabled = True
                if spread_bad and corr_bad:
                    disable_reason = "Guardrail breach: 2 consecutive negative spread and corr<-0.02"
                elif spread_bad:
                    disable_reason = "Guardrail breach: 2 consecutive negative spread lift"
                else:
                    disable_reason = "Guardrail breach: 2 consecutive corr lift below -0.02"

    tracking["secondary_variant"] = secondary
    tracking["secondary_enabled"] = bool(secondary is not None and not disabled)
    tracking["secondary_disable_reason"] = disable_reason
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_path.write_text(yaml.safe_dump(tracking, sort_keys=False), encoding="utf-8")

    return {
        "secondary_variant": secondary,
        "secondary_enabled": tracking["secondary_enabled"],
        "secondary_disable_reason": disable_reason,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Refresh COT stack and update tracking guardrails.")
    p.add_argument(
        "--sources",
        default="legacy,disagg,tff",
        help="Comma-separated COT source overlays to include.",
    )
    p.add_argument("--json", action="store_true", dest="json_output")
    args = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    overlays = _run_json(
        [
            str(PYTHON_BIN),
            "scripts/build_cot_source_overlays.py",
            "--sources",
            args.sources,
            "--json",
        ]
    )
    benchmark = _run_json(
        [
            str(PYTHON_BIN),
            "scripts/benchmark_cot_sources.py",
            "--sources",
            args.sources,
            "--json",
        ]
    )
    dataset_path = benchmark["ranked"][0]["dataset_path"]
    fusion = _run_json(
        [
            str(PYTHON_BIN),
            "scripts/build_cot_fusion_features.py",
            "--dataset",
            dataset_path,
            "--json",
        ]
    )
    ablation = _run_json(
        [
            str(PYTHON_BIN),
            "scripts/evaluate_cot_fusion_ablation.py",
            "--dataset",
            dataset_path,
            "--json",
        ]
    )
    regimes = _run_json(
        [
            str(PYTHON_BIN),
            "scripts/report_cot_regime_effects.py",
            "--dataset",
            dataset_path,
            "--json",
        ]
    )
    state_features = _run_json(
        [
            str(PYTHON_BIN),
            "scripts/build_cot_state_features.py",
            "--json",
        ]
    )
    state_ablation = _run_json(
        [
            str(PYTHON_BIN),
            "scripts/evaluate_cot_state_ablation.py",
            "--dataset",
            dataset_path,
            "--json",
        ]
    )

    secondary_variant = ablation.get("secondary_variant")
    sec_lifts = {}
    for row in ablation.get("variants", []):
        if row.get("variant") == secondary_variant:
            sec_lifts = {
                "lift_corr_vs_baseline": row.get("lift_corr_vs_baseline"),
                "lift_spread_vs_baseline": row.get("lift_spread_vs_baseline"),
                "lift_top_r_vs_baseline": row.get("lift_top_r_vs_baseline"),
            }
            break

    history_row = {
        "timestamp_utc": ts,
        "primary_variant": ablation.get("primary_variant"),
        "secondary_variant": secondary_variant,
        **sec_lifts,
    }
    _append_history(history_row)
    guardrail = _apply_guardrails(TRACKING_PATH, history_row)

    payload = {
        "timestamp_utc": ts,
        "dataset_path": dataset_path,
        "overlays": overlays.get("sources", []),
        "benchmark_recommended_source": benchmark.get("recommended_source"),
        "ablation_primary_variant": ablation.get("primary_variant"),
        "ablation_secondary_variant": secondary_variant,
        "secondary_lifts": sec_lifts,
        "guardrail": guardrail,
        "artifacts": {
            "fusion_features": fusion.get("out_path"),
            "ablation": str(ANALYSIS_DIR / "cot_fusion_ablation_latest.json"),
            "regimes": str(ANALYSIS_DIR / "cot_fusion_regime_effects_latest.json"),
            "state_features": state_features.get("out_path"),
            "state_ablation": str(ANALYSIS_DIR / "cot_state_ablation_latest.json"),
            "history": str(HISTORY_PATH),
            "tracking": str(TRACKING_PATH),
        },
        "state_ablation_rows": state_ablation.get("rows", []),
    }

    out_path = ANALYSIS_DIR / f"{ts}_cot_stack_refresh.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["summary_path"] = str(out_path)

    if args.json_output:
        print(json.dumps(payload, indent=2))
        return
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
