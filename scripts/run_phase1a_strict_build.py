#!/usr/bin/env python3
"""One-command strict Phase 1a dataset build.

This wrapper orchestrates strict Task 7/8 flow:
1) Ensure external data inputs exist (auto-fetch VIX from yfinance if missing).
2) Run full-universe immutable dataset build.
3) Fail hard on partial coverage unless explicitly allowed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _fetch_vix_csv(path: Path, start: str = "2015-01-01") -> None:
    import yfinance as yf  # local import to keep optional dependency behavior

    data = yf.download("^VIX", start=start, auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise RuntimeError("yfinance returned no ^VIX data")

    def _norm_col(col) -> str:
        if isinstance(col, tuple):
            return " ".join(str(x).strip().lower() for x in col if x is not None)
        return str(col).strip().lower()

    cols = {_norm_col(c): c for c in data.columns}
    close_col = cols.get("close")
    if close_col is None:
        # yfinance may return MultiIndex columns like ("Close", "^VIX")
        # Normalize by selecting the first column whose normalized name
        # begins with "close".
        for norm, original in cols.items():
            if norm.startswith("close"):
                close_col = original
                break
    if close_col is None:
        raise RuntimeError("Downloaded ^VIX data missing Close column")

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(data.index).tz_localize(None),
            "vix_close": pd.to_numeric(data[close_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["date", "vix_close"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Run strict Phase 1a one-command build")
    p.add_argument("--version", type=str, default="v1_full_universe_real")
    p.add_argument(
        "--cot-csv",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "external" / "cot_phase1a.csv",
        help="Preprocessed COT CSV (publication_date,symbol,commercial_net).",
    )
    p.add_argument(
        "--vix-csv",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "external" / "vix_phase1a.csv",
        help="VIX CSV path. Auto-fetched from yfinance if missing.",
    )
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow build to succeed even if some symbols are missing coverage.",
    )
    args = p.parse_args()

    # External data preflight.
    if not args.vix_csv.exists():
        print(f"[INFO] VIX CSV missing; fetching to {args.vix_csv}")
        _fetch_vix_csv(args.vix_csv)

    if not args.cot_csv.exists():
        raise SystemExit(
            f"[ERROR] Missing COT CSV: {args.cot_csv}\n"
            "Provide preprocessed COT data with columns: publication_date,symbol,commercial_net."
        )

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / "scripts" / "build_phase1a_dataset.py"),
        "--timeframe",
        "daily",
        "--version",
        args.version,
        "--cot-csv",
        str(args.cot_csv),
        "--vix-csv",
        str(args.vix_csv),
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    # Preserve logs on stderr for visibility.
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")

    if proc.returncode != 0:
        print(proc.stdout, end="")
        raise SystemExit(proc.returncode)

    payload = json.loads(proc.stdout)
    warnings = payload.get("warnings", [])
    if warnings and not args.allow_partial:
        print(json.dumps(payload, indent=2))
        raise SystemExit(
            "[ERROR] Strict build detected partial coverage warnings. "
            "Fix missing symbols, then rerun (or pass --allow-partial)."
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
