#!/usr/bin/env python3
"""Model card builder placeholder."""

import argparse
from datetime import datetime, timezone
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a placeholder model card.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "model_cards",
        help="Directory where model cards are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = out / f"model_card_{ts}.md"
    p.write_text("# Model Card\n\nTODO: populate run metadata and metrics.\n")
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
