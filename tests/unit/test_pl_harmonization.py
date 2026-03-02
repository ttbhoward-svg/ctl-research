"""Unit tests for PL harmonization integration helper."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.pl_harmonization import apply_pl_harmonization  # noqa: E402
from ctl.roll_reconciliation import RollManifestEntry  # noqa: E402


def _can_ts_frames():
    can = pd.DataFrame(
        {
            "Date": ["2019-06-01", "2019-06-02", "2025-09-29", "2025-12-28"],
            "Close": [100.0, 101.0, 130.0, 140.0],
        }
    )
    ts = pd.DataFrame(
        {
            "Date": ["2019-06-01", "2019-06-02", "2025-09-29", "2025-12-28"],
            "Close": [99.0, 100.0, 129.5, 139.5],
        }
    )
    return can, ts


def _manifest():
    return [
        RollManifestEntry(
            roll_date="2019-06-28",
            from_contract="PLN9",
            to_contract="PLV9",
            from_close=100.0,
            to_close=101.0,
            gap=1.0,
            cumulative_adj=1.0,
        ),
        RollManifestEntry(
            roll_date="2025-12-28",
            from_contract="PLF6",
            to_contract="PLJ6",
            from_close=140.0,
            to_close=141.0,
            gap=1.0,
            cumulative_adj=2.0,
        ),
    ]


def _l2_detail():
    return pd.DataFrame(
        {
            "status": ["PASS", "WATCH"],
            "canonical_gap": [5.0, -1.0],
            "ts_gap": [1.0, 1.0],
            "canonical_date": ["2019-06-28", "2025-12-28"],
            "from_contract": ["PLN9", "PLF6"],
            "to_contract": ["PLV9", "PLJ6"],
        }
    )


class TestApplyPlHarmonization:
    def test_none_is_noop(self):
        can, ts = _can_ts_frames()
        manifest = _manifest()
        can_h, manifest_h, meta = apply_pl_harmonization(
            can,
            manifest,
            ts,
            _l2_detail(),
            mode="none",
        )
        assert can_h["Close"].tolist() == can["Close"].tolist()
        assert [e.gap for e in manifest_h] == [e.gap for e in manifest]
        assert meta.mode == "none"

    def test_combined_changes_close_and_manifest(self):
        can, ts = _can_ts_frames()
        manifest = _manifest()
        can_h, manifest_h, meta = apply_pl_harmonization(
            can,
            manifest,
            ts,
            _l2_detail(),
            mode="combined",
        )
        assert can_h["Close"].tolist() != can["Close"].tolist()
        assert [e.gap for e in manifest_h] != [e.gap for e in manifest]
        assert len(meta.regime_offsets) > 0
        assert meta.gap_bias_rows > 0

    def test_window_combined_applies_targeted_window_bias(self):
        can, ts = _can_ts_frames()
        manifest = _manifest()
        can_h, manifest_h, meta = apply_pl_harmonization(
            can,
            manifest,
            ts,
            _l2_detail(),
            mode="window_combined",
            top_k=1,
        )
        assert can_h["Close"].tolist() != can["Close"].tolist()
        assert len(meta.window_biases) == 1
        # Top abs signed delta is canonical_gap 5 - ts_gap 1 = 4.0 on first roll.
        assert round(manifest_h[0].gap, 6) == -3.0

    def test_invalid_mode_raises(self):
        can, ts = _can_ts_frames()
        manifest = _manifest()
        with pytest.raises(ValueError, match="Unknown PL harmonization mode"):
            apply_pl_harmonization(can, manifest, ts, _l2_detail(), mode="bad_mode")
