"""Unit tests for Chart Study Session Infrastructure (Task 12)."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.chart_study import (
    PULLBACK_CHOICES,
    QUEUE_COLUMNS,
    VOLATILITY_CHOICES,
    TradeObservation,
    filter_dataset,
    generate_notes_template,
    generate_study_queue,
    load_observations,
    load_study_queue,
    make_trade_id,
    save_notes_template,
    save_observations,
    save_study_queue,
    select_bottom_n,
    select_stratified,
    select_top_n,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Minimal assembled-dataset-shaped DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="7D")
    tickers = ["/ES", "/GC", "/CL", "XLE"] * ((n // 4) + 1)
    tickers = tickers[:n]

    return pd.DataFrame({
        "Date": dates,
        "Ticker": tickers,
        "Timeframe": "daily",
        "SetupType": "B1",
        "EntryPrice": rng.uniform(100, 200, n),
        "StopPrice": rng.uniform(90, 100, n),
        "TP1": rng.uniform(110, 120, n),
        "TP2": rng.uniform(120, 130, n),
        "TP3": rng.uniform(130, 140, n),
        "TheoreticalR": np.linspace(-1.0, 3.0, n),
        "RMultiple_Actual": np.linspace(-1.0, 2.5, n),
        "TradeOutcome": np.where(np.linspace(-1, 3, n) > 0, "Win", "Loss"),
        "ExitReason": "TP1",
        "HoldBars": rng.integers(1, 30, n),
        "ScoreBucket": (["bottom"] * (n // 3)
                        + ["mid"] * (n // 3)
                        + ["top"] * (n - 2 * (n // 3))),
        "AssetCluster": "IDX_FUT",
        "BarsOfAir": rng.integers(3, 10, n).astype(float),
        "Slope_20": rng.uniform(5, 20, n),
    })


# ---------------------------------------------------------------------------
# Tests: trade ID
# ---------------------------------------------------------------------------

class TestMakeTradeId:
    def test_basic_format(self):
        row = pd.Series({
            "Date": pd.Timestamp("2020-06-15"),
            "Ticker": "/ES",
            "Timeframe": "daily",
            "SetupType": "B1",
        })
        tid = make_trade_id(row)
        assert tid == "2020-06-15__ES_daily_B1"

    def test_no_slash_in_id(self):
        row = pd.Series({
            "Date": "2020-06-15",
            "Ticker": "/GC",
            "Timeframe": "daily",
            "SetupType": "B1",
        })
        tid = make_trade_id(row)
        assert "/" not in tid

    def test_deterministic(self):
        row = pd.Series({
            "Date": pd.Timestamp("2021-01-01"),
            "Ticker": "XLE",
            "Timeframe": "weekly",
            "SetupType": "B1",
        })
        assert make_trade_id(row) == make_trade_id(row)


# ---------------------------------------------------------------------------
# Tests: observation schema
# ---------------------------------------------------------------------------

class TestTradeObservation:
    def test_default_blank(self):
        obs = TradeObservation(trade_id="test_001")
        assert obs.setup_quality is None
        assert obs.pullback_character is None
        assert obs.volatility_context is None
        assert obs.regime_alignment_note == ""

    def test_valid_observation(self):
        obs = TradeObservation(
            trade_id="test_001",
            setup_quality=4,
            pullback_character="textbook",
            volatility_context="normal",
            regime_alignment_note="Bullish trend confirmed",
        )
        assert obs.validate() == []

    def test_invalid_setup_quality(self):
        obs = TradeObservation(trade_id="t", setup_quality=6)
        errors = obs.validate()
        assert len(errors) == 1
        assert "setup_quality" in errors[0]

    def test_invalid_pullback_character(self):
        obs = TradeObservation(trade_id="t", pullback_character="bad")
        errors = obs.validate()
        assert len(errors) == 1
        assert "pullback_character" in errors[0]

    def test_invalid_volatility_context(self):
        obs = TradeObservation(trade_id="t", volatility_context="crazy")
        errors = obs.validate()
        assert len(errors) == 1
        assert "volatility_context" in errors[0]

    def test_to_dict_roundtrip(self):
        obs = TradeObservation(
            trade_id="test_001", setup_quality=3,
            pullback_character="deep", volatility_context="high",
            regime_alignment_note="aligned",
            execution_quality_note="good",
            post_trade_reflection="held too long",
        )
        d = obs.to_dict()
        restored = TradeObservation.from_dict(d)
        assert restored.trade_id == obs.trade_id
        assert restored.setup_quality == obs.setup_quality
        assert restored.pullback_character == obs.pullback_character
        assert restored.post_trade_reflection == obs.post_trade_reflection


# ---------------------------------------------------------------------------
# Tests: filtering
# ---------------------------------------------------------------------------

class TestFilterDataset:
    def test_filter_by_symbol(self):
        df = _make_dataset(20)
        sub = filter_dataset(df, symbol="/ES")
        assert (sub["Ticker"] == "/ES").all()
        assert len(sub) < len(df)

    def test_filter_by_timeframe(self):
        df = _make_dataset(20)
        sub = filter_dataset(df, timeframe="daily")
        assert len(sub) == 20  # all are daily

    def test_filter_by_date_range(self):
        df = _make_dataset(30)
        sub = filter_dataset(df, date_range=("2020-02-01", "2020-04-01"))
        dates = pd.to_datetime(sub["Date"])
        assert (dates >= "2020-02-01").all()
        assert (dates <= "2020-04-01").all()

    def test_filter_by_score_bucket(self):
        df = _make_dataset(30)
        sub = filter_dataset(df, score_bucket="top")
        assert (sub["ScoreBucket"] == "top").all()

    def test_combined_filters(self):
        df = _make_dataset(30)
        sub = filter_dataset(df, symbol="/ES", score_bucket="top")
        assert (sub["Ticker"] == "/ES").all()
        assert (sub["ScoreBucket"] == "top").all()

    def test_no_filters_returns_all(self):
        df = _make_dataset(20)
        sub = filter_dataset(df)
        assert len(sub) == 20

    def test_impossible_filter_returns_empty(self):
        df = _make_dataset(20)
        sub = filter_dataset(df, symbol="NONEXISTENT")
        assert len(sub) == 0


# ---------------------------------------------------------------------------
# Tests: selection — top/bottom
# ---------------------------------------------------------------------------

class TestSelectTopBottom:
    def test_top_n_returns_highest(self):
        df = _make_dataset(30)
        top5 = select_top_n(df, n=5, by="TheoreticalR")
        assert len(top5) == 5
        # All returned values should be >= the 5th largest in the full set.
        threshold = df["TheoreticalR"].nlargest(5).min()
        assert (top5["TheoreticalR"] >= threshold).all()

    def test_bottom_n_returns_lowest(self):
        df = _make_dataset(30)
        bot5 = select_bottom_n(df, n=5, by="TheoreticalR")
        assert len(bot5) == 5
        threshold = df["TheoreticalR"].nsmallest(5).max()
        assert (bot5["TheoreticalR"] <= threshold).all()

    def test_top_n_sorted_descending(self):
        df = _make_dataset(30)
        top10 = select_top_n(df, n=10)
        vals = top10["TheoreticalR"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_bottom_n_sorted_ascending(self):
        df = _make_dataset(30)
        bot10 = select_bottom_n(df, n=10)
        vals = bot10["TheoreticalR"].values
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

    def test_n_larger_than_dataset(self):
        df = _make_dataset(5)
        top10 = select_top_n(df, n=10)
        assert len(top10) == 5  # returns all available

    def test_missing_column_raises(self):
        df = _make_dataset(10)
        with pytest.raises(ValueError, match="NoSuchCol"):
            select_top_n(df, n=5, by="NoSuchCol")


# ---------------------------------------------------------------------------
# Tests: selection — stratified
# ---------------------------------------------------------------------------

class TestSelectStratified:
    def test_returns_from_all_terciles(self):
        df = _make_dataset(30)
        sel = select_stratified(df, n_per_tercile=3)
        assert "top" in sel["_tercile"].values
        assert "mid" in sel["_tercile"].values
        assert "bottom" in sel["_tercile"].values

    def test_correct_count_per_tercile(self):
        df = _make_dataset(30)
        sel = select_stratified(df, n_per_tercile=3)
        counts = sel["_tercile"].value_counts()
        for label in ("top", "mid", "bottom"):
            assert counts.get(label, 0) <= 3

    def test_deterministic(self):
        df = _make_dataset(30)
        s1 = select_stratified(df, n_per_tercile=3, seed=42)
        s2 = select_stratified(df, n_per_tercile=3, seed=42)
        pd.testing.assert_frame_equal(s1, s2)

    def test_different_seed_different_selection(self):
        df = _make_dataset(30)
        s1 = select_stratified(df, n_per_tercile=3, seed=42)
        s2 = select_stratified(df, n_per_tercile=3, seed=99)
        # May differ (with 30 trades, very likely)
        assert not s1.index.equals(s2.index) or len(df) <= 9

    def test_empty_dataset(self):
        df = _make_dataset(30).head(0)
        sel = select_stratified(df, n_per_tercile=3)
        assert len(sel) == 0

    def test_small_tercile_returns_all(self):
        # Only 6 trades → 2 per tercile; request 5 per tercile
        df = _make_dataset(6)
        sel = select_stratified(df, n_per_tercile=5)
        assert len(sel) == 6  # all included


# ---------------------------------------------------------------------------
# Tests: study queue
# ---------------------------------------------------------------------------

class TestStudyQueue:
    def test_queue_length_matches_input(self):
        df = _make_dataset(10)
        queue = generate_study_queue(df, selection_reason="test")
        assert len(queue) == 10

    def test_queue_entry_structure(self):
        df = _make_dataset(5)
        queue = generate_study_queue(df)
        entry = queue[0]
        assert "trade_id" in entry
        assert "selection_reason" in entry
        assert "context" in entry
        assert "observation" in entry

    def test_observation_has_all_fields(self):
        df = _make_dataset(5)
        queue = generate_study_queue(df)
        obs = queue[0]["observation"]
        assert "trade_id" in obs
        assert "setup_quality" in obs
        assert "pullback_character" in obs
        assert "volatility_context" in obs
        assert "regime_alignment_note" in obs
        assert "execution_quality_note" in obs
        assert "post_trade_reflection" in obs

    def test_context_has_key_columns(self):
        df = _make_dataset(5)
        queue = generate_study_queue(df)
        ctx = queue[0]["context"]
        for col in ["Ticker", "TheoreticalR", "TradeOutcome"]:
            assert col in ctx

    def test_json_serializable(self):
        df = _make_dataset(10)
        queue = generate_study_queue(df)
        # Should not raise.
        serialized = json.dumps(queue, default=str)
        assert len(serialized) > 0

    def test_save_and_load_roundtrip(self):
        df = _make_dataset(5)
        queue = generate_study_queue(df, selection_reason="roundtrip")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "queue.json"
            save_study_queue(queue, path)
            loaded = load_study_queue(path)
        assert len(loaded) == 5
        assert loaded[0]["selection_reason"] == "roundtrip"


# ---------------------------------------------------------------------------
# Tests: notes template
# ---------------------------------------------------------------------------

class TestNotesTemplate:
    def test_header_present(self):
        df = _make_dataset(3)
        md = generate_notes_template(df)
        assert "# Chart Study Notes" in md
        assert "Trades: 3" in md

    def test_one_section_per_trade(self):
        df = _make_dataset(5)
        md = generate_notes_template(df)
        assert md.count("### Observations") == 5

    def test_observation_fields_present(self):
        df = _make_dataset(2)
        md = generate_notes_template(df)
        assert "setup_quality" in md
        assert "pullback_character" in md
        assert "volatility_context" in md
        assert "regime_alignment_note" in md
        assert "execution_quality_note" in md
        assert "post_trade_reflection" in md

    def test_trade_context_in_template(self):
        df = _make_dataset(3)
        md = generate_notes_template(df)
        # Should contain ticker values from the dataset.
        assert "/ES" in md or "_ES" in md

    def test_save_creates_file(self):
        df = _make_dataset(2)
        md = generate_notes_template(df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "notes.md"
            result = save_notes_template(md, path)
            assert result.exists()
            content = result.read_text()
            assert "Chart Study Notes" in content


# ---------------------------------------------------------------------------
# Tests: observation I/O
# ---------------------------------------------------------------------------

class TestObservationIO:
    def test_save_and_load(self):
        obs_list = [
            TradeObservation(
                trade_id="t1", setup_quality=4,
                pullback_character="textbook",
                volatility_context="normal",
            ),
            TradeObservation(
                trade_id="t2", setup_quality=2,
                pullback_character="choppy",
                volatility_context="high",
                post_trade_reflection="Exited too early",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "obs.json"
            save_observations(obs_list, path)
            loaded = load_observations(path)
        assert len(loaded) == 2
        assert loaded[0].trade_id == "t1"
        assert loaded[0].setup_quality == 4
        assert loaded[1].post_trade_reflection == "Exited too early"

    def test_roundtrip_preserves_all_fields(self):
        obs = TradeObservation(
            trade_id="roundtrip_test",
            setup_quality=5,
            pullback_character="deep",
            volatility_context="extreme",
            regime_alignment_note="Strong alignment",
            execution_quality_note="Clean fill",
            post_trade_reflection="Would take again",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "obs.json"
            save_observations([obs], path)
            loaded = load_observations(path)
        restored = loaded[0]
        assert restored.trade_id == obs.trade_id
        assert restored.setup_quality == obs.setup_quality
        assert restored.pullback_character == obs.pullback_character
        assert restored.volatility_context == obs.volatility_context
        assert restored.regime_alignment_note == obs.regime_alignment_note
        assert restored.execution_quality_note == obs.execution_quality_note
        assert restored.post_trade_reflection == obs.post_trade_reflection

    def test_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            save_observations([], path)
            loaded = load_observations(path)
        assert loaded == []
