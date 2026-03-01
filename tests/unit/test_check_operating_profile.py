"""Unit tests for operating profile loader and gate logic (H.7)."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.operating_profile import (
    OperatingProfile,
    PolicyConstraints,
    PortfolioCheckResult,
    SymbolCheckResult,
    SymbolSetting,
    check_portfolio,
    check_symbol_status,
    discover_ts_custom_file,
    load_operating_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile_dict(**overrides) -> dict:
    """Build a valid operating profile dict with sensible defaults."""
    defaults = {
        "cycle_id": "test_v1",
        "locked_date": "2026-03-01",
        "portfolio_recommendation": "CONDITIONAL GO",
        "portfolio_scope": "futures_only",
        "gating_universe": ["ES", "CL", "PL"],
        "non_gating_symbols": ["PA"],
        "symbol_settings": {
            "ES": {
                "tick_size": 0.25,
                "max_day_delta": 3,
                "expected_status": "WATCH",
                "notes": "test note",
            },
            "CL": {
                "tick_size": 0.01,
                "max_day_delta": 3,
                "expected_status": "ACCEPT",
            },
            "PL": {
                "tick_size": 0.10,
                "max_day_delta": 2,
                "expected_status": "WATCH",
            },
        },
        "policy_constraints": {
            "thresholds_locked": True,
            "strategy_logic_locked": True,
        },
    }
    defaults.update(overrides)
    return defaults


def _write_profile(tmp_path: Path, data: dict) -> Path:
    """Write a YAML profile file and return the path."""
    path = tmp_path / "profile.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Tests: load_operating_profile
# ---------------------------------------------------------------------------

class TestLoadOperatingProfile:
    def test_valid_yaml_loads(self, tmp_path):
        data = _make_profile_dict()
        path = _write_profile(tmp_path, data)
        profile = load_operating_profile(path)

        assert profile.cycle_id == "test_v1"
        assert profile.locked_date == "2026-03-01"
        assert profile.portfolio_recommendation == "CONDITIONAL GO"
        assert profile.gating_universe == ["ES", "CL", "PL"]
        assert profile.non_gating_symbols == ["PA"]
        assert profile.portfolio_scope == "futures_only"

    def test_missing_required_key_raises(self, tmp_path):
        data = _make_profile_dict()
        del data["cycle_id"]
        path = _write_profile(tmp_path, data)

        with pytest.raises(ValueError, match="Missing required keys"):
            load_operating_profile(path)

    def test_missing_symbol_settings_raises(self, tmp_path):
        data = _make_profile_dict()
        del data["symbol_settings"]
        path = _write_profile(tmp_path, data)

        with pytest.raises(ValueError, match="Missing required keys"):
            load_operating_profile(path)

    def test_symbol_settings_parsed_correctly(self, tmp_path):
        data = _make_profile_dict()
        path = _write_profile(tmp_path, data)
        profile = load_operating_profile(path)

        es = profile.symbol_settings["ES"]
        assert es.tick_size == 0.25
        assert es.max_day_delta == 3
        assert es.expected_status == "WATCH"
        assert es.notes == "test note"

        cl = profile.symbol_settings["CL"]
        assert cl.tick_size == 0.01
        assert cl.expected_status == "ACCEPT"

    def test_defaults_for_optional_fields(self, tmp_path):
        data = _make_profile_dict()
        del data["non_gating_symbols"]
        del data["policy_constraints"]
        del data["portfolio_scope"]
        path = _write_profile(tmp_path, data)
        profile = load_operating_profile(path)

        assert profile.non_gating_symbols == []
        assert profile.policy_constraints.thresholds_locked is True
        assert profile.policy_constraints.strategy_logic_locked is True
        assert profile.portfolio_scope == "futures_only"

    def test_invalid_symbol_settings_type_raises(self, tmp_path):
        data = _make_profile_dict()
        data["symbol_settings"] = "not a dict"
        path = _write_profile(tmp_path, data)

        with pytest.raises(ValueError, match="symbol_settings must be a mapping"):
            load_operating_profile(path)

    def test_invalid_symbol_entry_type_raises(self, tmp_path):
        data = _make_profile_dict()
        data["symbol_settings"]["ES"] = "not a dict"
        path = _write_profile(tmp_path, data)

        with pytest.raises(ValueError, match="symbol_settings.*must be a mapping"):
            load_operating_profile(path)

    def test_notes_default_empty_string(self, tmp_path):
        data = _make_profile_dict()
        # CL has no notes in _make_profile_dict, so notes should default to ""
        assert "notes" not in data["symbol_settings"]["CL"]
        path = _write_profile(tmp_path, data)
        profile = load_operating_profile(path)

        assert profile.symbol_settings["CL"].notes == ""

    def test_policy_constraints_parsed(self, tmp_path):
        data = _make_profile_dict()
        data["policy_constraints"] = {
            "thresholds_locked": False,
            "strategy_logic_locked": False,
        }
        path = _write_profile(tmp_path, data)
        profile = load_operating_profile(path)

        assert profile.policy_constraints.thresholds_locked is False
        assert profile.policy_constraints.strategy_logic_locked is False


# ---------------------------------------------------------------------------
# Tests: check_symbol_status
# ---------------------------------------------------------------------------

class TestCheckSymbolStatus:
    def test_match_passes(self):
        result = check_symbol_status("ES", "WATCH", "WATCH")
        assert result.passed is True
        assert result.detail == ""

    def test_mismatch_fails(self):
        result = check_symbol_status("ES", "WATCH", "ACCEPT")
        assert result.passed is False
        assert "expected WATCH" in result.detail
        assert "got ACCEPT" in result.detail

    def test_accept_vs_reject(self):
        result = check_symbol_status("CL", "ACCEPT", "REJECT")
        assert result.passed is False
        assert result.symbol == "CL"

    def test_reject_matches(self):
        result = check_symbol_status("PL", "REJECT", "REJECT")
        assert result.passed is True

    def test_watch_vs_reject(self):
        result = check_symbol_status("ES", "WATCH", "REJECT")
        assert result.passed is False


# ---------------------------------------------------------------------------
# Tests: check_portfolio
# ---------------------------------------------------------------------------

class TestCheckPortfolio:
    def _make_profile(self) -> OperatingProfile:
        return OperatingProfile(
            cycle_id="test",
            locked_date="2026-03-01",
            portfolio_recommendation="CONDITIONAL GO",
            gating_universe=["ES", "CL", "PL"],
            non_gating_symbols=[],
            symbol_settings={},
            policy_constraints=PolicyConstraints(),
        )

    def test_all_match_passes(self):
        profile = self._make_profile()
        results = [
            SymbolCheckResult("ES", "WATCH", "WATCH", True),
            SymbolCheckResult("CL", "ACCEPT", "ACCEPT", True),
            SymbolCheckResult("PL", "WATCH", "WATCH", True),
        ]
        portfolio = check_portfolio(profile, results)
        assert portfolio.passed is True
        assert portfolio.recommendation == "CONDITIONAL GO"

    def test_one_mismatch_fails(self):
        profile = self._make_profile()
        results = [
            SymbolCheckResult("ES", "WATCH", "WATCH", True),
            SymbolCheckResult("CL", "ACCEPT", "REJECT", False, "expected ACCEPT, got REJECT"),
            SymbolCheckResult("PL", "WATCH", "WATCH", True),
        ]
        portfolio = check_portfolio(profile, results)
        assert portfolio.passed is False

    def test_all_mismatch_fails(self):
        profile = self._make_profile()
        results = [
            SymbolCheckResult("ES", "WATCH", "REJECT", False),
            SymbolCheckResult("CL", "ACCEPT", "REJECT", False),
            SymbolCheckResult("PL", "WATCH", "REJECT", False),
        ]
        portfolio = check_portfolio(profile, results)
        assert portfolio.passed is False

    def test_empty_results_passes(self):
        profile = self._make_profile()
        portfolio = check_portfolio(profile, [])
        assert portfolio.passed is True


# ---------------------------------------------------------------------------
# Tests: discover_ts_custom_file
# ---------------------------------------------------------------------------

class TestDiscoverTsCustomFile:
    def test_finds_file(self, tmp_path):
        f = tmp_path / "TS_ES_CUSTOM_ADJ_1D_20180101_20260101.csv"
        f.write_text("Date,Open,High,Low,Close,Volume\n")
        result = discover_ts_custom_file("ES", tmp_path, "ADJ")
        assert result == f

    def test_returns_none_when_missing(self, tmp_path):
        result = discover_ts_custom_file("ES", tmp_path, "ADJ")
        assert result is None

    def test_handles_multiple_matches(self, tmp_path):
        f1 = tmp_path / "TS_CL_CUSTOM_UNADJ_1D_20180101.csv"
        f2 = tmp_path / "TS_CL_CUSTOM_UNADJ_1D_20190101.csv"
        f1.write_text("data\n")
        f2.write_text("data\n")
        result = discover_ts_custom_file("CL", tmp_path, "UNADJ")
        # Should return first sorted match.
        assert result == f1

    def test_wrong_variant_not_matched(self, tmp_path):
        f = tmp_path / "TS_ES_CUSTOM_ADJ_1D_20180101.csv"
        f.write_text("data\n")
        result = discover_ts_custom_file("ES", tmp_path, "UNADJ")
        assert result is None

    def test_wrong_symbol_not_matched(self, tmp_path):
        f = tmp_path / "TS_CL_CUSTOM_ADJ_1D_20180101.csv"
        f.write_text("data\n")
        result = discover_ts_custom_file("ES", tmp_path, "ADJ")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: PortfolioCheckResult.to_dict
# ---------------------------------------------------------------------------

class TestPortfolioCheckResultDict:
    def test_to_dict_has_expected_keys(self):
        result = PortfolioCheckResult(
            passed=True,
            recommendation="CONDITIONAL GO",
            symbol_results=[
                SymbolCheckResult("ES", "WATCH", "WATCH", True),
            ],
        )
        d = result.to_dict()
        assert set(d.keys()) == {"passed", "recommendation", "symbol_results"}
        assert len(d["symbol_results"]) == 1

    def test_to_dict_json_serializable(self):
        result = PortfolioCheckResult(
            passed=False,
            recommendation="NO-GO",
            symbol_results=[
                SymbolCheckResult("ES", "WATCH", "REJECT", False, "mismatch"),
                SymbolCheckResult("CL", "ACCEPT", "ACCEPT", True),
            ],
        )
        # Must not raise.
        serialized = json.dumps(result.to_dict())
        parsed = json.loads(serialized)
        assert parsed["passed"] is False
        assert len(parsed["symbol_results"]) == 2


# ---------------------------------------------------------------------------
# Tests: SymbolCheckResult.to_dict
# ---------------------------------------------------------------------------

class TestSymbolCheckResultDict:
    def test_to_dict_correctness(self):
        result = SymbolCheckResult(
            symbol="ES",
            expected="WATCH",
            actual="WATCH",
            passed=True,
            detail="",
        )
        d = result.to_dict()
        assert d == {
            "symbol": "ES",
            "expected": "WATCH",
            "actual": "WATCH",
            "passed": True,
            "detail": "",
        }

    def test_to_dict_with_detail(self):
        result = SymbolCheckResult(
            symbol="CL",
            expected="ACCEPT",
            actual="REJECT",
            passed=False,
            detail="expected ACCEPT, got REJECT",
        )
        d = result.to_dict()
        assert d["passed"] is False
        assert "REJECT" in d["detail"]
