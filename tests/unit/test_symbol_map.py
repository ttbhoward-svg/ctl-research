"""Unit tests for symbol mapping + hash tracking (Data Cutover Task B)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.symbol_map import (
    DEFAULT_MAP_PATH,
    EXPECTED_SYMBOL_COUNT,
    PROVIDERS,
    all_ctl_symbols,
    get_ctl_symbol,
    get_provider_symbol,
    load_symbol_map,
    map_sha256,
    provider_symbols,
    validate_symbol_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smap():
    """Load the symbol map once for the whole test module."""
    return load_symbol_map()


@pytest.fixture(scope="module")
def sha(smap):
    """Compute the hash once."""
    return map_sha256()


# ---------------------------------------------------------------------------
# Tests: Load + Structure
# ---------------------------------------------------------------------------

class TestLoadAndStructure:
    def test_load_returns_dict(self, smap):
        assert isinstance(smap, dict)

    def test_has_version(self, smap):
        assert smap.get("version") == 1

    def test_has_providers_list(self, smap):
        assert set(smap.get("providers", [])) == set(PROVIDERS)

    def test_has_symbols_key(self, smap):
        assert "symbols" in smap


# ---------------------------------------------------------------------------
# Tests: 29/29 Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_exactly_29_symbols(self, smap):
        assert len(smap["symbols"]) == EXPECTED_SYMBOL_COUNT

    def test_validate_returns_no_errors(self, smap):
        errors = validate_symbol_map(smap)
        assert errors == [], f"Validation errors: {errors}"

    def test_all_symbols_have_asset_class(self, smap):
        for sym, entry in smap["symbols"].items():
            assert "asset_class" in entry, f"{sym} missing asset_class"

    def test_all_symbols_have_cluster(self, smap):
        for sym, entry in smap["symbols"].items():
            assert "cluster" in entry, f"{sym} missing cluster"

    def test_all_symbols_have_status(self, smap):
        for sym, entry in smap["symbols"].items():
            assert "status" in entry, f"{sym} missing status"

    def test_all_symbols_have_all_providers(self, smap):
        for sym, entry in smap["symbols"].items():
            providers = entry.get("providers", {})
            for p in PROVIDERS:
                assert p in providers, f"{sym} missing provider '{p}'"
                assert providers[p], f"{sym} has empty mapping for '{p}'"

    def test_15_futures(self, smap):
        futures = [s for s in smap["symbols"] if s.startswith("/")]
        assert len(futures) == 15

    def test_7_etfs(self, smap):
        etfs = [
            s for s, e in smap["symbols"].items()
            if e.get("asset_class") == "etf"
        ]
        assert len(etfs) == 7

    def test_7_equities(self, smap):
        equities = [
            s for s, e in smap["symbols"].items()
            if e.get("asset_class") == "equity"
        ]
        assert len(equities) == 7

    def test_28_tradable(self, smap):
        tradable = [
            s for s, e in smap["symbols"].items()
            if e.get("status") == "tradable"
        ]
        assert len(tradable) == 28

    def test_1_research_only(self, smap):
        ro = [
            s for s, e in smap["symbols"].items()
            if e.get("status") == "research_only"
        ]
        assert len(ro) == 1
        assert ro[0] == "SBSW"

    def test_missing_symbol_count_flagged(self):
        bad = {"symbols": {"/ES": {"asset_class": "futures", "cluster": "IDX_FUT", "status": "tradable", "providers": {p: "X" for p in PROVIDERS}}}}
        errors = validate_symbol_map(bad)
        assert any("Expected 29" in e for e in errors)

    def test_missing_provider_flagged(self):
        entry = {
            "asset_class": "futures", "cluster": "IDX_FUT",
            "status": "tradable",
            "providers": {"tradestation": "@ES"},  # missing 3
        }
        bad = {"symbols": {"/ES": entry}}
        errors = validate_symbol_map(bad)
        assert any("databento" in e for e in errors)

    def test_missing_metadata_flagged(self):
        entry = {
            "providers": {p: "X" for p in PROVIDERS},
        }
        bad = {"symbols": {"/ES": entry}}
        errors = validate_symbol_map(bad)
        assert any("asset_class" in e for e in errors)
        assert any("cluster" in e for e in errors)
        assert any("status" in e for e in errors)


# ---------------------------------------------------------------------------
# Tests: SHA-256 Hash
# ---------------------------------------------------------------------------

class TestHash:
    def test_hash_is_64_hex_chars(self, sha):
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_hash_is_deterministic(self):
        h1 = map_sha256()
        h2 = map_sha256()
        assert h1 == h2

    def test_hash_matches_file_content(self):
        import hashlib
        with open(DEFAULT_MAP_PATH, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        assert map_sha256() == expected


# ---------------------------------------------------------------------------
# Tests: Forward Lookup
# ---------------------------------------------------------------------------

class TestForwardLookup:
    def test_futures_tradestation(self, smap):
        assert get_provider_symbol(smap, "/ES", "tradestation") == "@ES"

    def test_futures_databento(self, smap):
        assert get_provider_symbol(smap, "/GC", "databento") == "GC.FUT"

    def test_futures_norgate(self, smap):
        assert get_provider_symbol(smap, "/CL", "norgate") == "&CL"

    def test_futures_ibkr(self, smap):
        assert get_provider_symbol(smap, "/ZN", "ibkr") == "ZN"

    def test_equity_tradestation(self, smap):
        assert get_provider_symbol(smap, "XOM", "tradestation") == "$XOM"

    def test_etf_same_across_providers(self, smap):
        for provider in PROVIDERS:
            assert get_provider_symbol(smap, "XLE", provider) == "XLE"

    def test_unknown_symbol_returns_none(self, smap):
        assert get_provider_symbol(smap, "AAPL", "tradestation") is None

    def test_unknown_provider_returns_none(self, smap):
        assert get_provider_symbol(smap, "/ES", "bloomberg") is None


# ---------------------------------------------------------------------------
# Tests: Reverse Lookup
# ---------------------------------------------------------------------------

class TestReverseLookup:
    def test_tradestation_to_ctl(self, smap):
        assert get_ctl_symbol(smap, "tradestation", "@ES") == "/ES"

    def test_databento_to_ctl(self, smap):
        assert get_ctl_symbol(smap, "databento", "GC.FUT") == "/GC"

    def test_norgate_to_ctl(self, smap):
        assert get_ctl_symbol(smap, "norgate", "&CL") == "/CL"

    def test_equity_reverse(self, smap):
        assert get_ctl_symbol(smap, "tradestation", "$XOM") == "XOM"

    def test_unknown_returns_none(self, smap):
        assert get_ctl_symbol(smap, "tradestation", "AAPL") is None


# ---------------------------------------------------------------------------
# Tests: Bulk Helpers
# ---------------------------------------------------------------------------

class TestBulkHelpers:
    def test_all_ctl_symbols_count(self, smap):
        syms = all_ctl_symbols(smap)
        assert len(syms) == EXPECTED_SYMBOL_COUNT

    def test_all_ctl_symbols_sorted(self, smap):
        syms = all_ctl_symbols(smap)
        assert syms == sorted(syms)

    def test_provider_symbols_count(self, smap):
        for p in PROVIDERS:
            ps = provider_symbols(smap, p)
            assert len(ps) == EXPECTED_SYMBOL_COUNT

    def test_provider_symbols_values_nonempty(self, smap):
        ps = provider_symbols(smap, "tradestation")
        for ctl, ts in ps.items():
            assert ts, f"{ctl} has empty tradestation symbol"


# ---------------------------------------------------------------------------
# Tests: Cross-validation with universe config
# ---------------------------------------------------------------------------

class TestCrossValidation:
    def test_symbols_match_universe(self, smap):
        """All 29 CTL symbols in the map should match symbols_phase1a.yaml."""
        from ctl.universe import Universe
        universe = Universe.from_yaml()
        map_syms = set(smap["symbols"].keys())
        uni_syms = set(universe.all_symbols)
        assert map_syms == uni_syms, (
            f"Mismatch — in map only: {map_syms - uni_syms}, "
            f"in universe only: {uni_syms - map_syms}"
        )

    def test_clusters_consistent(self, smap):
        """Cluster names should match between map and universe."""
        from ctl.universe import Universe
        universe = Universe.from_yaml()
        for sym, entry in smap["symbols"].items():
            uni_info = universe.symbols.get(sym)
            assert uni_info is not None, f"{sym} not in universe"
            assert entry["cluster"] == uni_info.cluster, (
                f"{sym}: map says {entry['cluster']}, "
                f"universe says {uni_info.cluster}"
            )
