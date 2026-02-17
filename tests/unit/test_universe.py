"""Tests for the universe configuration module."""

import sys
from pathlib import Path

# Add src/ to path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.universe import CLUSTER_NAMES, Universe


def test_universe_loads_from_yaml():
    u = Universe.from_yaml()
    assert len(u.all_symbols) == 29


def test_tradable_count():
    u = Universe.from_yaml()
    assert len(u.tradable) == 28


def test_research_only():
    u = Universe.from_yaml()
    assert u.research_only == ["SBSW"]


def test_sbsw_is_research_only():
    u = Universe.from_yaml()
    info = u.symbols["SBSW"]
    assert info.status == "research_only"
    assert info.cluster == "EQ_COMMODITY_LINKED"
    assert not info.is_tradable


def test_futures_identification():
    u = Universe.from_yaml()
    futs = u.futures()
    assert "/PA" in futs
    assert "/ES" in futs
    assert "XLE" not in futs
    assert "SBSW" not in futs


def test_equities_and_etfs():
    u = Universe.from_yaml()
    eq = u.equities_and_etfs()
    assert "XLE" in eq
    assert "GS" in eq
    assert "SBSW" in eq
    assert "/PA" not in eq


def test_cluster_count():
    u = Universe.from_yaml()
    # Phase 1a uses 8 of the 11 clusters.
    assert len(u.clusters) == 8


def test_all_clusters_in_canonical_list():
    u = Universe.from_yaml()
    for cluster_name in u.clusters:
        assert cluster_name in CLUSTER_NAMES, f"Unknown cluster: {cluster_name}"


def test_tick_values_for_futures():
    u = Universe.from_yaml()
    assert u.symbols["/PA"].tick_value == 5.0
    assert u.symbols["/ES"].tick_value == 12.50
    assert u.symbols["/ZB"].tick_value == 31.25


def test_tick_values_for_equities():
    u = Universe.from_yaml()
    assert u.symbols["XLE"].tick_value == 0.01
    assert u.symbols["GS"].tick_value == 0.01
