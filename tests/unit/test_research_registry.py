"""Tests for research ticker registry loader."""

from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ctl.research_registry import load_research_registry  # noqa: E402


def _write_yaml(path: Path, obj: dict) -> Path:
    path.write_text(yaml.safe_dump(obj), encoding="utf-8")
    return path


def test_load_valid_registry(tmp_path: Path):
    p = _write_yaml(
        tmp_path / "r.yaml",
        {
            "cycle_id": "cutover_v1",
            "registry_id": "r1",
            "profile_path": "configs/cutover/operating_profile_v1.yaml",
            "symbols": [
                {"symbol": "PA", "enabled": True, "slippage_per_side": 0.1, "tick_size": 0.1, "max_day_delta": 3},
                {"symbol": "xle", "enabled": False},
            ],
        },
    )
    r = load_research_registry(p)
    assert r.registry_id == "r1"
    assert r.enabled_symbols() == ["PA"]
    assert r.slippage_map() == {"PA": 0.1}
    assert r.symbols[0].tick_size == 0.1
    assert r.symbols[0].max_day_delta == 3


def test_reject_duplicate_symbol(tmp_path: Path):
    p = _write_yaml(
        tmp_path / "r.yaml",
        {
            "cycle_id": "cutover_v1",
            "registry_id": "r1",
            "profile_path": "configs/cutover/operating_profile_v1.yaml",
            "symbols": [{"symbol": "PA"}, {"symbol": "PA"}],
        },
    )
    with pytest.raises(ValueError):
        load_research_registry(p)


def test_reject_missing_required(tmp_path: Path):
    p = _write_yaml(tmp_path / "r.yaml", {"symbols": []})
    with pytest.raises(ValueError):
        load_research_registry(p)
