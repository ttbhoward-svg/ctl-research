"""Unit tests for scripts/build_model_card.py."""

from pathlib import Path
import sys

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_model_card  # noqa: E402


class TestBuildModelCard:
    def test_main_writes_model_card_to_out_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_model_card.py", "--out-dir", str(tmp_path)],
        )

        build_model_card.main()

        files = sorted(tmp_path.glob("model_card_*.md"))
        assert len(files) == 1
        assert files[0].read_text().startswith("# Model Card")

    def test_help_exits_without_writing_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            sys,
            "argv",
            ["build_model_card.py", "--help", "--out-dir", str(tmp_path)],
        )

        with pytest.raises(SystemExit) as excinfo:
            build_model_card.main()

        assert excinfo.value.code == 0
        assert list(tmp_path.glob("model_card_*.md")) == []
