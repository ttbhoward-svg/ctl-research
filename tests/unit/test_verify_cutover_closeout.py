"""Unit tests for cutover closeout verification script (H.12)."""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_cutover_closeout import REQUIRED_FILES, verify_files


# ---------------------------------------------------------------------------
# Tests: verify_files
# ---------------------------------------------------------------------------

class TestVerifyFiles:
    def test_all_present(self, tmp_path):
        manifest = ["a.txt", "b/c.txt"]
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b").mkdir()
        (tmp_path / "b" / "c.txt").write_text("y")

        result = verify_files(repo_root=tmp_path, required=manifest)
        assert result["passed"] is True
        assert result["present_count"] == 2
        assert result["missing_count"] == 0
        assert result["missing"] == []

    def test_some_missing(self, tmp_path):
        manifest = ["exists.txt", "missing.txt"]
        (tmp_path / "exists.txt").write_text("x")

        result = verify_files(repo_root=tmp_path, required=manifest)
        assert result["passed"] is False
        assert result["present_count"] == 1
        assert result["missing_count"] == 1
        assert "missing.txt" in result["missing"]

    def test_all_missing(self, tmp_path):
        manifest = ["a.txt", "b.txt"]
        result = verify_files(repo_root=tmp_path, required=manifest)
        assert result["passed"] is False
        assert result["missing_count"] == 2

    def test_empty_manifest(self, tmp_path):
        result = verify_files(repo_root=tmp_path, required=[])
        assert result["passed"] is True
        assert result["total"] == 0

    def test_result_is_json_serialisable(self, tmp_path):
        manifest = ["a.txt"]
        (tmp_path / "a.txt").write_text("x")
        result = verify_files(repo_root=tmp_path, required=manifest)
        serialized = json.dumps(result)
        parsed = json.loads(serialized)
        assert parsed["passed"] is True


# ---------------------------------------------------------------------------
# Tests: real repo manifest
# ---------------------------------------------------------------------------

class TestRealManifest:
    def test_required_files_list_is_nonempty(self):
        assert len(REQUIRED_FILES) >= 10

    def test_all_real_files_exist(self):
        """Validate that the manifest matches the actual repo."""
        repo_root = Path(__file__).resolve().parents[2]
        result = verify_files(repo_root=repo_root)
        assert result["passed"] is True, (
            f"Missing files: {result['missing']}"
        )
