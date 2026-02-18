"""Phase 1a Archive Package (Task 15).

Generates a deterministic archive manifest listing governance artifacts,
configs, and key outputs with SHA-256 hashes for reproducibility.

See docs/notes/Task15_assumptions.md for design rationale.

Usage
-----
>>> from ctl.archive import build_manifest, save_manifest
>>> manifest = build_manifest(repo_root=Path("."))
>>> save_manifest(manifest, Path("outputs/phase1a_manifest.json"))
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default artifacts to include in the Phase 1a archive.
DEFAULT_ARTIFACTS = [
    "configs/pre_registration_v1.yaml",
    "configs/phase1a.yaml",
    "configs/symbol_map_v1.yaml",
    "docs/governance/model_card_v1.md",
    "docs/governance/phase_gate_decision_v1.md",
    "docs/governance/CTL_Phase_Gate_Checklist_One_Page.md",
    "docs/governance/pre_registration_v1.md",
]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class ArtifactEntry:
    """A single file in the archive manifest."""

    path: str
    sha256: str
    size_bytes: int
    exists: bool


@dataclass
class ArchiveManifest:
    """Deterministic archive manifest for Phase 1a."""

    phase: str = "phase1a"
    timestamp: str = ""
    artifacts: List[ArtifactEntry] = field(default_factory=list)
    extra_files: List[ArtifactEntry] = field(default_factory=list)

    @property
    def n_found(self) -> int:
        return sum(1 for a in self.all_entries if a.exists)

    @property
    def n_missing(self) -> int:
        return sum(1 for a in self.all_entries if not a.exists)

    @property
    def all_entries(self) -> List[ArtifactEntry]:
        return self.artifacts + self.extra_files

    def to_dict(self) -> Dict:
        return {
            "phase": self.phase,
            "timestamp": self.timestamp,
            "n_found": self.n_found,
            "n_missing": self.n_missing,
            "artifacts": [
                {
                    "path": a.path, "sha256": a.sha256,
                    "size_bytes": a.size_bytes, "exists": a.exists,
                }
                for a in self.artifacts
            ],
            "extra_files": [
                {
                    "path": a.path, "sha256": a.sha256,
                    "size_bytes": a.size_bytes, "exists": a.exists,
                }
                for a in self.extra_files
            ],
        }

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "Phase 1a Archive Manifest",
            "=" * 55,
            f"Timestamp: {self.timestamp}",
            f"Found: {self.n_found}  Missing: {self.n_missing}",
            "",
        ]
        for a in self.all_entries:
            status = "OK" if a.exists else "MISSING"
            lines.append(f"  [{status:7s}] {a.path}")
            if a.exists:
                lines.append(f"            sha256: {a.sha256[:16]}...")
        lines.append("=" * 55)
        return "\n".join(lines)


def _scan_file(repo_root: Path, rel_path: str) -> ArtifactEntry:
    """Scan a single file and return its manifest entry."""
    full = repo_root / rel_path
    if full.exists() and full.is_file():
        return ArtifactEntry(
            path=rel_path,
            sha256=file_sha256(full),
            size_bytes=full.stat().st_size,
            exists=True,
        )
    return ArtifactEntry(
        path=rel_path, sha256="", size_bytes=0, exists=False,
    )


def build_manifest(
    repo_root: Path,
    artifact_paths: Optional[List[str]] = None,
    extra_paths: Optional[List[str]] = None,
) -> ArchiveManifest:
    """Build the Phase 1a archive manifest.

    Parameters
    ----------
    repo_root : Path
        Repository root directory.
    artifact_paths : list of str, optional
        Relative paths of core artifacts. Defaults to DEFAULT_ARTIFACTS.
    extra_paths : list of str, optional
        Additional files to include (e.g., gate decision JSON, dataset manifest).

    Returns
    -------
    ArchiveManifest
    """
    if artifact_paths is None:
        artifact_paths = DEFAULT_ARTIFACTS
    if extra_paths is None:
        extra_paths = []

    manifest = ArchiveManifest(
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    for rel in artifact_paths:
        manifest.artifacts.append(_scan_file(repo_root, rel))

    for rel in extra_paths:
        manifest.extra_files.append(_scan_file(repo_root, rel))

    return manifest


def save_manifest(manifest: ArchiveManifest, path: Path) -> Path:
    """Write archive manifest to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    return path


def load_manifest(path: Path) -> Dict:
    """Load a previously saved manifest."""
    with open(path) as f:
        return json.load(f)
