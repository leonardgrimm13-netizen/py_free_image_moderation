from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sdist_manifest_includes_env_example() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert ".env.example" in manifest.splitlines()[0].split()


def test_license_metadata_uses_supported_spdx_string() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["license"] == "LicenseRef-Proprietary"
    assert "setuptools>=77" in metadata["build-system"]["requires"]
