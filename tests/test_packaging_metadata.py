from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_SVG_DEPENDENCIES = {"resvg_py>=0.3,<1.0", "defusedxml>=0.7.1,<1.0"}
PILLOW_AVIF_REQUIREMENT = "Pillow>=11.3.0"


def test_sdist_manifest_includes_env_example() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert ".env.example" in manifest.splitlines()[0].split()


def test_license_metadata_uses_supported_spdx_string() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["license"] == "LicenseRef-Proprietary"
    assert "setuptools>=77" in metadata["build-system"]["requires"]


def test_svg_runtime_dependencies_are_in_core_manifests() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert CORE_SVG_DEPENDENCIES <= set(project["dependencies"])
    assert CORE_SVG_DEPENDENCIES <= requirements
    assert "SVG" in project["description"]


def test_derived_requirement_sets_include_core_svg_dependencies() -> None:
    for filename in ("requirements_api.txt", "requirements_dev.txt", "requirements_local.txt"):
        lines = {
            line.strip()
            for line in (ROOT / filename).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        assert "-r requirements.txt" in lines, f"{filename} must inherit the core runtime dependencies"


def test_builtin_avif_pillow_requirement_is_consistent_across_core_manifests() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(metadata["project"]["dependencies"])
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert PILLOW_AVIF_REQUIREMENT in dependencies
    assert PILLOW_AVIF_REQUIREMENT in requirements
    assert "AVIF" in metadata["project"]["description"]
    assert all("pillow-avif-plugin" not in dependency.lower() for dependency in dependencies | requirements)


def test_every_derived_requirement_set_inherits_core_avif_support() -> None:
    direct_core_inheritors = ("requirements_api.txt", "requirements_dev.txt", "requirements_local.txt")
    for filename in direct_core_inheritors:
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert "-r requirements.txt" in content.splitlines()

    all_requirements = (ROOT / "requirements_all.txt").read_text(encoding="utf-8").splitlines()
    assert "-r requirements_local.txt" in all_requirements
