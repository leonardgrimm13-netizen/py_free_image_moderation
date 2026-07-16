from __future__ import annotations

from modimg import resources


def test_bundled_resource_prefers_installation_over_cwd_shadow(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    install = tmp_path / "install"
    cwd = tmp_path / "cwd"
    relative = "models/model.pt"
    (install / relative).parent.mkdir(parents=True)
    (install / relative).write_bytes(b"trusted")
    (cwd / relative).parent.mkdir(parents=True)
    (cwd / relative).write_bytes(b"shadow")
    source.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(resources, "source_root", lambda: source)
    monkeypatch.setattr(resources.sys, "prefix", str(tmp_path / "prefix"))
    monkeypatch.setattr(resources.sysconfig, "get_path", lambda name: str(install) if name == "data" else "")
    monkeypatch.setattr(resources.site, "getuserbase", lambda: str(tmp_path / "userbase"))

    bundled = resources.resolve_bundled_resource_path(relative)
    explicit = resources.resolve_resource_path(relative)

    assert bundled == (install / relative).resolve()
    assert explicit == (cwd / relative).resolve()


def test_missing_bundled_resource_never_falls_back_to_cwd(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    install = tmp_path / "install"
    cwd = tmp_path / "cwd"
    relative = "models/forbidden_symbols_yolo.pt"
    source.mkdir()
    install.mkdir()
    (cwd / relative).parent.mkdir(parents=True)
    (cwd / relative).write_bytes(b"untrusted-cwd-model")

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(resources, "source_root", lambda: source)
    monkeypatch.setattr(resources, "_installation_roots", lambda: [install])

    bundled_candidates = resources.bundled_resource_candidates(relative)
    bundled = resources.resolve_bundled_resource_path(relative)
    explicit = resources.resolve_resource_path(relative)

    assert (cwd / relative) not in bundled_candidates
    assert bundled == (source / relative).resolve(strict=False)
    assert bundled.exists() is False
    assert explicit == (cwd / relative).resolve()


def test_explicit_relative_resource_prefers_cwd_over_source_root(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source"
    cwd = tmp_path / "cwd"
    relative = "models/custom.pt"
    (source / relative).parent.mkdir(parents=True)
    (source / relative).write_bytes(b"source")
    (cwd / relative).parent.mkdir(parents=True)
    (cwd / relative).write_bytes(b"cwd")

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(resources, "source_root", lambda: source)
    monkeypatch.setattr(resources, "_installation_roots", lambda: [])

    assert resources.resolve_resource_path(relative) == (cwd / relative).resolve()
    assert resources.resolve_bundled_resource_path(relative) == (source / relative).resolve()
