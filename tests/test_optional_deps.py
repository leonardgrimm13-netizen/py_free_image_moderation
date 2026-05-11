from __future__ import annotations

import builtins
from PIL import Image
from modimg.pipeline import build_main_engines, run_engines
from modimg.engines.yolo_weapons import YOLOWorldWeaponsEngine
from modimg.types import Frame


MISSING = {"nudenet", "opennsfw2", "open_nsfw2", "ultralytics", "pytesseract"}


def test_missing_optional_libs_are_skipped(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root in MISSING:
            raise ModuleNotFoundError(f"No module named '{root}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("OCR_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "0")
    monkeypatch.setattr(builtins, "__import__", fake_import)

    engines = build_main_engines(no_apis=True)
    results = run_engines(path="dummy.png", frames=[], engines=engines)

    by_name = {r.name: r for r in results}

    assert by_name["OCR text"].status == "skipped"
    assert by_name["NudeNet"].status == "skipped"
    assert by_name["OpenNSFW2"].status == "skipped"
    assert by_name["YOLO-World weapons"].status == "skipped"
    assert by_name["YOLO forbidden symbols"].status == "skipped"

    for result in by_name.values():
        assert result.status != "error"


def test_opennsfw2_disable_flag_short_circuits_backend_import(monkeypatch) -> None:
    from modimg.engines.opennsfw2_engine import OpenNSFW2Engine

    engine = OpenNSFW2Engine()

    def should_not_run():
        raise AssertionError("backend import should not be attempted when disabled")

    monkeypatch.setenv("OPENNSFW2_DISABLE", "1")
    monkeypatch.setattr(engine, "_import_backend", should_not_run)

    ok, reason = engine.available()

    assert ok is False
    assert reason == "disabled via OPENNSFW2_DISABLE=1"


def test_yolo_skips_when_default_model_path_missing(monkeypatch, tmp_path) -> None:
    engine = YOLOWorldWeaponsEngine()
    monkeypatch.setenv("YOLO_WORLD_MODEL", "")
    monkeypatch.setenv("YOLO_WEAPON_MODEL", "")
    monkeypatch.setenv("YOLO_WEAPONS_WEIGHTS", "")
    monkeypatch.setattr(engine, "available", lambda: (True, "ok"))
    monkeypatch.setattr("modimg.engines.yolo_weapons.project_root", lambda: str(tmp_path))

    frame = Frame(idx=0, pil=Image.new("RGB", (2, 2)))
    result = engine.run(path="dummy.png", frames=[frame])

    assert result.status == "skipped"
    assert result.error is not None
    assert "missing default YOLO model path:" in result.error
    assert "yolov8s-oiv7.pt" in result.error


def test_yolo_weapon_model_project_root_relative_path(monkeypatch, tmp_path) -> None:
    import sys
    import types

    from modimg.engines import yolo_weapons

    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    model_path = model_dir / "weapon.pt"
    model_path.write_bytes(b"fake weights")
    loaded: list[str] = []

    class FakeYOLO:
        names = {}

        def __init__(self, model_ref: str) -> None:
            loaded.append(model_ref)

        def predict(self, *args, **kwargs):
            return []

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setenv("YOLO_WORLD_MODEL", "")
    monkeypatch.setenv("YOLO_WEAPON_MODEL", "weights/weapon.pt")
    monkeypatch.setenv("YOLO_WEAPONS_WEIGHTS", "")
    monkeypatch.setattr(yolo_weapons, "project_root", lambda: str(tmp_path))
    yolo_weapons._YOLO_CACHE.clear()

    frame = Frame(idx=0, pil=Image.new("RGB", (2, 2)))
    result = YOLOWorldWeaponsEngine().run(path="dummy.png", frames=[frame])

    assert result.status == "ok"
    assert loaded == [str(model_path.resolve())]
    assert result.details["model"] == str(model_path.resolve())


def test_yolo_weapon_model_missing_explicit_path_skips_before_import(monkeypatch, tmp_path) -> None:
    import builtins

    from modimg.engines import yolo_weapons

    real_import = builtins.__import__

    def fail_ultralytics_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] == "ultralytics":
            raise AssertionError("ultralytics should not be imported for a missing explicit model path")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setenv("YOLO_WORLD_MODEL", "")
    monkeypatch.setenv("YOLO_WEAPON_MODEL", "weights/missing.pt")
    monkeypatch.setenv("YOLO_WEAPONS_WEIGHTS", "")
    monkeypatch.setattr(yolo_weapons, "project_root", lambda: str(tmp_path))
    monkeypatch.setattr(builtins, "__import__", fail_ultralytics_import)

    frame = Frame(idx=0, pil=Image.new("RGB", (2, 2)))
    result = YOLOWorldWeaponsEngine().run(path="dummy.png", frames=[frame])

    assert result.status == "skipped"
    assert result.error is not None
    assert "explicit YOLO weapons model path not found" in result.error
    assert "weights/missing.pt" in result.error
