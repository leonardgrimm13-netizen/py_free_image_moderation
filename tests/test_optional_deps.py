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
    monkeypatch.setattr(builtins, "__import__", fake_import)

    engines = build_main_engines(no_apis=True)
    results = run_engines(path="dummy.png", frames=[], engines=engines)

    by_name = {r.name: r for r in results}

    assert by_name["OCR text"].status == "skipped"
    assert by_name["NudeNet"].status == "skipped"
    assert by_name["OpenNSFW2"].status == "skipped"
    assert by_name["YOLO-World weapons"].status == "skipped"

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
