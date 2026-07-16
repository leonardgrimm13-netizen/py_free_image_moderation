from __future__ import annotations

import os
import sys
import types

from PIL import Image

from modimg.engines.ocr import OCREngine
from modimg.enums import EngineStatus
from modimg.types import Frame


def _install_fake_pytesseract(monkeypatch, text: str) -> None:
    fake_module = types.SimpleNamespace(
        image_to_string=lambda *args, **kwargs: text,
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
    monkeypatch.setenv("OCR_ENABLE", "1")
    monkeypatch.setattr(OCREngine, "_CACHE", {})


def _run_ocr(monkeypatch, tmp_path, blocklist_text: str, detected_text: str):
    blocklist = tmp_path / "ocr.txt"
    blocklist.write_text(blocklist_text, encoding="utf-8")
    _install_fake_pytesseract(monkeypatch, detected_text)
    engine = OCREngine()
    engine.blocklist_path = str(blocklist)
    frame = Frame(idx=0, pil=Image.new("RGB", (4, 4)))
    return engine.execute("dummy.png", [frame])


def test_ocr_re_prefix_compiles_the_expression_without_prefix(monkeypatch, tmp_path) -> None:
    result = _run_ocr(monkeypatch, tmp_path, r"re:\b(heil\s+hitler|hakenkreuz)\b" + "\n", "HEIL   HITLER")

    assert result.status == EngineStatus.OK
    assert result.scores["ocr_match"] == 1.0
    assert not result.details["hit"].startswith("re:")


def test_ocr_plain_blocklist_lines_are_literal(monkeypatch, tmp_path) -> None:
    no_match = _run_ocr(monkeypatch, tmp_path, "a.b\n", "axb")
    literal_match = _run_ocr(monkeypatch, tmp_path, "a.b\n", "a.b")

    assert no_match.scores["ocr_match"] == 0.0
    assert literal_match.scores["ocr_match"] == 1.0


def test_ocr_pattern_cache_is_scoped_to_each_file(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("other\n", encoding="utf-8")
    stat = first.stat()
    os.utime(second, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    monkeypatch.setattr(OCREngine, "_CACHE", {})

    first_engine = OCREngine()
    first_engine.blocklist_path = str(first)
    second_engine = OCREngine()
    second_engine.blocklist_path = str(second)

    assert first_engine._load_patterns()[0].pattern == "first"
    assert second_engine._load_patterns()[0].pattern == "other"


def test_ocr_restores_tesseract_command_before_releasing_lock(monkeypatch, tmp_path) -> None:
    blocklist = tmp_path / "ocr.txt"
    blocklist.write_text("blocked\n", encoding="utf-8")
    config = types.SimpleNamespace(tesseract_cmd="original-command")

    class TrackingLock:
        entered = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, exc_type, exc, tb):
            assert config.tesseract_cmd == "original-command"
            self.entered = False

    lock = TrackingLock()

    def image_to_string(*args, **kwargs):
        assert lock.entered is True
        assert config.tesseract_cmd == "custom-command"
        raise RuntimeError("ocr backend failed")

    fake_module = types.SimpleNamespace(image_to_string=image_to_string, pytesseract=config)
    monkeypatch.setitem(sys.modules, "pytesseract", fake_module)
    monkeypatch.setenv("OCR_ENABLE", "1")
    monkeypatch.setenv("TESSERACT_CMD", "custom-command")
    monkeypatch.setattr(OCREngine, "_TESSERACT_CONFIG_LOCK", lock)
    engine = OCREngine()
    engine.blocklist_path = str(blocklist)

    result = engine.execute("dummy.png", [Frame(idx=0, pil=Image.new("RGB", (4, 4)))])

    assert result.status == EngineStatus.ERROR
    assert config.tesseract_cmd == "original-command"
    assert lock.entered is False
