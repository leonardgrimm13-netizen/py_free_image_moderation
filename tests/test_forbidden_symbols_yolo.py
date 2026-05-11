from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from modimg.engines.forbidden_symbols_yolo import YOLOForbiddenSymbolsEngine, _FORBIDDEN_SYMBOLS_YOLO_CACHE
from modimg.enums import EngineStatus, VerdictLabel
from modimg.pipeline import build_local_engines, build_pre_engines
from modimg.types import EngineResult, Frame
from modimg.verdict import compute_verdict


def _frame() -> list[Frame]:
    return [Frame(idx=3, pil=Image.new("RGB", (200, 100), color=(240, 240, 240)))]


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    _FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()


def test_forbidden_symbols_engine_disabled(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "0")
    sys.modules.pop("ultralytics", None)

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.SKIPPED
    assert "FORBIDDEN_SYMBOLS_YOLO_ENABLE=0" in (result.error or "")
    assert "ultralytics" not in sys.modules


def test_forbidden_symbols_engine_missing_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(tmp_path / "missing.pt"))

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert "missing forbidden symbols YOLO model" in (result.error or "")


def test_forbidden_symbols_engine_model_pointer(monkeypatch, tmp_path) -> None:
    pointer = tmp_path / "model.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(pointer))

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert "model pointer file detected" in (result.error or "")
    assert "real model weights" in (result.error or "")


class FakeBoxes:
    cls = [0]
    xyxy = [[10, 20, 110, 80]]

    def __init__(self, confidence: float = 0.72) -> None:
        self.conf = [confidence]


class FakeResult:
    names = {0: "test_symbol"}

    def __init__(self, confidence: float = 0.72) -> None:
        self.boxes = FakeBoxes(confidence)


class FakeYOLO:
    names = {0: "test_symbol"}
    confidence = 0.72

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def predict(self, image, **kwargs):
        assert kwargs["conf"] == 0.2
        assert kwargs["iou"] == 0.45
        assert kwargs["imgsz"] == 960
        assert kwargs["max_det"] == 20
        assert kwargs["verbose"] is False
        return [FakeResult(self.confidence)]


def test_forbidden_symbols_engine_mock_detection(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    FakeYOLO.confidence = 0.72
    fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["forbidden_symbols_detected"] == 1.0
    assert result.scores["forbidden_symbols_max_conf"] == pytest.approx(0.72)
    assert result.scores["forbidden_symbols_review_hit"] == 1.0
    assert result.scores["forbidden_symbols_block_hit"] == 0.0
    assert result.details["top_label"] == "test_symbol"
    assert result.details["model_path"] == str(model.resolve())
    assert result.details["detections"][0]["bbox_xyxy"] == [10.0, 20.0, 110.0, 80.0]
    assert result.details["detections"][0]["bbox_norm_xyxy"] == pytest.approx([0.05, 0.2, 0.55, 0.8])


def test_forbidden_symbols_engine_mock_high_confidence_blocks(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    FakeYOLO.confidence = 0.93
    fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["forbidden_symbols_max_conf"] == pytest.approx(0.93)
    assert result.scores["forbidden_symbols_review_hit"] == 1.0
    assert result.scores["forbidden_symbols_block_hit"] == 1.0


def test_forbidden_symbols_engine_ignore_labels(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    FakeYOLO.confidence = 0.93
    fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS", "test_symbol")

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["forbidden_symbols_detected"] == 0.0
    assert result.scores["forbidden_symbols_detection_count"] == 0.0
    assert result.scores["forbidden_symbols_max_conf"] == 0.0
    assert result.details["top_label"] == ""


def _yolo_result(conf: float) -> EngineResult:
    return EngineResult(
        name="YOLO forbidden symbols",
        status=EngineStatus.OK,
        scores={"forbidden_symbols_max_conf": conf},
        details={"top_label": "test_symbol", "top_confidence": conf},
    )


def test_verdict_forbidden_symbols_block(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    verdict = compute_verdict([_yolo_result(0.93)])
    assert verdict.label == VerdictLabel.BLOCK
    assert verdict.hate_risk >= 1.0
    assert any("YOLO forbidden symbol" in reason for reason in verdict.reasons)


def test_verdict_forbidden_symbols_review(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.30")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    verdict = compute_verdict([_yolo_result(0.72)])
    assert verdict.label == VerdictLabel.REVIEW
    assert any("possible forbidden symbol" in reason for reason in verdict.reasons)


def test_verdict_forbidden_symbols_below_threshold(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.30")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    verdict = compute_verdict([_yolo_result(0.10)])
    assert verdict.label == VerdictLabel.OK
    assert not verdict.reasons


def test_pipeline_includes_forbidden_symbols_engine() -> None:
    assert "YOLO forbidden symbols" in [engine.name for engine in build_local_engines()]


def test_phash_short_circuit_pre_engine_order_unchanged() -> None:
    names = [engine.name for engine in build_pre_engines()]
    assert names == ["pHash blocklist", "pHash allowlist"]
    assert "YOLO forbidden symbols" not in names


def test_json_shape_for_forbidden_symbols() -> None:
    result = EngineResult(
        name="YOLO forbidden symbols",
        status=EngineStatus.OK,
        scores={"forbidden_symbols_max_conf": 0.0, "forbidden_symbols_detection_count": 0.0},
        details={"detections": [{"label": "test_symbol", "bbox_xyxy": [1.0, 2.0, 3.0, 4.0]}]},
    )
    payload = {**result.__dict__, "status": result.status.value}
    restored = json.loads(json.dumps(payload))
    assert restored["details"]["detections"][0]["bbox_xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert restored["scores"]["forbidden_symbols_detection_count"] == 0.0
