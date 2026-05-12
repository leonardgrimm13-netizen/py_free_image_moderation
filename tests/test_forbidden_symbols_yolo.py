from __future__ import annotations

import json
import sys
import types
import pytest
from PIL import Image

from modimg.engines.forbidden_symbols_yolo import YOLOForbiddenSymbolsEngine, _FORBIDDEN_SYMBOLS_YOLO_CACHE
from modimg.enums import EngineStatus, VerdictLabel
from modimg.pipeline import build_local_engines, build_pre_engines
from modimg.types import EngineResult, Frame
from modimg.utils import parse_label_thresholds
from modimg.verdict import compute_verdict


def _frame() -> list[Frame]:
    return [Frame(idx=3, pil=Image.new("RGB", (200, 100), color=(240, 240, 240)))]


@pytest.fixture(autouse=True)
def _clean_cache() -> None:
    _FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()


def test_parse_label_thresholds_empty() -> None:
    assert parse_label_thresholds("") == {}
    assert parse_label_thresholds(None) == {}


def test_parse_label_thresholds_basic_and_spaces() -> None:
    assert parse_label_thresholds("isis:0.75,swastika:0.50") == {"isis": 0.75, "swastika": 0.50}
    assert parse_label_thresholds("isis:0.75, swastika:0.50") == {"isis": 0.75, "swastika": 0.50}


def test_parse_label_thresholds_case_insensitive_labels() -> None:
    assert parse_label_thresholds("ISIS:0.75, SwAsTiKa:0.50") == {"isis": 0.75, "swastika": 0.50}


def test_parse_label_thresholds_ignores_invalid_entries() -> None:
    assert parse_label_thresholds("bad,label:notfloat,label:,:0.5,isis:0.75") == {"isis": 0.75}


def test_parse_label_thresholds_clamps_out_of_range_values() -> None:
    assert parse_label_thresholds("low:-0.5,high:1.5") == {"low": 0.0, "high": 1.0}


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


def _yolo_detection_result(label: str, confidence: float, *, include_detections: bool = True) -> EngineResult:
    details = {"top_label": label, "top_confidence": confidence}
    if include_detections:
        details["detections"] = [{"label": label, "confidence": confidence}]
    return EngineResult(
        name="YOLO forbidden symbols",
        status=EngineStatus.OK,
        scores={"forbidden_symbols_max_conf": confidence},
        details=details,
    )


def test_verdict_label_review_threshold_suppresses_isis_false_positive(monkeypatch) -> None:
    monkeypatch.delenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", raising=False)
    monkeypatch.delenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF", raising=False)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.30")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    assert compute_verdict([_yolo_detection_result("isis", 0.55)]).label == VerdictLabel.REVIEW

    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", "isis:0.75")
    verdict = compute_verdict([_yolo_detection_result("isis", 0.55)])

    assert verdict.label == VerdictLabel.OK
    assert not verdict.reasons


def test_verdict_label_review_threshold_allows_higher_isis_detection(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.30")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", "isis:0.75")

    verdict = compute_verdict([_yolo_detection_result("isis", 0.80)])

    assert verdict.label == VerdictLabel.REVIEW
    assert any("isis confidence=0.80 threshold=0.75" in reason for reason in verdict.reasons)


def test_verdict_label_block_threshold_blocks_isis(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.30")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", "isis:0.75")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF", "isis:0.92")

    verdict = compute_verdict([_yolo_detection_result("isis", 0.93)])

    assert verdict.label == VerdictLabel.BLOCK
    assert any("isis confidence=0.93 threshold=0.92" in reason for reason in verdict.reasons)


def test_verdict_label_threshold_case_insensitive_and_swastika_specific(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.80")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.95")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", "SwAsTiKa:0.50")

    verdict = compute_verdict([_yolo_detection_result("SWASTIKA", 0.60)])

    assert verdict.label == VerdictLabel.REVIEW
    assert any("swastika confidence=0.60 threshold=0.50" in reason for reason in verdict.reasons)


def test_verdict_unconfigured_label_uses_global_threshold(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.30")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", "isis:0.75")

    verdict = compute_verdict([_yolo_detection_result("other_symbol", 0.55)])

    assert verdict.label == VerdictLabel.REVIEW
    assert any("other_symbol confidence=0.55 threshold=0.30" in reason for reason in verdict.reasons)


def test_verdict_fallback_without_detections_uses_top_label_threshold(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.30")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", "isis:0.75")

    verdict = compute_verdict([_yolo_detection_result("isis", 0.80, include_detections=False)])

    assert verdict.label == VerdictLabel.REVIEW
    assert any("isis" in reason and "threshold=0.75" in reason for reason in verdict.reasons)


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


def test_forbidden_symbols_model_path_resolution_accepts_project_relative(monkeypatch, tmp_path) -> None:
    model = tmp_path / "rel_model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", "rel_model.pt")
    from modimg.engines.forbidden_symbols_yolo import _resolve_model_path

    assert _resolve_model_path() == model.resolve()


def test_forbidden_symbols_engine_max_frames_zero_skips_inference(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    FakeYOLO.confidence = 0.93
    fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "0")

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.details["max_frames"] == 0
    assert result.scores["forbidden_symbols_detection_count"] == 0.0


def test_forbidden_symbols_max_frames_zero_does_not_load_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "0")

    def fail_load(*args, **kwargs):
        raise AssertionError("YOLO model must not be loaded when max_frames <= 0")

    monkeypatch.setattr("modimg.engines.forbidden_symbols_yolo._load_model", fail_load)

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["forbidden_symbols_detection_count"] == 0.0
    assert result.details["inference_skipped"] is True
    assert result.details["skip_reason"] == "FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0"
