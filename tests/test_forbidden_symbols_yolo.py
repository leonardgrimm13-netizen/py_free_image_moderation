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
from modimg.verdict import compute_verdict


def _frame() -> list[Frame]:
    return [Frame(idx=3, pil=Image.new("RGB", (200, 100), color=(240, 240, 240)))]


def _frames(n: int) -> list[Frame]:
    return [Frame(idx=i, pil=Image.new("RGB", (200, 100), color=(240, 240, 240))) for i in range(n)]


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

    assert result.status == EngineStatus.SKIPPED
    assert "missing forbidden symbols YOLO model" in (result.error or "")


def test_forbidden_symbols_engine_model_pointer(monkeypatch, tmp_path) -> None:
    pointer = tmp_path / "model.pt"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(pointer))

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.SKIPPED
    assert "model pointer file detected" in (result.error or "")
    assert "real model weights" in (result.error or "")


def test_forbidden_symbols_engine_missing_ultralytics_skips(monkeypatch, tmp_path) -> None:
    import importlib.util

    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    sys.modules.pop("ultralytics", None)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "ultralytics" else real_find_spec(name))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.SKIPPED
    assert "ultralytics not available" in (result.error or "")


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


class FakeEmptyYOLO:
    names = {0: "test_symbol"}

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def predict(self, image, **kwargs):
        return []


class FakeBatchBoxes:
    def __init__(self, class_id: int, confidence: float) -> None:
        self.cls = [class_id]
        self.conf = [confidence]
        self.xyxy = [[10, 20, 110, 80]]


class FakeBatchResult:
    names = {0: "test_symbol", 1: "second_symbol"}

    def __init__(self, class_id: int, confidence: float) -> None:
        self.boxes = FakeBatchBoxes(class_id, confidence)


class FakeBatchYOLO:
    names = {0: "test_symbol", 1: "second_symbol"}
    result_count = 2
    instances: list["FakeBatchYOLO"] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.calls: list[tuple[bool, dict]] = []
        FakeBatchYOLO.instances.append(self)

    def predict(self, image, **kwargs):
        self.calls.append((isinstance(image, list), dict(kwargs)))
        if isinstance(image, list):
            return [FakeBatchResult(i, 0.70 + (i * 0.1)) for i in range(self.result_count)]
        return [FakeBatchResult(0, 0.70)]


class FakeSequentialStopYOLO:
    names = {0: "test_symbol"}
    instances: list["FakeSequentialStopYOLO"] = []

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.calls = 0
        FakeSequentialStopYOLO.instances.append(self)

    def predict(self, image, **kwargs):
        self.calls += 1
        return [FakeBatchResult(0, 0.95)]


def test_forbidden_symbols_engine_empty_detections(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeEmptyYOLO))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["forbidden_symbols_detected"] == 0.0
    assert result.scores["forbidden_symbols_max_conf"] == 0.0
    assert result.details["detections"] == []


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


def test_forbidden_symbols_batch_parses_multiple_frame_results(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    FakeBatchYOLO.instances = []
    FakeBatchYOLO.result_count = 2
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeBatchYOLO))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "2")

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frames(2))

    assert result.status == EngineStatus.OK
    assert result.details["batch_enabled"] is True
    assert result.details["result_count"] == 2
    assert [d["frame_idx"] for d in result.details["detections"]] == [0, 1]
    assert [d["label"] for d in result.details["detections"]] == ["test_symbol", "second_symbol"]
    assert len(FakeBatchYOLO.instances) == 1
    is_batch, kwargs = FakeBatchYOLO.instances[0].calls[0]
    assert is_batch is True
    assert kwargs["batch"] == 2
    assert kwargs["conf"] == pytest.approx(0.2)
    assert kwargs["iou"] == pytest.approx(0.45)
    assert kwargs["imgsz"] == 960
    assert kwargs["max_det"] == 20
    assert kwargs["verbose"] is False


def test_forbidden_symbols_batch_handles_fewer_results_than_frames(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    FakeBatchYOLO.instances = []
    FakeBatchYOLO.result_count = 1
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeBatchYOLO))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "2")

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frames(2))

    assert result.status == EngineStatus.OK
    assert result.details["result_count"] == 1
    assert [d["frame_idx"] for d in result.details["detections"]] == [0]


def test_forbidden_symbols_sequential_stop_after_block(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    FakeSequentialStopYOLO.instances = []
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeSequentialStopYOLO))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE", "0")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_STOP_AFTER_BLOCK", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "2")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frames(2))

    assert result.status == EngineStatus.OK
    assert result.scores["forbidden_symbols_block_hit"] == 1.0
    assert result.details["early_stopped"] is True
    assert result.details["processed_frames"] == [0]
    assert FakeSequentialStopYOLO.instances[0].calls == 1


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


def test_forbidden_symbols_engine_label_specific_block_threshold(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    FakeYOLO.confidence = 0.72
    fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF", "test_symbol:0.70")

    result = YOLOForbiddenSymbolsEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["forbidden_symbols_max_conf"] == pytest.approx(0.72)
    assert result.scores["forbidden_symbols_block_hit"] == 1.0
    assert result.details["block_detection"]["label"] == "test_symbol"


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


def test_verdict_forbidden_symbols_label_specific_block(monkeypatch) -> None:
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF", "test_symbol:0.70")

    verdict = compute_verdict([_yolo_result(0.72)])

    assert verdict.label == VerdictLabel.BLOCK
    assert verdict.hate_risk >= 1.0
    assert any("test_symbol" in reason for reason in verdict.reasons)


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


def test_forbidden_symbols_model_path_resolution_accepts_project_relative(monkeypatch, tmp_path) -> None:
    model = tmp_path / "rel_model.pt"
    model.write_bytes(b"not a model pointer" * 200)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", "rel_model.pt")
    from modimg.engines.forbidden_symbols_yolo import _resolve_model_path

    assert _resolve_model_path() == model.resolve()


def test_forbidden_symbols_model_path_resolution_accepts_installed_data_root(monkeypatch, tmp_path) -> None:
    install_root = tmp_path / "install"
    model = install_root / "models" / "forbidden_symbols_yolo.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"not a model pointer" * 200)
    (tmp_path / "elsewhere").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path / "elsewhere")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", "models/forbidden_symbols_yolo.pt")
    monkeypatch.setattr("modimg.engines.forbidden_symbols_yolo.project_root", lambda: str(tmp_path / "project"))
    monkeypatch.setattr("modimg.engines.forbidden_symbols_yolo.sys.prefix", str(tmp_path / "prefix"))
    monkeypatch.setattr("modimg.engines.forbidden_symbols_yolo.sysconfig.get_path", lambda name: str(install_root) if name == "data" else "")
    monkeypatch.setattr("modimg.engines.forbidden_symbols_yolo.site.getuserbase", lambda: str(tmp_path / "userbase"))
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
