from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from modimg.enums import EngineStatus
from modimg.engines.forbidden_symbols_yolo import FORBIDDEN_SYMBOL_CLASSES, YOLOForbiddenSymbolsEngine
from modimg.types import Frame
from modimg.utils import json_safe


if os.getenv("RUN_REAL_MODEL_SMOKE") != "1":
    pytest.skip("set RUN_REAL_MODEL_SMOKE=1 to run the bundled-model smoke test", allow_module_level=True)


def test_bundled_forbidden_symbols_onnx_model_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = Path(__file__).resolve().parents[1] / "models" / "forbidden_symbols_yolo26s_CPU_0.1.onnx"
    assert model_path.is_file()

    monkeypatch.delenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", raising=False)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BACKEND", "onnx")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ONNX_MODEL", str(model_path))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "cpu")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE", "0")

    frame = Frame(idx=0, pil=Image.new("RGB", (640, 480), (127, 127, 127)))
    try:
        result = YOLOForbiddenSymbolsEngine().execute(str(model_path), [frame])
    finally:
        frame.close()

    assert result.status == EngineStatus.OK
    assert result.error is None
    assert result.details["backend"] == "onnx"
    assert result.details["device_resolved"] == "cpu"
    assert result.details["model_path"] == str(model_path)
    assert result.details["class_count"] == len(FORBIDDEN_SYMBOL_CLASSES)
    assert result.details["class_names"] == {str(key): value for key, value in FORBIDDEN_SYMBOL_CLASSES.items()}
    assert result.details["processed_frames"] == [0]
    assert result.details["result_count"] == 1
    assert result.details["detection_count"] == len(result.details["detections"])
    json.dumps(json_safe(result.__dict__), allow_nan=False)
