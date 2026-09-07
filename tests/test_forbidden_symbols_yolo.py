from __future__ import annotations

import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from modimg.engines import forbidden_symbols_yolo as fs
from modimg.enums import EngineStatus, VerdictLabel
from modimg.pipeline import build_local_engines, build_pre_engines
from modimg.types import EngineResult, Frame
from modimg.verdict import compute_verdict

NAMES = dict(fs.FORBIDDEN_SYMBOL_CLASSES)


def _frames(count: int = 1) -> list[Frame]:
    return [Frame(idx=i + 3, pil=Image.new("RGBA", (200, 100), color=(240, 240, 240, 255))) for i in range(count)]


class Boxes:
    def __init__(self, rows):
        self.cls = [row[0] for row in rows]
        self.conf = [row[1] for row in rows]
        self.xyxy = [row[2] for row in rows]


class Result:
    names = NAMES

    def __init__(self, rows):
        self.boxes = Boxes(rows)


class FakeYOLO:
    names = NAMES
    task = "detect"
    rows = [(6, 0.72, [10, 20, 110, 80])]
    load_count = 0
    instances = []

    def __init__(self, model_path, task=None):
        assert task == "detect"
        self.model_path = model_path
        self.calls = []
        type(self).load_count += 1
        type(self).instances.append(self)

    def predict(self, source, **kwargs):
        self.calls.append((source, dict(kwargs)))
        count = len(source) if isinstance(source, list) else 1
        return [Result(list(type(self).rows)) for _ in range(count)]


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    fs._FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()
    fs._FORBIDDEN_SYMBOLS_YOLO_INFERENCE_LOCKS.clear()
    FakeYOLO.load_count = 0
    FakeYOLO.instances = []
    FakeYOLO.rows = [(6, 0.72, [10, 20, 110, 80])]
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "cpu")
    for key in (
        "FORBIDDEN_SYMBOLS_YOLO_BACKEND", "FORBIDDEN_SYMBOLS_YOLO_MODEL",
        "FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", "FORBIDDEN_SYMBOLS_YOLO_ONNX_MODEL",
        "FORBIDDEN_SYMBOLS_YOLO_CONF", "FORBIDDEN_SYMBOLS_YOLO_IOU",
        "FORBIDDEN_SYMBOLS_YOLO_IMGSZ", "FORBIDDEN_SYMBOLS_YOLO_MAX_DET",
        "FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS", "FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF",
        "FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF",
        "FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF",
        "FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE",
        "FORBIDDEN_SYMBOLS_YOLO_STOP_AFTER_BLOCK", "YOLO_AUTOINSTALL",
    ):
        monkeypatch.delenv(key, raising=False)


def _install_fake(monkeypatch, tmp_path, suffix=".pt", fake=FakeYOLO):
    model = tmp_path / f"model{suffix}"
    model.write_bytes(b"real-ish model bytes" * 200)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=fake))
    if suffix == ".onnx":
        monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace())
    return model


def test_disabled_does_not_import_ultralytics(monkeypatch):
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ENABLE", "0")
    sys.modules.pop("ultralytics", None)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.SKIPPED
    assert "ultralytics" not in sys.modules


@pytest.mark.parametrize("suffix", [".pt", ".onnx"])
def test_missing_model_skips(monkeypatch, tmp_path, suffix):
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(tmp_path / f"missing{suffix}"))
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.SKIPPED
    assert "missing forbidden symbols YOLO" in (result.error or "")
    json.dumps(result.__dict__)


@pytest.mark.parametrize("suffix", [".pt", ".onnx"])
def test_lfs_pointer_skips_before_load(monkeypatch, tmp_path, suffix):
    model = tmp_path / f"model{suffix}"
    model.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.SKIPPED
    assert "pointer file" in (result.error or "")


def test_missing_ultralytics_skips(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path)
    sys.modules.pop("ultralytics", None)
    monkeypatch.setattr(fs, "_module_available", lambda name: False if name == "ultralytics" else True)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.SKIPPED
    assert "ultralytics not available" in (result.error or "")


def test_model_load_disables_ultralytics_autoinstall(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path)
    utils = types.SimpleNamespace(AUTOINSTALL=True)
    checks = types.SimpleNamespace(AUTOINSTALL=True)
    monkeypatch.setitem(sys.modules, "ultralytics.utils", utils)
    monkeypatch.setitem(sys.modules, "ultralytics.utils.checks", checks)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.OK
    assert utils.AUTOINSTALL is False and checks.AUTOINSTALL is False
    assert __import__("os").environ["YOLO_AUTOINSTALL"] == "false"


def test_explicit_onnx_without_runtime_skips(monkeypatch, tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx" * 1000)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setattr(fs, "_module_available", lambda name: False if name == "onnxruntime" else True)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.SKIPPED
    assert "onnxruntime" in (result.error or "")


def test_defaults_rgb_original_size_and_one_detection(monkeypatch, tmp_path):
    model = _install_fake(monkeypatch, tmp_path)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.OK
    source, kwargs = FakeYOLO.instances[0].calls[0]
    assert source.mode == "RGB" and source.size == (200, 100)
    assert kwargs == {"conf": 0.25, "iou": 0.45, "imgsz": 640, "max_det": 20, "device": "cpu", "verbose": False}
    assert result.details["model_path"] == str(model.resolve())
    assert result.details["top_label"] == "swastika"
    detection = result.details["detections"][0]
    assert detection["frame_idx"] == 3
    assert detection["class_id"] == 6
    assert detection["label"] == "swastika"
    assert detection["confidence"] == pytest.approx(0.72)
    assert detection["bbox_xyxy"] == [10.0, 20.0, 110.0, 80.0]
    assert detection["bbox_norm_xyxy"] == pytest.approx([0.05, 0.2, 0.55, 0.8])
    assert detection["area_ratio"] == pytest.approx(0.3)
    assert detection["image_size"] == [200, 100]
    json.dumps(result.__dict__)


def test_canonical_default_model_paths():
    assert fs.DEFAULT_FORBIDDEN_SYMBOLS_PT_MODEL == "models/forbidden_symbols_yolo26s_GPU_0.1.pt"
    assert fs.DEFAULT_FORBIDDEN_SYMBOLS_ONNX_MODEL == "models/forbidden_symbols_yolo26s_CPU_0.1.onnx"
    assert fs.DEFAULT_FORBIDDEN_SYMBOLS_MODEL == fs.DEFAULT_FORBIDDEN_SYMBOLS_PT_MODEL


def test_zero_and_multiple_detections(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path)
    FakeYOLO.rows = []
    empty = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert empty.status == EngineStatus.OK
    assert empty.details["detections"] == []
    assert empty.scores["forbidden_symbols_detection_count"] == 0.0
    json.dumps(empty.__dict__)
    fs._FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()
    FakeYOLO.rows = [(1, 0.71, [1, 2, 30, 40]), (6, 0.68, [30, 5, 70, 60]), (5, 0.64, [50, 10, 90, 80])]
    multiple = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert [d["label"] for d in multiple.details["detections"]] == ["black_sun", "swastika", "ss_skull"]
    assert multiple.details["detection_count"] == len(multiple.details["detections"])
    assert multiple.details["top_confidence"] == max(d["confidence"] for d in multiple.details["detections"])
    json.dumps(multiple.__dict__)


def test_all_class_ids_use_fixed_order(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path)
    FakeYOLO.rows = [(i, 0.5, [0, 0, 10, 10]) for i in range(7)]
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert [(d["class_id"], d["label"]) for d in result.details["detections"]] == list(NAMES.items())


def test_wrong_mapping_unknown_id_and_inconsistent_arrays(monkeypatch, tmp_path):
    class WrongMapping(FakeYOLO):
        names = {0: "wrong"}

    _install_fake(monkeypatch, tmp_path, fake=WrongMapping)
    assert fs.YOLOForbiddenSymbolsEngine().execute("x", _frames()).status == EngineStatus.ERROR
    fs._FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()
    _install_fake(monkeypatch, tmp_path)
    FakeYOLO.rows = [(7, 0.8, [0, 0, 1, 1])]
    unknown = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert unknown.status == EngineStatus.ERROR
    assert "unknown class ID" in (unknown.error or "")

    class Broken(FakeYOLO):
        def predict(self, source, **kwargs):
            result = Result([(0, 0.5, [0, 0, 1, 1])])
            result.boxes.conf = []
            return [result]

    fs._FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()
    _install_fake(monkeypatch, tmp_path, fake=Broken)
    broken = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert broken.status == EngineStatus.ERROR
    assert "inconsistent detection arrays" in (broken.error or "")


def test_result_metadata_conflict_is_an_error(monkeypatch, tmp_path):
    class WrongResultMetadata(FakeYOLO):
        def predict(self, source, **kwargs):
            result = Result([(0, 0.5, [0, 0, 1, 1])])
            result.names = {0: "wrong"}
            return [result]

    _install_fake(monkeypatch, tmp_path, fake=WrongResultMetadata)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert "result.names" in (result.error or "")


def test_wrong_task_and_class_count_are_errors(monkeypatch, tmp_path):
    class WrongTask(FakeYOLO):
        task = "classify"

    _install_fake(monkeypatch, tmp_path, fake=WrongTask)
    wrong_task = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert wrong_task.status == EngineStatus.ERROR
    assert "expected detect" in (wrong_task.error or "")

    class WrongCount(FakeYOLO):
        def __init__(self, model_path, task=None):
            super().__init__(model_path, task)
            self.model = types.SimpleNamespace(nc=8)

    fs._FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()
    _install_fake(monkeypatch, tmp_path, fake=WrongCount)
    wrong_count = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert wrong_count.status == EngineStatus.ERROR
    assert "class count" in (wrong_count.error or "")


def test_invalid_imgsz_is_visible_error(monkeypatch):
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_IMGSZ", "960")
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert "must be 640" in (result.error or "")


def test_pt_batches_but_onnx_is_sequential(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path, ".pt")
    pt = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames(2))
    assert pt.details["batch_enabled"] is True
    assert isinstance(FakeYOLO.instances[0].calls[0][0], list)
    assert [d["frame_idx"] for d in pt.details["detections"]] == [3, 4]
    fs._FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()
    FakeYOLO.instances = []
    _install_fake(monkeypatch, tmp_path, ".onnx")
    onnx = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames(2))
    assert onnx.status == EngineStatus.OK
    assert onnx.details["backend"] == "onnx"
    assert onnx.details["device_resolved"] == "cpu"
    assert onnx.details["batch_enabled"] is False
    assert onnx.details["batch_disabled_reason"] == "onnx model has fixed batch size 1"
    assert len(FakeYOLO.instances[0].calls) == 2


def test_pt_and_onnx_normalize_identically(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path, ".pt")
    pt = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames()).details["detections"]
    fs._FORBIDDEN_SYMBOLS_YOLO_CACHE.clear()
    _install_fake(monkeypatch, tmp_path, ".onnx")
    onnx = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames()).details["detections"]
    assert pt == onnx


def test_auto_backend_priority(monkeypatch, tmp_path):
    pt, onnx = tmp_path / "model.pt", tmp_path / "model.onnx"
    pt.write_bytes(b"pt" * 1000)
    onnx.write_bytes(b"onnx" * 1000)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", str(pt))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ONNX_MODEL", str(onnx))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "auto")
    monkeypatch.setattr(fs, "_module_available", lambda name: True)
    monkeypatch.setattr(fs, "_cuda_available", lambda: True)
    assert fs._select_model().backend == "pt"
    monkeypatch.setattr(fs, "_cuda_available", lambda: False)
    assert fs._select_model().backend == "onnx"
    onnx.unlink()
    assert (fs._select_model().backend, fs._select_model().device_resolved) == ("pt", "cpu")
    onnx.write_bytes(b"onnx" * 1000)
    pt.unlink()
    monkeypatch.setattr(fs, "_cuda_available", lambda: True)
    assert (fs._select_model().backend, fs._select_model().device_resolved) == ("onnx", "cpu")


def test_auto_cpu_falls_back_to_pt_when_onnx_runtime_is_missing(monkeypatch, tmp_path):
    pt, onnx = tmp_path / "model.pt", tmp_path / "model.onnx"
    pt.write_bytes(b"pt" * 1000)
    onnx.write_bytes(b"onnx" * 1000)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", str(pt))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ONNX_MODEL", str(onnx))
    monkeypatch.setattr(fs, "_module_available", lambda name: name != "onnxruntime")
    assert fs._select_model().backend == "pt"


def test_auto_only_onnx_reports_missing_runtime(monkeypatch, tmp_path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx" * 1000)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_ONNX_MODEL", str(onnx))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "auto")
    monkeypatch.setattr(fs, "_cuda_available", lambda: False)
    monkeypatch.setattr(fs, "_module_available", lambda name: name != "onnxruntime")
    with pytest.raises(fs.OptionalModelUnavailable, match="onnxruntime"):
        fs._select_model()


@pytest.mark.parametrize(
    "backend,env_name,wrong_suffix,expected_suffix",
    [
        ("pt", "FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", ".onnx", ".pt"),
        ("onnx", "FORBIDDEN_SYMBOLS_YOLO_ONNX_MODEL", ".pt", ".onnx"),
        ("pt", "FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", ".bin", ".pt"),
        ("onnx", "FORBIDDEN_SYMBOLS_YOLO_ONNX_MODEL", ".bin", ".onnx"),
    ],
)
def test_backend_specific_model_path_rejects_wrong_format(
    monkeypatch, tmp_path, backend, env_name, wrong_suffix, expected_suffix,
):
    wrong_model = tmp_path / f"wrong{wrong_suffix}"
    wrong_model.write_bytes(b"wrong model format" * 200)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BACKEND", backend)
    monkeypatch.setenv(env_name, str(wrong_model))
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert f"{env_name} must point to a {expected_suffix} model" in (result.error or "")


@pytest.mark.parametrize("suffix,backend", [(".pt", "pt"), (".onnx", "onnx")])
def test_legacy_override_selects_format(monkeypatch, tmp_path, suffix, backend):
    model = tmp_path / f"legacy{suffix}"
    model.write_bytes(b"model" * 1000)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", str(model))
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace())
    assert fs._select_model().backend == backend


def test_explicit_cuda_unavailable_and_onnx_cuda_conflict_skip(monkeypatch, tmp_path):
    pt = tmp_path / "m.pt"
    pt.write_bytes(b"pt" * 1000)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", str(pt))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "cuda:0")
    monkeypatch.setattr(fs, "_visible_cuda_device_count", lambda: 0)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.SKIPPED and "not available" in (result.error or "")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BACKEND", "onnx")
    conflict = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert conflict.status == EngineStatus.SKIPPED and "CPU-only" in (conflict.error or "")


def test_numeric_cuda_device_selects_pt_and_is_normalized(monkeypatch, tmp_path):
    pt = tmp_path / "m.pt"
    pt.write_bytes(b"pt" * 1000)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", str(pt))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "0")
    monkeypatch.setattr(fs, "_visible_cuda_device_count", lambda: 1)
    selection = fs._select_model()
    assert selection.backend == "pt"
    assert selection.device_resolved == "cuda:0"


@pytest.mark.parametrize(
    "device,device_count,expected_status,expected_resolved",
    [
        ("0", 1, EngineStatus.OK, "cuda:0"),
        ("cuda:0", 1, EngineStatus.OK, "cuda:0"),
        ("cuda", 1, EngineStatus.OK, "cuda:0"),
        ("1", 1, EngineStatus.SKIPPED, None),
        ("cuda:1", 1, EngineStatus.SKIPPED, None),
        ("2", 2, EngineStatus.SKIPPED, None),
    ],
)
def test_explicit_cuda_index_is_checked_against_visible_devices(
    monkeypatch, tmp_path, device, device_count, expected_status, expected_resolved,
):
    _install_fake(monkeypatch, tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    fake_cuda = types.SimpleNamespace(is_available=lambda: True, device_count=lambda: device_count)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", device)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == expected_status
    if expected_resolved is None:
        assert "not available or visible" in (result.error or "")
        assert FakeYOLO.load_count == 0
    else:
        assert result.details["device_resolved"] == expected_resolved


def test_cache_and_per_model_inference_lock(monkeypatch, tmp_path):
    class Concurrent(FakeYOLO):
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def predict(self, source, **kwargs):
            with self.state_lock:
                type(self).active += 1
                type(self).max_active = max(type(self).max_active, type(self).active)
            try:
                time.sleep(0.04)
                return super().predict(source, **kwargs)
            finally:
                with self.state_lock:
                    type(self).active -= 1

    _install_fake(monkeypatch, tmp_path, fake=Concurrent)
    barrier = threading.Barrier(2)

    def run():
        barrier.wait(timeout=2)
        return fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result(timeout=3) for future in (pool.submit(run), pool.submit(run))]
    assert all(result.status == EngineStatus.OK for result in results)
    assert Concurrent.load_count == 1 and Concurrent.max_active == 1


def test_different_model_keys_have_independent_inference_locks():
    first = fs._inference_lock("pt:/one:cpu")
    second = fs._inference_lock("onnx:/two:cpu")
    assert first is not second

    barrier = threading.Barrier(2)
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def enter(lock):
        nonlocal active, max_active
        barrier.wait(timeout=2)
        with lock:
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(enter, first), pool.submit(enter, second)]
        for future in futures:
            future.result(timeout=2)
    assert max_active == 2


@pytest.mark.parametrize("suffix", [".pt", ".onnx"])
def test_corrupt_existing_model_is_error(monkeypatch, tmp_path, suffix):
    class CorruptYOLO:
        def __init__(self, model_path, task=None):
            raise ValueError("invalid model archive")

    _install_fake(monkeypatch, tmp_path, suffix, CorruptYOLO)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert "invalid model archive" in (result.error or "")
    json.dumps(result.__dict__)


def test_policy_all_detections_and_ignore_semantics(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path)
    FakeYOLO.rows = [(1, 0.85, [0, 0, 20, 20]), (3, 0.80, [20, 20, 40, 40])]
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF", "isis:0.75")
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.details["top_label"] == "black_sun"
    assert result.details["block_detection"]["label"] == "isis"
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS", "black_sun,isis")
    ignored = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert len(ignored.details["detections"]) == 2
    assert ignored.scores["forbidden_symbols_detected"] == 1.0
    assert ignored.scores["forbidden_symbols_block_hit"] == 0.0
    assert ignored.details["ignored_detection_count"] == 2


@pytest.mark.parametrize("suffix", [".pt", ".onnx"])
def test_sequential_stop_after_policy_block(monkeypatch, tmp_path, suffix):
    _install_fake(monkeypatch, tmp_path, suffix)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE", "0")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_STOP_AFTER_BLOCK", "1")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "2")
    FakeYOLO.rows = [(6, 0.95, [1, 1, 20, 20])]
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames(2))
    assert result.status == EngineStatus.OK
    assert result.details["early_stopped"] is True
    assert result.details["processed_frames"] == [3]
    assert len(FakeYOLO.instances[0].calls) == 1


@pytest.mark.parametrize("confidence,review,block", [(0.2, False, False), (0.5, True, False), (0.95, True, True)])
def test_policy_thresholds(confidence, review, block):
    policy = fs.evaluate_forbidden_symbol_policy(
        [{"label": "swastika", "confidence": confidence}],
        review_conf=0.3, block_conf=0.9, label_review_conf={}, label_block_conf={}, ignore_labels=set(),
    )
    assert policy["review_hit"] is review and policy["block_hit"] is block


def test_label_specific_review_threshold():
    policy = fs.evaluate_forbidden_symbol_policy(
        [{"label": "isis", "confidence": 0.4}],
        review_conf=0.6, block_conf=0.9,
        label_review_conf={"isis": 0.35}, label_block_conf={}, ignore_labels=set(),
    )
    assert policy["review_hit"] is True
    assert policy["block_hit"] is False


def test_threshold_configuration_validation(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.91")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", "0.90")
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert "must not exceed" in (result.error or "")

    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "0.30")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", "isis:0.95")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF", "isis:0.90")
    assert fs.YOLOForbiddenSymbolsEngine().execute("x", _frames()).status == EngineStatus.ERROR


@pytest.mark.parametrize(
    "value,error_part",
    [("foobar:0.5", "unknown"), ("isis:nope", "invalid threshold"), ("isis:0.4, ISIS:0.5", "duplicate")],
)
def test_label_threshold_parser_rejects_invalid_entries(monkeypatch, tmp_path, value, error_part):
    _install_fake(monkeypatch, tmp_path)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", value)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert error_part in (result.error or "")


def test_label_threshold_parser_accepts_spaces_and_case(monkeypatch):
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF", " ISIS : 0.75 , swastika:0.50 ")
    assert fs._strict_label_thresholds("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF") == {"isis": 0.75, "swastika": 0.5}


def test_unknown_ignore_label_is_tolerated(monkeypatch):
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS", "foobar")
    policy = fs.evaluate_forbidden_symbol_policy(
        [{"label": "isis", "confidence": 0.8}],
        review_conf=0.3,
        block_conf=0.9,
        label_review_conf={},
        label_block_conf={},
    )
    assert policy["review_hit"] is True


@pytest.mark.parametrize("confidence", [float("nan"), float("inf"), -0.1, 1.5])
def test_invalid_confidence_is_rejected(monkeypatch, tmp_path, confidence):
    _install_fake(monkeypatch, tmp_path)
    FakeYOLO.rows = [(1, confidence, [0, 0, 10, 10])]
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert "invalid confidence" in (result.error or "")


@pytest.mark.parametrize(
    "box,error_part",
    [
        ([0, 0, float("nan"), 2], "invalid bounding box"),
        ([0, 0, float("inf"), 2], "invalid bounding box"),
        ([10, 0, 5, 2], "non-positive"),
        ([0, 10, 2, 5], "non-positive"),
        ([1, 0, 1, 2], "non-positive"),
        ([0, 1, 2, 1], "non-positive"),
        ([201, 0, 202, 2], "outside"),
        ([-1000000, 20, 50, 80], "outside the source image tolerance"),
        ([20, -1000000, 80, 50], "outside the source image tolerance"),
        ([20, 20, 1000000, 80], "outside the source image tolerance"),
    ],
)
def test_invalid_boxes_are_rejected(monkeypatch, tmp_path, box, error_part):
    _install_fake(monkeypatch, tmp_path)
    FakeYOLO.rows = [(1, 0.5, box)]
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert error_part in (result.error or "")


def test_small_box_boundary_overflow_is_clamped(monkeypatch, tmp_path):
    _install_fake(monkeypatch, tmp_path)
    FakeYOLO.rows = [(1, 0.5, [-0.001, -0.001, 200.001, 100.001])]
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.details["detections"][0]["bbox_xyxy"] == [0.0, 0.0, 200.0, 100.0]


@pytest.mark.parametrize(
    "name,value,error_part",
    [
        ("FORBIDDEN_SYMBOLS_YOLO_IMGSZ", "not-an-int", "must be an integer"),
        ("FORBIDDEN_SYMBOLS_YOLO_MAX_DET", "0", "greater than zero"),
        ("FORBIDDEN_SYMBOLS_YOLO_MAX_DET", "2.5", "must be an integer"),
        ("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "2.5", "must be an integer"),
    ],
)
def test_integer_configuration_is_strict(monkeypatch, tmp_path, name, value, error_part):
    _install_fake(monkeypatch, tmp_path)
    monkeypatch.setenv(name, value)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert error_part in (result.error or "")


@pytest.mark.parametrize("device", ["cuda:", "cuda:-1", "cuda:foo", "-1", "gpu:0"])
def test_invalid_device_configuration_is_rejected(monkeypatch, tmp_path, device):
    _install_fake(monkeypatch, tmp_path)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", device)
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.ERROR
    assert "DEVICE must be" in (result.error or "")


@pytest.mark.parametrize("class_id,valid", [(1, True), (1.0, True), (1.8, False), (-1, False), (7, False), (float("nan"), False)])
def test_class_id_validation(monkeypatch, tmp_path, class_id, valid):
    _install_fake(monkeypatch, tmp_path)
    FakeYOLO.rows = [(class_id, 0.5, [0, 0, 10, 10])]
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert (result.status == EngineStatus.OK) is valid


def test_verdict_trusts_policy_hits_and_legacy_falls_back():
    explicit = EngineResult(
        name="YOLO forbidden symbols", status=EngineStatus.OK,
        scores={"forbidden_symbols_max_conf": 0.99, "forbidden_symbols_review_hit": 0.0, "forbidden_symbols_block_hit": 0.0},
        details={"top_label": "swastika"},
    )
    assert compute_verdict([explicit]).label == VerdictLabel.OK
    review = EngineResult(
        name="YOLO forbidden symbols", status=EngineStatus.OK,
        scores={"forbidden_symbols_max_conf": 0.4, "forbidden_symbols_review_hit": 1.0, "forbidden_symbols_block_hit": 0.0},
        details={"review_detection": {"label": "black_sun", "confidence": 0.4}},
    )
    assert compute_verdict([review]).label == VerdictLabel.REVIEW
    legacy = EngineResult(
        name="YOLO forbidden symbols", status=EngineStatus.OK,
        scores={"forbidden_symbols_max_conf": 0.95}, details={"top_label": "swastika"},
    )
    assert compute_verdict([legacy]).label == VerdictLabel.BLOCK


def test_max_frames_zero_never_loads(monkeypatch):
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "0")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", "invalid-unused-policy")
    monkeypatch.delitem(sys.modules, "ultralytics", raising=False)
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(fs, "_load_model", lambda *args: pytest.fail("must not load"))
    result = fs.YOLOForbiddenSymbolsEngine().execute("x", _frames())
    assert result.status == EngineStatus.SKIPPED
    assert result.details["skip_reason"] == "FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0"
    assert "ultralytics" not in sys.modules and "onnxruntime" not in sys.modules


def test_pipeline_order_and_json_compatibility():
    assert "YOLO forbidden symbols" in [engine.name for engine in build_local_engines()]
    assert [engine.name for engine in build_pre_engines()] == ["pHash blocklist", "pHash allowlist"]
    payload = {"detections": [{"label": "swastika", "bbox_xyxy": [1.0, 2.0, 3.0, 4.0]}]}
    assert json.loads(json.dumps(payload)) == payload
