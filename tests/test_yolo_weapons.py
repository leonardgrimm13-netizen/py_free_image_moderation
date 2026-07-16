from __future__ import annotations

import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from modimg.engines import yolo_weapons
from modimg.engines.yolo_weapons import YOLOWorldWeaponsEngine
from modimg.enums import EngineStatus
from modimg.types import Frame
from modimg.verdict import compute_verdict


class FakeBoxes:
    def __init__(self, class_id: int, confidence: float) -> None:
        self.cls = [class_id]
        self.conf = [confidence]


class FakeResult:
    def __init__(self, class_id: int, confidence: float) -> None:
        self.boxes = FakeBoxes(class_id, confidence)


class FakeWeaponModel:
    names = {0: "firearm", 1: "knife"}

    def __init__(self) -> None:
        self.calls: list[tuple[bool, dict]] = []

    def predict(self, image, **kwargs):
        self.calls.append((isinstance(image, list), dict(kwargs)))
        if isinstance(image, list):
            return [FakeResult(0, 0.61), FakeResult(1, 0.72)]
        return [FakeResult(0, 0.61)]


def test_yolo_weapons_batch_parses_multiple_frame_results(monkeypatch) -> None:
    fake_model = FakeWeaponModel()
    monkeypatch.setenv("YOLO_BATCH_ENABLE", "1")
    monkeypatch.setenv("YOLO_MAX_FRAMES", "2")
    monkeypatch.setattr("modimg.engines.yolo_weapons._resolve_model_reference", lambda: ("fake.pt", True, None))
    monkeypatch.setattr("modimg.engines.yolo_weapons._load_model", lambda model_ref: fake_model)

    frames = [
        Frame(idx=0, pil=Image.new("RGB", (16, 16), color=(1, 2, 3))),
        Frame(idx=1, pil=Image.new("RGB", (16, 16), color=(4, 5, 6))),
    ]

    result = YOLOWorldWeaponsEngine().run("dummy.png", frames)

    assert result.status == EngineStatus.OK
    assert result.scores["yolo_firearm"] == 0.61
    assert result.scores["yolo_knife"] == 0.72
    assert result.details["batch_enabled"] is True
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0][0] is True
    assert fake_model.calls[0][1]["batch"] == 2


def test_yolo_weapons_rejects_incomplete_batch_results(monkeypatch) -> None:
    class IncompleteBatchModel(FakeWeaponModel):
        def predict(self, image, **kwargs):
            return [FakeResult(0, 0.61)]

    monkeypatch.setenv("YOLO_BATCH_ENABLE", "1")
    monkeypatch.setenv("YOLO_MAX_FRAMES", "2")
    monkeypatch.setattr("modimg.engines.yolo_weapons._resolve_model_reference", lambda: ("fake.pt", True, None))
    monkeypatch.setattr("modimg.engines.yolo_weapons._load_model", lambda model_ref: IncompleteBatchModel())
    frames = [
        Frame(idx=0, pil=Image.new("RGB", (16, 16))),
        Frame(idx=1, pil=Image.new("RGB", (16, 16))),
    ]
    engine = YOLOWorldWeaponsEngine()
    monkeypatch.setattr(engine, "available", lambda: (True, ""))

    result = engine.execute("dummy.png", frames)

    assert result.status == EngineStatus.ERROR
    assert "returned 1 results for 2 frames" in (result.error or "")


def test_yolo_weapons_serializes_predict_on_cached_model(monkeypatch) -> None:
    class ConcurrentFakeWeaponModel(FakeWeaponModel):
        def __init__(self) -> None:
            super().__init__()
            self.state_lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def predict(self, image, **kwargs):
            with self.state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.05)
                return super().predict(image, **kwargs)
            finally:
                with self.state_lock:
                    self.active -= 1

    fake_model = ConcurrentFakeWeaponModel()
    yolo_weapons._YOLO_INFERENCE_LOCKS.clear()
    monkeypatch.setenv("YOLO_BATCH_ENABLE", "1")
    monkeypatch.setenv("YOLO_MAX_FRAMES", "2")
    monkeypatch.setattr("modimg.engines.yolo_weapons._resolve_model_reference", lambda: ("fake.pt", True, None))
    monkeypatch.setattr("modimg.engines.yolo_weapons._load_model", lambda model_ref: fake_model)

    frames = [
        Frame(idx=0, pil=Image.new("RGB", (16, 16), color=(1, 2, 3))),
        Frame(idx=1, pil=Image.new("RGB", (16, 16), color=(4, 5, 6))),
    ]
    start_barrier = threading.Barrier(2)

    def run_once():
        start_barrier.wait(timeout=2)
        return YOLOWorldWeaponsEngine().run("dummy.png", frames)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = [future.result(timeout=3) for future in [executor.submit(run_once), executor.submit(run_once)]]

    assert [result.status for result in results] == [EngineStatus.OK, EngineStatus.OK]
    assert fake_model.max_active == 1
    assert len(fake_model.calls) == 2
    assert all(is_batch is True for is_batch, _ in fake_model.calls)


def test_yolo_toy_gun_is_not_classified_as_realistic(monkeypatch) -> None:
    class ToyModel(FakeWeaponModel):
        names = {0: "toy gun"}

        def predict(self, image, **kwargs):
            return [FakeResult(0, 0.8)]

    monkeypatch.setenv("ALLOW_TOY_GUN", "1")
    monkeypatch.setattr("modimg.engines.yolo_weapons._resolve_model_reference", lambda: ("toy.pt", True, None))
    monkeypatch.setattr("modimg.engines.yolo_weapons._load_model", lambda model_ref: ToyModel())
    frame = Frame(idx=0, pil=Image.new("RGB", (16, 16)))

    result = YOLOWorldWeaponsEngine().run("dummy.png", [frame])
    verdict = compute_verdict([result])

    assert result.scores["yolo_firearm_toy"] == 0.8
    assert result.scores["yolo_firearm_realistic"] == 0.0
    assert verdict.label.value == "OK"


def test_yolo_weapons_rejects_inconsistent_detection_arrays(monkeypatch) -> None:
    class BrokenBoxes:
        cls = [0]
        conf = []

    class BrokenModel:
        names = {0: "gun"}

        def predict(self, image, **kwargs):
            return [types.SimpleNamespace(boxes=BrokenBoxes())]

    engine = YOLOWorldWeaponsEngine()
    monkeypatch.setattr(engine, "available", lambda: (True, ""))
    monkeypatch.setattr("modimg.engines.yolo_weapons._resolve_model_reference", lambda: ("broken.pt", True, None))
    monkeypatch.setattr("modimg.engines.yolo_weapons._load_model", lambda model_ref: BrokenModel())

    result = engine.execute("dummy.png", [Frame(idx=0, pil=Image.new("RGB", (4, 4)))])

    assert result.status == EngineStatus.ERROR
    assert "inconsistent detection arrays" in (result.error or "")
