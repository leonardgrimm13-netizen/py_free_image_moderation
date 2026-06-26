from __future__ import annotations

from PIL import Image

from modimg.engines.yolo_weapons import YOLOWorldWeaponsEngine
from modimg.enums import EngineStatus
from modimg.types import Frame


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
