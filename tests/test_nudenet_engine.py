from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from modimg.engines.nudenet_engine import NudeNetEngine
from modimg.enums import EngineStatus
from modimg.types import Frame


class TrackingRLock:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state_lock = threading.Lock()
        self.attempts = 0
        self.two_attempts = threading.Event()

    def __enter__(self):
        with self._state_lock:
            self.attempts += 1
            if self.attempts >= 2:
                self.two_attempts.set()
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._lock.release()
        return False


def test_nudenet_detector_initialization_runtime_failure_is_error(monkeypatch) -> None:
    def fail_detector():
        raise RuntimeError("corrupt model")

    monkeypatch.setattr(NudeNetEngine, "_DETECTOR", None)
    monkeypatch.setitem(sys.modules, "nudenet", types.SimpleNamespace(NudeDetector=fail_detector))

    result = NudeNetEngine().execute("dummy.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])

    assert result.status == EngineStatus.ERROR
    assert "detector unavailable" in (result.error or "")


def test_nudenet_serializes_inference_on_cached_detector(monkeypatch) -> None:
    inference_lock = TrackingRLock()

    class Detector:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.state_lock = threading.Lock()

        def detect(self, path: str):
            with self.state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                assert inference_lock.two_attempts.wait(timeout=2)
                return []
            finally:
                with self.state_lock:
                    self.active -= 1

    detector = Detector()
    monkeypatch.setattr(NudeNetEngine, "_DETECTOR", detector)
    monkeypatch.setattr(NudeNetEngine, "_INFERENCE_LOCK", inference_lock)
    monkeypatch.setitem(sys.modules, "nudenet", types.SimpleNamespace(NudeDetector=lambda: detector))
    start = threading.Barrier(2)

    def run_once(index: int):
        start.wait(timeout=2)
        frame = Frame(idx=index, pil=Image.new("RGB", (2, 2)))
        return NudeNetEngine().execute("missing.png", [frame])

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_once, range(2)))

    assert [result.status for result in results] == [EngineStatus.OK, EngineStatus.OK]
    assert inference_lock.attempts == 2
    assert detector.max_active == 1
