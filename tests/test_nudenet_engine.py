from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def test_nudenet_avif_uses_lazy_rgb_jpeg_fallback_from_decoded_frame(monkeypatch) -> None:
    seen_paths: list[str] = []
    seen_pixels: list[tuple[int, int, int]] = []

    class Detector:
        def detect(self, path: str):
            seen_paths.append(path)
            assert Path(path).suffix == ".jpg"
            assert Path(path).is_file()
            with Image.open(path) as image:
                assert image.format == "JPEG"
                assert image.mode == "RGB"
                seen_pixels.append(image.getpixel((0, 0)))
            return []

    detector = Detector()
    monkeypatch.setattr(NudeNetEngine, "_DETECTOR", detector)
    monkeypatch.setitem(sys.modules, "nudenet", types.SimpleNamespace(NudeDetector=lambda: detector))
    frame = Frame(idx=0, pil=Image.new("RGB", (6, 4), color=(40, 90, 150)), source_format="avif")

    try:
        result = NudeNetEngine().run("original.avif", [frame])

        assert result.status == EngineStatus.OK
        assert len(seen_paths) == 1
        assert Path(seen_paths[0]).exists()
        # JPEG is lossy, but it must come from the already decoded frame.
        assert all(abs(actual - expected) <= 5 for actual, expected in zip(seen_pixels[0], (40, 90, 150), strict=True))
        assert frame.temporary_file_paths() == (seen_paths[0],)
    finally:
        frame.close()

    assert Path(seen_paths[0]).exists() is False


def test_nudenet_non_avif_single_frame_keeps_original_path_without_fallback(monkeypatch, tmp_path) -> None:
    original = tmp_path / "ordinary.png"
    Image.new("RGB", (4, 3), color=(1, 2, 3)).save(original)
    seen_paths: list[str] = []

    class Detector:
        def detect(self, path: str):
            seen_paths.append(path)
            return []

    detector = Detector()
    monkeypatch.setattr(NudeNetEngine, "_DETECTOR", detector)
    monkeypatch.setitem(sys.modules, "nudenet", types.SimpleNamespace(NudeDetector=lambda: detector))
    frame = Frame(idx=0, pil=Image.new("RGB", (4, 3)), source_format="png")

    try:
        result = NudeNetEngine().run(str(original), [frame])
        assert result.status == EngineStatus.OK
        assert seen_paths == [str(original)]
        assert frame.temporary_file_paths() == ()
    finally:
        frame.close()
