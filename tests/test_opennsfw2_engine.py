from __future__ import annotations

import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from modimg.engines.opennsfw2_engine import OpenNSFW2Engine
from modimg.enums import EngineStatus
from modimg.types import EngineResult, Frame


def _frame() -> list[Frame]:
    return [Frame(idx=0, pil=Image.new("RGB", (2, 2), color=(1, 2, 3)))]


def test_opennsfw2_in_process_mode_defaults_to_subprocess(monkeypatch) -> None:
    monkeypatch.delenv("OPENNSFW2_IN_PROCESS", raising=False)
    assert OpenNSFW2Engine._in_process_mode() == "0"

    monkeypatch.setenv("OPENNSFW2_IN_PROCESS", "0")
    assert OpenNSFW2Engine._in_process_mode() == "0"

    monkeypatch.setenv("OPENNSFW2_IN_PROCESS", "auto")
    assert OpenNSFW2Engine._in_process_mode() == "auto"

    monkeypatch.setenv("OPENNSFW2_IN_PROCESS", "invalid")
    assert OpenNSFW2Engine._in_process_mode() == "0"


def test_opennsfw2_run_uses_subprocess_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENNSFW2_IN_PROCESS", raising=False)
    eng = OpenNSFW2Engine()
    calls: list[str] = []

    def fake_in_process(path, frames, start):
        raise AssertionError("in-process should not be used by default")

    def fake_subprocess(path, start):
        calls.append("subprocess")
        return EngineResult(name=eng.name, status=EngineStatus.OK, scores={"nsfw_probability": 0.1})

    monkeypatch.setattr(eng, "_predict_in_process", fake_in_process)
    monkeypatch.setattr(eng, "_predict_in_subprocess", fake_subprocess)

    result = eng.run("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert calls == ["subprocess"]


def test_opennsfw2_auto_falls_back_to_subprocess_on_error(monkeypatch) -> None:
    monkeypatch.setenv("OPENNSFW2_IN_PROCESS", "auto")
    eng = OpenNSFW2Engine()
    calls: list[str] = []

    def fake_in_process(path, frames, start):
        calls.append("in_process")
        return EngineResult(name=eng.name, status=EngineStatus.ERROR, error="boom")

    def fake_subprocess(path, start):
        calls.append("subprocess")
        return EngineResult(name=eng.name, status=EngineStatus.OK, scores={"nsfw_probability": 0.2})

    monkeypatch.setattr(eng, "_predict_in_process", fake_in_process)
    monkeypatch.setattr(eng, "_predict_in_subprocess", fake_subprocess)

    result = eng.run("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["nsfw_probability"] == 0.2
    assert calls == ["in_process", "subprocess"]


def test_opennsfw2_rejects_non_finite_probabilities() -> None:
    engine = OpenNSFW2Engine()

    for probability in (float("nan"), float("inf"), float("-inf")):
        result = engine._result_from_probability(probability, "fake", 0)
        assert result.status == EngineStatus.ERROR
        assert "non-finite" in (result.error or "")


def test_opennsfw2_subprocess_does_not_inherit_api_credentials(monkeypatch) -> None:
    captured_env: dict[str, str] = {}

    def fake_run(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return types.SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "backend": "fake", "probability": 0.2}) + "\n",
            stderr="",
        )

    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("SIGHTENGINE_USER", "sight-user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "sight-secret")
    monkeypatch.setattr("modimg.engines.opennsfw2_engine.subprocess.run", fake_run)

    result = OpenNSFW2Engine()._predict_in_subprocess("dummy.png", 0)

    assert result.status == EngineStatus.OK
    assert "OPENAI_API_KEY" not in captured_env
    assert "SIGHTENGINE_USER" not in captured_env
    assert "SIGHTENGINE_SECRET" not in captured_env


def test_opennsfw2_serializes_in_process_inference(monkeypatch) -> None:
    class TrackingRLock:
        def __init__(self) -> None:
            self.lock = threading.RLock()
            self.state_lock = threading.Lock()
            self.attempts = 0
            self.two_attempts = threading.Event()

        def __enter__(self):
            with self.state_lock:
                self.attempts += 1
                if self.attempts >= 2:
                    self.two_attempts.set()
            self.lock.acquire()
            return self

        def __exit__(self, exc_type, exc, traceback):
            self.lock.release()
            return False

    inference_lock = TrackingRLock()
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def predict_image(path: str) -> float:
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            assert inference_lock.two_attempts.wait(timeout=2)
            return 0.2
        finally:
            with state_lock:
                active -= 1

    backend = types.SimpleNamespace(predict_image=predict_image)
    monkeypatch.setenv("OPENNSFW2_IN_PROCESS", "1")
    monkeypatch.setitem(sys.modules, "opennsfw2", backend)
    monkeypatch.setattr(OpenNSFW2Engine, "_BACKEND", ("opennsfw2", backend))
    monkeypatch.setattr(OpenNSFW2Engine, "_INFERENCE_LOCK", inference_lock)
    start = threading.Barrier(2)

    def run_once(index: int):
        start.wait(timeout=2)
        return OpenNSFW2Engine().execute(
            "dummy.png",
            [Frame(idx=index, pil=Image.new("RGB", (2, 2)))],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_once, range(2)))

    assert [result.status for result in results] == [EngineStatus.OK, EngineStatus.OK]
    assert inference_lock.attempts == 2
    assert max_active == 1
