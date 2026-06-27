from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from modimg.engines.sightengine import SightengineEngine
from modimg.enums import EngineStatus
from modimg.types import Frame


class FakeResponse:
    def __init__(self, status_code: int, text: str = "", headers: dict[str, str] | None = None, data=None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"content-type": "application/json"}
        self._data = data

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _frame() -> list[Frame]:
    return [Frame(idx=0, pil=Image.new("RGB", (4, 4), color=(1, 2, 3)))]


def test_sightengine_invalid_json_returns_error(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    fake_requests = types.SimpleNamespace(post=lambda *a, **k: FakeResponse(200, data=ValueError("bad json")))
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert "invalid JSON response" in (result.error or "")


def test_sightengine_http_error_returns_error(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    fake_requests = types.SimpleNamespace(post=lambda *a, **k: FakeResponse(500, text="oops", headers={"content-type": "text/plain"}))
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert result.error == "http error 500"


def test_sightengine_request_exception_returns_error(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")

    def fail_post(*args, **kwargs):
        raise TimeoutError("timed out")

    fake_requests = types.SimpleNamespace(post=fail_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert "request failed: TimeoutError: timed out" == result.error


def test_sightengine_uses_thread_local_sessions(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    monkeypatch.setattr(SightengineEngine, "_SESSION_LOCAL", threading.local())
    created = []
    created_lock = threading.Lock()

    class FakeSession:
        def __init__(self) -> None:
            self.posts = 0
            with created_lock:
                created.append(self)

        def post(self, *args, **kwargs):
            self.posts += 1
            return FakeResponse(200, data={"status": "success"})

    fake_requests = types.SimpleNamespace(Session=FakeSession)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    start_barrier = threading.Barrier(2)

    def run_twice():
        start_barrier.wait(timeout=2)
        first = SightengineEngine().run("dummy.png", _frame())
        second = SightengineEngine().run("dummy.png", _frame())
        return first.status, second.status

    with ThreadPoolExecutor(max_workers=2) as executor:
        statuses = [future.result(timeout=3) for future in [executor.submit(run_twice), executor.submit(run_twice)]]

    assert statuses == [(EngineStatus.OK, EngineStatus.OK), (EngineStatus.OK, EngineStatus.OK)]
    assert len(created) == 2
    assert sorted(session.posts for session in created) == [2, 2]
