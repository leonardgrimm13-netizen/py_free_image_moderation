from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from modimg.engines.sightengine import SightengineEngine, SightengineRunState
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


def test_sightengine_disable_flag_short_circuits_credentials(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_DISABLE", "1")
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")

    result = SightengineEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.SKIPPED
    assert result.error == "disabled via SIGHTENGINE_DISABLE=1"


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
            return FakeResponse(200, data={"status": "success", "nudity": {"safe": 1.0}})

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


def test_sightengine_success_without_moderation_scores_is_error(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=lambda *args, **kwargs: FakeResponse(200, data={"status": "success"})),
    )

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert "no recognized moderation scores" in (result.error or "")


def test_sightengine_rejects_non_finite_only_scores(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", "nudity": {"raw": float("nan")}})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR


def test_sightengine_clamps_scores_to_probability_range(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", "nudity": {"raw": 2.0, "safe": -1.0}})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["nudity_raw"] == 1.0
    assert result.scores["nudity_safe"] == 0.0


def test_sightengine_request_error_redacts_credentials_and_url_query(monkeypatch) -> None:
    secret = "sightengine-top-secret"
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", secret)

    def fail_post(*args, **kwargs):
        raise RuntimeError(f"failed {secret} at https://example.test/a?token=private")

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=fail_post))

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert secret not in (result.error or "")
    assert "private" not in (result.error or "")


@pytest.mark.parametrize("status_code", [402, 403, 429])
def test_sightengine_quota_disable_is_shared_only_within_run(monkeypatch, status_code: int) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    monkeypatch.setattr(SightengineEngine, "_SESSION_LOCAL", threading.local())
    responses = [
        FakeResponse(status_code),
        FakeResponse(200, data={"status": "success", "nudity": {"safe": 1.0}}),
    ]
    post_count = 0

    def post(*args, **kwargs):
        nonlocal post_count
        post_count += 1
        return responses.pop(0)

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=post))
    first_run = SightengineRunState()

    first = SightengineEngine(run_state=first_run).execute("first.png", _frame())
    same_run_new_instance = SightengineEngine(run_state=first_run).execute("second.png", _frame())
    separate_run = SightengineEngine(run_state=SightengineRunState()).execute("third.png", _frame())

    assert first.status == EngineStatus.SKIPPED
    assert same_run_new_instance.status == EngineStatus.SKIPPED
    assert f"http={status_code}" in (same_run_new_instance.error or "")
    assert separate_run.status == EngineStatus.OK
    assert post_count == 2
