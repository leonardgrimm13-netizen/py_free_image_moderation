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
from modimg.verdict import compute_verdict


class FakeResponse:
    def __init__(self, status_code: int, text: str = "", headers: dict[str, str] | None = None, data=None) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {"content-type": "application/json"}
        self._data = data
        self.closed = False

    def json(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data

    def close(self) -> None:
        self.closed = True


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
    response = FakeResponse(200, data=ValueError("bad json"))
    fake_requests = types.SimpleNamespace(post=lambda *a, **k: response)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert "invalid JSON response" in (result.error or "")
    assert response.closed is True


def test_sightengine_http_error_returns_error(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(500, text="oops", headers={"content-type": "text/plain"})
    fake_requests = types.SimpleNamespace(post=lambda *a, **k: response)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert result.error == "http error 500"
    assert response.closed is True


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
    response = FakeResponse(200, data={"status": "success"})
    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(post=lambda *args, **kwargs: response),
    )

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert "no recognized moderation scores" in (result.error or "")
    assert response.closed is True


@pytest.mark.parametrize("nudity", [{}, {"none": float("nan")}, {"sexual_activity": True}, {"unknown": 0.9}, {"suggestive_classes": {"test": float("inf")}}])
def test_sightengine_invalid_advanced_nudity_is_error(monkeypatch, nudity) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", "nudity": nudity})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert result.error
    assert response.closed is True


@pytest.mark.parametrize("nudity", [{"none": 0.0}, {"suggestive_classes": {"test": 0.0}}])
def test_sightengine_valid_zero_advanced_nudity_is_success(monkeypatch, nudity) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", "nudity": nudity})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores == {"nudity_safe": 0.0, "nudity_raw": 0.0, "nudity_partial": 0.0}


def test_sightengine_rejects_non_finite_only_scores(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", "nudity": {"raw": float("nan")}})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR


def test_sightengine_rejects_boolean_scores(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", "offensive": True})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert "invalid moderation probability" in (result.error or "")
    assert response.closed is True


def test_sightengine_rejects_scores_outside_probability_range(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", "nudity": {"raw": 2.0, "safe": -1.0}})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().run("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert response.closed is True


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), -0.1, 1.1, None, "0.9"])
@pytest.mark.parametrize(
    "model,field",
    [
        ("nudity", "raw"),
        ("nudity", "sexual_activity"),
        ("nudity", "suggestive_classes"),
        ("weapon", "firearm"),
        ("weapon", "classes"),
        ("weapon", "firearm_type"),
        ("gore", "prob"),
        ("violence", "physical_violence"),
        ("offensive", "classes"),
    ],
)
def test_sightengine_invalid_probability_is_error_even_with_valid_scores(monkeypatch, model, field, value) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    if field in {"classes", "suggestive_classes"}:
        value = {"firearm" if model == "weapon" else "suggestive": value}
    elif field == "firearm_type":
        value = {"animated": value}
    response = FakeResponse(200, data={"status": "success", "nudity": {"safe": 1.0}, model: {field: value}})
    # Keep the valid score in the same response even when the malformed model is nudity.
    response._data["gore"] = {"prob": 0.0} if model != "gore" else response._data["gore"]
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))
    frames = _frame()
    try:
        result = SightengineEngine().execute("dummy.png", frames)
    finally:
        for frame in frames:
            frame.close()

    assert result.status == EngineStatus.ERROR
    assert response.closed is True


def test_sightengine_probability_validation_allows_response_metadata(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(
        200,
        data={
            "status": "success",
            "operations": 5,
            "nudity": {"none": 1.0, "model_version": "2.1"},
            "gore": {"prob": 0.0, "model_version": "2.0", "request_id": "request-123"},
            "offensive": {"prob": 0.0, "model_version": 2, "request_id": "request-123"},
        },
    )
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["nudity_safe"] == 1.0
    assert result.scores["gore_prob"] == 0.0
    assert result.scores["offensive_max"] == 0.0
    assert result.scores["operations_used"] == 5.0


def test_sightengine_accepts_nested_firearm_gesture_schema(monkeypatch) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", "weapon": {"firearm": 0.1, "firearm_gesture": {"aiming": 0.2}}})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert result.scores["weapon_firearm_action_aiming"] == 0.2


@pytest.mark.parametrize(
    "models",
    [
        {"nudity": {"safe": 1.0, "sexual_activity": float("nan")}},
        {"nudity": {"none": 1.0, "suggestive_classes": float("nan")}},
        {"nudity": {"safe": 1.0}, "weapon": {"classes": ["invalid"]}},
        {"nudity": {"safe": 1.0}, "offensive": {"nazi": "0.99"}},
    ],
)
def test_sightengine_rejects_malformed_fields_that_used_to_be_ignored(monkeypatch, models) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    response = FakeResponse(200, data={"status": "success", **models})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))
    frames = _frame()
    try:
        result = SightengineEngine().execute("dummy.png", frames)
    finally:
        for frame in frames:
            frame.close()

    assert result.status == EngineStatus.ERROR
    assert compute_verdict([result]).label.value == "REVIEW"
    assert response.closed is True


@pytest.mark.parametrize("value", [None, True, [], "0.9"])
@pytest.mark.parametrize(
    "model,container",
    [
        ("nudity", None),
        ("weapon", None),
        ("gore", None),
        ("violence", None),
        ("offensive", None),
        ("nudity", "suggestive_classes"),
        ("weapon", "classes"),
        ("weapon", "firearm_type"),
        ("weapon", "firearm_action"),
        ("gore", "classes"),
        ("violence", "classes"),
        ("offensive", "classes"),
    ],
)
def test_sightengine_present_models_and_containers_cannot_be_silently_ignored(monkeypatch, model, container, value) -> None:
    if model == "nudity" and container == "suggestive_classes" and value == []:
        # Lists of nested class objects are a supported container shape.
        value = 0.5
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    models = {"nudity": {"safe": 1.0}, "gore": {"prob": 0.0}}
    models[model] = value if container is None else {container: value}
    response = FakeResponse(200, data={"status": "success", **models})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.ERROR
    assert response.closed is True


@pytest.mark.parametrize(
    "nudity,label",
    [
        ({"safe": 1.0, "sexual_activity": 0.99}, "BLOCK"),
        ({"raw": 0.0, "none": 0.0, "very_suggestive": 0.8}, "REVIEW"),
        ({"partial": 0.8, "safe": 0.0, "none": 1.0}, "REVIEW"),
        ({"safe": 1.0, "suggestive_classes": {"suggestive": 0.8}}, "REVIEW"),
    ],
)
def test_sightengine_combines_legacy_and_advanced_nudity_risks(monkeypatch, nudity, label) -> None:
    monkeypatch.setenv("SIGHTENGINE_USER", "user")
    monkeypatch.setenv("SIGHTENGINE_SECRET", "secret")
    monkeypatch.setenv("FINAL_REVIEW_THRESHOLD", "0.4")
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.85")
    from modimg.config import get_config
    get_config(reload=True)
    response = FakeResponse(200, data={"status": "success", "nudity": nudity})
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=lambda *args, **kwargs: response))

    result = SightengineEngine().execute("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert compute_verdict([result]).label.value == label


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
    quota_response = FakeResponse(status_code)
    success_response = FakeResponse(200, data={"status": "success", "nudity": {"safe": 1.0}})
    responses = [quota_response, success_response]
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
    assert quota_response.closed is True
    assert success_response.closed is True
