from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
import types

import pytest
from PIL import Image

from modimg.enums import EngineStatus, VerdictLabel
from modimg.engines import openai_mod
from modimg.engines.openai_mod import OpenAIModerationEngine, OpenAIRunState
from modimg.types import Frame
from modimg.verdict import compute_verdict


def _reset_cache_state() -> None:
    OpenAIModerationEngine._CACHE = None
    OpenAIModerationEngine._CACHE_PATH = None
    OpenAIModerationEngine._CACHE_DIR_READY = False
    OpenAIModerationEngine._CACHE_DIRTY = False
    OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH = 0
    OpenAIModerationEngine._CACHE_DIR_ERROR = False
    OpenAIModerationEngine._CACHE_DIR_ERROR_REASON = None
    OpenAIModerationEngine._CACHE_DIR_ERROR_TIME = 0.0
    OpenAIModerationEngine._CACHE_DIR_RETRY_DELAY = 2.0
    OpenAIModerationEngine._CLIENT = None
    OpenAIModerationEngine._CLIENT_TIMEOUT = None
    OpenAIModerationEngine._CLIENT_API_KEY = None


@pytest.fixture(autouse=True)
def _isolate_openai_class_state():
    _reset_cache_state()
    yield
    _reset_cache_state()


def _cache_key_for(engine: OpenAIModerationEngine, frame: Frame, model: str = "omni-moderation-latest") -> str:
    return engine._cache_key(model, [frame])


def _valid_cache_entry(model: str = "omni-moderation-latest") -> dict:
    return {
        "schema_version": 1,
        "model": model,
        "scores": {"sexual": 0.2, "max_any_category": 0.2, "flagged": 0.0},
        "details": {"categories": {"sexual": False}},
    }


def _valid_api_response(score: float = 0.1) -> dict:
    return {
        "results": [
            {
                "flagged": False,
                "categories": {"sexual": False},
                "category_scores": {"sexual": score},
            }
        ]
    }


def test_openai_cache_save_is_reentrant_under_cache_lock(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(tmp_path / "openai_cache.json"))

    OpenAIModerationEngine._CACHE = {"k": {"scores": {}, "details": {}}}
    OpenAIModerationEngine._CACHE_PATH = None
    OpenAIModerationEngine._CACHE_DIR_READY = False
    OpenAIModerationEngine._CACHE_DIRTY = True
    OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH = 1

    eng = OpenAIModerationEngine()

    with OpenAIModerationEngine._CACHE_LOCK:
        eng._save_cache(force=True)

    assert (tmp_path / "openai_cache.json").exists()


def test_openai_load_cache_reloads_after_cache_path_change(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"old": {"scores": {"x": 1.0}, "details": {}}}', encoding="utf-8")
    second.write_text('{"new": {"scores": {"y": 2.0}, "details": {}}}', encoding="utf-8")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(first))
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    _reset_cache_state()

    eng = OpenAIModerationEngine()
    first_cache = eng._load_cache()
    assert "old" in first_cache
    assert "new" not in first_cache
    assert OpenAIModerationEngine._CACHE_PATH == str(first)

    OpenAIModerationEngine._CACHE_DIR_READY = True
    OpenAIModerationEngine._CACHE_DIRTY = True
    OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH = 1

    monkeypatch.setenv("OPENAI_CACHE_PATH", str(second))

    second_cache = eng._load_cache()

    assert second_cache == {"new": {"scores": {"y": 2.0}, "details": {}}}
    assert "old" not in second_cache
    assert OpenAIModerationEngine._CACHE_PATH == str(second)
    assert OpenAIModerationEngine._CACHE_DIR_READY is False
    assert OpenAIModerationEngine._CACHE_DIRTY is False
    assert OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH == 0


def test_openai_missing_api_key_is_skipped(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_DISABLE", "0")

    result = OpenAIModerationEngine().execute("dummy.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])

    assert result.status == EngineStatus.SKIPPED
    assert result.error == "OPENAI_API_KEY not set"


def test_openai_client_is_reused_until_timeout_or_key_changes(monkeypatch) -> None:
    class FakeOpenAI:
        instances: list["FakeOpenAI"] = []

        def __init__(self, timeout):
            self.timeout = timeout
            FakeOpenAI.instances.append(self)

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.setenv("OPENAI_API_KEY", "key-1")
    _reset_cache_state()

    first = OpenAIModerationEngine._client_for_timeout(10.0)
    second = OpenAIModerationEngine._client_for_timeout(10.0)

    assert first is second
    assert len(FakeOpenAI.instances) == 1

    third = OpenAIModerationEngine._client_for_timeout(20.0)
    assert third is not first
    assert len(FakeOpenAI.instances) == 2

    monkeypatch.setenv("OPENAI_API_KEY", "key-2")
    fourth = OpenAIModerationEngine._client_for_timeout(20.0)
    assert fourth is not third
    assert len(FakeOpenAI.instances) == 3


def test_openai_cache_ignores_non_object_root(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text('[{"unexpected": true}]', encoding="utf-8")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(cache_path))
    _reset_cache_state()

    assert OpenAIModerationEngine()._load_cache() == {}


def test_openai_cache_ignores_corrupt_json(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text('{"broken":', encoding="utf-8")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(cache_path))
    _reset_cache_state()

    assert OpenAIModerationEngine()._load_cache() == {}


def test_openai_cache_size_limit_is_enforced(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "cache.json"
    cache_path.write_text('{"too-large": "value"}', encoding="utf-8")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("OPENAI_CACHE_MAX_BYTES", "8")
    _reset_cache_state()

    assert OpenAIModerationEngine()._load_cache() == {}


def test_openai_valid_cache_hit_needs_no_key_package_or_client(monkeypatch, tmp_path) -> None:
    _reset_cache_state()
    frame = Frame(idx=0, pil=Image.new("RGB", (2, 2)))
    run_state = OpenAIRunState()
    run_state.disable("disabled earlier in this run")
    engine = OpenAIModerationEngine(run_state=run_state)
    cache_key = _cache_key_for(engine, frame)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({cache_key: _valid_cache_entry()}), encoding="utf-8")
    monkeypatch.setenv("OPENAI_DISABLE", "0")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(cache_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(engine, "available", lambda: pytest.fail("availability must not be checked on a cache hit"))
    monkeypatch.setattr(engine, "_client_for_timeout", lambda timeout: pytest.fail("client must not be created on a cache hit"))

    result = engine.execute("dummy.png", [frame])

    assert result.status == EngineStatus.OK
    assert result.scores["sexual"] == 0.2
    assert result.details["cache_hit"] is True


def test_openai_auth_disable_is_scoped_to_one_run(monkeypatch) -> None:
    class AuthError(RuntimeError):
        status_code = 401

    _reset_cache_state()
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace())
    monkeypatch.setenv("OPENAI_DISABLE", "0")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "0")
    monkeypatch.setenv("OPENAI_MIN_INTERVAL_SEC", "0")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")
    monkeypatch.setenv("OPENAI_API_KEY", "first-key")

    first_state = OpenAIRunState()
    first_engine = OpenAIModerationEngine(run_state=first_state)
    first_calls = 0

    def auth_failure(**kwargs):
        nonlocal first_calls
        first_calls += 1
        raise AuthError("invalid api key")

    monkeypatch.setattr(
        first_engine,
        "_client_for_timeout",
        lambda timeout: types.SimpleNamespace(moderations=types.SimpleNamespace(create=auth_failure)),
    )
    first_result = first_engine.execute("first.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])

    monkeypatch.setenv("OPENAI_API_KEY", "replacement-key")
    same_run_engine = OpenAIModerationEngine(run_state=first_state)
    monkeypatch.setattr(
        same_run_engine,
        "_client_for_timeout",
        lambda timeout: pytest.fail("disabled run must not create another client"),
    )
    same_run_result = same_run_engine.execute("second.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])

    second_state = OpenAIRunState()
    second_engine = OpenAIModerationEngine(run_state=second_state)
    second_calls = 0

    def success(**kwargs):
        nonlocal second_calls
        second_calls += 1
        return _valid_api_response()

    monkeypatch.setattr(
        second_engine,
        "_client_for_timeout",
        lambda timeout: types.SimpleNamespace(moderations=types.SimpleNamespace(create=success)),
    )
    second_result = second_engine.execute("third.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])

    assert first_result.status == EngineStatus.SKIPPED
    assert same_run_result.status == EngineStatus.SKIPPED
    assert second_result.status == EngineStatus.OK
    assert first_calls == 1
    assert second_calls == 1
    assert first_state.disabled_reason
    assert second_state.disabled_reason is None


def test_openai_parallel_files_share_run_auth_disable(monkeypatch) -> None:
    class AuthError(RuntimeError):
        status_code = 403

    _reset_cache_state()
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace())
    monkeypatch.setenv("OPENAI_API_KEY", "shared-run-key")
    monkeypatch.setenv("OPENAI_DISABLE", "0")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "0")
    monkeypatch.setenv("OPENAI_MIN_INTERVAL_SEC", "0")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")

    run_state = OpenAIRunState()
    auth_engine = OpenAIModerationEngine(run_state=run_state)
    later_engine = OpenAIModerationEngine(run_state=run_state)
    request_started = threading.Event()
    release_failure = threading.Event()
    auth_finished = threading.Event()
    later_client_calls = 0

    def auth_failure(**kwargs):
        request_started.set()
        assert release_failure.wait(timeout=2)
        raise AuthError("forbidden")

    def later_client(timeout):
        nonlocal later_client_calls
        later_client_calls += 1
        return types.SimpleNamespace(
            moderations=types.SimpleNamespace(create=lambda **kwargs: _valid_api_response())
        )

    monkeypatch.setattr(
        auth_engine,
        "_client_for_timeout",
        lambda timeout: types.SimpleNamespace(moderations=types.SimpleNamespace(create=auth_failure)),
    )
    monkeypatch.setattr(later_engine, "_client_for_timeout", later_client)

    def run_auth_file():
        try:
            return auth_engine.execute("auth.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])
        finally:
            auth_finished.set()

    def run_later_file():
        assert request_started.wait(timeout=2)
        release_failure.set()
        assert auth_finished.wait(timeout=2)
        return later_engine.execute("later.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])

    with ThreadPoolExecutor(max_workers=2) as executor:
        auth_future = executor.submit(run_auth_file)
        later_future = executor.submit(run_later_file)
        auth_result = auth_future.result(timeout=3)
        later_result = later_future.result(timeout=3)

    assert auth_result.status == EngineStatus.SKIPPED
    assert later_result.status == EngineStatus.SKIPPED
    assert later_client_calls == 0
    assert run_state.disabled_reason


@pytest.mark.parametrize(
    ("platform", "environment", "relative_root"),
    [
        ("linux", {"XDG_CACHE_HOME": "xdg-cache"}, ("xdg-cache",)),
        ("win32", {"LOCALAPPDATA": "local-app-data"}, ("local-app-data",)),
        ("darwin", {}, ("home", "Library", "Caches")),
    ],
)
def test_openai_default_cache_path_uses_platform_user_cache(
    monkeypatch,
    tmp_path,
    platform,
    environment,
    relative_root,
) -> None:
    _reset_cache_state()
    monkeypatch.delenv("OPENAI_CACHE_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(openai_mod.sys, "platform", platform)
    for key, relative in environment.items():
        monkeypatch.setenv(key, str(tmp_path / relative))

    expected = tmp_path.joinpath(*relative_root, "py-free-image-moderation", "openai_moderation_cache.json")

    assert OpenAIModerationEngine()._cache_path() == str(expected)


def test_openai_default_cache_path_uses_home_when_xdg_is_unset(monkeypatch, tmp_path) -> None:
    _reset_cache_state()
    monkeypatch.delenv("OPENAI_CACHE_PATH", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(openai_mod.sys, "platform", "linux")

    expected = tmp_path / "home" / ".cache" / "py-free-image-moderation" / "openai_moderation_cache.json"

    assert OpenAIModerationEngine()._cache_path() == str(expected)


def test_openai_explicit_relative_cache_path_keeps_project_root_semantics(monkeypatch, tmp_path) -> None:
    _reset_cache_state()
    project = tmp_path / "project"
    monkeypatch.setenv("OPENAI_CACHE_PATH", "custom/cache.json")
    monkeypatch.setattr(openai_mod, "project_root", lambda: str(project))

    assert OpenAIModerationEngine()._cache_path() == str(project / "custom" / "cache.json")


@pytest.mark.parametrize(
    "entry",
    [
        {"scores": {"unknown/category": 0.9, "flagged": 0.0}, "details": {}},
        {"scores": {"sexual": float("nan"), "flagged": 0.0}, "details": {}},
        {"scores": {"sexual": "0.2", "flagged": 0.0}, "details": {}},
        {"scores": {"sexual": 1.2, "flagged": 0.0}, "details": {}},
        {"scores": {"sexual": 0.2, "flagged": 0.5}, "details": {}},
        {"scores": {"sexual": 0.2}, "details": {}},
        {"scores": {"sexual": 0.2, "flagged": 0.0}, "details": []},
        {"schema_version": 999, "scores": {"sexual": 0.2, "flagged": 0.0}, "details": {}},
        {"model": "different-model", "scores": {"sexual": 0.2, "flagged": 0.0}, "details": {}},
        {"scores": [0.2], "details": {}},
    ],
)
def test_openai_invalid_cache_entries_are_removed(monkeypatch, tmp_path, entry) -> None:
    _reset_cache_state()
    frame = Frame(idx=0, pil=Image.new("RGB", (2, 2)))
    engine = OpenAIModerationEngine()
    cache_key = _cache_key_for(engine, frame)
    monkeypatch.setenv("OPENAI_DISABLE", "0")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(tmp_path / "cache.json"))
    OpenAIModerationEngine._CACHE = {cache_key: entry}
    monkeypatch.setattr(engine, "available", lambda: (False, "offline"))

    result = engine.execute("dummy.png", [frame])

    assert result.status == EngineStatus.SKIPPED
    assert result.error == "offline"
    assert cache_key not in OpenAIModerationEngine._CACHE


def test_openai_cache_write_evicts_oldest_entries_to_byte_limit(monkeypatch, tmp_path) -> None:
    _reset_cache_state()
    cache_path = tmp_path / "cache.json"
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(cache_path))
    monkeypatch.setenv("OPENAI_CACHE_MAX_BYTES", "420")
    monkeypatch.setenv("OPENAI_CACHE_MAX_ITEMS", "0")
    padded_entries = {
        key: {
            **_valid_cache_entry(),
            "details": {"padding": key * 140},
        }
        for key in ("a", "b", "c")
    }
    OpenAIModerationEngine._CACHE = padded_entries
    OpenAIModerationEngine._CACHE_DIRTY = True

    OpenAIModerationEngine()._save_cache(force=True)

    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache_path.stat().st_size <= 420
    assert list(persisted) == ["c"]
    assert list(OpenAIModerationEngine._CACHE) == ["c"]


def _run_openai_with(monkeypatch, *, response=None, error: Exception | None = None):
    class Moderations:
        @staticmethod
        def create(**kwargs):
            if error is not None:
                raise error
            return response

    engine = OpenAIModerationEngine()
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-secret")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "0")
    monkeypatch.setenv("OPENAI_MIN_INTERVAL_SEC", "0")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")
    monkeypatch.setattr(engine, "available", lambda: (True, ""))
    monkeypatch.setattr(
        engine,
        "_client_for_timeout",
        lambda timeout: types.SimpleNamespace(moderations=Moderations()),
    )
    return engine.run("dummy.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])


def test_openai_ordinary_request_error_is_engine_error_and_redacted(monkeypatch) -> None:
    _reset_cache_state()
    result = _run_openai_with(
        monkeypatch,
        error=RuntimeError("request with test-openai-secret failed"),
    )

    assert result.status == EngineStatus.ERROR
    assert "test-openai-secret" not in (result.error or "")
    assert compute_verdict([result]).label == VerdictLabel.REVIEW


def test_openai_malformed_success_response_is_error(monkeypatch) -> None:
    _reset_cache_state()
    result = _run_openai_with(monkeypatch, response={"results": []})

    assert result.status == EngineStatus.ERROR
    assert "did not contain a moderation result" in (result.error or "")


def test_openai_non_finite_score_is_error(monkeypatch) -> None:
    _reset_cache_state()
    response = {
        "results": [
            {
                "flagged": False,
                "categories": {},
                "category_scores": {"sexual": float("nan")},
            }
        ]
    }

    result = _run_openai_with(monkeypatch, response=response)

    assert result.status == EngineStatus.ERROR
    assert "non-finite score" in (result.error or "")


def test_openai_cache_write_failure_does_not_discard_valid_result(monkeypatch, tmp_path) -> None:
    _reset_cache_state()
    response = {
        "results": [
            {
                "flagged": False,
                "categories": {"sexual": False},
                "category_scores": {"sexual": 0.1},
            }
        ]
    }
    engine = OpenAIModerationEngine()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "1")
    monkeypatch.setenv("OPENAI_CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setenv("OPENAI_MIN_INTERVAL_SEC", "0")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")
    monkeypatch.setattr(OpenAIModerationEngine, "_CACHE_FLUSH_EVERY_N", 1)
    monkeypatch.setattr(engine, "available", lambda: (True, ""))
    monkeypatch.setattr(
        engine,
        "_client_for_timeout",
        lambda timeout: types.SimpleNamespace(
            moderations=types.SimpleNamespace(create=lambda **kwargs: response)
        ),
    )

    def fail_write(*args, **kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr("modimg.engines.openai_mod.atomic_write_text", fail_write)

    result = engine.run("dummy.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])

    assert result.status == EngineStatus.OK
    assert result.scores["sexual"] == 0.1
    assert OpenAIModerationEngine._CACHE_DIR_ERROR is True
