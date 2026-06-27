from __future__ import annotations

import sys
import types

from PIL import Image

from modimg.enums import EngineStatus
from modimg.engines.openai_mod import OpenAIModerationEngine
from modimg.types import Frame


def _reset_cache_state() -> None:
    OpenAIModerationEngine._CACHE = None
    OpenAIModerationEngine._CACHE_PATH = None
    OpenAIModerationEngine._CACHE_DIR_READY = False
    OpenAIModerationEngine._CACHE_DIRTY = False
    OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH = 0
    OpenAIModerationEngine._CACHE_DIR_ERROR = False
    OpenAIModerationEngine._CACHE_DIR_ERROR_REASON = None
    OpenAIModerationEngine._CLIENT = None
    OpenAIModerationEngine._CLIENT_TIMEOUT = None
    OpenAIModerationEngine._CLIENT_API_KEY = None


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
    OpenAIModerationEngine._DISABLED_REASON = None

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
