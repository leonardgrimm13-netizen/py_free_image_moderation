from __future__ import annotations

from PIL import Image

from modimg.enums import EngineStatus
from modimg.engines.openai_mod import OpenAIModerationEngine
from modimg.types import Frame


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


def test_openai_missing_api_key_is_skipped(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_DISABLE", "0")
    OpenAIModerationEngine._DISABLED_REASON = None

    result = OpenAIModerationEngine().execute("dummy.png", [Frame(idx=0, pil=Image.new("RGB", (2, 2)))])

    assert result.status == EngineStatus.SKIPPED
    assert result.error == "OPENAI_API_KEY not set"
