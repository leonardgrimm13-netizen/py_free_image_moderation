from __future__ import annotations

import sys
import types

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
