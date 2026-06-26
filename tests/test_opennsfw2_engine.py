from __future__ import annotations

from PIL import Image

from modimg.engines.opennsfw2_engine import OpenNSFW2Engine
from modimg.enums import EngineStatus
from modimg.types import EngineResult, Frame


def _frame() -> list[Frame]:
    return [Frame(idx=0, pil=Image.new("RGB", (2, 2), color=(1, 2, 3)))]


def test_opennsfw2_in_process_mode_defaults_to_enabled(monkeypatch) -> None:
    monkeypatch.delenv("OPENNSFW2_IN_PROCESS", raising=False)
    assert OpenNSFW2Engine._in_process_mode() == "1"

    monkeypatch.setenv("OPENNSFW2_IN_PROCESS", "0")
    assert OpenNSFW2Engine._in_process_mode() == "0"

    monkeypatch.setenv("OPENNSFW2_IN_PROCESS", "auto")
    assert OpenNSFW2Engine._in_process_mode() == "auto"

    monkeypatch.setenv("OPENNSFW2_IN_PROCESS", "invalid")
    assert OpenNSFW2Engine._in_process_mode() == "1"


def test_opennsfw2_run_uses_in_process_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENNSFW2_IN_PROCESS", raising=False)
    eng = OpenNSFW2Engine()
    calls: list[str] = []

    def fake_in_process(path, frames, start):
        calls.append("in_process")
        return EngineResult(name=eng.name, status=EngineStatus.OK, scores={"nsfw_probability": 0.1})

    def fake_subprocess(path, start):
        raise AssertionError("subprocess should not be used by default")

    monkeypatch.setattr(eng, "_predict_in_process", fake_in_process)
    monkeypatch.setattr(eng, "_predict_in_subprocess", fake_subprocess)

    result = eng.run("dummy.png", _frame())

    assert result.status == EngineStatus.OK
    assert calls == ["in_process"]


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
