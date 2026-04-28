from __future__ import annotations

from modimg.config import get_config
from modimg.pipeline import run_engines
from modimg.types import Engine, EngineResult


class SlowEngine(Engine):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, path, frames, max_api_frames=3):
        import time

        time.sleep(0.05)
        return EngineResult(name=self.name, status="ok", scores={"x": 1.0})


def test_get_config_disable_flag_parsing(monkeypatch) -> None:
    monkeypatch.setenv("OPENNSFW2_DISABLE", "1")
    cfg = get_config(reload=True)
    assert cfg.opennsfw2_disable is True


def test_run_engines_parallel_preserves_order(monkeypatch) -> None:
    monkeypatch.setenv("MODIMG_PARALLEL_ENGINES", "1")
    monkeypatch.setenv("MODIMG_PARALLEL_WORKERS", "2")
    get_config(reload=True)

    engines = [SlowEngine("a"), SlowEngine("b")]
    results = run_engines("dummy", [], engines)

    assert [r.name for r in results] == ["a", "b"]
