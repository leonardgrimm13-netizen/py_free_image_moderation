from __future__ import annotations

import json
import subprocess
import sys

from PIL import Image

from modimg.config import get_config
from modimg.pipeline import maybe_auto_learn, run_engines
from modimg.types import Engine, EngineResult
from modimg.verdict import compute_verdict
from modimg.enums import VerdictLabel
from modimg.types import Frame, Verdict


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


def test_maybe_auto_learn_processes_all_hashes(monkeypatch) -> None:
    monkeypatch.setenv("PHASH_AUTO_LEARN_ENABLE", "1")
    monkeypatch.setenv("PHASH_GIF_LEARN_FIRST_LAST", "1")
    monkeypatch.setenv("PHASH_AUTO_ALLOW_APPEND", "1")
    calls: list[str] = []

    monkeypatch.setattr("modimg.pipeline.frame_phash_hex_int", lambda fr: (f"h{fr.idx}", 1))

    def fake_append(hx: str, path: str, label: str) -> bool:
        calls.append(hx)
        return hx == "h0"

    monkeypatch.setattr("modimg.pipeline.append_phash_to_allowlist", fake_append)
    monkeypatch.setattr("modimg.pipeline.get_allowlist_path", lambda: "allow.txt")

    frames = [Frame(idx=0, pil=Image.new("RGB", (2, 2))), Frame(idx=5, pil=Image.new("RGB", (2, 2)))]
    verdict = Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, [])

    msg = maybe_auto_learn(verdict, frames)

    assert calls == ["h0", "h5"]
    assert msg == "Auto-added pHash to allowlist (allow.txt)"


def test_cli_json_serializes_enum_values(tmp_path) -> None:
    img_path = tmp_path / "sample.png"
    out_path = tmp_path / "report.json"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(img_path)

    proc = subprocess.run(
        [sys.executable, "-m", "modimg.cli", str(img_path), "--no-apis", "--json", str(out_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode in (0, 2)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data["verdict"]["label"], str)
    assert data["verdict"]["label"] in {"OK", "REVIEW", "BLOCK"}
    assert all(isinstance(r["status"], str) for r in data["results"])


def test_final_block_threshold_uses_config(monkeypatch) -> None:
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.95")
    get_config(reload=True)
    verdict = compute_verdict([EngineResult(name="OpenNSFW2", status="ok", scores={"nsfw_probability": 0.90})])
    assert verdict.label == VerdictLabel.REVIEW
