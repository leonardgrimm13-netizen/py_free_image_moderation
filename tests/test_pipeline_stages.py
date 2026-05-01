from __future__ import annotations

from pathlib import Path

from PIL import Image

from modimg.config import get_config
from modimg.enums import EngineStatus, VerdictLabel
from modimg.pipeline import build_main_engines, run_on_input
from modimg.types import Engine, EngineResult


class FakeEngine(Engine):
    def __init__(self, name: str, score: float = 0.0) -> None:
        super().__init__()
        self.name = name
        self.score = score
        self.ran = False

    def run(self, path, frames, max_api_frames=3):
        self.ran = True
        return EngineResult(name=self.name, status=EngineStatus.OK, scores={"nsfw_probability": self.score})


def _mk_img(tmp_path: Path) -> str:
    p = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color=(10, 10, 10)).save(p)
    return str(p)


def test_api_policy_default_is_always(monkeypatch) -> None:
    monkeypatch.delenv("API_POLICY", raising=False)
    assert get_config(reload=True).api_policy == "always"


def test_api_policy_aliases(monkeypatch) -> None:
    monkeypatch.setenv("API_POLICY", "review")
    assert get_config(reload=True).api_policy == "on_review"
    monkeypatch.setenv("API_POLICY", "uncertain")
    assert get_config(reload=True).api_policy == "on_review"
    monkeypatch.setenv("API_POLICY", "on-review")
    assert get_config(reload=True).api_policy == "on_review"
    monkeypatch.setenv("API_POLICY", "onreview")
    assert get_config(reload=True).api_policy == "on_review"
    monkeypatch.setenv("API_POLICY", "only-review")
    assert get_config(reload=True).api_policy == "on_review"
    monkeypatch.setenv("API_POLICY", "when_review")
    assert get_config(reload=True).api_policy == "on_review"
    monkeypatch.setenv("API_POLICY", "off")
    assert get_config(reload=True).api_policy == "never"
    monkeypatch.setenv("API_POLICY", "disabled")
    assert get_config(reload=True).api_policy == "never"
    monkeypatch.setenv("API_POLICY", "no-api")
    assert get_config(reload=True).api_policy == "never"
    monkeypatch.setenv("API_POLICY", "no-apis")
    assert get_config(reload=True).api_policy == "never"
    monkeypatch.setenv("API_POLICY", "noapis")
    assert get_config(reload=True).api_policy == "never"
    monkeypatch.setenv("API_POLICY", "false")
    assert get_config(reload=True).api_policy == "never"
    monkeypatch.setenv("API_POLICY", "0")
    assert get_config(reload=True).api_policy == "never"
    monkeypatch.setenv("API_POLICY", "all")
    assert get_config(reload=True).api_policy == "always"
    monkeypatch.setenv("API_POLICY", "bad-value")
    assert get_config(reload=True).api_policy == "always"


def test_build_main_engines_kept_for_backward_compatibility(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_DISABLE", "0")
    monkeypatch.setenv("SIGHTENGINE_DISABLE", "0")
    get_config(reload=True)
    names = [e.name for e in build_main_engines(no_apis=False)]
    assert names == [
        "OCR text",
        "NudeNet",
        "OpenNSFW2",
        "YOLO-World weapons",
        "OpenAI Moderation",
        "Sightengine",
    ]


def test_run_on_input_no_apis_skips_api_stage(monkeypatch, tmp_path) -> None:
    img = _mk_img(tmp_path)
    pre = FakeEngine("pre")
    local = FakeEngine("local")
    api = FakeEngine("api")
    monkeypatch.setattr("modimg.pipeline.build_pre_engines", lambda **kwargs: [pre])
    monkeypatch.setattr("modimg.pipeline.build_local_engines", lambda **kwargs: [local])
    monkeypatch.setattr("modimg.pipeline.build_api_engines", lambda **kwargs: [api])

    run_on_input(img, no_apis=True)
    assert local.ran is True
    assert api.ran is False


def test_api_policy_never_skips_api_stage(monkeypatch, tmp_path) -> None:
    img = _mk_img(tmp_path)
    monkeypatch.setenv("API_POLICY", "never")
    get_config(reload=True)
    pre = FakeEngine("pre")
    local = FakeEngine("local")
    api = FakeEngine("api")
    monkeypatch.setattr("modimg.pipeline.build_pre_engines", lambda **kwargs: [pre])
    monkeypatch.setattr("modimg.pipeline.build_local_engines", lambda **kwargs: [local])
    monkeypatch.setattr("modimg.pipeline.build_api_engines", lambda **kwargs: [api])

    run_on_input(img, no_apis=False)
    assert api.ran is False


def test_api_policy_on_review_runs_api_only_for_review(monkeypatch, tmp_path) -> None:
    img = _mk_img(tmp_path)
    monkeypatch.setenv("API_POLICY", "on_review")
    get_config(reload=True)
    pre = FakeEngine("pre", score=0.0)
    local = FakeEngine("local", score=0.0)
    api = FakeEngine("api", score=0.1)
    monkeypatch.setattr("modimg.pipeline.build_pre_engines", lambda **kwargs: [pre])
    monkeypatch.setattr("modimg.pipeline.build_local_engines", lambda **kwargs: [local])
    monkeypatch.setattr("modimg.pipeline.build_api_engines", lambda **kwargs: [api])

    def fake_compute(results):
        from modimg.types import Verdict

        if len(results) == 2:
            return Verdict(VerdictLabel.REVIEW, 0.0, 0.0, 0.0, ["review"])
        return Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, ["ok"])

    monkeypatch.setattr("modimg.pipeline.compute_verdict", fake_compute)

    run_on_input(img, no_apis=False)
    assert api.ran is True


def test_api_policy_on_review_skips_api_for_ok(monkeypatch, tmp_path) -> None:
    img = _mk_img(tmp_path)
    monkeypatch.setenv("API_POLICY", "on_review")
    get_config(reload=True)
    pre = FakeEngine("pre", score=0.0)
    local = FakeEngine("local", score=0.0)
    api = FakeEngine("api", score=0.1)
    monkeypatch.setattr("modimg.pipeline.build_pre_engines", lambda **kwargs: [pre])
    monkeypatch.setattr("modimg.pipeline.build_local_engines", lambda **kwargs: [local])
    monkeypatch.setattr("modimg.pipeline.build_api_engines", lambda **kwargs: [api])

    run_on_input(img, no_apis=False)
    assert api.ran is False


def test_api_policy_always_runs_api_even_if_local_ok(monkeypatch, tmp_path) -> None:
    img = _mk_img(tmp_path)
    monkeypatch.setenv("API_POLICY", "always")
    get_config(reload=True)
    pre = FakeEngine("pre", score=0.0)
    local = FakeEngine("local", score=0.0)
    api = FakeEngine("api", score=0.1)
    monkeypatch.setattr("modimg.pipeline.build_pre_engines", lambda **kwargs: [pre])
    monkeypatch.setattr("modimg.pipeline.build_local_engines", lambda **kwargs: [local])
    monkeypatch.setattr("modimg.pipeline.build_api_engines", lambda **kwargs: [api])

    out = run_on_input(img, no_apis=False)
    assert api.ran is True
    assert out["verdict"].label in (VerdictLabel.OK, VerdictLabel.REVIEW, VerdictLabel.BLOCK, "OK", "REVIEW", "BLOCK")


def test_phash_short_circuit_skips_local_and_api_stages(monkeypatch, tmp_path) -> None:
    img = _mk_img(tmp_path)
    monkeypatch.setenv("SHORT_CIRCUIT_PHASH", "1")
    monkeypatch.setenv("API_POLICY", "always")
    get_config(reload=True)

    class PreEngine(FakeEngine):
        def run(self, path, frames, max_api_frames=3):
            self.ran = True
            return EngineResult(
                name="pHash blocklist",
                status=EngineStatus.OK,
                scores={"phash_block_match": 1.0},
                details={"distance": 0},
            )

    pre = PreEngine("pre")
    local = FakeEngine("local", score=0.0)
    api = FakeEngine("api", score=0.0)
    monkeypatch.setattr("modimg.pipeline.build_pre_engines", lambda **kwargs: [pre])
    monkeypatch.setattr("modimg.pipeline.build_local_engines", lambda **kwargs: [local])
    monkeypatch.setattr("modimg.pipeline.build_api_engines", lambda **kwargs: [api])

    out = run_on_input(img, no_apis=False)
    assert local.ran is False
    assert api.ran is False
    assert out["verdict"].label == VerdictLabel.BLOCK
    assert len(out["results"]) == 1
    assert out["results"][0].name == "pHash blocklist"


def test_auto_learn_uses_final_api_verdict(monkeypatch, tmp_path) -> None:
    img = _mk_img(tmp_path)
    monkeypatch.setenv("API_POLICY", "always")
    get_config(reload=True)
    pre = FakeEngine("pre", score=0.0)
    local = FakeEngine("local", score=0.0)
    api = FakeEngine("api", score=0.9)
    monkeypatch.setattr("modimg.pipeline.build_pre_engines", lambda **kwargs: [pre])
    monkeypatch.setattr("modimg.pipeline.build_local_engines", lambda **kwargs: [local])
    monkeypatch.setattr("modimg.pipeline.build_api_engines", lambda **kwargs: [api])

    verdict_calls = {"count": 0}

    def fake_compute(results):
        from modimg.types import Verdict

        verdict_calls["count"] += 1
        if len(results) == 2 and [r.name for r in results] == ["pre", "local"]:
            return Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, ["local_ok"])
        if len(results) == 3 and [r.name for r in results] == ["pre", "local", "api"]:
            return Verdict(VerdictLabel.BLOCK, 1.0, 1.0, 1.0, ["api_block"])
        raise AssertionError("unexpected compute_verdict input")

    learn_labels = []

    def fake_auto_learn(verdict, frames):
        learn_labels.append(verdict.label)
        return "learned"

    monkeypatch.setattr("modimg.pipeline.compute_verdict", fake_compute)
    monkeypatch.setattr("modimg.pipeline.maybe_auto_learn", fake_auto_learn)

    out = run_on_input(img, no_apis=False)
    assert api.ran is True
    assert verdict_calls["count"] == 2
    assert len(learn_labels) == 1
    assert learn_labels[0] == VerdictLabel.BLOCK
    assert out["auto_learn"] == "learned"
    assert out["verdict"].label == VerdictLabel.BLOCK
