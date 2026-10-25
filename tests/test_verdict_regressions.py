from __future__ import annotations

import pytest
from PIL import Image

from modimg import config, pipeline
from modimg.enums import EngineStatus, VerdictLabel
from modimg.types import Engine, EngineResult
from modimg.verdict import compute_verdict


@pytest.fixture(autouse=True)
def verdict_environment(monkeypatch):
    monkeypatch.setattr(config, "_CURRENT_CONFIG", None)
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.85")
    monkeypatch.setenv("FINAL_REVIEW_THRESHOLD", "0.40")
    monkeypatch.setenv("ENGINE_ERROR_POLICY", "review")
    monkeypatch.setenv("CORE_ENGINES", "ocr")
    monkeypatch.setenv("PHASH_AUTO_LEARN_ENABLE", "0")


@pytest.mark.parametrize(
    ("policy", "threshold", "label", "risks"),
    [
        ("ignore", "0.20", VerdictLabel.OK, (0.1, 0.0, 0.0)),
        ("ignore", "0.40", VerdictLabel.OK, (0.1, 0.0, 0.0)),
        ("ignore", "0.60", VerdictLabel.OK, (0.1, 0.0, 0.0)),
        ("review", "0.20", VerdictLabel.REVIEW, (0.4, 0.4, 0.4)),
        ("review", "0.40", VerdictLabel.REVIEW, (0.4, 0.4, 0.4)),
        ("review", "0.60", VerdictLabel.REVIEW, (0.4, 0.4, 0.4)),
        ("review", "0.90", VerdictLabel.REVIEW, (0.4, 0.4, 0.4)),
        ("block", "0.20", VerdictLabel.BLOCK, (0.5, 0.5, 0.5)),
        ("block", "0.40", VerdictLabel.BLOCK, (0.5, 0.5, 0.5)),
        ("block", "0.60", VerdictLabel.BLOCK, (0.5, 0.5, 0.5)),
    ],
)
def test_core_error_policy_respects_configured_review_threshold(monkeypatch, policy, threshold, label, risks) -> None:
    monkeypatch.setenv("ENGINE_ERROR_POLICY", policy)
    monkeypatch.setenv("FINAL_REVIEW_THRESHOLD", threshold)

    verdict = compute_verdict(
        [
            EngineResult(name="OCR text", status=EngineStatus.ERROR, error="OCR failed"),
            EngineResult(name="OpenNSFW2", status=EngineStatus.OK, scores={"nsfw_probability": 0.1}),
        ]
    )

    assert verdict.label == label
    assert (verdict.nudity_risk, verdict.violence_risk, verdict.hate_risk) == risks


@pytest.mark.parametrize("policy", ["ignore", "review", "block"])
@pytest.mark.parametrize(
    ("name", "scores"),
    [
        ("pHash blocklist", {"phash_block_match": 1.0}),
        ("OCR text", {"ocr_match": 1.0}),
        ("OpenNSFW2", {"nsfw_probability": 0.9}),
    ],
)
def test_core_error_policy_preserves_content_blocks(monkeypatch, policy, name, scores) -> None:
    monkeypatch.setenv("ENGINE_ERROR_POLICY", policy)
    monkeypatch.setenv("FINAL_REVIEW_THRESHOLD", "0.60")

    verdict = compute_verdict(
        [
            EngineResult(name="OCR text", status=EngineStatus.ERROR, error="OCR failed"),
            EngineResult(name=name, status=EngineStatus.OK, scores=scores),
        ]
    )

    assert verdict.label == VerdictLabel.BLOCK


def test_review_error_policy_preserves_existing_risk_floor_block(monkeypatch) -> None:
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.35")
    monkeypatch.setenv("FINAL_REVIEW_THRESHOLD", "0.60")

    verdict = compute_verdict(
        [
            EngineResult(name="OCR text", status=EngineStatus.ERROR, error="OCR failed"),
            EngineResult(name="OpenNSFW2", status=EngineStatus.OK, scores={"nsfw_probability": 0.1}),
        ]
    )

    assert verdict.label == VerdictLabel.BLOCK
    assert (verdict.nudity_risk, verdict.violence_risk, verdict.hate_risk) == (0.4, 0.4, 0.4)


def test_on_review_api_policy_runs_after_core_error_with_raised_review_threshold(monkeypatch, tmp_path) -> None:
    path = tmp_path / "safe.png"
    Image.new("RGB", (2, 2), color=(30, 60, 90)).save(path)
    monkeypatch.setenv("FINAL_REVIEW_THRESHOLD", "0.60")
    monkeypatch.setenv("API_POLICY", "on_review")

    class ResultEngine(Engine):
        def __init__(self, result: EngineResult) -> None:
            super().__init__()
            self.name = result.name
            self.result = result

        def run(self, path, frames, max_api_frames=3):
            return self.result

    monkeypatch.setattr(pipeline, "build_pre_engines", lambda **kwargs: [])
    monkeypatch.setattr(
        pipeline,
        "build_local_engines",
        lambda **kwargs: [
            ResultEngine(EngineResult(name="OCR text", status=EngineStatus.ERROR, error="OCR failed")),
            ResultEngine(EngineResult(name="OpenNSFW2", status=EngineStatus.OK, scores={"nsfw_probability": 0.1})),
        ],
    )
    monkeypatch.setattr(
        pipeline,
        "build_api_engines",
        lambda **kwargs: [
            ResultEngine(EngineResult(name="OpenAI Moderation", status=EngineStatus.OK, scores={"violence": 0.95})),
        ],
    )

    report = pipeline.run_on_input(str(path))

    assert report["verdict"].label == VerdictLabel.BLOCK
    assert report["verdict"].violence_risk == 0.95
    assert [result.name for result in report["results"]] == ["OCR text", "OpenNSFW2", "OpenAI Moderation"]
