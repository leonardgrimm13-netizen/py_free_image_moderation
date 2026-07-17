from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from modimg import pipeline, svg as svg_module
from modimg.enums import EngineStatus, VerdictLabel
from modimg.types import Engine, EngineResult, Verdict


SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def _svg(*, body: str = '<rect width="48" height="24" fill="red"/>') -> bytes:
    return f'<svg xmlns="{SVG_NAMESPACE}" width="48" height="24">{body}</svg>'.encode()


class RecordingEngine(Engine):
    def __init__(self, name: str, seen: list[dict[str, Any]], *, expose_path: bool = False) -> None:
        super().__init__()
        self.name = name
        self._seen = seen
        self._expose_path = expose_path

    def run(self, path: str, frames, max_api_frames: int = 3) -> EngineResult:
        processing_path = Path(path)
        observation = {
            "path": path,
            "exists": processing_path.exists(),
            "suffix": processing_path.suffix,
            "frame_modes": [frame.pil.mode for frame in frames],
            "frame_sizes": [frame.pil.size for frame in frames],
        }
        self._seen.append(observation)
        details = {"internal_processing_path": path} if self._expose_path else {}
        return EngineResult(name=self.name, status=EngineStatus.OK, details=details)


def _install_pipeline_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pre_engines: list[Engine] | None = None,
    local_engines: list[Engine] | None = None,
    api_engines: list[Engine] | None = None,
    api_policy: str = "never",
) -> None:
    config = SimpleNamespace(
        parallel_engines=False,
        parallel_workers=2,
        short_circuit_phash=False,
        api_policy=api_policy,
        debug=False,
    )
    monkeypatch.setattr(pipeline, "get_config", lambda: config)
    monkeypatch.setattr(pipeline, "build_pre_engines", lambda **kwargs: list(pre_engines or []))
    monkeypatch.setattr(pipeline, "build_local_engines", lambda **kwargs: list(local_engines or []))
    monkeypatch.setattr(pipeline, "build_api_engines", lambda **kwargs: list(api_engines or []))
    monkeypatch.setattr(
        pipeline,
        "compute_verdict",
        lambda results: Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, []),
    )
    monkeypatch.setattr(pipeline, "maybe_auto_learn", lambda verdict, frames: None)


def _status_value(status: EngineStatus | str) -> str:
    return status.value if isinstance(status, EngineStatus) else str(status).lower()


def test_pipeline_passes_same_existing_normalized_png_to_every_engine_and_cleans_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.SVG"
    original.write_bytes(_svg())
    seen: list[dict[str, Any]] = []
    pre = RecordingEngine("pre-path-engine", seen)
    local = RecordingEngine("local-path-engine", seen)
    api = RecordingEngine("api-path-engine", seen)
    _install_pipeline_doubles(
        monkeypatch,
        pre_engines=[pre],
        local_engines=[local],
        api_engines=[api],
        api_policy="always",
    )

    report = pipeline.run_on_input(str(original), no_apis=False)

    assert len(seen) == 3
    assert all(observation["exists"] is True for observation in seen)
    assert all(observation["suffix"] == ".png" for observation in seen)
    assert all(observation["frame_modes"] == ["RGB"] for observation in seen)
    assert all(observation["frame_sizes"] == [(48, 24)] for observation in seen)
    assert len({observation["path"] for observation in seen}) == 1
    normalized_path = Path(seen[0]["path"])
    assert normalized_path != original
    assert normalized_path.exists() is False
    assert original.exists() is True

    assert report["name"] == str(original)
    assert report["path"] == str(original)
    assert report["preprocessing"] == {
        "source_format": "svg",
        "normalized_format": "png",
        "renderer": "resvg_py",
        "render_width": 48,
        "render_height": 24,
        "background": "#ffffff",
    }
    assert str(normalized_path) not in repr(report)


def test_pipeline_report_sanitizes_internal_temp_paths_exposed_by_an_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "image.svg"
    original.write_bytes(_svg())
    seen: list[dict[str, Any]] = []
    engine = RecordingEngine("path-reporting-engine", seen, expose_path=True)
    _install_pipeline_doubles(monkeypatch, local_engines=[engine])

    report = pipeline.run_on_input(str(original), no_apis=True)

    normalized_path = seen[0]["path"]
    result = report["results"][0]
    assert normalized_path not in repr(report)
    assert result.details["internal_processing_path"] == "<temporary-file>"
    assert Path(normalized_path).exists() is False


def test_svg_engine_error_redacts_normalized_path_from_report_and_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = tmp_path / "engine-error.svg"
    original.write_bytes(_svg())
    seen_paths: list[str] = []

    class FailingEngine(Engine):
        name = "failing-svg-path"

        def run(self, path: str, frames, max_api_frames: int = 3) -> EngineResult:
            seen_paths.append(path)
            assert Path(path).exists()
            raise RuntimeError(f"backend rejected {path}")

    _install_pipeline_doubles(monkeypatch, local_engines=[FailingEngine()])
    stdout_handler = next(
        handler
        for handler in logging.getLogger("modimg").handlers
        if isinstance(handler, logging.StreamHandler)
    )
    previous_stream = stdout_handler.setStream(sys.stdout)
    try:
        report = pipeline.run_on_input(str(original), no_apis=True)
        log_output = capsys.readouterr().out
    finally:
        stdout_handler.setStream(previous_stream)

    assert len(seen_paths) == 1
    normalized_path = seen_paths[0]
    assert Path(normalized_path).exists() is False
    assert normalized_path not in repr(report)
    assert normalized_path not in log_output
    assert "engine failed: RuntimeError: backend rejected <temporary-file>" in log_output
    assert report["results"][0].status == EngineStatus.ERROR
    assert report["results"][0].error == "RuntimeError: backend rejected <temporary-file>"
    assert report["preprocessing"]["source_format"] == "svg"


def test_pipeline_sanitizes_temp_paths_copied_into_every_public_result_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "echo-path.svg"
    original.write_bytes(_svg())
    seen_paths: list[str] = []

    class PathEchoEngine(Engine):
        name = "path-echo"

        def run(self, path: str, frames, max_api_frames: int = 3) -> EngineResult:
            seen_paths.append(path)
            return EngineResult(
                name=path,
                status=path,
                scores={path: 1.0},
                details={"match_label": path},
                error=path,
            )

    _install_pipeline_doubles(monkeypatch, local_engines=[PathEchoEngine()])
    monkeypatch.setattr(
        pipeline,
        "compute_verdict",
        lambda results: Verdict(results[0].details["match_label"], 0.0, 0.0, 0.0, [results[0].details["match_label"]]),
    )
    monkeypatch.setattr(pipeline, "maybe_auto_learn", lambda verdict, frames: seen_paths[0])

    report = pipeline.run_on_input(str(original), no_apis=True)

    normalized_path = seen_paths[0]
    assert normalized_path not in repr(report)
    assert report["verdict"].label == "<temporary-file>"
    assert report["verdict"].reasons == ["<temporary-file>"]
    assert report["results"][0].name == "<temporary-file>"
    assert report["results"][0].status == "<temporary-file>"
    assert report["results"][0].scores == {"<temporary-file>": 1.0}
    assert report["results"][0].error == "<temporary-file>"
    assert report["auto_learn"] == "<temporary-file>"
    assert Path(normalized_path).exists() is False


def test_pipeline_cleans_svg_png_when_an_engine_raises_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "interrupt.svg"
    original.write_bytes(_svg())
    seen_paths: list[str] = []

    class InterruptingEngine(Engine):
        name = "interrupting"

        def execute(self, path: str, frames, max_api_frames: int = 3) -> EngineResult:
            assert Path(path).exists()
            seen_paths.append(path)
            raise KeyboardInterrupt

    _install_pipeline_doubles(monkeypatch, local_engines=[InterruptingEngine()])

    with pytest.raises(KeyboardInterrupt):
        pipeline.run_on_input(str(original), no_apis=True)

    assert len(seen_paths) == 1
    assert Path(seen_paths[0]).exists() is False
    assert original.exists() is True


def test_pipeline_loader_failure_after_rasterization_is_review_and_cleans_png(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "decoder-failure.svg"
    original.write_bytes(_svg())
    processing_paths: list[str] = []

    def fail_load_frames(path: str, sample_frames: int):
        processing_paths.append(path)
        assert Path(path).exists()
        assert Path(path).suffix == ".png"
        raise ValueError("forced decoder failure")

    _install_pipeline_doubles(monkeypatch)
    monkeypatch.setattr(pipeline, "load_frames", fail_load_frames)

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert report["verdict"].label == VerdictLabel.REVIEW
    assert report["results"][0].name == "Loader"
    assert _status_value(report["results"][0].status) == EngineStatus.ERROR.value
    assert "forced decoder failure" in (report["results"][0].error or "")
    assert report["preprocessing"]["source_format"] == "svg"
    assert report["preprocessing"]["normalized_format"] == "png"
    assert len(processing_paths) == 1
    assert Path(processing_paths[0]).exists() is False
    assert processing_paths[0] not in repr(report)


@pytest.mark.parametrize(
    ("filename", "data", "error_fragment"),
    [
        ("malformed.svg", b"<svg>", "malformed XML"),
        ("wrong-root.svg", b"<html></html>", "root element is not svg"),
        (
            "script.svg",
            f'<svg xmlns="{SVG_NAMESPACE}"><script>alert(1)</script></svg>'.encode(),
            "script elements are not allowed",
        ),
        (
            "external.svg",
            f'<svg xmlns="{SVG_NAMESPACE}"><image href="file:///etc/passwd"/></svg>'.encode(),
            "external resource reference is not allowed",
        ),
    ],
)
def test_invalid_svg_becomes_controlled_loader_review_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
    data: bytes,
    error_fragment: str,
) -> None:
    original = tmp_path / filename
    original.write_bytes(data)
    _install_pipeline_doubles(monkeypatch)

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert report["path"] == str(original)
    assert report["verdict"].label == VerdictLabel.REVIEW
    assert len(report["results"]) == 1
    result = report["results"][0]
    assert result.name == "Loader"
    assert _status_value(result.status) == EngineStatus.ERROR.value
    assert error_fragment in (result.error or "")
    assert "Traceback" not in (result.error or "")


def test_url_svg_download_and_normalized_png_are_both_cleaned_while_url_is_preserved_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloaded = tmp_path / "downloaded-svg"
    downloaded.write_bytes(_svg())
    seen: list[dict[str, Any]] = []
    engine = RecordingEngine("url-path-engine", seen)
    _install_pipeline_doubles(monkeypatch, local_engines=[engine])
    monkeypatch.setattr(pipeline, "is_url", lambda value: True)
    monkeypatch.setattr(
        pipeline,
        "download_url_to_temp",
        lambda value, **kwargs: (str(downloaded), "asset.svg"),
    )
    original_url = "https://example.test/assets/asset.svg?token=top-secret#fragment"

    report = pipeline.run_on_input(original_url, no_apis=True)

    normalized_path = seen[0]["path"]
    assert seen[0]["suffix"] == ".png"
    assert seen[0]["exists"] is True
    assert downloaded.exists() is False
    assert Path(normalized_path).exists() is False
    assert report["name"] == "asset.svg"
    assert report["path"] == "https://example.test/assets/asset.svg?<redacted>"
    assert "top-secret" not in repr(report)
    assert "fragment" not in repr(report)
    assert str(downloaded) not in repr(report)
    assert normalized_path not in repr(report)
    assert report["preprocessing"]["source_format"] == "svg"
    assert report["preprocessing"]["normalized_format"] == "png"


def test_url_svg_download_is_cleaned_when_svg_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloaded = tmp_path / "downloaded.svg"
    downloaded.write_bytes(b"<svg>")
    _install_pipeline_doubles(monkeypatch)
    monkeypatch.setattr(pipeline, "is_url", lambda value: True)
    monkeypatch.setattr(
        pipeline,
        "download_url_to_temp",
        lambda value, **kwargs: (str(downloaded), "downloaded.svg"),
    )

    report = pipeline.run_on_input("https://example.test/downloaded.svg", no_apis=True)

    assert downloaded.exists() is False
    assert report["verdict"].label == VerdictLabel.REVIEW
    assert _status_value(report["results"][0].status) == EngineStatus.ERROR.value
    assert "malformed XML" in (report["results"][0].error or "")


@pytest.mark.parametrize("mode", ["renderer-error", "invalid-png", "missing-renderer"])
def test_renderer_failures_become_controlled_pipeline_loader_reviews(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    original = tmp_path / f"{mode}.svg"
    original.write_bytes(_svg())
    _install_pipeline_doubles(monkeypatch)
    real_import_module = svg_module.importlib.import_module

    if mode == "renderer-error":
        def svg_to_bytes(**kwargs: object) -> bytes:
            raise ValueError("renderer private failure")

        replacement: object = SimpleNamespace(svg_to_bytes=svg_to_bytes)
        expected = "SVG rendering failed"
    elif mode == "invalid-png":
        replacement = SimpleNamespace(svg_to_bytes=lambda **kwargs: b"not a png")
        expected = "invalid PNG"
    else:
        replacement = ModuleNotFoundError("No module named 'resvg_py'")
        expected = "renderer unavailable"

    def import_module(name: str):
        if name == "resvg_py":
            if isinstance(replacement, BaseException):
                raise replacement
            return replacement
        return real_import_module(name)

    monkeypatch.setattr(svg_module.importlib, "import_module", import_module)

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert report["verdict"].label == VerdictLabel.REVIEW
    assert _status_value(report["results"][0].status) == EngineStatus.ERROR.value
    assert expected in (report["results"][0].error or "")
    assert "Traceback" not in (report["results"][0].error or "")


def test_extensionless_svg_pipeline_preserves_original_path_and_still_normalizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "extensionless"
    original.write_bytes(b"\xef\xbb\xbf<!-- before root -->" + _svg())
    seen: list[dict[str, Any]] = []
    _install_pipeline_doubles(monkeypatch, local_engines=[RecordingEngine("extensionless-engine", seen)])

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert seen[0]["suffix"] == ".png"
    assert seen[0]["exists"] is True
    assert Path(seen[0]["path"]).exists() is False
    assert report["path"] == str(original)
    assert report["name"] == str(original)
    assert report["preprocessing"]["source_format"] == "svg"


def test_normal_raster_processing_path_and_report_remain_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from PIL import Image

    original = tmp_path / "ordinary.png"
    Image.new("RGB", (7, 5), color="blue").save(original)
    seen: list[dict[str, Any]] = []
    _install_pipeline_doubles(monkeypatch, local_engines=[RecordingEngine("raster-engine", seen)])

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert seen[0]["path"] == str(original)
    assert seen[0]["exists"] is True
    assert seen[0]["suffix"] == ".png"
    assert seen[0]["frame_modes"] == ["RGB"]
    assert seen[0]["frame_sizes"] == [(7, 5)]
    assert original.exists() is True
    assert report["path"] == str(original)
    assert "preprocessing" not in report
