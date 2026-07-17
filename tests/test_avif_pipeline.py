from __future__ import annotations

import base64
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from modimg import pipeline
from modimg.engines import forbidden_symbols_yolo, yolo_weapons
from modimg.engines.forbidden_symbols_yolo import YOLOForbiddenSymbolsEngine
from modimg.engines.nudenet_engine import NudeNetEngine
from modimg.engines.ocr import OCREngine
from modimg.engines.openai_mod import OpenAIModerationEngine
from modimg.engines.opennsfw2_engine import OpenNSFW2Engine
from modimg.engines.phash_allow import PHashAllowlistEngine
from modimg.engines.phash_block import PHashBlocklistEngine
from modimg.engines.sightengine import SightengineEngine
from modimg.engines.yolo_weapons import YOLOWorldWeaponsEngine
from modimg.enums import EngineStatus, VerdictLabel
from modimg.phash import frame_phash_hex_int
from modimg.types import Engine, EngineResult, Frame, Verdict


def _make_avif(path: Path, *, size: tuple[int, int] = (18, 12), color: tuple[int, int, int] = (30, 80, 160)) -> None:
    with Image.new("RGB", size, color=color) as image:
        image.save(path, format="AVIF", quality=90)


def _install_pipeline_doubles(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pre_engines: list[Engine] | None = None,
    local_engines: list[Engine] | None = None,
    api_engines: list[Engine] | None = None,
    api_policy: str = "never",
    parallel_engines: bool = False,
) -> None:
    config = SimpleNamespace(
        parallel_engines=parallel_engines,
        parallel_workers=4,
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


class _NativeFrameEngine(Engine):
    def __init__(self, name: str, observations: list[dict[str, Any]]) -> None:
        super().__init__()
        self.name = name
        self._observations = observations

    def run(self, path: str, frames: list[Frame], max_api_frames: int = 3) -> EngineResult:
        self._observations.append(
            {
                "path": path,
                "modes": [frame.pil.mode for frame in frames],
                "sizes": [frame.pil.size for frame in frames],
                "formats": [frame.source_format for frame in frames],
                "fallbacks": [frame.temporary_file_paths() for frame in frames],
            }
        )
        return EngineResult(name=self.name, status=EngineStatus.OK)


def test_avif_pipeline_native_decode_keeps_original_path_without_eager_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "extension-is-wrong.bin"
    _make_avif(original)
    observations: list[dict[str, Any]] = []
    engines = [_NativeFrameEngine(stage, observations) for stage in ("pre", "local", "api")]
    _install_pipeline_doubles(
        monkeypatch,
        pre_engines=[engines[0]],
        local_engines=[engines[1]],
        api_engines=[engines[2]],
        api_policy="always",
    )

    def unexpected_jpeg_encoding(image: Image.Image, quality: int = 90) -> bytes:
        raise AssertionError("native AVIF frame engines must not create a JPEG fallback")

    monkeypatch.setattr("modimg.utils.pil_to_jpeg_bytes", unexpected_jpeg_encoding)

    report = pipeline.run_on_input(str(original), no_apis=False)

    assert len(observations) == 3
    assert all(item["path"] == str(original) for item in observations)
    assert all(item["modes"] == ["RGB"] for item in observations)
    assert all(item["sizes"] == [(18, 12)] for item in observations)
    assert all(item["formats"] == ["avif"] for item in observations)
    assert all(item["fallbacks"] == [()] for item in observations)
    assert original.exists()
    assert report["name"] == str(original)
    assert report["path"] == str(original)
    assert report["preprocessing"] == {"source_format": "avif", "native_decode": True}
    assert "engine_fallback" not in repr(report)


def test_avif_pipeline_opens_and_decodes_source_only_once_without_path_engines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "decode-once.avif"
    _make_avif(original)
    observations: list[dict[str, Any]] = []
    _install_pipeline_doubles(monkeypatch, local_engines=[_NativeFrameEngine("native", observations)])
    real_open = Image.open
    source_open_count = 0

    def counted_open(fp: Any, *args: Any, **kwargs: Any):
        nonlocal source_open_count
        if str(fp) == str(original):
            source_open_count += 1
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr("modimg.frames.Image.open", counted_open)

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert source_open_count == 1
    assert observations[0]["fallbacks"] == [()]
    assert report["preprocessing"] == {"source_format": "avif", "native_decode": True}


def test_disabled_path_engines_do_not_trigger_avif_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "disabled-path-engines.avif"
    _make_avif(original)
    monkeypatch.setenv("NUDENET_DISABLE", "1")
    monkeypatch.setenv("OPENNSFW2_DISABLE", "1")
    _install_pipeline_doubles(
        monkeypatch,
        local_engines=[NudeNetEngine(), OpenNSFW2Engine()],
    )

    def unexpected_fallback(self: Frame) -> str:
        raise AssertionError("disabled path engines must not request a fallback")

    monkeypatch.setattr(Frame, "get_jpeg_path", unexpected_fallback)

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert [result.status for result in report["results"]] == [EngineStatus.SKIPPED, EngineStatus.SKIPPED]
    assert report["preprocessing"] == {"source_format": "avif", "native_decode": True}
    assert original.exists()


def test_parallel_avif_path_engines_share_one_atomic_jpeg_and_report_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "parallel.avif"
    _make_avif(original, size=(22, 14), color=(120, 30, 60))
    start = threading.Barrier(2)
    observations: list[dict[str, Any]] = []
    observation_lock = threading.Lock()
    encode_count = 0
    encode_lock = threading.Lock()

    from modimg import utils

    real_encode = utils.pil_to_jpeg_bytes

    def counted_encode(image: Image.Image, quality: int = 90) -> bytes:
        nonlocal encode_count
        with encode_lock:
            encode_count += 1
        time.sleep(0.02)
        return real_encode(image, quality=quality)

    monkeypatch.setattr(utils, "pil_to_jpeg_bytes", counted_encode)

    class LazyPathEngine(Engine):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def run(self, path: str, frames: list[Frame], max_api_frames: int = 3) -> EngineResult:
            start.wait(timeout=2)
            compatible_path = frames[0].compatible_file_path(path)
            fallback = Path(compatible_path)
            with Image.open(fallback) as image:
                observation = {
                    "processing_path": path,
                    "fallback_path": compatible_path,
                    "exists": fallback.exists(),
                    "format": image.format,
                    "mode": image.mode,
                    "size": image.size,
                }
            with observation_lock:
                observations.append(observation)
            return EngineResult(
                name=self.name,
                status=EngineStatus.OK,
                details={"internal_input": compatible_path},
            )

    _install_pipeline_doubles(
        monkeypatch,
        local_engines=[LazyPathEngine("path-a"), LazyPathEngine("path-b")],
        parallel_engines=True,
    )

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert encode_count == 1
    assert len(observations) == 2
    assert {item["processing_path"] for item in observations} == {str(original)}
    assert len({item["fallback_path"] for item in observations}) == 1
    assert all(item["exists"] is True for item in observations)
    assert all(item["format"] == "JPEG" for item in observations)
    assert all(item["mode"] == "RGB" for item in observations)
    assert all(item["size"] == (22, 14) for item in observations)
    fallback_path = Path(observations[0]["fallback_path"])
    assert fallback_path.exists() is False
    assert str(fallback_path) not in repr(report)
    assert all(result.details["internal_input"] == "<temporary-file>" for result in report["results"])
    assert report["preprocessing"] == {
        "source_format": "avif",
        "native_decode": True,
        "engine_fallback_format": "jpeg",
        "engine_fallback_created": True,
    }
    assert original.exists()


def test_avif_fallback_is_cleaned_and_sanitized_after_engine_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original = tmp_path / "engine-error.avif"
    _make_avif(original)
    seen_paths: list[str] = []

    class FailingPathEngine(Engine):
        name = "failing-path"

        def run(self, path: str, frames: list[Frame], max_api_frames: int = 3) -> EngineResult:
            fallback = frames[0].compatible_file_path(path)
            seen_paths.append(fallback)
            assert Path(fallback).exists()
            raise RuntimeError(f"backend rejected {fallback}")

    _install_pipeline_doubles(monkeypatch, local_engines=[FailingPathEngine()])

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
    assert Path(seen_paths[0]).exists() is False
    assert seen_paths[0] not in repr(report)
    assert seen_paths[0] not in repr(report["results"][0])
    assert seen_paths[0] not in log_output
    assert "engine failed: RuntimeError: backend rejected <temporary-file>" in log_output
    assert report["results"][0].status == EngineStatus.ERROR
    assert report["results"][0].error == "RuntimeError: backend rejected <temporary-file>"
    assert report["preprocessing"] == {
        "source_format": "avif",
        "native_decode": True,
        "engine_fallback_format": "jpeg",
        "engine_fallback_created": True,
    }


def test_avif_fallback_is_cleaned_after_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "interrupt.avif"
    _make_avif(original)
    seen_paths: list[str] = []

    class InterruptingPathEngine(Engine):
        name = "interrupting-path"

        def execute(self, path: str, frames: list[Frame], max_api_frames: int = 3) -> EngineResult:
            fallback = frames[0].compatible_file_path(path)
            seen_paths.append(fallback)
            assert Path(fallback).exists()
            raise KeyboardInterrupt

    _install_pipeline_doubles(monkeypatch, local_engines=[InterruptingPathEngine()])

    with pytest.raises(KeyboardInterrupt):
        pipeline.run_on_input(str(original), no_apis=True)

    assert len(seen_paths) == 1
    assert Path(seen_paths[0]).exists() is False
    assert original.exists()


def test_url_avif_download_is_native_and_cleaned_while_report_url_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    downloaded = tmp_path / "downloaded.avif"
    _make_avif(downloaded)
    observations: list[dict[str, Any]] = []
    _install_pipeline_doubles(
        monkeypatch,
        local_engines=[_NativeFrameEngine("url-frame", observations)],
    )
    monkeypatch.setattr(pipeline, "is_url", lambda value: True)
    monkeypatch.setattr(
        pipeline,
        "download_url_to_temp",
        lambda value, **kwargs: (str(downloaded), "remote.avif"),
    )

    report = pipeline.run_on_input(
        "https://example.test/assets?id=public&token=top-secret#hidden",
        no_apis=True,
    )

    assert observations[0]["path"] == str(downloaded)
    assert observations[0]["formats"] == ["avif"]
    assert downloaded.exists() is False
    assert report["name"] == "remote.avif"
    assert report["path"] == "https://example.test/assets?<redacted>"
    assert "top-secret" not in repr(report)
    assert str(downloaded) not in repr(report)
    assert report["preprocessing"] == {"source_format": "avif", "native_decode": True}


def test_url_avif_engine_error_redacts_download_path_from_report_and_stdout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    downloaded = tmp_path / "downloaded-error.avif"
    _make_avif(downloaded)
    seen_paths: list[str] = []

    class FailingEngine(Engine):
        name = "failing-url-path"

        def run(self, path: str, frames: list[Frame], max_api_frames: int = 3) -> EngineResult:
            seen_paths.append(path)
            assert Path(path).exists()
            raise RuntimeError(f"backend rejected {path}")

    _install_pipeline_doubles(monkeypatch, local_engines=[FailingEngine()])
    monkeypatch.setattr(pipeline, "is_url", lambda value: True)
    monkeypatch.setattr(
        pipeline,
        "download_url_to_temp",
        lambda value, **kwargs: (str(downloaded), "remote.avif"),
    )
    stdout_handler = next(
        handler
        for handler in logging.getLogger("modimg").handlers
        if isinstance(handler, logging.StreamHandler)
    )
    previous_stream = stdout_handler.setStream(sys.stdout)
    try:
        report = pipeline.run_on_input("https://example.test/image.avif?token=secret", no_apis=True)
        log_output = capsys.readouterr().out
    finally:
        stdout_handler.setStream(previous_stream)

    assert seen_paths == [str(downloaded)]
    assert downloaded.exists() is False
    assert str(downloaded) not in repr(report)
    assert str(downloaded) not in log_output
    assert "engine failed: RuntimeError: backend rejected <temporary-file>" in log_output
    assert report["results"][0].error == "RuntimeError: backend rejected <temporary-file>"
    assert report["preprocessing"] == {"source_format": "avif", "native_decode": True}


def test_avif_auto_learn_receives_native_rgb_frame_without_creating_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "auto-learn.avif"
    _make_avif(original)
    _install_pipeline_doubles(monkeypatch)
    observations: list[tuple[str, str, tuple[str, ...]]] = []

    def fake_auto_learn(verdict: Verdict, frames: list[Frame]) -> str:
        assert verdict.label == VerdictLabel.OK
        observations.append((frames[0].pil.mode, frames[0].source_format, frames[0].temporary_file_paths()))
        assert frame_phash_hex_int(frames[0])[0]
        return "learned AVIF pHash"

    monkeypatch.setattr(pipeline, "maybe_auto_learn", fake_auto_learn)

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert observations == [("RGB", "avif", ())]
    assert report["auto_learn"] == "learned AVIF pHash"
    assert report["preprocessing"] == {"source_format": "avif", "native_decode": True}


def test_openai_and_sightengine_share_one_avif_frame_jpeg_cache_and_use_jpeg_mime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = Frame(idx=0, pil=Image.new("RGB", (9, 7), color=(30, 80, 160)), source_format="avif")
    encode_count = 0
    encode_lock = threading.Lock()
    from modimg import utils

    real_encode = utils.pil_to_jpeg_bytes

    def counted_encode(image: Image.Image, quality: int = 90) -> bytes:
        nonlocal encode_count
        with encode_lock:
            encode_count += 1
        time.sleep(0.02)
        return real_encode(image, quality=quality)

    monkeypatch.setattr(utils, "pil_to_jpeg_bytes", counted_encode)
    monkeypatch.setenv("OPENAI_CACHE_ENABLE", "0")
    monkeypatch.setenv("OPENAI_MIN_INTERVAL_SEC", "0")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "0")

    openai_inputs: list[list[dict[str, Any]]] = []

    class Moderations:
        @staticmethod
        def create(**kwargs: Any) -> dict[str, Any]:
            openai_inputs.append(kwargs["input"])
            return {
                "results": [
                    {
                        "flagged": False,
                        "categories": {"sexual": False},
                        "category_scores": {"sexual": 0.01},
                    }
                ]
            }

    openai_engine = OpenAIModerationEngine()
    monkeypatch.setattr(openai_engine, "available", lambda: (True, ""))
    monkeypatch.setattr(
        openai_engine,
        "_client_for_timeout",
        lambda timeout: SimpleNamespace(moderations=Moderations()),
    )

    sight_uploads: list[tuple[str, bytes, str]] = []

    class SightResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        text = ""

        @staticmethod
        def json() -> dict[str, Any]:
            return {"status": "success", "nudity": {"safe": 1.0, "raw": 0.0}}

    class SightSession:
        @staticmethod
        def post(url: str, *, data: dict[str, str], files: dict[str, tuple[str, bytes, str]], timeout: float) -> SightResponse:
            sight_uploads.append(files["media"])
            return SightResponse()

    sightengine = SightengineEngine()
    monkeypatch.setattr(sightengine, "available", lambda: (True, ""))
    monkeypatch.setattr(sightengine, "_session", lambda: SightSession())

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            openai_future = executor.submit(openai_engine.run, "original.avif", [frame])
            sight_future = executor.submit(sightengine.run, "original.avif", [frame])
            openai_result = openai_future.result(timeout=3)
            sight_result = sight_future.result(timeout=3)

        assert openai_result.status == EngineStatus.OK
        assert sight_result.status == EngineStatus.OK
        assert encode_count == 1
        assert frame.temporary_file_paths() == ()
        assert len(openai_inputs) == 1
        image_url = openai_inputs[0][0]["image_url"]["url"]
        assert image_url.startswith("data:image/jpeg;base64,")
        openai_bytes = base64.b64decode(image_url.split(",", 1)[1], validate=True)
        assert len(sight_uploads) == 1
        filename, sight_bytes, mime = sight_uploads[0]
        assert filename == "frame.jpg"
        assert mime == "image/jpeg"
        assert sight_bytes == openai_bytes
        assert sight_bytes.startswith(b"\xff\xd8\xff")
    finally:
        frame.close()


def test_real_frame_engines_use_rgb_frame_without_requesting_avif_path_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = Frame(idx=0, pil=Image.new("RGB", (10, 8), color=(10, 20, 30)), source_format="avif")
    original = tmp_path / "frame-engines.avif"
    _make_avif(original, size=frame.pil.size)

    def unexpected_fallback() -> str:
        raise AssertionError("frame-based engine requested a file fallback")

    frame.get_jpeg_path = unexpected_fallback  # type: ignore[method-assign]

    blocklist = tmp_path / "ocr-blocklist.txt"
    blocklist.write_text("forbidden\n", encoding="utf-8")
    monkeypatch.setenv("OCR_ENABLE", "1")
    def fake_ocr(image: Image.Image, lang: str, timeout: float) -> str:
        assert image is frame.pil
        assert image.mode == "RGB"
        assert timeout > 0
        return "safe text"

    fake_pytesseract = SimpleNamespace(
        pytesseract=SimpleNamespace(tesseract_cmd="tesseract"),
        image_to_string=fake_ocr,
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)
    ocr = OCREngine()
    ocr.blocklist_path = str(blocklist)

    seen_yolo_images: list[Image.Image] = []

    def fake_predict(model: Any, image: Image.Image, **kwargs: Any) -> list[Any]:
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        seen_yolo_images.append(image)
        return []

    monkeypatch.setattr(yolo_weapons, "_resolve_model_reference", lambda: ("fake-weapons.pt", True, None))
    monkeypatch.setattr(yolo_weapons, "_load_model", lambda model_ref: SimpleNamespace(names={}))
    monkeypatch.setattr(yolo_weapons, "_inference_lock", lambda model_ref: threading.RLock())
    monkeypatch.setattr(yolo_weapons, "_predict", fake_predict)

    model_path = tmp_path / "symbols.pt"
    model_path.write_bytes(b"fake-model")
    model_path_path = model_path
    monkeypatch.setattr(forbidden_symbols_yolo, "_resolve_model_path", lambda model_path=None: model_path_path)
    monkeypatch.setattr(forbidden_symbols_yolo, "_load_model", lambda model_path=None: SimpleNamespace(names={}))
    monkeypatch.setattr(forbidden_symbols_yolo, "_inference_lock", lambda model_key: threading.RLock())
    monkeypatch.setattr(forbidden_symbols_yolo, "_predict", fake_predict)
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", "1")

    phash_hex, _phash_int = frame_phash_hex_int(frame)
    allowlist = tmp_path / "allow.csv"
    allowlist.write_text(f"{phash_hex},test\n", encoding="utf-8")
    blocklist_path = tmp_path / "block.csv"
    blocklist_path.write_text(f"{phash_hex},test\n", encoding="utf-8")

    try:
        results = [
            ocr.run(str(original), [frame]),
            YOLOWorldWeaponsEngine().run(str(original), [frame]),
            YOLOForbiddenSymbolsEngine().run(str(original), [frame]),
            PHashAllowlistEngine(allowlist_path=str(allowlist), max_distance=0).run(str(original), [frame]),
            PHashBlocklistEngine(blocklist_path=str(blocklist_path), max_distance=0).run(str(original), [frame]),
        ]
        assert [result.status for result in results] == [
            EngineStatus.OK,
            EngineStatus.OK,
            EngineStatus.OK,
            EngineStatus.OK,
            EngineStatus.OK,
        ]
        assert results[-2].scores["phash_allow_match"] == 1.0
        assert results[-1].scores["phash_block_match"] == 1.0
        assert len(seen_yolo_images) == 2
        assert frame.temporary_file_paths() == ()
    finally:
        frame.close()


def test_invalid_avif_pipeline_returns_controlled_loader_review(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "broken.avif"
    invalid.write_bytes(b"not an ISO-BMFF image")
    _install_pipeline_doubles(monkeypatch)

    report = pipeline.run_on_input(str(invalid), no_apis=True)

    assert report["verdict"].label == VerdictLabel.REVIEW
    assert report["results"][0].name == "Loader"
    assert report["results"][0].status == EngineStatus.ERROR
    assert "Traceback" not in (report["results"][0].error or "")
    assert report["path"] == str(invalid)


@pytest.mark.parametrize("path_name", ["asset.avif", "asset.AVIF", "asset.AvIf", "asset-without-extension"])
def test_avif_pipeline_preserves_all_original_local_path_spellings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path_name: str,
) -> None:
    original = tmp_path / path_name
    _make_avif(original)
    observations: list[dict[str, Any]] = []
    _install_pipeline_doubles(monkeypatch, local_engines=[_NativeFrameEngine("native", observations)])

    report = pipeline.run_on_input(str(original), no_apis=True)

    assert observations[0]["path"] == str(original)
    assert report["path"] == str(original)
    assert report["name"] == str(original)
    assert report["preprocessing"]["native_decode"] is True


def test_avif_report_contains_no_accidental_temporary_filename_pattern(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original = tmp_path / "clean-report.avif"
    _make_avif(original)
    fallback_paths: list[str] = []

    class PathEngine(Engine):
        name = "path-engine"

        def run(self, path: str, frames: list[Frame], max_api_frames: int = 3) -> EngineResult:
            fallback = frames[0].compatible_file_path(path)
            fallback_paths.append(fallback)
            return EngineResult(name=self.name, status=EngineStatus.OK, details={"path": fallback})

    _install_pipeline_doubles(monkeypatch, local_engines=[PathEngine()])
    report = pipeline.run_on_input(str(original), no_apis=True)

    assert len(fallback_paths) == 1
    assert fallback_paths[0] not in repr(report)
    assert report["results"][0].details["path"] == "<temporary-file>"
