from __future__ import annotations

import os
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config
from .enums import EngineStatus, VerdictLabel
from .frames import load_frames
from .logging_utils import get_logger
from .preprocessing import prepare_image
from .types import Engine, EngineResult, Verdict, Frame, engine_temporary_path_scope
from .utils import (
    download_url_to_temp,
    env_bool,
    env_float,
    env_int,
    is_url,
    redact_sensitive_text,
    redact_url,
    status_value,
)
from .verdict import compute_verdict
from .phash import (
    append_phash_to_allowlist,
    append_phash_to_blocklist,
    frame_phash_hex_int,
    get_allowlist_path,
    get_blocklist_path,
)
from .engines import (
    PHashAllowlistEngine,
    PHashBlocklistEngine,
    OCREngine,
    NudeNetEngine,
    OpenNSFW2Engine,
    YOLOWorldWeaponsEngine,
    YOLOForbiddenSymbolsEngine,
    OpenAIModerationEngine,
    OpenAIRunState,
    SightengineEngine,
    SightengineRunState,
)

LOGGER = get_logger("pipeline")


def build_pre_engines(*, no_apis: bool = False) -> List[Engine]:
    return [PHashBlocklistEngine(), PHashAllowlistEngine()]


def build_local_engines(*, no_apis: bool = False) -> List[Engine]:
    return [OCREngine(), NudeNetEngine(), OpenNSFW2Engine(), YOLOWorldWeaponsEngine(), YOLOForbiddenSymbolsEngine()]


def build_api_engines(
    *,
    no_apis: bool = False,
    openai_run_state: OpenAIRunState | None = None,
    sightengine_run_state: SightengineRunState | None = None,
) -> List[Engine]:
    engines: List[Engine] = []
    cfg = get_config()
    if (not no_apis) and (not cfg.openai_disable):
        engines.append(OpenAIModerationEngine(run_state=openai_run_state))
    if (not no_apis) and (not cfg.sightengine_disable):
        engines.append(SightengineEngine(run_state=sightengine_run_state))
    return engines


def build_main_engines(
    *,
    no_apis: bool = False,
    openai_run_state: OpenAIRunState | None = None,
    sightengine_run_state: SightengineRunState | None = None,
) -> List[Engine]:
    return build_local_engines(no_apis=no_apis) + build_api_engines(
        no_apis=no_apis,
        openai_run_state=openai_run_state,
        sightengine_run_state=sightengine_run_state,
    )


def _run_single_engine(
    path: str,
    frames: List[Frame],
    engine: Engine,
    pipeline_temporary_paths: tuple[str, ...] = (),
) -> EngineResult:
    with engine_temporary_path_scope(pipeline_temporary_paths):
        try:
            return engine.execute(path, frames)
        except Exception as exc:  # last-resort protection
            temporary_paths = [
                *pipeline_temporary_paths,
                *(
                    temporary_path
                    for frame in frames
                    for temporary_path in frame.temporary_file_paths()
                ),
            ]
            details: Dict[str, Any] = {}
            if get_config().debug:
                details["trace"] = _sanitize_internal_paths(traceback.format_exc()[-2000:], temporary_paths)
            return EngineResult(
                name=getattr(engine, "name", "engine"),
                status=EngineStatus.ERROR,
                error=_sanitize_internal_paths(f"{type(exc).__name__}: {exc}", temporary_paths),
                details=details,
            )


def run_engines(
    path: str,
    frames: List[Frame],
    engines: List[Engine],
    *,
    temporary_paths: List[str] | None = None,
) -> List[EngineResult]:
    cfg = get_config()
    internal_paths = tuple(temporary_paths or ())
    if not cfg.parallel_engines or len(engines) <= 1:
        return [_run_single_engine(path, frames, eng, internal_paths) for eng in engines]

    results: List[EngineResult] = []
    with ThreadPoolExecutor(max_workers=min(cfg.parallel_workers, len(engines))) as executor:
        futures = {
            executor.submit(_run_single_engine, path, frames, eng, internal_paths): idx
            for idx, eng in enumerate(engines)
        }
        ordered: Dict[int, EngineResult] = {}
        for fut in as_completed(futures):
            ordered[futures[fut]] = fut.result()
        for idx in range(len(engines)):
            results.append(ordered[idx])
    return results


def _short_circuit_from_phash(results: List[EngineResult]) -> Optional[Verdict]:
    block: Optional[Verdict] = None
    allow: Optional[Verdict] = None
    for r in results:
        if status_value(r.status) != EngineStatus.OK.value:
            continue
        if r.name == "pHash blocklist" and r.scores.get("phash_block_match") == 1.0:
            block = Verdict(VerdictLabel.BLOCK, 1.0, 1.0, 1.0, [f"Blocklist match (distance={r.details.get('distance')})"])
        if r.name == "pHash allowlist" and r.scores.get("phash_allow_match") == 1.0:
            allow = Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, [f"Allowlist match (distance={r.details.get('distance')})"])
    return block or allow


def maybe_auto_learn(verdict: Verdict, frames: List[Frame]) -> Optional[str]:
    try:
        if not frames:
            return None
        auto_learn_is_set = os.getenv("PHASH_AUTO_LEARN_ENABLE") is not None
        auto_learn = env_bool("PHASH_AUTO_LEARN_ENABLE", False)
        if auto_learn_is_set and not auto_learn:
            return None
        if not auto_learn_is_set:
            legacy_any = env_bool("PHASH_AUTO_APPEND", False) or env_bool("PHASH_AUTO_ALLOW_APPEND", False)
            if not legacy_any:
                return None
        learn_first_last = env_bool("PHASH_GIF_LEARN_FIRST_LAST", False)
        frs = [frames[0], frames[-1]] if learn_first_last and len(frames) > 1 else [frames[0]]
        hashes = [frame_phash_hex_int(fr)[0] for fr in frs]

        allow_append = os.getenv("PHASH_AUTO_ALLOW_APPEND", "").strip()
        block_append = os.getenv("PHASH_AUTO_BLOCK_APPEND", "").strip()
        if auto_learn:
            if allow_append == "":
                allow_append = "1"
            if block_append == "":
                block_append = "1"

        if verdict.label == VerdictLabel.OK and env_bool("PHASH_AUTO_ALLOW_APPEND", allow_append == "1"):
            label = os.getenv("PHASH_AUTO_ALLOW_LABEL", os.getenv("PHASH_AUTO_LABEL", "ok")).strip() or "ok"
            apath = get_allowlist_path()
            added_any = False
            for hx in hashes:
                added_any = append_phash_to_allowlist(hx, apath, label) or added_any
            if added_any:
                return f"Auto-added pHash to allowlist ({apath})"

        if verdict.label == VerdictLabel.BLOCK and env_bool("PHASH_AUTO_BLOCK_APPEND", block_append == "1"):
            label = os.getenv("PHASH_AUTO_BLOCK_LABEL", os.getenv("PHASH_AUTO_LABEL", "not_ok")).strip() or "not_ok"
            bpath = get_blocklist_path()
            added_any = False
            for hx in hashes:
                added_any = append_phash_to_blocklist(hx, bpath, label) or added_any
            if added_any:
                return f"Auto-added pHash to blocklist ({bpath})"
    except (ValueError, OSError):
        return None
    return None


def _sanitize_internal_paths(value: Any, temporary_paths: List[str]) -> Any:
    """Recursively remove internal temporary paths from public report values."""
    if isinstance(value, str):
        sanitized = value
        for temporary_path in sorted(set(temporary_paths), key=len, reverse=True):
            if temporary_path:
                sanitized = sanitized.replace(temporary_path, "<temporary-file>")
        return redact_sensitive_text(sanitized)
    if isinstance(value, Path):
        return _sanitize_internal_paths(str(value), temporary_paths)
    if isinstance(value, dict):
        return {
            _sanitize_internal_paths(key, temporary_paths): _sanitize_internal_paths(item, temporary_paths)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_internal_paths(item, temporary_paths) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_internal_paths(item, temporary_paths) for item in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(_sanitize_internal_paths(item, temporary_paths) for item in value)
    return value


def _sanitize_engine_results(results: List[EngineResult], temporary_paths: List[str]) -> None:
    for result in results:
        result.name = _sanitize_internal_paths(result.name, temporary_paths)
        if type(result.status) is str:
            result.status = _sanitize_internal_paths(result.status, temporary_paths)
        result.scores = _sanitize_internal_paths(result.scores, temporary_paths)
        if result.error:
            result.error = _sanitize_internal_paths(result.error, temporary_paths)
        result.details = _sanitize_internal_paths(result.details, temporary_paths)


def _sanitize_verdict(verdict: Verdict, temporary_paths: List[str]) -> None:
    """Remove internal paths that an engine may have copied into a verdict."""
    if type(verdict.label) is str:
        verdict.label = _sanitize_internal_paths(verdict.label, temporary_paths)
    verdict.reasons = _sanitize_internal_paths(verdict.reasons, temporary_paths)


def _register_frame_temporary_paths(frames: List[Frame], temporary_paths: List[str]) -> bool:
    """Register lazy frame-owned files for report sanitizing and final cleanup."""
    registered = False
    known = set(temporary_paths)
    for frame in frames:
        for temporary_path in frame.temporary_file_paths():
            registered = True
            if temporary_path not in known:
                known.add(temporary_path)
                temporary_paths.append(temporary_path)
    return registered


def run_on_input(
    inp: str,
    *,
    no_apis: bool = False,
    sample_frames: int = 12,
    openai_run_state: OpenAIRunState | None = None,
    sightengine_run_state: SightengineRunState | None = None,
) -> Dict[str, Any]:
    original_input = inp
    temporary_paths: List[str] = []
    frames: List[Frame] = []
    remote_input = is_url(original_input)
    report_path = redact_url(original_input) if remote_input else original_input
    display_name = report_path
    preprocessing: Dict[str, Any] | None = None
    processing_path = original_input
    source_format = "unknown"
    avif_fallback_created = False

    try:
        try:
            if remote_input:
                downloaded_path, display_name = download_url_to_temp(
                    original_input,
                    max_bytes=max(1, env_int("MODIMG_MAX_DOWNLOAD_BYTES", 25_000_000)),
                    timeout_sec=env_float("MODIMG_URL_TIMEOUT_SEC", 20.0, min_value=0.1),
                    max_redirects=max(0, env_int("MODIMG_MAX_URL_REDIRECTS", 5)),
                )
                temporary_paths.append(downloaded_path)
                local_path = downloaded_path
            else:
                local_path = original_input

            prepared = prepare_image(local_path)
            temporary_paths.extend(prepared.temporary_paths)
            processing_path = prepared.processing_path
            source_format = prepared.source_format
            preprocessing = dict(prepared.metadata) if prepared.metadata else None
            frames = load_frames(processing_path, sample_frames=sample_frames)
            if source_format == "avif":
                preprocessing = {"source_format": "avif", "native_decode": True}
        except Exception as e:
            error = redact_sensitive_text(
                f"{type(e).__name__}: {e}",
                extra_secrets=(original_input,) if remote_input else (),
            )
            error = _sanitize_internal_paths(error, temporary_paths)
            v = Verdict(VerdictLabel.REVIEW, 0.0, 0.0, 0.0, [f"loader_failure: {error}"])
            report: Dict[str, Any] = {
                "name": display_name,
                "path": report_path,
                "verdict": v,
                "results": [EngineResult(name="Loader", status=EngineStatus.ERROR, error=f"failed to load image: {error}")],
                "auto_learn": "",
            }
            if preprocessing is not None:
                report["preprocessing"] = preprocessing
            return report

        pre_results = run_engines(
            processing_path,
            frames,
            build_pre_engines(no_apis=no_apis),
            temporary_paths=temporary_paths,
        )
        avif_fallback_created = _register_frame_temporary_paths(frames, temporary_paths)
        sc = _short_circuit_from_phash(pre_results)

        cfg = get_config()
        if sc is not None and cfg.short_circuit_phash:
            results = pre_results
            v = sc
        else:
            local_results = run_engines(
                processing_path,
                frames,
                build_local_engines(no_apis=no_apis),
                temporary_paths=temporary_paths,
            )
            avif_fallback_created = (
                _register_frame_temporary_paths(frames, temporary_paths) or avif_fallback_created
            )
            local_with_pre = pre_results + local_results
            local_verdict = compute_verdict(local_with_pre)

            should_run_apis = False
            if (not no_apis) and cfg.api_policy != "never":
                api_engines = build_api_engines(
                    no_apis=no_apis,
                    openai_run_state=openai_run_state,
                    sightengine_run_state=sightengine_run_state,
                )
                if cfg.api_policy == "always":
                    should_run_apis = len(api_engines) > 0
                elif cfg.api_policy == "on_review":
                    label = local_verdict.label
                    should_run_apis = len(api_engines) > 0 and (
                        label == VerdictLabel.REVIEW or str(label).upper() == VerdictLabel.REVIEW.value
                    )
                if should_run_apis:
                    api_results = run_engines(
                        processing_path,
                        frames,
                        api_engines,
                        temporary_paths=temporary_paths,
                    )
                    avif_fallback_created = (
                        _register_frame_temporary_paths(frames, temporary_paths) or avif_fallback_created
                    )
                    results = local_with_pre + api_results
                    v = compute_verdict(results)
                else:
                    results = local_with_pre
                    v = local_verdict
            else:
                results = local_with_pre
                v = local_verdict

        avif_fallback_created = _register_frame_temporary_paths(frames, temporary_paths) or avif_fallback_created
        if source_format == "avif" and preprocessing is not None and avif_fallback_created:
            preprocessing.update(
                {
                    "engine_fallback_format": "jpeg",
                    "engine_fallback_created": True,
                }
            )
        _sanitize_engine_results(results, temporary_paths)
        _sanitize_verdict(v, temporary_paths)
        learn_msg = maybe_auto_learn(v, frames)
        if learn_msg:
            learn_msg = _sanitize_internal_paths(learn_msg, temporary_paths)
        report = {"name": display_name, "path": report_path, "verdict": v, "results": results, "auto_learn": learn_msg}
        if preprocessing is not None:
            report["preprocessing"] = preprocessing
        return report
    finally:
        # Capture lazily created compatibility files even when a BaseException
        # interrupted an engine stage before the normal registration point.
        _register_frame_temporary_paths(frames, temporary_paths)
        for frame in frames:
            try:
                frame.close()
            except Exception as exc:
                LOGGER.warning("failed to close decoded frame: %s", type(exc).__name__)
        cleaned: set[str] = set()
        for temporary_path in reversed(temporary_paths):
            if not temporary_path or temporary_path in cleaned:
                continue
            cleaned.add(temporary_path)
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("failed to remove an internal temporary file: %s", type(exc).__name__)
