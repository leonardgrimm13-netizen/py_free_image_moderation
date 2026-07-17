"""Dataclasses and engine base types."""
from __future__ import annotations

import dataclasses
import os
import tempfile
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .enums import EngineStatus, VerdictLabel
from .logging_utils import get_logger
from .utils import now_ms, redact_sensitive_text


_FRAME_LOCK_CREATION_LOCK = threading.Lock()
_FRAME_LOGGER = get_logger("frames")
_ENGINE_TEMPORARY_PATHS: ContextVar[tuple[str, ...]] = ContextVar(
    "modimg_engine_temporary_paths",
    default=(),
)


@contextmanager
def engine_temporary_path_scope(paths: Iterable[str]) -> Iterator[None]:
    """Make pipeline-owned temporary paths available to engine error sanitizing."""
    inherited = _ENGINE_TEMPORARY_PATHS.get()
    combined = tuple(dict.fromkeys((*inherited, *(str(path) for path in paths if path))))
    token = _ENGINE_TEMPORARY_PATHS.set(combined)
    try:
        yield
    finally:
        _ENGINE_TEMPORARY_PATHS.reset(token)


def redact_engine_output(value: Any, frames: List["Frame"]) -> str:
    """Redact all known temporary paths before engine output or logging."""
    text = redact_sensitive_text(value)
    temporary_paths = set(_ENGINE_TEMPORARY_PATHS.get())
    temporary_paths.update(
        temporary_path
        for frame in frames
        for temporary_path in frame.temporary_file_paths()
        if temporary_path
    )
    for temporary_path in sorted(temporary_paths, key=len, reverse=True):
        text = text.replace(temporary_path, "<temporary-file>")
    return text


@dataclass
class Frame:
    """A sampled frame used by moderation engines."""

    idx: int
    pil: Image.Image
    _jpeg_bytes: Optional[bytes] = None
    source_format: str = ""
    _jpeg_path: Optional[str] = dataclasses.field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cache_lock = threading.RLock()

    def cache_lock(self) -> threading.RLock:
        """Return the per-frame lock used by lazily computed caches."""
        lock = getattr(self, "_cache_lock", None)
        if lock is not None:
            return lock
        with _FRAME_LOCK_CREATION_LOCK:
            lock = getattr(self, "_cache_lock", None)
            if lock is None:
                lock = threading.RLock()
                self._cache_lock = lock
            return lock

    def get_jpeg_bytes(self) -> bytes:
        """Return cached JPEG bytes shared by API and legacy path engines."""
        with self.cache_lock():
            if self._jpeg_bytes is None:
                from .utils import pil_to_jpeg_bytes

                self._jpeg_bytes = pil_to_jpeg_bytes(self.pil)
            return self._jpeg_bytes

    def get_jpeg_path(self) -> str:
        """Return one fully written, thread-safe temporary JPEG for this frame."""
        with self.cache_lock():
            if self._jpeg_path and os.path.isfile(self._jpeg_path):
                return self._jpeg_path
            self._jpeg_path = None
            temporary_path = ""
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temporary:
                    temporary_path = temporary.name
                    temporary.write(self.get_jpeg_bytes())
                self._jpeg_path = temporary_path
                return temporary_path
            except BaseException as exc:
                if temporary_path:
                    try:
                        os.remove(temporary_path)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        _FRAME_LOGGER.warning(
                            "failed to remove an incomplete frame JPEG: %s",
                            type(cleanup_error).__name__,
                        )
                if isinstance(exc, Exception):
                    raise RuntimeError("failed to create a compatible frame JPEG") from exc
                raise

    def compatible_file_path(self, original_path: str) -> str:
        """Use a cached JPEG only when an AVIF-incompatible path engine needs it."""
        if self.source_format.strip().lower() == "avif":
            return self.get_jpeg_path()
        return original_path

    def temporary_file_paths(self) -> tuple[str, ...]:
        """Expose owned temporary paths for central report sanitizing and cleanup."""
        with self.cache_lock():
            return (self._jpeg_path,) if self._jpeg_path else ()

    def close(self) -> None:
        """Release decoded pixels, cached bytes, and any owned JPEG fallback."""
        with self.cache_lock():
            temporary_path = self._jpeg_path
            self._jpeg_path = None
            try:
                self.pil.close()
            finally:
                self._jpeg_bytes = None
                if temporary_path:
                    try:
                        os.remove(temporary_path)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        _FRAME_LOGGER.warning(
                            "failed to remove a frame JPEG fallback: %s",
                            type(cleanup_error).__name__,
                        )


@dataclass
class EngineResult:
    """Single engine outcome."""

    name: str
    status: EngineStatus | str
    scores: Dict[str, float] = dataclasses.field(default_factory=dict)
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    error: Optional[str] = None
    took_ms: Optional[int] = None


@dataclass
class Verdict:
    """Aggregated moderation verdict."""

    label: VerdictLabel | str
    nudity_risk: float
    violence_risk: float
    hate_risk: float
    reasons: List[str]


class Engine:
    """Base engine interface and helpers."""

    name: str = "engine"

    def __init__(self) -> None:
        self.disabled_reason: Optional[str] = None
        self.logger = get_logger(self.name.lower().replace(" ", "_"))

    def available(self) -> Tuple[bool, str]:
        if self.disabled_reason:
            return False, self.disabled_reason
        return True, ""

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        raise NotImplementedError

    def execute(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        """Run the engine with availability checks and standardized error handling."""
        t0 = now_ms()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=why, took_ms=now_ms() - t0)
        try:
            result = self.run(path, frames, max_api_frames=max_api_frames)
            if result.took_ms is None:
                result.took_ms = now_ms() - t0
            if result.error:
                result.error = redact_engine_output(result.error, frames)
            return result
        except Exception as exc:
            error = redact_engine_output(f"{type(exc).__name__}: {exc}", frames)
            self.logger.warning("engine failed: %s", error)
            return EngineResult(name=self.name, status=EngineStatus.ERROR, error=error, took_ms=now_ms() - t0)

    def disable(self, why: str) -> None:
        self.disabled_reason = why


def mk_skipped(engine: Engine, why: str, took_ms: Optional[int] = None) -> EngineResult:
    """Build a skipped engine result."""
    return EngineResult(name=engine.name, status=EngineStatus.SKIPPED, error=why, took_ms=took_ms)
