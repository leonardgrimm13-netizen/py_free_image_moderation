"""Dataclasses and engine base types."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .enums import EngineStatus, VerdictLabel
from .logging_utils import get_logger
from .utils import now_ms


@dataclass
class Frame:
    """A sampled frame used by moderation engines."""

    idx: int
    pil: Image.Image
    _jpeg_bytes: Optional[bytes] = None

    def get_jpeg_bytes(self) -> bytes:
        """Return cached JPEG bytes for API-based engines."""
        if self._jpeg_bytes is None:
            from .utils import pil_to_jpeg_bytes

            self._jpeg_bytes = pil_to_jpeg_bytes(self.pil)
        return self._jpeg_bytes


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
            return result
        except (ValueError, RuntimeError, OSError) as exc:
            self.logger.warning("engine failed: %s", exc)
            return EngineResult(name=self.name, status=EngineStatus.ERROR, error=f"{type(exc).__name__}: {exc}", took_ms=now_ms() - t0)

    def disable(self, why: str) -> None:
        self.disabled_reason = why


def mk_skipped(engine: Engine, why: str, took_ms: Optional[int] = None) -> EngineResult:
    """Build a skipped engine result."""
    return EngineResult(name=engine.name, status=EngineStatus.SKIPPED, error=why, took_ms=took_ms)
