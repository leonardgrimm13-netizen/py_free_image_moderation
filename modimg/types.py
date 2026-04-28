"""Dataclasses and engine base types."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

@dataclass
class Frame:
    idx: int
    pil: Image.Image
    # Some API engines need JPEG bytes. To reduce CPU when scanning many images,
    # we compute them lazily on demand.
    _jpeg_bytes: Optional[bytes] = None

    def get_jpeg_bytes(self) -> bytes:
        """Return JPEG bytes for this frame (computed once)."""
        if self._jpeg_bytes is None:
            from .utils import pil_to_jpeg_bytes
            self._jpeg_bytes = pil_to_jpeg_bytes(self.pil)
        return self._jpeg_bytes

@dataclass
class EngineResult:
    name: str
    status: str  # "ok" | "skipped" | "error"
    scores: Dict[str, float] = dataclasses.field(default_factory=dict)
    details: Dict[str, Any] = dataclasses.field(default_factory=dict)
    error: Optional[str] = None
    took_ms: Optional[int] = None

@dataclass
class Verdict:
    label: str  # "OK" | "REVIEW" | "BLOCK"
    nudity_risk: float
    violence_risk: float
    hate_risk: float
    reasons: List[str]

class Engine:
    """Base engine interface."""
    name: str = "engine"

    def __init__(self) -> None:
        self.disabled_reason: Optional[str] = None

    def available(self) -> Tuple[bool, str]:
        if self.disabled_reason:
            return False, self.disabled_reason
        return True, ""

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        raise NotImplementedError

    def disable(self, why: str) -> None:
        self.disabled_reason = why

def mk_skipped(engine: Engine, why: str, took_ms: Optional[int] = None) -> EngineResult:
    return EngineResult(name=engine.name, status="skipped", error=why, took_ms=took_ms)
