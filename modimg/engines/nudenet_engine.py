from __future__ import annotations

import os
from typing import Any, List, Tuple, Optional
from PIL import Image

from ..enums import EngineStatus
from ..types import Engine, EngineResult
from ..utils import now_ms

class NudeNetEngine(Engine):
    """Offline nudity detection via NudeNet (optional)."""
    name = "NudeNet"

    _DETECTOR = None

    def available(self) -> Tuple[bool, str]:
        if (os.getenv("NUDENET_DISABLE", "0") or "0").strip() == "1":
            return False, "disabled via NUDENET_DISABLE=1"
        try:
            from nudenet import NudeDetector  # noqa: F401
            return True, EngineStatus.OK.value
        except Exception as e:
            return False, f"nudenet not available: {type(e).__name__}"

    def run(self, path: str, frames: List[Any], max_api_frames: Optional[int] = None) -> EngineResult:
        start = now_ms()
        from nudenet import NudeDetector
        import numpy as np  # type: ignore

        if NudeNetEngine._DETECTOR is None:
            NudeNetEngine._DETECTOR = NudeDetector()
        detector = NudeNetEngine._DETECTOR
        exposed_max = 0.0
        covered_max = 0.0

        def _to_pil(x: Any) -> Image.Image:
            if hasattr(x, "pil"):
                return getattr(x, "pil")
            return x

        frames_use = frames[:1] if not frames else ([frames[0], frames[-1]] if len(frames) > 1 else frames)
        for fr in frames_use:
            im = _to_pil(fr)
            arr = np.array(im.convert("RGB"))
            try:
                dets = detector.detect(arr) or []
            except Exception:
                dets = []
            for d in dets:
                cls = str(d.get("class", "")).upper()
                score = float(d.get("score", 0.0) or 0.0)
                if "EXPOSED" in cls:
                    exposed_max = max(exposed_max, score)
                elif "COVERED" in cls:
                    covered_max = max(covered_max, score)

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={
                "nudity_exposed": float(max(0.0, min(1.0, exposed_max))),
                "nudity_covered": float(max(0.0, min(1.0, covered_max))),
            },
            took_ms=now_ms()-start,
        )
