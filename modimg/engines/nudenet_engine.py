from __future__ import annotations

import os
import tempfile
from typing import Any, List, Tuple, Optional
from PIL import Image

from ..enums import EngineStatus
from ..types import Engine, EngineResult
from ..utils import env_bool, now_ms

class NudeNetEngine(Engine):
    """Offline nudity detection via NudeNet (optional)."""
    name = "NudeNet"

    _DETECTOR = None

    def available(self) -> Tuple[bool, str]:
        if env_bool("NUDENET_DISABLE", False):
            return False, "disabled via NUDENET_DISABLE=1"
        try:
            from nudenet import NudeDetector  # noqa: F401
            return True, EngineStatus.OK.value
        except Exception as e:
            return False, f"nudenet not available: {type(e).__name__}"

    def run(self, path: str, frames: List[Any], max_api_frames: Optional[int] = None) -> EngineResult:
        start = now_ms()
        if not frames:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error="no frames", took_ms=now_ms() - start)

        try:
            from nudenet import NudeDetector
        except Exception as exc:
            return EngineResult(
                name=self.name,
                status=EngineStatus.SKIPPED,
                error=f"nudenet not available: {type(exc).__name__}: {exc}",
                took_ms=now_ms() - start,
            )

        if NudeNetEngine._DETECTOR is None:
            try:
                NudeNetEngine._DETECTOR = NudeDetector()
            except Exception as exc:
                return EngineResult(
                    name=self.name,
                    status=EngineStatus.SKIPPED,
                    error=f"nudenet detector unavailable: {type(exc).__name__}: {exc}",
                    took_ms=now_ms()-start,
                )
        detector = NudeNetEngine._DETECTOR
        exposed_max = 0.0
        covered_max = 0.0
        temp_paths: list[str] = []

        def _to_pil(x: Any) -> Image.Image:
            if hasattr(x, "pil"):
                return getattr(x, "pil")
            return x

        def _detect_input_for_frame(frame: Any, frame_count: int) -> str:
            if frame_count == 1 and path and os.path.exists(path):
                return path
            im = _to_pil(frame).convert("RGB")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            tmp.close()
            im.save(tmp.name, format="JPEG", quality=90)
            temp_paths.append(tmp.name)
            return tmp.name

        frames_use = frames[:1] if not frames else ([frames[0], frames[-1]] if len(frames) > 1 else frames)
        try:
            for fr in frames_use:
                detect_input = _detect_input_for_frame(fr, len(frames_use))
                try:
                    dets = detector.detect(detect_input) or []
                except Exception as exc:
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error=f"nudenet detection failed: {type(exc).__name__}: {exc}",
                        details={"frame_idx": int(getattr(fr, "idx", 0)), "input": detect_input},
                        took_ms=now_ms() - start,
                    )
                for d in dets:
                    if not isinstance(d, dict):
                        continue
                    cls = str(d.get("class", "")).upper()
                    try:
                        score = float(d.get("score", 0.0) or 0.0)
                    except Exception:
                        continue
                    if "EXPOSED" in cls:
                        exposed_max = max(exposed_max, score)
                    elif "COVERED" in cls:
                        covered_max = max(covered_max, score)
        finally:
            for tmp_path in temp_paths:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={
                "nudity_exposed": float(max(0.0, min(1.0, exposed_max))),
                "nudity_covered": float(max(0.0, min(1.0, covered_max))),
            },
            took_ms=now_ms()-start,
        )
