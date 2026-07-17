from __future__ import annotations

import os
import tempfile
import threading
from typing import Any, List, Tuple, Optional
from PIL import Image

from ..enums import EngineStatus
from ..types import Engine, EngineResult, Frame
from ..utils import env_bool, now_ms, redact_sensitive_text, safe_float01

class NudeNetEngine(Engine):
    """Offline nudity detection via NudeNet (optional)."""
    name = "NudeNet"

    _DETECTOR = None
    _DETECTOR_LOCK = threading.RLock()
    _INFERENCE_LOCK = threading.RLock()

    @staticmethod
    def _is_missing_dependency_error(exc: Exception) -> bool:
        message = f"{type(exc).__name__}: {exc}".lower()
        return isinstance(exc, (ImportError, ModuleNotFoundError)) or "no module named" in message

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

        with NudeNetEngine._DETECTOR_LOCK:
            if NudeNetEngine._DETECTOR is None:
                try:
                    NudeNetEngine._DETECTOR = NudeDetector()
                except Exception as exc:
                    status = EngineStatus.SKIPPED if self._is_missing_dependency_error(exc) else EngineStatus.ERROR
                    return EngineResult(
                        name=self.name,
                        status=status,
                        error=redact_sensitive_text(f"nudenet detector unavailable: {type(exc).__name__}: {exc}"),
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
            if isinstance(frame, Frame):
                if frame_count == 1:
                    return frame.compatible_file_path(path)
                return frame.get_jpeg_path()
            if frame_count == 1 and path and os.path.exists(path):
                return path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                temp_path = tmp.name
                temp_paths.append(temp_path)
            with _to_pil(frame).convert("RGB") as im:
                im.save(temp_path, format="JPEG", quality=90)
            return temp_path

        frames_use = frames[:1] if not frames else ([frames[0], frames[-1]] if len(frames) > 1 else frames)
        try:
            for fr in frames_use:
                detect_input = _detect_input_for_frame(fr, len(frames_use))
                try:
                    with NudeNetEngine._INFERENCE_LOCK:
                        dets = detector.detect(detect_input) or []
                except Exception as exc:
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error=redact_sensitive_text(f"nudenet detection failed: {type(exc).__name__}: {exc}"),
                        details={"frame_idx": int(getattr(fr, "idx", 0))},
                        took_ms=now_ms() - start,
                    )
                for d in dets:
                    if not isinstance(d, dict):
                        continue
                    cls = str(d.get("class", "")).upper()
                    try:
                        score = safe_float01(d.get("score", 0.0) or 0.0)
                    except (TypeError, ValueError, OverflowError):
                        continue
                    if "EXPOSED" in cls:
                        exposed_max = max(exposed_max, score)
                    elif "COVERED" in cls:
                        covered_max = max(covered_max, score)
        finally:
            for tmp_path in temp_paths:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    self.logger.warning("failed to remove NudeNet temporary file: %s", type(exc).__name__)

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={
                "nudity_exposed": safe_float01(exposed_max),
                "nudity_covered": safe_float01(covered_max),
            },
            took_ms=now_ms()-start,
        )
