from __future__ import annotations

import os
from typing import Any, List, Tuple, Optional
from PIL import Image

from ..types import Engine, EngineResult
from ..utils import now_ms

class OpenNSFW2Engine(Engine):
    """Offline NSFW probability via OpenNSFW2.

    Supports the PyPI package **opennsfw2** (recommended) and also keeps compatibility
    with older code that imports **open_nsfw2**.
    """
    name = "OpenNSFW2"

    _BACKEND = None  # (name, module)

    def _import_backend(self):
        if OpenNSFW2Engine._BACKEND is not None:
            return OpenNSFW2Engine._BACKEND
        # Preferred official package name on PyPI is `opennsfw2`.
        try:
            import opennsfw2 as n2  # type: ignore
            OpenNSFW2Engine._BACKEND = ("opennsfw2", n2)
            return OpenNSFW2Engine._BACKEND
        except Exception:
            pass
        # Back-compat name some projects use:
        import open_nsfw2 as n2  # type: ignore
        OpenNSFW2Engine._BACKEND = ("open_nsfw2", n2)
        return OpenNSFW2Engine._BACKEND

    def available(self) -> Tuple[bool, str]:
        if os.getenv("OPENNSFW2_DISABLE", "0").strip() == "1":
            return False, "disabled via OPENNSFW2_DISABLE=1"
        try:
            self._import_backend()
            return True, "ok"
        except Exception as e:
            return False, f"opennsfw2/open_nsfw2 not available: {type(e).__name__}"

    def run(self, path: str, frames: List[Any], max_api_frames: Optional[int] = None) -> EngineResult:
        start = now_ms()
        backend_name, n2 = self._import_backend()

        def _to_pil(x: Any) -> Image.Image:
            if hasattr(x, "pil"):
                return getattr(x, "pil")
            return x

        im = _to_pil(frames[0]).convert("RGB")

        prob = None
        try:
            # opennsfw2 docs: predict_image accepts path or PIL
            if hasattr(n2, "predict_image"):
                prob = n2.predict_image(im)
            elif hasattr(n2, "predict"):
                prob = n2.predict(im)
            elif hasattr(n2, "predict_images"):
                prob = (n2.predict_images([im]) or [0.0])[0]
        except Exception as e:
            return EngineResult(
                name=self.name,
                status="error",
                error=f"{backend_name} prediction failed: {type(e).__name__}: {e}",
                took_ms=now_ms() - start,
            )

        if prob is None:
            return EngineResult(
                name=self.name,
                status="error",
                error=f"{backend_name} installed but no compatible predict_* function found",
                took_ms=now_ms() - start,
            )

        try:
            p = float(prob)
        except Exception:
            p = 0.0
        p = float(max(0.0, min(1.0, p)))
        return EngineResult(name=self.name, status="ok", scores={"nsfw_probability": p}, took_ms=now_ms() - start)
