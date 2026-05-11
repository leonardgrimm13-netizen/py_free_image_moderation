from __future__ import annotations

import os
import re
from typing import List, Tuple


from ..enums import EngineStatus
from ..types import Engine, EngineResult, Frame
from ..utils import env_int, now_ms
from ..config import project_root

class OCREngine(Engine):
    name = "OCR text"

    # Cache compiled patterns per process to reduce CPU.
    _CACHE: tuple[float, List[re.Pattern]] = (0.0, [])

    def __init__(self) -> None:
        super().__init__()
        self.blocklist_path = os.path.join(project_root(), "data", "ocr_text_blocklist.txt")

    def available(self) -> Tuple[bool, str]:
        if os.getenv("OCR_ENABLE", "0").strip() != "1":
            return False, "disabled (set OCR_ENABLE=1)"
        try:
            import pytesseract  # noqa
        except Exception as e:
            return False, f"pytesseract not available: {type(e).__name__}"
        if not os.path.exists(self.blocklist_path):
            return False, f"blocklist not found ({self.blocklist_path})"
        return True, ""

    def _load_patterns(self) -> List[re.Pattern]:
        try:
            mtime = os.path.getmtime(self.blocklist_path)
        except Exception:
            return []
        cached_mtime, cached_pats = OCREngine._CACHE
        if cached_pats and cached_mtime == mtime:
            return cached_pats

        pats: List[re.Pattern] = []
        try:
            with open(self.blocklist_path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    try:
                        pats.append(re.compile(s, re.IGNORECASE))
                    except re.error:
                        # treat as literal
                        pats.append(re.compile(re.escape(s), re.IGNORECASE))
        except Exception:
            pats = []
        OCREngine._CACHE = (mtime, pats)
        return pats

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        start = now_ms()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=why, took_ms=now_ms()-start)

        import pytesseract
        # optional custom tesseract path
        tess = os.getenv("TESSERACT_CMD", "").strip()
        if tess:
            pytesseract.pytesseract.tesseract_cmd = tess

        lang = os.getenv("OCR_LANG", "eng").strip() or "eng"
        max_frames = env_int("OCR_MAX_FRAMES", 2)
        min_len = env_int("OCR_MIN_LEN", 3)

        patterns = self._load_patterns()
        if not patterns:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error="ocr blocklist empty", took_ms=now_ms()-start)

        text_all: List[str] = []
        use = frames[:max_frames] if max_frames > 0 else frames[:1]
        for fr in use:
            try:
                txt = pytesseract.image_to_string(fr.pil, lang=lang) or ""
            except Exception:
                txt = ""
            if txt:
                text_all.append(txt)

        joined = "\n".join(text_all).strip()
        if len(joined) < min_len:
            return EngineResult(name=self.name, status=EngineStatus.OK, scores={"ocr_match": 0.0}, details={"text": ""}, took_ms=now_ms()-start)

        hit = None
        for pat in patterns:
            m = pat.search(joined)
            if m:
                hit = pat.pattern
                break

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={"ocr_match": 1.0 if hit else 0.0},
            details={"hit": hit, "text": joined[:2000]},
            took_ms=now_ms()-start,
        )
