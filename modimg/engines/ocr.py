from __future__ import annotations

import os
import re
import threading
from typing import List, Tuple


from ..enums import EngineStatus
from ..resources import resolve_bundled_resource_path, resolve_resource_path
from ..types import Engine, EngineResult, Frame
from ..utils import env_bool, env_int, now_ms, redact_sensitive_text

class OCREngine(Engine):
    name = "OCR text"

    # Cache compiled patterns by file so custom per-instance blocklists cannot collide.
    _CACHE: dict[str, tuple[tuple[int, int], List[re.Pattern]]] = {}
    _CACHE_LOCK = threading.RLock()
    _TESSERACT_CONFIG_LOCK = threading.RLock()

    def __init__(self) -> None:
        super().__init__()
        self.blocklist_path = str(resolve_bundled_resource_path(os.path.join("data", "ocr_text_blocklist.txt")))

    def available(self) -> Tuple[bool, str]:
        if not env_bool("OCR_ENABLE", False):
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
            stat_result = os.stat(self.blocklist_path)
        except OSError:
            return []
        max_bytes = max(1, env_int("OCR_BLOCKLIST_MAX_BYTES", 1_000_000))
        if stat_result.st_size > max_bytes:
            return []
        signature = (stat_result.st_mtime_ns, stat_result.st_size)
        cache_key = str(resolve_resource_path(self.blocklist_path))
        with OCREngine._CACHE_LOCK:
            cached = OCREngine._CACHE.get(cache_key)
            if cached is not None and cached[0] == signature:
                return cached[1]

        pats: List[re.Pattern] = []
        max_patterns = max(1, env_int("OCR_BLOCKLIST_MAX_PATTERNS", 10_000))
        max_pattern_chars = max(1, env_int("OCR_MAX_PATTERN_CHARS", 1_000))
        try:
            with open(self.blocklist_path, "rb") as f:
                raw = f.read(max_bytes + 1)
            if len(raw) > max_bytes:
                return []
            for line in raw.decode("utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                is_regex = s.lower().startswith("re:")
                pattern_text = s[3:] if is_regex else re.escape(s)
                if not pattern_text or len(pattern_text) > max_pattern_chars:
                    continue
                try:
                    pats.append(re.compile(pattern_text, re.IGNORECASE))
                except re.error:
                    pats.append(re.compile(re.escape(s[3:] if is_regex else s), re.IGNORECASE))
                if len(pats) >= max_patterns:
                    break
        except (OSError, UnicodeError):
            pats = []
        with OCREngine._CACHE_LOCK:
            OCREngine._CACHE[cache_key] = (signature, pats)
        return pats

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        start = now_ms()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=why, took_ms=now_ms()-start)

        import pytesseract
        # Optional custom tesseract path. pytesseract stores it in module-global state.
        tess = os.getenv("TESSERACT_CMD", "").strip()

        lang = os.getenv("OCR_LANG", "eng").strip() or "eng"
        max_frames = env_int("OCR_MAX_FRAMES", 2)
        min_len = env_int("OCR_MIN_LEN", 3)

        patterns = self._load_patterns()
        if not patterns:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error="ocr blocklist empty", took_ms=now_ms()-start)

        text_all: List[str] = []
        use = frames[:max_frames] if max_frames > 0 else frames[:1]
        errors: List[str] = []
        for fr in use:
            try:
                with OCREngine._TESSERACT_CONFIG_LOCK:
                    original_tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
                    try:
                        if tess:
                            pytesseract.pytesseract.tesseract_cmd = tess
                        txt = pytesseract.image_to_string(fr.pil, lang=lang) or ""
                    finally:
                        pytesseract.pytesseract.tesseract_cmd = original_tesseract_cmd
            except Exception as exc:
                msg = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
                errors.append(msg)
                if "tesseract" in msg.lower():
                    return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=f"tesseract unavailable: {msg}", took_ms=now_ms()-start)
                txt = ""
            if txt:
                text_all.append(txt)

        if errors and not text_all:
            return EngineResult(name=self.name, status=EngineStatus.ERROR, error=f"ocr failed: {errors[0]}", took_ms=now_ms()-start)

        max_text_chars = max(1, env_int("OCR_MAX_TEXT_CHARS", 20_000))
        joined = "\n".join(text_all).strip()[:max_text_chars]
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
