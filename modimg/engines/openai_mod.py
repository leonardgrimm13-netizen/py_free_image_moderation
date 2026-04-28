from __future__ import annotations

import os
import json
import time
import random
import base64
import threading
import atexit
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple

from ..types import Engine, EngineResult, Frame
from ..config import project_root
from ..utils import now_ms, safe_model_dump


def _read_text(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read()


def _write_text(p: str, s: str) -> None:
    with open(p, "w", encoding="utf-8") as f:
        f.write(s)


class OpenAIModerationEngine(Engine):
    name = "OpenAI Moderation"

    # Global rate-limiter shared across instances (important when scanning many files)
    _GLOBAL_LOCK = threading.Lock()
    _GLOBAL_LAST_CALL_MONO: float = 0.0

    # Simple on-disk cache to avoid re-calling OpenAI for the same bytes
    _CACHE_LOCK = threading.RLock()
    _CACHE: Optional[Dict[str, Any]] = None
    _CACHE_PATH: Optional[str] = None
    _CACHE_DIR_READY: bool = False
    _CACHE_DIR_ERROR: bool = False
    _CACHE_DIR_ERROR_REASON: Optional[str] = None
    _CACHE_DIR_ERROR_TIME: float = 0.0
    _CACHE_DIR_RETRY_DELAY: float = 2.0
    _CACHE_DIR_RETRY_MULT: float = 2.0
    _CACHE_DIR_RETRY_MAX: float = 60.0

    # Reduce IO: flush cache periodically (and on process exit)
    _CACHE_DIRTY: bool = False
    _CACHE_WRITES_SINCE_FLUSH: int = 0
    _CACHE_FLUSH_EVERY_N: int = 25
    _ATEXIT_REGISTERED: bool = False

    # If we detect a permanent auth problem (401/403), disable OpenAI for the remainder of the run
    _DISABLED_REASON: Optional[str] = None

    def __init__(self, extra_text: str = "") -> None:
        super().__init__()
        self.extra_text = (extra_text or "").strip()

    def available(self) -> Tuple[bool, str]:
        if os.getenv("OPENAI_DISABLE", "0").strip() == "1":
            return False, "disabled via OPENAI_DISABLE=1"
        if OpenAIModerationEngine._DISABLED_REASON:
            return False, OpenAIModerationEngine._DISABLED_REASON

        key = (os.getenv("OPENAI_API_KEY") or "").strip()
        # Treat common placeholders / empty as not set
        if not key or key.lower() in {"changeme", "your_key_here", "your-api-key", "none"}:
            return False, "OPENAI_API_KEY not set"
        try:
            import openai  # noqa: F401
            return True, ""
        except Exception as e:
            return False, f"missing dependency (pip install openai): {e}"

    def _cache_enabled(self) -> bool:
        return os.getenv("OPENAI_CACHE_ENABLE", "1").strip() == "1"

    def _cache_path(self) -> str:
        # Resolve relative path from project root for predictable behavior.
        if OpenAIModerationEngine._CACHE_PATH:
            return OpenAIModerationEngine._CACHE_PATH
        raw = os.getenv("OPENAI_CACHE_PATH", ".cache/openai_moderation_cache.json")
        raw = raw.strip() or ".cache/openai_moderation_cache.json"
        if not os.path.isabs(raw):
            raw = os.path.join(project_root(), raw)
        OpenAIModerationEngine._CACHE_PATH = raw
        return raw

    def _ensure_cache_dir(self) -> None:
        if OpenAIModerationEngine._CACHE_DIR_READY:
            return
        path = self._cache_path()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        OpenAIModerationEngine._CACHE_DIR_READY = True

    def _load_cache(self) -> Dict[str, Any]:
        with OpenAIModerationEngine._CACHE_LOCK:
            if OpenAIModerationEngine._CACHE is not None:
                return OpenAIModerationEngine._CACHE
            if not self._cache_enabled():
                OpenAIModerationEngine._CACHE = {}
                return OpenAIModerationEngine._CACHE
            path = self._cache_path()
            try:
                if os.path.exists(path):
                    OpenAIModerationEngine._CACHE = json.loads(_read_text(path))
                else:
                    OpenAIModerationEngine._CACHE = {}
            except Exception:
                OpenAIModerationEngine._CACHE = {}
            # Ensure we flush cache on process exit
            if not OpenAIModerationEngine._ATEXIT_REGISTERED:
                atexit.register(self._flush_cache_at_exit)
                OpenAIModerationEngine._ATEXIT_REGISTERED = True
            return OpenAIModerationEngine._CACHE

    def _flush_cache_at_exit(self) -> None:
        try:
            self._save_cache(force=True)
        except Exception:
            pass

    def _save_cache(self, force: bool = False) -> None:
        if not self._cache_enabled():
            return
        if not force and not OpenAIModerationEngine._CACHE_DIRTY:
            return
        self._ensure_cache_dir()
        path = self._cache_path()
        with OpenAIModerationEngine._CACHE_LOCK:
            data = OpenAIModerationEngine._CACHE or {}
            tmp = path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
                os.replace(tmp, path)
                OpenAIModerationEngine._CACHE_DIRTY = False
                OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH = 0
            except Exception:
                # Cache is best-effort
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass

    @staticmethod
    def _is_429(err: Exception) -> bool:
        # Works across openai SDK versions and generic errors
        try:
            sc = getattr(err, "status_code", None)
            if sc == 429:
                return True
        except Exception:
            pass
        try:
            resp = getattr(err, "response", None)
            sc2 = getattr(resp, "status_code", None)
            if sc2 == 429:
                return True
        except Exception:
            pass
        msg = str(err)
        return ("Error code: 429" in msg) or ("Too Many Requests" in msg) or ("rate" in msg.lower() and "429" in msg)

    @staticmethod
    def _status_code(err: Exception) -> Optional[int]:
        """Best-effort HTTP status code extraction across OpenAI SDK versions."""
        for attr in ("status_code", "status"):
            try:
                sc = getattr(err, attr, None)
                if isinstance(sc, int):
                    return sc
            except Exception:
                pass
        for obj in (getattr(err, "response", None), getattr(err, "http_response", None)):
            try:
                sc = getattr(obj, "status_code", None)
                if isinstance(sc, int):
                    return sc
            except Exception:
                pass
        # Fallback: parse from message
        m = re.search(r"Error code:\s*(\d{3})", str(err))
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    @classmethod
    def _is_auth_error(cls, err: Exception) -> bool:
        sc = cls._status_code(err)
        if sc in (401, 403):
            return True
        msg = str(err).lower()
        return ("deactivated" in msg) or ("invalid api key" in msg) or ("unauthorized" in msg)

    @staticmethod
    def _retry_after_seconds(err: Exception) -> Optional[float]:
        # Prefer Retry-After header when available
        for obj in (getattr(err, "response", None), getattr(err, "http_response", None)):
            try:
                if obj is None:
                    continue
                headers = getattr(obj, "headers", None)
                if not headers:
                    continue
                ra = headers.get("retry-after") or headers.get("Retry-After")
                if ra:
                    s = str(ra).strip().lower()
                    if s.endswith("s"):
                        s = s[:-1]
                    return float(s)
            except Exception:
                continue
        return None

    def _throttle_global(self) -> None:
        # Ensures spacing between calls EVEN if each image gets a new engine instance
        try:
            min_interval = float(os.getenv("OPENAI_MIN_INTERVAL_SEC", "1.0"))
        except Exception:
            min_interval = 1.0
        if min_interval <= 0:
            return
        while True:
            with OpenAIModerationEngine._GLOBAL_LOCK:
                now = time.monotonic()
                wait = (OpenAIModerationEngine._GLOBAL_LAST_CALL_MONO + min_interval) - now
                if wait <= 0:
                    # Reserve slot now (so retry loops can’t hammer)
                    OpenAIModerationEngine._GLOBAL_LAST_CALL_MONO = now
                    return
            time.sleep(min(wait, 5.0))

    def _cache_key(self, model_name: str, use_frames: List[Frame]) -> str:
        # Stable key based on bytes + text + model
        h = hashlib.sha256()
        h.update(model_name.encode("utf-8"))
        h.update(b"\n")
        h.update(self.extra_text.encode("utf-8"))
        h.update(b"\n")
        for fr in use_frames:
            # hashing bytes directly avoids pHash collisions in API cache
            h.update(hashlib.sha256(fr.get_jpeg_bytes()).digest())
        return h.hexdigest()

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        start = now_ms()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status="skipped", error=why, took_ms=now_ms() - start)
        if not frames:
            return EngineResult(name=self.name, status="skipped", error="no frames", took_ms=now_ms() - start)

        try:
            from openai import OpenAI

            # Client timeout (prevents extremely long hangs)
            try:
                timeout = float(os.getenv("OPENAI_REQUEST_TIMEOUT_SEC", "20"))
            except Exception:
                timeout = 20.0
            client = OpenAI(timeout=timeout)

            use_n = max(1, int(max_api_frames or 1))
            use_frames = frames[:use_n]

            model_name = os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")

            # Cache
            cache = self._load_cache()
            ck = self._cache_key(model_name, use_frames)
            if self._cache_enabled() and ck in cache:
                cached = cache.get(ck) or {}
                return EngineResult(
                    name=self.name,
                    status="ok",
                    scores=cached.get("scores") or {},
                    details=cached.get("details") or {"cache_hit": True},
                    took_ms=now_ms() - start,
                )

            inputs: List[Dict[str, Any]] = []
            if self.extra_text:
                inputs.append({"type": "text", "text": self.extra_text})
            for fr in use_frames:
                b64 = base64.b64encode(fr.get_jpeg_bytes()).decode("ascii")
                inputs.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

            # Retry / backoff policy
            try:
                max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "6"))
            except Exception:
                max_retries = 6
            try:
                base_sleep = float(os.getenv("OPENAI_BACKOFF_BASE_SEC", "1.0"))
            except Exception:
                base_sleep = 1.0
            try:
                max_sleep = float(os.getenv("OPENAI_BACKOFF_MAX_SEC", "10"))
            except Exception:
                max_sleep = 10.0
            try:
                max_total_sleep = float(os.getenv("OPENAI_MAX_TOTAL_SLEEP_SEC", "30"))
            except Exception:
                max_total_sleep = 30.0
            policy = os.getenv("OPENAI_429_POLICY", "retry").strip().lower()  # retry | skip
            try:
                max_429_retries = int(os.getenv("OPENAI_MAX_429_RETRIES", "3"))
            except Exception:
                max_429_retries = 3

            total_slept = 0.0
            last_err: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    self._throttle_global()
                    resp = client.moderations.create(model=model_name, input=inputs)

                    d = safe_model_dump(resp)
                    r0 = (d.get("results") or [{}])[0]
                    cats = (r0.get("categories") or {})
                    scores = (r0.get("category_scores") or {})

                    wanted = [
                        "sexual",
                        "sexual/minors",
                        "violence",
                        "violence/graphic",
                        "self-harm",
                        "self-harm/intent",
                        "self-harm/instructions",
                        "hate",
                        "hate/threatening",
                        "harassment",
                        "harassment/threatening",
                        "illicit",
                        "illicit/violent",
                    ]
                    out_scores: Dict[str, float] = {}
                    for k in wanted:
                        v = scores.get(k, 0.0)
                        try:
                            out_scores[k] = float(v)
                        except Exception:
                            out_scores[k] = 0.0
                    max_any = max(out_scores.values()) if out_scores else 0.0
                    out_scores["max_any_category"] = float(max_any)
                    out_scores["flagged"] = 1.0 if bool(r0.get("flagged")) else 0.0

                    details = {
                        "categories": cats,
                        "frames_used": [f.idx for f in use_frames],
                        "has_text": bool(self.extra_text),
                        "category_applied_input_types": r0.get("category_applied_input_types"),
                    }

                    # Write cache
                    if self._cache_enabled():
                        with OpenAIModerationEngine._CACHE_LOCK:
                            cache[ck] = {"scores": out_scores, "details": details}
                            # Cap cache size (dict is insertion-ordered)
                            try:
                                cap = int(os.getenv("OPENAI_CACHE_MAX_ITEMS", "2000"))
                            except Exception:
                                cap = 2000
                            if cap > 0 and len(cache) > cap:
                                # Evict oldest
                                for _ in range(len(cache) - cap):
                                    try:
                                        cache.pop(next(iter(cache)))
                                    except Exception:
                                        break
                            OpenAIModerationEngine._CACHE = cache
                        OpenAIModerationEngine._CACHE_DIRTY = True
                        OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH += 1
                        if OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH >= OpenAIModerationEngine._CACHE_FLUSH_EVERY_N:
                            self._save_cache(force=True)

                    return EngineResult(name=self.name, status="ok", scores=out_scores, details=details, took_ms=now_ms() - start)

                except Exception as e:
                    last_err = e
                    # Permanent auth problems: disable immediately (prevents long waits when scanning folders)
                    msg = str(e)
                    sc = getattr(e, "status_code", None)
                    if sc in (401, 403) or ("Error code: 401" in msg) or ("Error code: 403" in msg) or ("deactivated" in msg.lower()):
                        OpenAIModerationEngine._DISABLED_REASON = "OpenAI disabled: invalid/deactivated API key (401/403). Remove OPENAI_API_KEY or set OPENAI_DISABLE=1"
                        break
                    # Fast handling for 429
                    if self._is_429(e):
                        if policy == "skip":
                            break
                        if attempt >= max_429_retries:
                            break
                        ra = self._retry_after_seconds(e)
                        if ra is None:
                            sleep = base_sleep * (2 ** attempt)
                            sleep = sleep * (0.75 + random.random() * 0.5)
                        else:
                            sleep = ra
                        sleep = min(float(sleep), max_sleep)
                        # Don’t stall forever on quota exhaustion
                        if total_slept + sleep > max_total_sleep:
                            break
                        time.sleep(max(0.0, sleep))
                        total_slept += max(0.0, sleep)
                        continue
                    # Anything else: stop retrying
                    break

            # If we hit a permanent auth failure, expose a clean "skipped" reason.
            if OpenAIModerationEngine._DISABLED_REASON:
                return EngineResult(
                    name=self.name,
                    status="skipped",
                    error=OpenAIModerationEngine._DISABLED_REASON,
                    took_ms=now_ms() - start,
                )

            return EngineResult(
                name=self.name,
                status="skipped",
                error=f"rate/quota or error: {last_err}",
                took_ms=now_ms() - start,
            )

        except Exception as e:
            return EngineResult(name=self.name, status="error", error=str(e), took_ms=now_ms() - start)
