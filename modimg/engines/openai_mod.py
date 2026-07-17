from __future__ import annotations

import os
import json
import time
import random
import base64
import threading
import atexit
import hashlib
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..enums import EngineStatus
from ..types import Engine, EngineResult, Frame, redact_engine_output
from ..config import project_root
from ..utils import (
    atomic_write_text,
    env_bool,
    env_float,
    env_int,
    json_dumps_safe,
    now_ms,
    redact_sensitive_text,
    safe_float01,
    safe_model_dump,
)


_CACHE_SCHEMA_VERSION = 1
_KNOWN_MODERATION_CATEGORIES = (
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
)
_AUTH_DISABLED_REASON = (
    "OpenAI disabled: invalid/deactivated API key (401/403). "
    "Remove OPENAI_API_KEY or set OPENAI_DISABLE=1"
)


class OpenAIRunState:
    """Share authentication disablement within one pipeline run."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._disabled_reason: Optional[str] = None

    @property
    def disabled_reason(self) -> Optional[str]:
        with self._lock:
            return self._disabled_reason

    def disable(self, reason: str) -> None:
        with self._lock:
            if self._disabled_reason is None:
                self._disabled_reason = reason


def _default_user_cache_path(*, platform: Optional[str] = None, home: Optional[Path] = None) -> Path:
    """Return a per-user cache path without requiring a platform helper dependency."""
    runtime_platform = sys.platform if platform is None else platform
    home_path = Path.home() if home is None else home
    if runtime_platform.startswith("win"):
        configured = (os.getenv("LOCALAPPDATA") or "").strip()
        base = Path(configured).expanduser() if configured else home_path / "AppData" / "Local"
    elif runtime_platform == "darwin":
        base = home_path / "Library" / "Caches"
    else:
        configured = (os.getenv("XDG_CACHE_HOME") or "").strip()
        xdg_path = Path(configured).expanduser() if configured else None
        base = xdg_path if xdg_path is not None and xdg_path.is_absolute() else home_path / ".cache"
    return base / "py-free-image-moderation" / "openai_moderation_cache.json"


def _read_text(p: str, max_bytes: int) -> str:
    with open(p, "rb") as f:
        raw = f.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"cache exceeds {max_bytes} bytes")
    return raw.decode("utf-8")


def _best_effort_attr(obj: Any, name: str) -> Any:
    """Read attributes from third-party SDK objects whose descriptors may raise."""
    if obj is None:
        return None
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _best_effort_retry_after(obj: Any) -> Optional[float]:
    """Parse Retry-After without letting a malformed SDK response mask the API error."""
    headers = _best_effort_attr(obj, "headers")
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if not raw:
            return None
        value_text = str(raw).strip().lower()
        if value_text.endswith("s"):
            value_text = value_text[:-1]
        value = float(value_text)
    except Exception:
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


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

    _CLIENT_LOCK = threading.RLock()
    _CLIENT: Optional[Any] = None
    _CLIENT_TIMEOUT: Optional[float] = None
    _CLIENT_API_KEY: Optional[str] = None

    def __init__(self, extra_text: str = "", *, run_state: Optional[OpenAIRunState] = None) -> None:
        super().__init__()
        self.extra_text = (extra_text or "").strip()
        self._run_state = run_state or OpenAIRunState()

    def available(self) -> Tuple[bool, str]:
        if env_bool("OPENAI_DISABLE", False):
            return False, "disabled via OPENAI_DISABLE=1"
        disabled_reason = self._run_state.disabled_reason
        if disabled_reason:
            return False, disabled_reason

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
        return env_bool("OPENAI_CACHE_ENABLE", True)

    def _cache_path(self) -> str:
        configured = os.getenv("OPENAI_CACHE_PATH")
        if configured is not None and configured.strip():
            raw = configured.strip()
            # Preserve the historical project-root semantics for explicit
            # relative OPENAI_CACHE_PATH values.
            if not os.path.isabs(raw):
                raw = str(Path(project_root()) / raw)
        else:
            raw = str(_default_user_cache_path())
        with OpenAIModerationEngine._CACHE_LOCK:
            if OpenAIModerationEngine._CACHE_PATH and OpenAIModerationEngine._CACHE_PATH != raw:
                OpenAIModerationEngine._CACHE = None
                OpenAIModerationEngine._CACHE_DIR_READY = False
                OpenAIModerationEngine._CACHE_DIRTY = False
                OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH = 0
                OpenAIModerationEngine._CACHE_DIR_ERROR = False
                OpenAIModerationEngine._CACHE_DIR_ERROR_REASON = None
                OpenAIModerationEngine._CACHE_DIR_ERROR_TIME = 0.0
                OpenAIModerationEngine._CACHE_DIR_RETRY_DELAY = 2.0
            if not OpenAIModerationEngine._CACHE_PATH or OpenAIModerationEngine._CACHE_PATH != raw:
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
            path = self._cache_path()
            if OpenAIModerationEngine._CACHE is not None:
                return OpenAIModerationEngine._CACHE
            if not self._cache_enabled():
                OpenAIModerationEngine._CACHE = {}
                return OpenAIModerationEngine._CACHE
            try:
                if os.path.exists(path):
                    max_bytes = max(1, env_int("OPENAI_CACHE_MAX_BYTES", 10_000_000))
                    if os.path.getsize(path) > max_bytes:
                        self.logger.warning("OpenAI cache ignored because it exceeds %s bytes", max_bytes)
                        loaded: Any = {}
                    else:
                        loaded = json.loads(_read_text(path, max_bytes))
                    OpenAIModerationEngine._CACHE = loaded if isinstance(loaded, dict) else {}
                else:
                    OpenAIModerationEngine._CACHE = {}
            except (OSError, UnicodeError, ValueError, RecursionError) as exc:
                self.logger.warning("OpenAI cache ignored: %s", redact_sensitive_text(f"{type(exc).__name__}: {exc}"))
                OpenAIModerationEngine._CACHE = {}
            # Ensure we flush cache on process exit
            if not OpenAIModerationEngine._ATEXIT_REGISTERED:
                atexit.register(self._flush_cache_at_exit)
                OpenAIModerationEngine._ATEXIT_REGISTERED = True
            return OpenAIModerationEngine._CACHE

    def _flush_cache_at_exit(self) -> None:
        try:
            with OpenAIModerationEngine._CACHE_LOCK:
                if not OpenAIModerationEngine._CACHE_DIRTY:
                    return
            self._save_cache(force=True)
        except Exception as exc:
            self.logger.warning("OpenAI cache exit flush failed: %s", redact_sensitive_text(exc))

    def _cache_payload_with_limits(self, data: Dict[str, Any]) -> str:
        """Serialize cache after deterministic oldest-first item and byte eviction."""
        item_limit = max(0, env_int("OPENAI_CACHE_MAX_ITEMS", 2000))
        while item_limit > 0 and len(data) > item_limit:
            oldest = next(iter(data))
            data.pop(oldest, None)

        byte_limit = max(1, env_int("OPENAI_CACHE_MAX_BYTES", 10_000_000))
        if byte_limit < len("{}".encode("utf-8")):
            raise ValueError("OPENAI_CACHE_MAX_BYTES is too small for a JSON cache")

        while True:
            payload = json_dumps_safe(
                data,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if len(payload.encode("utf-8")) <= byte_limit:
                return payload
            oldest = next(iter(data), None)
            if oldest is None:
                raise ValueError("OPENAI_CACHE_MAX_BYTES is too small for a JSON cache")
            data.pop(oldest, None)

    def _save_cache(self, force: bool = False) -> None:
        if not self._cache_enabled():
            return
        with OpenAIModerationEngine._CACHE_LOCK:
            if not force and not OpenAIModerationEngine._CACHE_DIRTY:
                return
            if (
                not force
                and OpenAIModerationEngine._CACHE_DIR_ERROR
                and time.monotonic() - OpenAIModerationEngine._CACHE_DIR_ERROR_TIME
                < OpenAIModerationEngine._CACHE_DIR_RETRY_DELAY
            ):
                return
            try:
                self._ensure_cache_dir()
                path = self._cache_path()
                data = OpenAIModerationEngine._CACHE or {}
                payload = self._cache_payload_with_limits(data)
                OpenAIModerationEngine._CACHE = data
                atomic_write_text(path, payload)
                OpenAIModerationEngine._CACHE_DIRTY = False
                OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH = 0
                OpenAIModerationEngine._CACHE_DIR_ERROR = False
                OpenAIModerationEngine._CACHE_DIR_ERROR_REASON = None
                OpenAIModerationEngine._CACHE_DIR_ERROR_TIME = 0.0
                OpenAIModerationEngine._CACHE_DIR_RETRY_DELAY = 2.0
            except (OSError, TypeError, ValueError, RecursionError) as exc:
                reason = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
                OpenAIModerationEngine._CACHE_DIR_ERROR = True
                OpenAIModerationEngine._CACHE_DIR_ERROR_REASON = reason
                OpenAIModerationEngine._CACHE_DIR_ERROR_TIME = time.monotonic()
                OpenAIModerationEngine._CACHE_DIR_RETRY_DELAY = min(
                    OpenAIModerationEngine._CACHE_DIR_RETRY_MAX,
                    max(2.0, OpenAIModerationEngine._CACHE_DIR_RETRY_DELAY * OpenAIModerationEngine._CACHE_DIR_RETRY_MULT),
                )
                self.logger.warning("OpenAI cache save failed; moderation result remains valid: %s", reason)

    @staticmethod
    def _is_429(err: Exception) -> bool:
        # Works across openai SDK versions and generic errors
        if _best_effort_attr(err, "status_code") == 429:
            return True
        response = _best_effort_attr(err, "response")
        if _best_effort_attr(response, "status_code") == 429:
            return True
        msg = str(err)
        return ("Error code: 429" in msg) or ("Too Many Requests" in msg) or ("rate" in msg.lower() and "429" in msg)

    @staticmethod
    def _status_code(err: Exception) -> Optional[int]:
        """Best-effort HTTP status code extraction across OpenAI SDK versions."""
        for attr in ("status_code", "status"):
            sc = _best_effort_attr(err, attr)
            if isinstance(sc, int):
                return sc
        for obj in (_best_effort_attr(err, "response"), _best_effort_attr(err, "http_response")):
            sc = _best_effort_attr(obj, "status_code")
            if isinstance(sc, int):
                return sc
        # Fallback: parse from message
        m = re.search(r"Error code:\s*(\d{3})", str(err))
        if m:
            return int(m.group(1))
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
        for obj in (_best_effort_attr(err, "response"), _best_effort_attr(err, "http_response")):
            retry_after = _best_effort_retry_after(obj)
            if retry_after is not None:
                return retry_after
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

    @staticmethod
    def _validated_cache_entry(entry: Any, model_name: str) -> Optional[Dict[str, Any]]:
        """Return a normalized trusted cache entry, or None for malformed data."""
        if not isinstance(entry, dict):
            return None

        schema_version = entry.get("schema_version")
        if schema_version is not None and (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != _CACHE_SCHEMA_VERSION
        ):
            return None

        cached_model = entry.get("model")
        if cached_model is not None and (not isinstance(cached_model, str) or cached_model != model_name):
            return None

        raw_scores = entry.get("scores")
        if not isinstance(raw_scores, dict):
            return None
        if not any(category in raw_scores for category in _KNOWN_MODERATION_CATEGORIES):
            return None
        if "flagged" not in raw_scores:
            return None

        scores: Dict[str, float] = {}
        for key, value in raw_scores.items():
            if not isinstance(key, str):
                return None
            if isinstance(value, bool):
                if key != "flagged":
                    return None
                parsed = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                parsed = float(value)
            else:
                return None
            if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
                return None
            if key == "flagged" and parsed not in (0.0, 1.0):
                return None
            scores[key] = parsed

        known_scores = [scores[key] for key in _KNOWN_MODERATION_CATEGORIES if key in scores]
        scores["max_any_category"] = max(known_scores)

        raw_details = entry.get("details", {})
        if not isinstance(raw_details, dict):
            return None
        return {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "model": model_name,
            "scores": scores,
            "details": dict(raw_details),
        }

    def _cache_hit_result(
        self,
        cache: Dict[str, Any],
        cache_key: str,
        model_name: str,
        start: int,
    ) -> Optional[EngineResult]:
        with OpenAIModerationEngine._CACHE_LOCK:
            if cache_key not in cache:
                return None
            normalized = self._validated_cache_entry(cache.get(cache_key), model_name)
            if normalized is None:
                cache.pop(cache_key, None)
                OpenAIModerationEngine._CACHE = cache
                OpenAIModerationEngine._CACHE_DIRTY = True
                OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH += 1
                return None

            # Dict order is persisted, so moving a hit to the end implements a
            # deterministic least-recently-used eviction policy.
            cache.pop(cache_key, None)
            cache[cache_key] = normalized
            OpenAIModerationEngine._CACHE = cache
            OpenAIModerationEngine._CACHE_DIRTY = True
            details = dict(normalized["details"])
            details["cache_hit"] = True
            scores = dict(normalized["scores"])

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores=scores,
            details=details,
            took_ms=now_ms() - start,
        )

    @classmethod
    def _client_for_timeout(cls, timeout: float):
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        with cls._CLIENT_LOCK:
            if cls._CLIENT is not None and cls._CLIENT_TIMEOUT == timeout and cls._CLIENT_API_KEY == api_key:
                return cls._CLIENT
            from openai import OpenAI

            cls._CLIENT = OpenAI(timeout=timeout)
            cls._CLIENT_TIMEOUT = timeout
            cls._CLIENT_API_KEY = api_key
            return cls._CLIENT

    def execute(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        """Check the cache before applying API availability requirements."""
        start = now_ms()
        try:
            result = self.run(path, frames, max_api_frames=max_api_frames)
            if result.took_ms is None:
                result.took_ms = now_ms() - start
            if result.error:
                result.error = redact_engine_output(result.error, frames)
            return result
        except Exception as exc:
            error = redact_engine_output(f"{type(exc).__name__}: {exc}", frames)
            self.logger.warning("engine failed: %s", error)
            return EngineResult(name=self.name, status=EngineStatus.ERROR, error=error, took_ms=now_ms() - start)

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        start = now_ms()
        if env_bool("OPENAI_DISABLE", False):
            return EngineResult(
                name=self.name,
                status=EngineStatus.SKIPPED,
                error="disabled via OPENAI_DISABLE=1",
                took_ms=now_ms() - start,
            )
        if not frames:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error="no frames", took_ms=now_ms() - start)

        try:
            use_n = max(1, int(max_api_frames or 1))
            use_frames = frames[:use_n]

            model_name = os.getenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest").strip() or "omni-moderation-latest"

            # Cache
            cache: Dict[str, Any] = {}
            ck: Optional[str] = None
            if self._cache_enabled():
                cache = self._load_cache()
                ck = self._cache_key(model_name, use_frames)
                cached_result = self._cache_hit_result(cache, ck, model_name, start)
                if cached_result is not None:
                    return cached_result

            ok, why = self.available()
            if not ok:
                return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=why, took_ms=now_ms() - start)

            # A client is needed only after a validated cache miss.
            timeout = env_float("OPENAI_REQUEST_TIMEOUT_SEC", 20.0, min_value=0.1)
            client = self._client_for_timeout(timeout)

            inputs: List[Dict[str, Any]] = []
            if self.extra_text:
                inputs.append({"type": "text", "text": self.extra_text})
            for fr in use_frames:
                b64 = base64.b64encode(fr.get_jpeg_bytes()).decode("ascii")
                inputs.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

            # Retry / backoff policy
            max_retries = max(0, env_int("OPENAI_MAX_RETRIES", 6))
            base_sleep = env_float("OPENAI_BACKOFF_BASE_SEC", 1.0, min_value=0.0)
            max_sleep = env_float("OPENAI_BACKOFF_MAX_SEC", 10.0, min_value=0.0)
            max_total_sleep = env_float("OPENAI_MAX_TOTAL_SLEEP_SEC", 30.0, min_value=0.0)
            policy = os.getenv("OPENAI_429_POLICY", "retry").strip().lower()  # retry | skip
            max_429_retries = max(0, env_int("OPENAI_MAX_429_RETRIES", 3))

            total_slept = 0.0
            last_err: Optional[Exception] = None
            last_was_429 = False

            for attempt in range(max_retries + 1):
                try:
                    disabled_reason = self._run_state.disabled_reason
                    if disabled_reason:
                        return EngineResult(
                            name=self.name,
                            status=EngineStatus.SKIPPED,
                            error=disabled_reason,
                            took_ms=now_ms() - start,
                        )
                    self._throttle_global()
                    resp = client.moderations.create(model=model_name, input=inputs)

                    d = safe_model_dump(resp)
                    if not isinstance(d, dict):
                        raise RuntimeError("OpenAI returned a non-object response")
                    raw_results = d.get("results")
                    if not isinstance(raw_results, list) or not raw_results or not isinstance(raw_results[0], dict):
                        raise RuntimeError("OpenAI response did not contain a moderation result")
                    r0 = raw_results[0]
                    cats = r0.get("categories") if isinstance(r0.get("categories"), dict) else {}
                    scores = r0.get("category_scores")
                    if not isinstance(scores, dict):
                        raise RuntimeError("OpenAI response did not contain category_scores")

                    wanted = _KNOWN_MODERATION_CATEGORIES
                    if not any(key in scores for key in wanted):
                        raise RuntimeError("OpenAI response contained no recognized moderation scores")
                    out_scores: Dict[str, float] = {}
                    for k in wanted:
                        v = scores.get(k, 0.0)
                        if isinstance(v, bool):
                            raise RuntimeError(f"OpenAI returned a non-numeric score for {k}")
                        try:
                            parsed = float(v)
                        except (TypeError, ValueError, OverflowError) as exc:
                            raise RuntimeError(f"OpenAI returned a non-numeric score for {k}") from exc
                        if not math.isfinite(parsed):
                            raise RuntimeError(f"OpenAI returned a non-finite score for {k}")
                        out_scores[k] = safe_float01(parsed)
                    max_any = max(out_scores.values()) if out_scores else 0.0
                    out_scores["max_any_category"] = float(max_any)
                    raw_flagged = r0.get("flagged")
                    if not isinstance(raw_flagged, bool):
                        raise RuntimeError("OpenAI returned an invalid flagged value")
                    out_scores["flagged"] = 1.0 if raw_flagged else 0.0

                    details = {
                        "categories": cats,
                        "frames_used": [f.idx for f in use_frames],
                        "has_text": bool(self.extra_text),
                        "category_applied_input_types": r0.get("category_applied_input_types"),
                    }

                    # Write cache
                    if self._cache_enabled() and ck is not None:
                        with OpenAIModerationEngine._CACHE_LOCK:
                            cache.pop(ck, None)
                            cache[ck] = {
                                "schema_version": _CACHE_SCHEMA_VERSION,
                                "model": model_name,
                                "scores": out_scores,
                                "details": details,
                            }
                            # Cap cache size (dict is insertion-ordered)
                            cap = max(0, env_int("OPENAI_CACHE_MAX_ITEMS", 2000))
                            if cap > 0 and len(cache) > cap:
                                # Evict oldest
                                for _ in range(len(cache) - cap):
                                    oldest = next(iter(cache), None)
                                    if oldest is None:
                                        break
                                    cache.pop(oldest, None)
                            OpenAIModerationEngine._CACHE = cache
                            OpenAIModerationEngine._CACHE_DIRTY = True
                            OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH += 1
                            should_flush = (
                                OpenAIModerationEngine._CACHE_WRITES_SINCE_FLUSH
                                >= OpenAIModerationEngine._CACHE_FLUSH_EVERY_N
                            )
                        if should_flush:
                            self._save_cache(force=True)

                    return EngineResult(name=self.name, status=EngineStatus.OK, scores=out_scores, details=details, took_ms=now_ms() - start)

                except Exception as e:
                    last_err = e
                    last_was_429 = self._is_429(e)
                    # Permanent auth problems: disable immediately (prevents long waits when scanning folders)
                    if self._is_auth_error(e):
                        self._run_state.disable(_AUTH_DISABLED_REASON)
                        break
                    # Fast handling for 429
                    if last_was_429:
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
            disabled_reason = self._run_state.disabled_reason
            if disabled_reason:
                return EngineResult(
                    name=self.name,
                    status=EngineStatus.SKIPPED,
                    error=disabled_reason,
                    took_ms=now_ms() - start,
                )

            if last_was_429:
                return EngineResult(
                    name=self.name,
                    status=EngineStatus.SKIPPED,
                    error=redact_sensitive_text(f"rate/quota error: {last_err}"),
                    took_ms=now_ms() - start,
                )
            return EngineResult(
                name=self.name,
                status=EngineStatus.ERROR,
                error=redact_sensitive_text(f"request failed: {type(last_err).__name__}: {last_err}"),
                took_ms=now_ms() - start,
            )

        except Exception as e:
            return EngineResult(
                name=self.name,
                status=EngineStatus.ERROR,
                error=redact_sensitive_text(f"{type(e).__name__}: {e}"),
                took_ms=now_ms() - start,
            )
