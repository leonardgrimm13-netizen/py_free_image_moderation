from __future__ import annotations

import os
import json
import math
import threading
from typing import Any, Dict, List, Optional, Tuple

from ..enums import EngineStatus
from ..types import Engine, EngineResult, Frame, mk_skipped
from ..utils import env_bool, env_float, now_ms, redact_sensitive_text, safe_float01


class SightengineRunState:
    """Share quota disablement within one pipeline run, never across runs."""

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


class SightengineEngine(Engine):
    name = "Sightengine"
    _SESSION_LOCAL = threading.local()

    def __init__(self, models: Optional[str] = None, *, run_state: Optional[SightengineRunState] = None) -> None:
        super().__init__()
        self._run_state = run_state or SightengineRunState()
        # Which Sightengine models to call. Keep default simple but useful.
        raw = models if models is not None else os.getenv("SIGHTENGINE_MODELS", "nudity-2.1,weapon,violence,gore-2.0,offensive-2.0")
        self.models = self._normalize_models(raw)
        # Credentials are read from env; refresh before every call (so .env and runtime $env:... both work).
        self.api_user = os.getenv("SIGHTENGINE_USER", "").strip()
        self.api_secret = os.getenv("SIGHTENGINE_SECRET", "").strip()

    @staticmethod
    def _normalize_models(raw: Any) -> str:
        """Accept comma-separated strings or list-like strings from .env.

        Examples that should work:
          nudity-2.1,weapon
          ['nudity-2.1', 'weapon']
          ["nudity-2.1","weapon"]
        """
        if raw is None:
            return ""
        if isinstance(raw, (list, tuple)):
            items = [str(x) for x in raw]
        else:
            s = str(raw).strip()
            # Strip surrounding brackets if it looks like a list
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip()
            items = [p.strip() for p in s.split(",")]
        out: List[str] = []
        for m in items:
            mm = str(m).strip()
            # Remove common wrappers/noise
            mm = mm.strip().strip('"').strip("'")
            mm = mm.strip().lstrip("[").rstrip("]")
            mm = mm.strip().strip('"').strip("'")
            if mm:
                out.append(mm)
        # Deduplicate while preserving order
        seen = set()
        uniq: List[str] = []
        for m in out:
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        return ",".join(uniq)

    def _refresh_creds(self) -> None:
        self.api_user = os.getenv("SIGHTENGINE_USER", "").strip()
        self.api_secret = os.getenv("SIGHTENGINE_SECRET", "").strip()

    @classmethod
    def _session(cls) -> Any:
        import requests  # type: ignore

        module_id = id(requests)
        session = getattr(cls._SESSION_LOCAL, "session", None)
        session_module_id = getattr(cls._SESSION_LOCAL, "module_id", None)
        if session is not None and session_module_id == module_id:
            return session

        session_factory = getattr(requests, "Session", None)
        session = session_factory() if callable(session_factory) else requests
        cls._SESSION_LOCAL.session = session
        cls._SESSION_LOCAL.module_id = module_id
        return session

    def available(self) -> Tuple[bool, str]:
        if env_bool("SIGHTENGINE_DISABLE", False):
            return False, "disabled via SIGHTENGINE_DISABLE=1"
        run_disabled_reason = self._run_state.disabled_reason
        if run_disabled_reason:
            self.disabled_reason = run_disabled_reason
        if self.disabled_reason:
            return False, self.disabled_reason
        # Ensure attributes exist + pick up any late env changes.
        self._refresh_creds()
        if not (self.api_user and self.api_secret):
            return False, "SIGHTENGINE_USER / SIGHTENGINE_SECRET not set"
        return True, ""

    def _disable_for_run(self, reason: str) -> None:
        self._run_state.disable(reason)
        self.disable(reason)

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        start = now_ms()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=why, took_ms=now_ms() - start)

        try:
            session = self._session()
        except Exception as e:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=f"missing dependency (pip install -U requests): {e}", took_ms=now_ms() - start)

        if not frames:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error="no frames", took_ms=now_ms() - start)

        try:
            limit = max(1, int(max_api_frames or 1))
        except Exception:
            limit = 1
        use_frames = frames[:limit]
        url = "https://api.sightengine.com/1.0/check.json"

        # credentials already refreshed in available()
        params_base = {
            "models": self.models,
            "api_user": self.api_user,
            "api_secret": self.api_secret,
        }

        def _extract_scores(data: Dict[str, Any]) -> Dict[str, float]:
            scores: Dict[str, float] = {}
            missing = object()
            metadata_keys = {"id", "request_id", "model", "model_version", "version", "status", "type", "metadata"}

            def _finite_number(value: Any) -> float | None:
                # ``bool`` is an ``int`` subclass in Python but is not a
                # valid probability from the external API.
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    return None
                try:
                    parsed = float(value)
                except (OverflowError, ValueError):
                    return None
                return parsed if math.isfinite(parsed) else None

            def _probability(value: Any) -> float:
                parsed = _finite_number(value)
                if parsed is None or not 0.0 <= parsed <= 1.0:
                    raise ValueError("Sightengine returned an invalid moderation probability; expected a finite number in [0, 1]")
                return parsed

            def _record_score(key: str, value: Any) -> float | None:
                if value is missing:
                    return None
                parsed = _probability(value)
                scores[key] = parsed
                return parsed

            # Total operations used (Sightengine sometimes counts per-model operations)
            ops = data.get("operations")
            parsed_ops = _finite_number(ops)
            if parsed_ops is not None:
                scores["operations_used"] = max(0.0, parsed_ops)

            def _pick_model(*names: str) -> Any:
                for n in names:
                    if n in data:
                        return data[n]
                return missing

            def _score_mapping(value: Any, prefix: str) -> None:
                if value is missing:
                    return
                if not isinstance(value, dict):
                    raise ValueError("Sightengine returned an invalid moderation score container; expected an object")
                for key, probability in value.items():
                    if key not in metadata_keys:
                        _record_score(f"{prefix}_{key}", probability)

            # Legacy and advanced nudity fields can coexist. Evaluate both
            # families before combining risk so neither silently hides the other.
            nud = _pick_model("nudity", "nudity-2.1", "nudity_2_1")
            if nud is not missing:
                if not isinstance(nud, dict):
                    raise ValueError("Sightengine returned an invalid nudity model; expected an object")
                legacy_present = any(key in nud for key in ("raw", "partial", "safe"))
                for key in ("raw", "partial", "safe"):
                    _record_score(f"nudity_{key}", nud.get(key, missing))

                advanced_present = False

                def _num(value: Any) -> float:
                    nonlocal advanced_present
                    if value is missing:
                        return 0.0
                    parsed = _probability(value)
                    advanced_present = True
                    return parsed

                safe = _num(nud.get("none", missing))
                raw = max(_num(nud.get(key, missing)) for key in ("sexual_activity", "sexual_display", "erotica"))
                partial = max(_num(nud.get(key, missing)) for key in ("very_suggestive", "suggestive", "mildly_suggestive"))

                suggestive_max = 0.0
                safe_labels = {"none", "safe", "neutral", "other", "non_suggestive", "normal", "ok", "no_nudity", "non_nudity", "clothed", "fully_clothed", "covered", "not_nude", "nonnude"}

                def _walk_max(obj: Any, include_risk: bool = True) -> None:
                    nonlocal suggestive_max, advanced_present
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            label = str(key).strip().lower()
                            if label not in metadata_keys:
                                _walk_max(value, include_risk and label not in safe_labels)
                    elif isinstance(obj, (list, tuple)):
                        for value in obj:
                            _walk_max(value, include_risk)
                    else:
                        probability = _probability(obj)
                        advanced_present = True
                        if include_risk:
                            suggestive_max = max(suggestive_max, probability)

                suggestive = nud.get("suggestive_classes", missing)
                if suggestive is not missing:
                    if not isinstance(suggestive, (dict, list, tuple)):
                        raise ValueError("Sightengine returned an invalid suggestive_classes container")
                    _walk_max(suggestive)
                partial = max(partial, suggestive_max)
                if safe > 0.0:
                    partial = min(partial, max(0.0, 1.0 - safe))

                if advanced_present:
                    if legacy_present:
                        legacy_safe = scores.get("nudity_safe", 0.0)
                        legacy_partial = scores.get("nudity_partial", 0.0)
                        if legacy_safe > 0.0:
                            legacy_partial = min(legacy_partial, max(0.0, 1.0 - legacy_safe))
                        raw = max(raw, scores.get("nudity_raw", 0.0))
                        partial = max(partial, legacy_partial)
                        safe = min(safe, legacy_safe)
                    scores.update(nudity_safe=safe, nudity_raw=raw, nudity_partial=partial)

            # Weapon scores support nested class/type/action mappings and flat
            # probabilities. Explicit nulls or malformed containers are errors.
            wpn = _pick_model("weapon", "weapons")
            if wpn is not missing:
                if not isinstance(wpn, dict):
                    raise ValueError("Sightengine returned an invalid weapon model; expected an object")
                _score_mapping(wpn.get("classes", missing), "weapon")
                for key in ("firearm", "knife", "firearm_toy"):
                    _record_score(f"weapon_{key}", wpn.get(key, missing))
                _score_mapping(wpn.get("firearm_type", missing), "weapon_firearm_type")
                _score_mapping(wpn.get("firearm_action", missing), "weapon_firearm_action")
                gesture = wpn.get("firearm_gesture", missing)
                if isinstance(gesture, dict):
                    _score_mapping(gesture, "weapon_firearm_action")
                else:
                    _record_score("weapon_firearm_gesture", gesture)

            def _parse_prob_classes(model_obj: Any, prefix: str) -> None:
                if model_obj is missing:
                    return
                if not isinstance(model_obj, dict):
                    # Older/alternate schemas return a single probability.
                    _record_score(f"{prefix}_prob", model_obj)
                    return
                _record_score(f"{prefix}_prob", model_obj.get("prob", missing))
                _score_mapping(model_obj.get("classes", missing), prefix)
                for key, value in model_obj.items():
                    if key not in {"prob", "classes"} | metadata_keys:
                        _record_score(f"{prefix}_{key}", value)

            _parse_prob_classes(_pick_model("gore", "gore-2.0", "gore_2_0"), "gore")
            _parse_prob_classes(_pick_model("violence", "violence-2.0", "violence_2_0"), "violence")
            _parse_prob_classes(_pick_model("offensive", "offensive-2.0", "offensive_2_0"), "offensive")
            offensive_scores = [value for key, value in scores.items() if key.startswith("offensive_")]
            if offensive_scores:
                scores["offensive_max"] = max(offensive_scores)

            normalized: Dict[str, float] = {}
            for key, value in scores.items():
                parsed = _finite_number(value)
                if parsed is None:
                    continue
                normalized[key] = max(0.0, parsed) if key == "operations_used" else safe_float01(parsed)
            return normalized

        best_scores: Dict[str, float] = {}
        per_frame: List[Dict[str, Any]] = []

        for fr in use_frames:
            files = {"media": ("frame.jpg", fr.get_jpeg_bytes(), "image/jpeg")}
            try:
                timeout = env_float("SIGHTENGINE_TIMEOUT_SEC", 60.0, min_value=0.1)
                r = session.post(url, data=params_base, files=files, timeout=timeout)
            except Exception as exc:
                return EngineResult(
                    name=self.name,
                    status=EngineStatus.ERROR,
                    error=redact_sensitive_text(f"request failed: {type(exc).__name__}: {exc}"),
                    took_ms=now_ms() - start,
                )

            try:
                try:
                    status_code = int(r.status_code)
                except (TypeError, ValueError, OverflowError):
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error="Sightengine returned an invalid HTTP status",
                        took_ms=now_ms() - start,
                    )
                if status_code in (402, 403, 429):
                    self._disable_for_run(f"quota/limit http={status_code}")
                    return mk_skipped(self, self.disabled_reason or "quota/limit", took_ms=now_ms() - start)
                if status_code >= 400:
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error=f"http error {status_code}",
                        took_ms=now_ms() - start,
                    )

                try:
                    headers = getattr(r, "headers", {}) or {}
                    data = (
                        r.json()
                        if "application/json" in str(headers.get("content-type", "")).lower()
                        else json.loads(r.text or "{}")
                    )
                except Exception as exc:
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error=redact_sensitive_text(f"invalid JSON response: {type(exc).__name__}: {exc}"),
                        took_ms=now_ms() - start,
                    )
                if not isinstance(data, dict):
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error="invalid JSON response: expected an object",
                        took_ms=now_ms() - start,
                    )
                if str(data.get("status") or "").lower() != "success":
                    err = data.get("error") or data.get("message") or str(data)
                    if "quota" in str(err).lower() or "limit" in str(err).lower():
                        self._disable_for_run(f"quota/limit: {redact_sensitive_text(err)[:200]}")
                        return mk_skipped(self, self.disabled_reason or "quota/limit", took_ms=now_ms() - start)
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error=redact_sensitive_text(err)[:400],
                        details={"api_status": str(data.get("status") or "")[:80]},
                        took_ms=now_ms() - start,
                    )

                try:
                    sc = _extract_scores(data)
                except ValueError as exc:
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error=str(exc),
                        took_ms=now_ms() - start,
                    )
                if not any(key != "operations_used" for key in sc):
                    return EngineResult(
                        name=self.name,
                        status=EngineStatus.ERROR,
                        error="Sightengine success response contained no recognized moderation scores",
                        took_ms=now_ms() - start,
                    )
            finally:
                close_response = getattr(r, "close", None)
                if callable(close_response):
                    try:
                        close_response()
                    except Exception as exc:
                        self.logger.warning("failed to close Sightengine response: %s", type(exc).__name__)
            per_frame.append({"frame": int(fr.idx), "scores": sc})
            for k, v in sc.items():
                if isinstance(v, (int, float)):
                    best_scores[k] = max(float(best_scores.get(k, 0.0)), float(v))

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={k: float(v) for k, v in best_scores.items()},
            details={"per_frame": per_frame, "frames_used": [int(fr.idx) for fr in use_frames], "models": self.models},
            took_ms=now_ms() - start,
        )
