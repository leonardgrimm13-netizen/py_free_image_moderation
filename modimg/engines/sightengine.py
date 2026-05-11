from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

from ..enums import EngineStatus
from ..types import Engine, EngineResult, Frame, mk_skipped
from ..utils import now_ms


class SightengineEngine(Engine):
    name = "Sightengine"

    def __init__(self, models: Optional[str] = None) -> None:
        super().__init__()
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

    def available(self) -> Tuple[bool, str]:
        # Ensure attributes exist + pick up any late env changes
        self._refresh_creds()
        if not (self.api_user and self.api_secret):
            return False, "SIGHTENGINE_USER / SIGHTENGINE_SECRET not set"
        return True, ""

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        start = now_ms()
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=why, took_ms=now_ms() - start)

        try:
            import requests  # type: ignore
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

            # Total operations used (Sightengine sometimes counts per-model operations)
            ops = data.get("operations")
            if isinstance(ops, (int, float)):
                scores["operations_used"] = float(ops)

            def _pick_model(*names: str) -> Any:
                for n in names:
                    if n in data:
                        return data.get(n)
                return None

            # Nudity: supports both legacy schema (raw/partial/safe) and advanced nudity-2.1 (intensity + suggestive classes)
            nud = _pick_model("nudity", "nudity-2.1", "nudity_2_1")
            if isinstance(nud, dict):
                legacy_found = False
                for kk in ("raw", "partial", "safe"):
                    vv = nud.get(kk)
                    if isinstance(vv, (int, float)):
                        scores[f"nudity_{kk}"] = float(vv)
                        legacy_found = True

                if not legacy_found:
                    def _num(v: Any) -> float:
                        return float(v) if isinstance(v, (int, float)) else 0.0

                    # Intensity classes (docs): sexual_activity, sexual_display, erotica, very_suggestive, suggestive, mildly_suggestive, none
                    safe = _num(nud.get("none", nud.get("safe", 0.0)))
                    raw = max(_num(nud.get("sexual_activity")), _num(nud.get("sexual_display")), _num(nud.get("erotica")))
                    partial_intensity = max(_num(nud.get("very_suggestive")), _num(nud.get("suggestive")), _num(nud.get("mildly_suggestive")))

                    # Suggestive classes live under nudity.suggestive_classes.* (nested dicts)
                    sugg_max = 0.0
                    def _walk_max(obj: Any) -> None:
                        nonlocal sugg_max
                        if isinstance(obj, dict):
                            for kk, vv in obj.items():
                                kl = str(kk).strip().lower()
                                # Skip safe/non-suggestive labels often present in nested structures
                                if kl in {"none", "safe", "neutral", "other", "non_suggestive", "normal", "ok", "no_nudity", "non_nudity", "clothed", "fully_clothed", "covered", "not_nude", "nonnude"}:
                                    continue
                                if isinstance(vv, (int, float)):
                                    val = float(vv)
                                    if val > sugg_max:
                                        sugg_max = val
                                else:
                                    _walk_max(vv)
                        elif isinstance(obj, (list, tuple)):
                            for vv in obj:
                                _walk_max(vv)

                    _walk_max(nud.get("suggestive_classes"))

                    partial = max(partial_intensity, sugg_max)

                    # Clamp to [0,1] just in case
                    safe = max(0.0, min(1.0, safe))
                    raw = max(0.0, min(1.0, raw))
                    partial = max(0.0, min(1.0, partial))
                    # If 'safe/none' is high, partial should be low. Cap it to (1-safe) to avoid false positives.
                    if safe > 0.0:
                        partial = min(partial, max(0.0, 1.0 - safe))

                    scores["nudity_safe"] = safe
                    scores["nudity_raw"] = raw
                    scores["nudity_partial"] = partial

            # Weapon model: dict with classes + firearm_type + firearm_action (schema can vary slightly)
            wpn = _pick_model("weapon", "weapons")
            if isinstance(wpn, dict):
                # common schema: weapon.classes.*
                classes = wpn.get("classes")
                if isinstance(classes, dict):
                    for kk, vv in classes.items():
                        if isinstance(vv, (int, float)):
                            scores[f"weapon_{kk}"] = float(vv)

                # some responses put scores directly under weapon.*
                for kk in ("firearm", "knife", "firearm_toy", "firearm_gesture"):
                    vv = wpn.get(kk)
                    if isinstance(vv, (int, float)):
                        scores[f"weapon_{kk}"] = float(vv)

                ft = wpn.get("firearm_type")
                if isinstance(ft, dict):
                    for kk, vv in ft.items():
                        if isinstance(vv, (int, float)):
                            scores[f"weapon_firearm_type_{kk}"] = float(vv)

                fa = wpn.get("firearm_action") or wpn.get("firearm_gesture")  # some variants
                if isinstance(fa, dict):
                    for kk, vv in fa.items():
                        if isinstance(vv, (int, float)):
                            scores[f"weapon_firearm_action_{kk}"] = float(vv)

            def _parse_prob_classes(model_obj: Any, prefix: str) -> None:
                if isinstance(model_obj, (int, float)):
                    # Some older/alternate schemas return a single float
                    scores[f"{prefix}_prob"] = float(model_obj)
                    return
                if not isinstance(model_obj, dict):
                    return

                prob = model_obj.get("prob")
                if isinstance(prob, (int, float)):
                    scores[f"{prefix}_prob"] = float(prob)

                # Newer schemas: {prefix: {classes: {...}}}
                classes = model_obj.get("classes")
                if isinstance(classes, dict):
                    for kk, vv in classes.items():
                        if isinstance(vv, (int, float)):
                            scores[f"{prefix}_{kk}"] = float(vv)

                # Some schemas flatten class scores at the top-level
                for kk, vv in model_obj.items():
                    if kk in ("prob", "classes"):
                        continue
                    if isinstance(vv, (int, float)):
                        scores[f"{prefix}_{kk}"] = float(vv)

            _parse_prob_classes(_pick_model("gore", "gore-2.0", "gore_2_0"), "gore")
            _parse_prob_classes(_pick_model("violence", "violence-2.0", "violence_2_0"), "violence")

            # Offensive: we also compute a stable offensive_max for downstream logic
            off = _pick_model("offensive", "offensive-2.0", "offensive_2_0")
            if isinstance(off, (int, float)):
                scores["offensive_max"] = float(off)
            elif isinstance(off, dict):
                _parse_prob_classes(off, "offensive")
                vals = []
                prob = off.get("prob")
                if isinstance(prob, (int, float)):
                    vals.append(float(prob))
                classes = off.get("classes")
                if isinstance(classes, dict):
                    for vv in classes.values():
                        if isinstance(vv, (int, float)):
                            vals.append(float(vv))
                # Fallback: any numeric top-level fields
                for kk, vv in off.items():
                    if kk in ("prob", "classes"):
                        continue
                    if isinstance(vv, (int, float)):
                        vals.append(float(vv))
                if vals:
                    scores["offensive_max"] = float(max(vals))

            return scores

        best_scores: Dict[str, float] = {}
        per_frame: List[Dict[str, Any]] = []

        for fr in use_frames:
            files = {"media": ("frame.jpg", fr.get_jpeg_bytes(), "image/jpeg")}
            r = requests.post(url, data=params_base, files=files, timeout=60)

            if r.status_code in (402, 403, 429):
                self.disable(f"quota/limit http={r.status_code}")
                return mk_skipped(self, self.disabled_reason or "quota/limit", took_ms=now_ms() - start)
            if r.status_code >= 400:
                return EngineResult(name=self.name, status=EngineStatus.ERROR, error=f"http error {r.status_code}", took_ms=now_ms() - start)

            try:
                data = r.json() if "application/json" in r.headers.get("content-type", "") else json.loads(r.text or "{}")
            except Exception as exc:
                return EngineResult(name=self.name, status=EngineStatus.ERROR, error=f"invalid JSON response: {type(exc).__name__}: {exc}", took_ms=now_ms() - start)
            if data.get("status") != "success":
                err = data.get("error") or data.get("message") or str(data)
                if "quota" in str(err).lower() or "limit" in str(err).lower():
                    self.disable(f"quota/limit: {str(err)[:200]}")
                    return mk_skipped(self, self.disabled_reason or "quota/limit", took_ms=now_ms() - start)
                return EngineResult(name=self.name, status=EngineStatus.ERROR, error=str(err)[:400], details={"raw": data}, took_ms=now_ms() - start)

            sc = _extract_scores(data)
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
