from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..config import project_root
from ..enums import EngineStatus
from ..types import Engine, EngineResult, Frame
from ..utils import env_float, env_int, now_ms, safe_float01

_YOLO_CACHE: Dict[Tuple[str, str], Any] = {}
_PLACEHOLDER_NAMES = {"yolo-world", "yolo_world"}


def _configured_model_name() -> Tuple[str, bool]:
    model_name = (
        os.getenv("YOLO_WORLD_MODEL", "").strip()
        or os.getenv("YOLO_WEAPON_MODEL", "").strip()
        or os.getenv("YOLO_WEAPONS_WEIGHTS", "").strip()
    )
    if model_name.strip().lower() in _PLACEHOLDER_NAMES:
        return "", False
    return model_name, bool(model_name)


def _default_model_path() -> str:
    return os.path.join(project_root(), ".cache", "ultralytics", "weights", "yolov8s-oiv7.pt")


def _looks_like_path(model_name: str) -> bool:
    p = Path(model_name).expanduser()
    return p.is_absolute() or any(sep in model_name for sep in ("/", "\\")) or model_name.startswith(".")


def _candidate_model_paths(model_name: str) -> list[Path]:
    p = Path(model_name).expanduser()
    if p.is_absolute():
        return [p]
    candidates = [Path(project_root()) / p, Path.cwd() / p]
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _resolve_model_reference() -> Tuple[str, bool, str | None]:
    """Return (model reference for Ultralytics, explicit, skip reason)."""
    configured, explicit = _configured_model_name()
    if explicit:
        candidates = _candidate_model_paths(configured)
        for candidate in candidates:
            if candidate.exists():
                return str(candidate.resolve()), True, None
        if _looks_like_path(configured):
            searched = ", ".join(str(c.resolve(strict=False)) for c in candidates)
            return configured, True, f"explicit YOLO weapons model path not found: {configured} (searched: {searched})"
        # Bare names such as yolov8n.pt are valid Ultralytics model names; keep them working.
        return configured, True, None

    default_model = Path(_default_model_path())
    if not default_model.exists():
        return str(default_model), False, f"missing default YOLO model path: {default_model}"
    return str(default_model.resolve()), False, None


def _load_model(model_ref: str) -> Any:
    backend = os.getenv("YOLO_BACKEND", "ultralytics").strip().lower()
    key = (backend, model_ref)
    if key in _YOLO_CACHE:
        return _YOLO_CACHE[key]
    from ultralytics import YOLO  # heavy import

    mdl = YOLO(model_ref)
    _YOLO_CACHE[key] = mdl
    return mdl


class YOLOWorldWeaponsEngine(Engine):
    """Offline weapon detection via Ultralytics YOLO weights (optional)."""

    name = "YOLO-World weapons"

    def available(self):
        try:
            import ultralytics  # noqa
        except Exception as e:
            return False, f"ultralytics not available: {type(e).__name__}"
        return True, "ok"

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 2) -> EngineResult:
        start = now_ms()
        model_ref, explicit, skip_reason = _resolve_model_reference()
        if skip_reason is not None:
            return EngineResult(
                name=self.name,
                status=EngineStatus.SKIPPED,
                error=skip_reason,
                details={"model": model_ref, "explicit_model": explicit},
                took_ms=now_ms() - start,
            )

        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error=why, details={"model": model_ref}, took_ms=now_ms() - start)

        mdl = _load_model(model_ref)
        conf = env_float("YOLO_CONF", 0.25, min_value=0.0, max_value=1.0)
        iou = env_float("YOLO_IOU", 0.45, min_value=0.0, max_value=1.0)
        imgsz = env_int("YOLO_IMGSZ", 640)
        max_det = env_int("YOLO_MAX_DET", 50)
        device = os.getenv("YOLO_DEVICE", "").strip() or None
        max_frames = env_int("YOLO_MAX_FRAMES", 2)
        use = frames[:max_frames] if max_frames > 0 else frames[:1]

        firearm = firearm_real = firearm_toy = 0.0
        knife = knife_danger = 0.0

        names = getattr(mdl, "names", None)

        def _name_for(cls_id: int) -> str:
            if isinstance(names, dict):
                return str(names.get(int(cls_id), ""))
            if isinstance(names, list) and 0 <= int(cls_id) < len(names):
                return str(names[int(cls_id)])
            return ""

        for fr in use:
            try:
                res = mdl.predict(fr.pil, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
            except TypeError:
                res = mdl.predict(fr.pil, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False)

            if not res:
                continue
            r0 = res[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is None:
                continue
            cls_ids = getattr(boxes, "cls", None)
            confs = getattr(boxes, "conf", None)
            if cls_ids is None or confs is None:
                continue
            try:
                cls_list = cls_ids.tolist()
                conf_list = confs.tolist()
            except Exception:
                cls_list = list(cls_ids)
                conf_list = list(confs)

            for cid, cprob in zip(cls_list, conf_list):
                nm = _name_for(int(cid)).lower()
                p = float(cprob)
                if "firearm" in nm or "gun" in nm or "rifle" in nm or "pistol" in nm:
                    firearm = max(firearm, p)
                    firearm_real = max(firearm_real, p)
                if "toy" in nm and ("gun" in nm or "firearm" in nm):
                    firearm_toy = max(firearm_toy, p)
                if "knife" in nm or "dagger" in nm:
                    knife = max(knife, p)
                    knife_danger = max(knife_danger, p)

        firearm_any = max(firearm, firearm_real, firearm_toy)

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={
                "yolo_firearm_realistic": safe_float01(firearm_real),
                "yolo_firearm_toy": safe_float01(firearm_toy),
                "yolo_firearm": safe_float01(firearm),
                "yolo_knife": safe_float01(knife),
                "yolo_knife_dangerous": safe_float01(knife_danger),
                "yolo_firearm_any": safe_float01(firearm_any),
            },
            details={"model": model_ref, "explicit_model": explicit},
            took_ms=now_ms() - start,
        )
