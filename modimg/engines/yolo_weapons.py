from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple


from ..types import Engine, EngineResult, Frame
from ..utils import env_int, now_ms, safe_float01
from ..config import project_root

_YOLO_CACHE: Dict[Tuple[str, str], Any] = {}

def _resolve_model_name() -> Tuple[str, bool]:
    model_name = (
        os.getenv("YOLO_WORLD_MODEL", "").strip()
        or os.getenv("YOLO_WEAPON_MODEL", "").strip()
        or os.getenv("YOLO_WEAPONS_WEIGHTS", "").strip()
    )
    explicit = bool(model_name)
    return model_name, explicit


def _default_model_path() -> str:
    return os.path.join(project_root(), ".cache", "ultralytics", "weights", "yolov8s-oiv7.pt")


def _load_model() -> Any:
    backend = os.getenv("YOLO_BACKEND", "ultralytics").strip().lower()
    # Keep a simple cache key; backend kept for future
    model_name, _ = _resolve_model_name()
    # Safety: older templates accidentally used the literal string "yolo-world"
    # as a placeholder. Ultralytics' YOLO() expects a valid model name/path.
    # Treat that placeholder as "unset" and fall back to our default weights.
    if model_name.strip().lower() in {"yolo-world", "yolo_world"}:
        model_name = ""
    if not model_name:
        model_name = _default_model_path()
    key = (backend, model_name)
    if key in _YOLO_CACHE:
        return _YOLO_CACHE[key]
    from ultralytics import YOLO  # heavy import
    mdl = YOLO(model_name)
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
        ok, why = self.available()
        if not ok:
            return EngineResult(name=self.name, status="skipped", error=why, took_ms=now_ms()-start)
        model_name, explicit = _resolve_model_name()
        if model_name.strip().lower() in {"yolo-world", "yolo_world"}:
            explicit = False
            model_name = ""
        if not explicit:
            default_model = _default_model_path()
            if not os.path.exists(default_model):
                return EngineResult(
                    name=self.name,
                    status="skipped",
                    error=f"missing default YOLO model path: {default_model}",
                    took_ms=now_ms() - start,
                )

        mdl = _load_model()
        conf = float(os.getenv("YOLO_CONF", "0.25").strip() or 0.25)
        iou = float(os.getenv("YOLO_IOU", "0.45").strip() or 0.45)
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
            # ultralytics accepts numpy arrays / PIL
            try:
                res = mdl.predict(fr.pil, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, verbose=False)
            except TypeError:
                # older versions: imgsz named img_size etc. fallback
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
                # very loose name matching for OpenImages weights
                if "firearm" in nm or "gun" in nm or "rifle" in nm or "pistol" in nm:
                    firearm = max(firearm, p)
                    firearm_real = max(firearm_real, p)
                if "toy" in nm and ("gun" in nm or "firearm" in nm):
                    firearm_toy = max(firearm_toy, p)
                if "knife" in nm or "dagger" in nm:
                    knife = max(knife, p)
                    # dangerous-knife heuristic: treat high confidence as dangerous
                    knife_danger = max(knife_danger, p)

        firearm_any = max(firearm, firearm_real, firearm_toy)

        return EngineResult(
            name=self.name,
            status="ok",
            scores={
                "yolo_firearm_realistic": safe_float01(firearm_real),
                "yolo_firearm_toy": safe_float01(firearm_toy),
                "yolo_firearm": safe_float01(firearm),
                "yolo_knife": safe_float01(knife),
                "yolo_knife_dangerous": safe_float01(knife_danger),
                "yolo_firearm_any": safe_float01(firearm_any),
            },
            took_ms=now_ms()-start,
        )
