from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from ..config import project_root
from ..enums import EngineStatus
from ..types import Engine, EngineResult, Frame
from ..utils import env_bool, env_int, now_ms, safe_float01

_FORBIDDEN_SYMBOLS_YOLO_CACHE: Dict[str, Any] = {}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except Exception:
        return float(default)


def _env_label_set(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name, default) or default
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def _resolve_model_path(model_path: str | None = None) -> Path:
    raw = (model_path or os.getenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", "models/forbidden_symbols_yolo.pt") or "models/forbidden_symbols_yolo.pt").strip()
    p = Path(raw)
    if not p.is_absolute():
        p = Path(project_root()) / p
    return p.resolve()


def _looks_like_model_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        head = path.read_text("utf-8", errors="ignore")[:200]
        return "git-lfs.github.com/spec" in head
    except OSError:
        return False


def _load_model(model_path: str | None = None) -> Any:
    resolved = _resolve_model_path(model_path)
    if not resolved.exists():
        raise RuntimeError(f"missing forbidden symbols YOLO model: {resolved}")
    if _looks_like_model_pointer(resolved):
        raise RuntimeError(f"model pointer file detected instead of real model weights: {resolved}")

    key = str(resolved)
    if key in _FORBIDDEN_SYMBOLS_YOLO_CACHE:
        return _FORBIDDEN_SYMBOLS_YOLO_CACHE[key]

    if "ultralytics" not in sys.modules and importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError("ultralytics not available for forbidden symbols YOLO")
    YOLO = getattr(importlib.import_module("ultralytics"), "YOLO")  # heavy optional import; local inference only

    model = YOLO(str(resolved))
    _FORBIDDEN_SYMBOLS_YOLO_CACHE[key] = model
    return model


def _tolist(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        out = value.tolist()
        return out if isinstance(out, list) else [out]
    try:
        return list(value)
    except TypeError:
        return [value]


def _name_for(class_id: int, *name_sources: Any) -> str:
    for names in name_sources:
        if isinstance(names, dict):
            if class_id in names:
                return str(names[class_id])
            if str(class_id) in names:
                return str(names[str(class_id)])
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
    return str(class_id)


def _predict(model: Any, image: Any, *, conf: float, iou: float, imgsz: int, max_det: int, device: str | None) -> Any:
    kwargs: Dict[str, Any] = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    try:
        return model.predict(image, **kwargs)
    except TypeError:
        kwargs.pop("imgsz", None)
        try:
            return model.predict(image, **kwargs)
        except TypeError:
            kwargs.pop("max_det", None)
            return model.predict(image, **kwargs)


class YOLOForbiddenSymbolsEngine(Engine):
    """Local forbidden/harmful-symbol detection using the bundled YOLO model."""

    name = "YOLO forbidden symbols"

    def available(self):
        if not env_bool("FORBIDDEN_SYMBOLS_YOLO_ENABLE", True):
            return False, "FORBIDDEN_SYMBOLS_YOLO_ENABLE=0"
        return True, "ok"

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        start = now_ms()
        model_path = _resolve_model_path()
        model_exists = model_path.exists()
        model_size = model_path.stat().st_size if model_exists else 0
        model_pointer = _looks_like_model_pointer(model_path) if model_exists else False

        model = _load_model(str(model_path))

        conf = _env_float("FORBIDDEN_SYMBOLS_YOLO_CONF", 0.20)
        iou = _env_float("FORBIDDEN_SYMBOLS_YOLO_IOU", 0.45)
        imgsz = env_int("FORBIDDEN_SYMBOLS_YOLO_IMGSZ", 960)
        max_det = env_int("FORBIDDEN_SYMBOLS_YOLO_MAX_DET", 20)
        max_frames = env_int("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", 2)
        review_conf = _env_float("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", 0.30)
        block_conf = _env_float("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", 0.90)
        device_raw = (os.getenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "auto") or "auto").strip()
        device = None if device_raw.lower() in ("", "auto") else device_raw
        include_boxes = env_bool("FORBIDDEN_SYMBOLS_YOLO_INCLUDE_BOXES", True)
        ignore_labels = _env_label_set("FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS", "")

        selected_frames = frames[:max_frames] if max_frames > 0 else frames
        detections: list[dict[str, Any]] = []
        names = getattr(model, "names", None)

        for fr in selected_frames:
            image = fr.pil.convert("RGB")
            width, height = image.size
            results = _predict(model, image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device)
            for result in list(results or []):
                result_names = getattr(result, "names", None)
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                class_ids = _tolist(getattr(boxes, "cls", None))
                confidences = _tolist(getattr(boxes, "conf", None))
                xyxy_values = _tolist(getattr(boxes, "xyxy", None))
                if not xyxy_values:
                    xyxy_values = [[] for _ in class_ids]

                for raw_cid, raw_conf, raw_box in zip(class_ids, confidences, xyxy_values):
                    class_id = int(raw_cid)
                    label = _name_for(class_id, result_names, names)
                    if label.strip().lower() in ignore_labels:
                        continue
                    det_conf = safe_float01(raw_conf)
                    box = [float(v) for v in list(raw_box)[:4]] if raw_box is not None else []
                    if len(box) < 4:
                        box = [0.0, 0.0, 0.0, 0.0]
                    x1, y1, x2, y2 = box
                    norm = [
                        safe_float01(x1 / width if width else 0.0),
                        safe_float01(y1 / height if height else 0.0),
                        safe_float01(x2 / width if width else 0.0),
                        safe_float01(y2 / height if height else 0.0),
                    ]
                    area_ratio = safe_float01(max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(width * height or 1))
                    det: dict[str, Any] = {
                        "frame_idx": int(fr.idx),
                        "class_id": class_id,
                        "label": label,
                        "confidence": det_conf,
                        "image_size": [int(width), int(height)],
                        "area_ratio": area_ratio,
                    }
                    if include_boxes:
                        det["bbox_xyxy"] = box
                        det["bbox_norm_xyxy"] = norm
                    detections.append(det)

        max_conf = max((float(d["confidence"]) for d in detections), default=0.0)
        top = max(detections, key=lambda d: float(d["confidence"])) if detections else None
        top_label = str(top.get("label", "")) if top else ""

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={
                "forbidden_symbols_detected": 1.0 if detections else 0.0,
                "forbidden_symbols_max_conf": safe_float01(max_conf),
                "forbidden_symbols_review_hit": 1.0 if max_conf >= review_conf else 0.0,
                "forbidden_symbols_block_hit": 1.0 if max_conf >= block_conf else 0.0,
                "forbidden_symbols_detection_count": float(len(detections)),
                "forbidden_symbols_top_conf": safe_float01(max_conf),
            },
            details={
                "model_path": str(model_path),
                "model_exists": bool(model_exists),
                "model_size_bytes": int(model_size),
                "imgsz": int(imgsz),
                "conf": float(conf),
                "iou": float(iou),
                "max_det": int(max_det),
                "max_frames": int(max_frames),
                "review_conf": float(review_conf),
                "block_conf": float(block_conf),
                "device": device_raw,
                "detection_count": len(detections),
                "top_label": top_label,
                "top_confidence": safe_float01(max_conf),
                "detections": detections,
                "model_pointer_detected": bool(model_pointer),
            },
            took_ms=now_ms() - start,
        )
