from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List

from ..enums import EngineStatus
from ..resources import bundled_resource_candidates, resource_candidates
from ..types import Engine, EngineResult, Frame
from ..utils import env_bool, env_float, env_int, env_label_float_map, env_label_set, now_ms, safe_float01

_FORBIDDEN_SYMBOLS_YOLO_CACHE: Dict[str, Any] = {}
_FORBIDDEN_SYMBOLS_YOLO_INFERENCE_LOCKS: Dict[str, threading.RLock] = {}
_FORBIDDEN_SYMBOLS_YOLO_CACHE_LOCK = threading.RLock()


DEFAULT_FORBIDDEN_SYMBOLS_MODEL = "models/forbidden_symbols_yolo.pt"


class OptionalModelUnavailable(RuntimeError):
    """Raised when optional local model weights are absent or not real weights."""


def _candidate_model_paths(raw: str) -> list[Path]:
    return resource_candidates(raw)


def _resolve_model_path(model_path: str | None = None) -> Path:
    """Resolve forbidden-symbols model path from absolute or project-root-relative input."""
    raw = (model_path or os.getenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", DEFAULT_FORBIDDEN_SYMBOLS_MODEL) or DEFAULT_FORBIDDEN_SYMBOLS_MODEL).strip()
    is_bundled_default = model_path is None and raw == DEFAULT_FORBIDDEN_SYMBOLS_MODEL
    candidates = bundled_resource_candidates(raw) if is_bundled_default else _candidate_model_paths(raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def _missing_model_message(path: Path) -> str:
    return (
        f"missing forbidden symbols YOLO model: {path}. "
        f"Expected the bundled model at {DEFAULT_FORBIDDEN_SYMBOLS_MODEL}; run from the project root, "
        "install package data, or set FORBIDDEN_SYMBOLS_YOLO_MODEL to an absolute path."
    )


def _pointer_model_message(path: Path) -> str:
    return (
        f"model pointer file detected instead of real model weights: {path}. "
        "This looks like a Git-LFS pointer; run `git lfs pull` and retry."
    )


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
        raise OptionalModelUnavailable(_missing_model_message(resolved))
    if _looks_like_model_pointer(resolved):
        raise OptionalModelUnavailable(_pointer_model_message(resolved))

    key = str(resolved)
    with _FORBIDDEN_SYMBOLS_YOLO_CACHE_LOCK:
        if key in _FORBIDDEN_SYMBOLS_YOLO_CACHE:
            return _FORBIDDEN_SYMBOLS_YOLO_CACHE[key]

        if "ultralytics" not in sys.modules and importlib.util.find_spec("ultralytics") is None:
            raise ImportError("ultralytics not available for forbidden symbols YOLO")
        YOLO = getattr(importlib.import_module("ultralytics"), "YOLO")  # heavy optional import; local inference only

        _FORBIDDEN_SYMBOLS_YOLO_CACHE[key] = YOLO(str(resolved))
        _FORBIDDEN_SYMBOLS_YOLO_INFERENCE_LOCKS.setdefault(key, threading.RLock())
        return _FORBIDDEN_SYMBOLS_YOLO_CACHE[key]


def _inference_lock(model_key: str) -> threading.RLock:
    with _FORBIDDEN_SYMBOLS_YOLO_CACHE_LOCK:
        lock = _FORBIDDEN_SYMBOLS_YOLO_INFERENCE_LOCKS.get(model_key)
        if lock is None:
            lock = threading.RLock()
            _FORBIDDEN_SYMBOLS_YOLO_INFERENCE_LOCKS[model_key] = lock
        return lock


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


def _predict(
    model: Any,
    image: Any,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    device: str | None,
    batch: int | None = None,
    lock: threading.RLock | None = None,
) -> Any:
    kwargs: Dict[str, Any] = {
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "max_det": max_det,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    if batch is not None:
        kwargs["batch"] = batch

    variants: list[Dict[str, Any]] = [dict(kwargs)]
    if "batch" in kwargs:
        without_batch = dict(kwargs)
        without_batch.pop("batch", None)
        variants.append(without_batch)
    without_imgsz = dict(variants[-1])
    without_imgsz.pop("imgsz", None)
    variants.append(without_imgsz)
    without_max_det = dict(without_imgsz)
    without_max_det.pop("max_det", None)
    variants.append(without_max_det)

    last_type_error: TypeError | None = None
    for variant in variants:
        try:
            if lock is None:
                return model.predict(image, **variant)
            with lock:
                return model.predict(image, **variant)
        except TypeError as exc:
            last_type_error = exc
    if last_type_error is not None:
        raise last_type_error
    if lock is None:
        return model.predict(image, **kwargs)
    with lock:
        return model.predict(image, **kwargs)


def _results_list(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, list):
        return results
    try:
        return list(results)
    except TypeError:
        return [results]


def _threshold_for_label(label: str, default: float, overrides: dict[str, float]) -> float:
    return float(overrides.get(label.strip().lower(), default))


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

        conf = env_float("FORBIDDEN_SYMBOLS_YOLO_CONF", 0.20, min_value=0.0, max_value=1.0)
        iou = env_float("FORBIDDEN_SYMBOLS_YOLO_IOU", 0.45, min_value=0.0, max_value=1.0)
        imgsz = max(32, env_int("FORBIDDEN_SYMBOLS_YOLO_IMGSZ", 960))
        max_det = max(1, env_int("FORBIDDEN_SYMBOLS_YOLO_MAX_DET", 20))
        max_frames = env_int("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", 2)
        review_conf = env_float("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", 0.30, min_value=0.0, max_value=1.0)
        block_conf = env_float("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", 0.90, min_value=0.0, max_value=1.0)
        label_review_conf = env_label_float_map("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF")
        label_block_conf = env_label_float_map("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF")
        if max_frames <= 0:
            return EngineResult(
                name=self.name,
                status=EngineStatus.SKIPPED,
                error="inference disabled via FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0",
                scores={
                    "forbidden_symbols_detected": 0.0,
                    "forbidden_symbols_max_conf": 0.0,
                    "forbidden_symbols_review_hit": 0.0,
                    "forbidden_symbols_block_hit": 0.0,
                    "forbidden_symbols_detection_count": 0.0,
                    "forbidden_symbols_top_conf": 0.0,
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
                    "label_review_conf": label_review_conf,
                    "label_block_conf": label_block_conf,
                    "detection_count": 0,
                    "top_label": "",
                    "top_confidence": 0.0,
                    "detections": [],
                    "model_pointer_detected": bool(model_pointer),
                    "inference_skipped": True,
                    "skip_reason": "FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0",
                },
                took_ms=now_ms() - start,
            )

        try:
            model = _load_model(str(model_path))
        except (OptionalModelUnavailable, ImportError) as exc:
            return EngineResult(
                name=self.name,
                status=EngineStatus.SKIPPED,
                error=str(exc),
                details={
                    "model_path": str(model_path),
                    "model_exists": bool(model_exists),
                    "model_size_bytes": int(model_size),
                    "model_pointer_detected": bool(model_pointer),
                },
                took_ms=now_ms() - start,
            )
        predict_lock = _inference_lock(str(model_path))

        device_raw = (os.getenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "auto") or "auto").strip()
        device = None if device_raw.lower() in ("", "auto") else device_raw
        include_boxes = env_bool("FORBIDDEN_SYMBOLS_YOLO_INCLUDE_BOXES", True)
        ignore_labels = env_label_set("FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS", "")
        batch_requested = env_bool("FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE", True)
        stop_after_block = env_bool("FORBIDDEN_SYMBOLS_YOLO_STOP_AFTER_BLOCK", True)

        selected_frames = frames[:max_frames] if max_frames > 0 else []
        detections: list[dict[str, Any]] = []
        processed_frame_indices: list[int] = []
        names = getattr(model, "names", None)

        def _append_detections(fr: Frame, image: Any, result: Any) -> bool:
            block_hit = False
            width, height = image.size
            result_names = getattr(result, "names", None)
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                return False
            class_ids = _tolist(getattr(boxes, "cls", None))
            confidences = _tolist(getattr(boxes, "conf", None))
            xyxy_values = _tolist(getattr(boxes, "xyxy", None))
            if not xyxy_values:
                xyxy_values = [[] for _ in class_ids]

            if len(class_ids) != len(confidences) or len(class_ids) != len(xyxy_values):
                raise RuntimeError(
                    "forbidden symbols YOLO returned inconsistent detection arrays: "
                    f"{len(class_ids)} classes, {len(confidences)} confidences, "
                    f"{len(xyxy_values)} boxes"
                )

            for raw_cid, raw_conf, raw_box in zip(class_ids, confidences, xyxy_values, strict=True):
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
                if det_conf >= _threshold_for_label(label, block_conf, label_block_conf):
                    block_hit = True
            return block_hit

        batch_enabled = bool(batch_requested and len(selected_frames) > 1)
        result_count = 0
        early_stopped = False

        if batch_enabled:
            images = [fr.pil.convert("RGB") for fr in selected_frames]
            try:
                try:
                    results = _results_list(
                        _predict(
                            model,
                            images,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            max_det=max_det,
                            device=device,
                            batch=len(images),
                            lock=predict_lock,
                        )
                    )
                except TypeError:
                    results = []
                    for fr, image in zip(selected_frames, images, strict=True):
                        frame_results = _results_list(
                            _predict(
                                model,
                                image,
                                conf=conf,
                                iou=iou,
                                imgsz=imgsz,
                                max_det=max_det,
                                device=device,
                                lock=predict_lock,
                            )
                        )
                        result_count += len(frame_results)
                        processed_frame_indices.append(int(fr.idx))
                        for result in frame_results:
                            _append_detections(fr, image, result)
                else:
                    result_count = len(results)
                    if result_count != len(selected_frames):
                        return EngineResult(
                            name=self.name,
                            status=EngineStatus.ERROR,
                            error=(
                                "forbidden symbols YOLO returned an unexpected batch result count: "
                                f"{result_count} for {len(selected_frames)} frames"
                            ),
                            details={"model_path": str(model_path), "result_count": result_count},
                            took_ms=now_ms() - start,
                        )
                    for fr, image, result in zip(selected_frames, images, results, strict=True):
                        processed_frame_indices.append(int(fr.idx))
                        _append_detections(fr, image, result)
            finally:
                for image in images:
                    image.close()
        else:
            for fr in selected_frames:
                with fr.pil.convert("RGB") as image:
                    frame_results = _results_list(
                        _predict(model, image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, lock=predict_lock)
                    )
                    result_count += len(frame_results)
                    processed_frame_indices.append(int(fr.idx))
                    frame_block_hit = False
                    for result in frame_results:
                        frame_block_hit = _append_detections(fr, image, result) or frame_block_hit
                if stop_after_block and frame_block_hit:
                    early_stopped = True
                    break

        max_conf = max((float(d["confidence"]) for d in detections), default=0.0)
        top = max(detections, key=lambda d: float(d["confidence"])) if detections else None
        top_label = str(top.get("label", "")) if top else ""
        review_detections = [
            d
            for d in detections
            if float(d["confidence"]) >= _threshold_for_label(str(d.get("label", "")), review_conf, label_review_conf)
        ]
        block_detections = [
            d
            for d in detections
            if float(d["confidence"]) >= _threshold_for_label(str(d.get("label", "")), block_conf, label_block_conf)
        ]
        review_top = max(review_detections, key=lambda d: float(d["confidence"])) if review_detections else None
        block_top = max(block_detections, key=lambda d: float(d["confidence"])) if block_detections else None

        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={
                "forbidden_symbols_detected": 1.0 if detections else 0.0,
                "forbidden_symbols_max_conf": safe_float01(max_conf),
                "forbidden_symbols_review_hit": 1.0 if (review_top is not None or block_top is not None) else 0.0,
                "forbidden_symbols_block_hit": 1.0 if block_top is not None else 0.0,
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
                "label_review_conf": label_review_conf,
                "label_block_conf": label_block_conf,
                "device": device_raw,
                "detection_count": len(detections),
                "top_label": top_label,
                "top_confidence": safe_float01(max_conf),
                "review_detection": review_top,
                "block_detection": block_top,
                "detections": detections,
                "model_pointer_detected": bool(model_pointer),
                "inference_skipped": False,
                "batch_enabled": batch_enabled,
                "batch_requested": batch_requested,
                "result_count": int(result_count),
                "processed_frames": processed_frame_indices,
                "early_stop_after_block": bool(stop_after_block and not batch_enabled),
                "early_stopped": early_stopped,
            },
            took_ms=now_ms() - start,
        )
