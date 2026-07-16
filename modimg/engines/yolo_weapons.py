from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..enums import EngineStatus
from ..resources import resource_candidates, resolve_bundled_resource_path
from ..types import Engine, EngineResult, Frame
from ..utils import env_bool, env_float, env_int, now_ms, safe_float01

_YOLO_CACHE: Dict[Tuple[str, str], Any] = {}
_YOLO_INFERENCE_LOCKS: Dict[Tuple[str, str], threading.RLock] = {}
_YOLO_CACHE_LOCK = threading.RLock()
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
    return str(resolve_bundled_resource_path(Path("models") / "yolov8s-oiv7.pt"))


def _looks_like_path(model_name: str) -> bool:
    p = Path(model_name).expanduser()
    return p.is_absolute() or any(sep in model_name for sep in ("/", "\\")) or model_name.startswith(".")


def _candidate_model_paths(model_name: str) -> list[Path]:
    return resource_candidates(model_name)


def _looks_like_model_pointer(path: Path) -> bool:
    try:
        if path.stat().st_size > 1024:
            return False
        head = path.read_text("utf-8", errors="ignore")[:200]
        return "git-lfs.github.com/spec" in head
    except OSError:
        return False


def _pointer_model_message(path: Path) -> str:
    return (
        f"model pointer file detected instead of real YOLO weights: {path}. "
        "This looks like a Git-LFS pointer; run `git lfs pull` and retry."
    )


def _resolve_model_reference() -> Tuple[str, bool, str | None]:
    """Return (model reference for Ultralytics, explicit, skip reason)."""
    configured, explicit = _configured_model_name()
    if explicit:
        candidates = _candidate_model_paths(configured)
        for candidate in candidates:
            if candidate.exists():
                if _looks_like_model_pointer(candidate):
                    return str(candidate.resolve()), True, _pointer_model_message(candidate.resolve())
                return str(candidate.resolve()), True, None
        if _looks_like_path(configured):
            searched = ", ".join(str(c.resolve(strict=False)) for c in candidates)
            return configured, True, f"explicit YOLO weapons model path not found: {configured} (searched: {searched})"
        # Bare names such as yolov8n.pt are valid Ultralytics model names; keep them working.
        return configured, True, None

    default_model = Path(_default_model_path())
    if not default_model.exists():
        return str(default_model), False, f"missing default YOLO model path: {default_model}"
    if _looks_like_model_pointer(default_model):
        return str(default_model.resolve()), False, _pointer_model_message(default_model.resolve())
    return str(default_model.resolve()), False, None


def _model_cache_key(model_ref: str) -> Tuple[str, str]:
    backend = os.getenv("YOLO_BACKEND", "ultralytics").strip().lower()
    return (backend, model_ref)


def _inference_lock(model_ref: str) -> threading.RLock:
    key = _model_cache_key(model_ref)
    with _YOLO_CACHE_LOCK:
        lock = _YOLO_INFERENCE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _YOLO_INFERENCE_LOCKS[key] = lock
        return lock


def _load_model(model_ref: str) -> Any:
    key = _model_cache_key(model_ref)
    with _YOLO_CACHE_LOCK:
        if key in _YOLO_CACHE:
            return _YOLO_CACHE[key]

        from ultralytics import YOLO  # heavy import

        _YOLO_CACHE[key] = YOLO(model_ref)
        _YOLO_INFERENCE_LOCKS.setdefault(key, threading.RLock())
        return _YOLO_CACHE[key]


def _predict(
    model: Any,
    image_or_images: Any,
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
                return model.predict(image_or_images, **variant)
            with lock:
                return model.predict(image_or_images, **variant)
        except TypeError as exc:
            last_type_error = exc
    if last_type_error is not None:
        raise last_type_error
    if lock is None:
        return model.predict(image_or_images, **kwargs)
    with lock:
        return model.predict(image_or_images, **kwargs)


def _results_list(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, list):
        return results
    try:
        return list(results)
    except TypeError:
        return [results]


class YOLOWorldWeaponsEngine(Engine):
    """Offline weapon detection via Ultralytics YOLO weights (optional)."""

    name = "YOLO-World weapons"

    def available(self):
        backend = os.getenv("YOLO_BACKEND", "ultralytics").strip().lower()
        if backend != "ultralytics":
            return False, f"unsupported YOLO_BACKEND={backend or '<empty>'}; expected ultralytics"
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

        mdl = _load_model(model_ref)
        predict_lock = _inference_lock(model_ref)
        conf = env_float("YOLO_CONF", 0.25, min_value=0.0, max_value=1.0)
        iou = env_float("YOLO_IOU", 0.45, min_value=0.0, max_value=1.0)
        imgsz = max(32, env_int("YOLO_IMGSZ", 640))
        max_det = max(1, env_int("YOLO_MAX_DET", 50))
        device = os.getenv("YOLO_DEVICE", "").strip() or None
        max_frames = env_int("YOLO_MAX_FRAMES", 2)
        use = frames[:max_frames] if max_frames > 0 else frames[:1]
        batch_enabled = env_bool("YOLO_BATCH_ENABLE", True)

        firearm = firearm_real = firearm_toy = 0.0
        knife = knife_danger = 0.0

        names = getattr(mdl, "names", None)

        def _name_for(cls_id: int) -> str:
            if isinstance(names, dict):
                return str(names.get(int(cls_id), ""))
            if isinstance(names, list) and 0 <= int(cls_id) < len(names):
                return str(names[int(cls_id)])
            return ""

        def _consume_result(r0: Any) -> None:
            nonlocal firearm, firearm_real, firearm_toy, knife, knife_danger
            boxes = getattr(r0, "boxes", None)
            if boxes is None:
                return
            cls_ids = getattr(boxes, "cls", None)
            confs = getattr(boxes, "conf", None)
            if cls_ids is None or confs is None:
                return
            try:
                cls_list = cls_ids.tolist()
                conf_list = confs.tolist()
            except Exception:
                cls_list = list(cls_ids)
                conf_list = list(confs)

            if len(cls_list) != len(conf_list):
                raise RuntimeError(
                    "YOLO weapons returned inconsistent detection arrays: "
                    f"{len(cls_list)} classes for {len(conf_list)} confidences"
                )
            for cid, cprob in zip(cls_list, conf_list, strict=True):
                nm = _name_for(int(cid)).lower()
                p = safe_float01(cprob)
                firearm_like = "firearm" in nm or "gun" in nm or "rifle" in nm or "pistol" in nm
                toy_firearm = "toy" in nm and firearm_like
                if firearm_like:
                    firearm = max(firearm, p)
                    if toy_firearm:
                        firearm_toy = max(firearm_toy, p)
                    else:
                        firearm_real = max(firearm_real, p)
                if "knife" in nm or "dagger" in nm:
                    knife = max(knife, p)
                    knife_danger = max(knife_danger, p)

        if batch_enabled and len(use) > 1:
            images = [fr.pil.convert("RGB") for fr in use]
            try:
                try:
                    batch_results = _results_list(
                        _predict(
                            mdl,
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
                    batch_results = []
                    for image in images:
                        batch_results.extend(
                            _results_list(
                                _predict(
                                    mdl,
                                    image,
                                    conf=conf,
                                    iou=iou,
                                    imgsz=imgsz,
                                    max_det=max_det,
                                    device=device,
                                    lock=predict_lock,
                                )
                            )
                        )
                if len(batch_results) != len(use):
                    raise RuntimeError(
                        f"YOLO weapons returned {len(batch_results)} results for {len(use)} frames"
                    )
                for result in batch_results[: len(use)]:
                    _consume_result(result)
            finally:
                for image in images:
                    image.close()
        else:
            for fr in use:
                with fr.pil.convert("RGB") as image:
                    for result in _results_list(
                        _predict(mdl, image, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device=device, lock=predict_lock)
                    ):
                        _consume_result(result)

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
            details={"model": model_ref, "explicit_model": explicit, "batch_enabled": bool(batch_enabled and len(use) > 1)},
            took_ms=now_ms() - start,
        )
