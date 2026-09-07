from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from ..enums import EngineStatus
from ..resources import bundled_resource_candidates, resource_candidates
from ..types import Engine, EngineResult, Frame
from ..utils import env_bool, env_label_set, now_ms, safe_float01

FORBIDDEN_SYMBOL_INPUT_SIZE = 640
BOUNDING_BOX_ABSOLUTE_TOLERANCE_PX = 0.01
FORBIDDEN_SYMBOL_CLASSES = {
    0: "Identitare Bewegung",
    1: "black_sun",
    2: "confederate-flag",
    3: "isis",
    4: "siegrune",
    5: "ss_skull",
    6: "swastika",
}
DEFAULT_FORBIDDEN_SYMBOLS_PT_MODEL = "models/forbidden_symbols_yolo26s_GPU_0.1.pt"
DEFAULT_FORBIDDEN_SYMBOLS_ONNX_MODEL = "models/forbidden_symbols_yolo26s_CPU_0.1.onnx"
DEFAULT_FORBIDDEN_SYMBOLS_MODEL = DEFAULT_FORBIDDEN_SYMBOLS_PT_MODEL  # compatibility

_FORBIDDEN_SYMBOLS_YOLO_CACHE: Dict[str, Any] = {}
_FORBIDDEN_SYMBOLS_YOLO_INFERENCE_LOCKS: Dict[str, threading.RLock] = {}
_FORBIDDEN_SYMBOLS_YOLO_CACHE_LOCK = threading.RLock()


class OptionalModelUnavailable(RuntimeError):
    """An optional model/runtime is unavailable rather than invalid."""


@dataclass(frozen=True)
class _ModelSelection:
    backend: str
    path: Path
    device_requested: str
    device_resolved: str

    @property
    def cache_key(self) -> str:
        return f"{self.backend}:{self.path}:{self.device_resolved}"


def _candidate_model_paths(raw: str, *, bundled_default: bool = False) -> list[Path]:
    return bundled_resource_candidates(raw) if bundled_default else resource_candidates(raw)


def _first_path(raw: str, *, bundled_default: bool) -> Path:
    candidates = _candidate_model_paths(raw, bundled_default=bundled_default)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def _resolve_model_path(model_path: str | None = None, *, default_path: str = DEFAULT_FORBIDDEN_SYMBOLS_PT_MODEL) -> Path:
    """Resolve explicit paths normally and bundled defaults without cwd shadowing."""
    if model_path is not None:
        raw = str(model_path).strip() or default_path
        return _first_path(raw, bundled_default=False)
    legacy = (os.getenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", "") or "").strip()
    raw = legacy or default_path
    return _first_path(raw, bundled_default=not legacy and raw in {DEFAULT_FORBIDDEN_SYMBOLS_PT_MODEL, DEFAULT_FORBIDDEN_SYMBOLS_ONNX_MODEL})


def _configured_path(env_name: str, default_path: str) -> Path:
    raw = (os.getenv(env_name, default_path) or default_path).strip()
    return _first_path(raw, bundled_default=raw == default_path)


def _configured_model_path(env_name: str, default_path: str, expected_suffix: str) -> Path:
    raw = (os.getenv(env_name, default_path) or default_path).strip()
    if Path(raw).suffix.lower() != expected_suffix:
        raise ValueError(f"{env_name} must point to a {expected_suffix} model")
    return _first_path(raw, bundled_default=raw == default_path)


def _looks_like_model_pointer(path: Path) -> bool:
    try:
        return path.stat().st_size <= 1024 and "git-lfs.github.com/spec" in path.read_text("utf-8", errors="ignore")[:200]
    except OSError:
        return False


def _usable_model(path: Path) -> bool:
    return path.is_file() and not _looks_like_model_pointer(path)


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _visible_cuda_device_count() -> int:
    """Return the lazy Torch-visible logical CUDA device count."""
    if (os.getenv("CUDA_VISIBLE_DEVICES", "") or "").strip().lower() in {"-1", "none", "void"}:
        return 0
    if not _module_available("torch"):
        return 0
    try:
        cuda = importlib.import_module("torch").cuda
        if not bool(cuda.is_available()):
            return 0
        return max(0, int(cuda.device_count()))
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return 0


def _cuda_available() -> bool:
    """Check lazily while respecting the repository's CUDA visibility policy."""
    return _visible_cuda_device_count() > 0


def _onnx_runtime_available() -> bool:
    """Return whether the CPU inference runtime required by Ultralytics is installed."""
    return _module_available("onnxruntime")


def _is_cuda_device(value: str) -> bool:
    value = value.strip().lower()
    return value.isascii() and (value.isdigit() or value == "cuda" or (value.startswith("cuda:") and value[5:].isdigit()))


def _cuda_device_index(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "cuda":
        return 0
    return int(normalized[5:] if normalized.startswith("cuda:") else normalized)


def _unavailable_model(path: Path, backend: str) -> OptionalModelUnavailable:
    if _looks_like_model_pointer(path):
        return OptionalModelUnavailable(
            f"model pointer file detected instead of real model weights: {path}. "
            "This looks like a Git-LFS pointer; run `git lfs pull` and retry."
        )
    return OptionalModelUnavailable(
        f"missing forbidden symbols YOLO {backend.upper()} model: {path}. "
        "Provide real local weights at the configured path; models are never downloaded automatically."
    )


def _select_model() -> _ModelSelection:
    backend_request = (os.getenv("FORBIDDEN_SYMBOLS_YOLO_BACKEND", "auto") or "auto").strip().lower()
    if backend_request not in {"auto", "pt", "onnx"}:
        raise ValueError("FORBIDDEN_SYMBOLS_YOLO_BACKEND must be one of: auto, pt, onnx")
    device_request = (os.getenv("FORBIDDEN_SYMBOLS_YOLO_DEVICE", "auto") or "auto").strip()
    device = device_request.lower()
    if device not in {"auto", "cpu", "cuda"} and not _is_cuda_device(device):
        raise ValueError("FORBIDDEN_SYMBOLS_YOLO_DEVICE must be auto, cpu, a CUDA index, or cuda:<index>")
    explicit_cuda = _is_cuda_device(device_request) and device != "auto"

    legacy = (os.getenv("FORBIDDEN_SYMBOLS_YOLO_MODEL", "") or "").strip()
    if legacy:
        suffix = Path(legacy).suffix.lower()
        if suffix not in {".pt", ".onnx"}:
            raise ValueError("FORBIDDEN_SYMBOLS_YOLO_MODEL must point to a .pt or .onnx model")
        backend = suffix[1:]
        if backend_request != "auto" and backend_request != backend:
            raise ValueError("FORBIDDEN_SYMBOLS_YOLO_MODEL format conflicts with FORBIDDEN_SYMBOLS_YOLO_BACKEND")
        path = _resolve_model_path(legacy)
    else:
        pt_path = _configured_model_path("FORBIDDEN_SYMBOLS_YOLO_PT_MODEL", DEFAULT_FORBIDDEN_SYMBOLS_PT_MODEL, ".pt")
        onnx_path = _configured_model_path("FORBIDDEN_SYMBOLS_YOLO_ONNX_MODEL", DEFAULT_FORBIDDEN_SYMBOLS_ONNX_MODEL, ".onnx")
        if backend_request == "onnx" and explicit_cuda:
            raise OptionalModelUnavailable(
                "ONNX forbidden-symbol inference is CPU-only; use device=cpu/auto or select backend=pt"
            )
        if explicit_cuda:
            backend = "pt"
        elif backend_request != "auto":
            backend = backend_request
        elif device == "cpu":
            backend = "onnx" if _usable_model(onnx_path) and (_onnx_runtime_available() or not _usable_model(pt_path)) else "pt"
        elif _cuda_available():
            backend = "pt" if _usable_model(pt_path) else "onnx"
        elif _usable_model(onnx_path) and (_onnx_runtime_available() or not _usable_model(pt_path)):
            backend = "onnx"
        else:
            backend = "pt"
        path = pt_path if backend == "pt" else onnx_path

    if not _usable_model(path):
        raise _unavailable_model(path, backend)
    if backend == "onnx" and not _onnx_runtime_available():
        raise OptionalModelUnavailable("onnxruntime is required for forbidden symbols YOLO ONNX inference")
    if backend == "onnx" and explicit_cuda:
        raise OptionalModelUnavailable("ONNX forbidden-symbol inference is CPU-only; a CUDA device was explicitly requested")
    if explicit_cuda:
        device_index = _cuda_device_index(device_request)
        visible_device_count = _visible_cuda_device_count()
        if device_index >= visible_device_count:
            raise OptionalModelUnavailable(
                f"requested CUDA device {device_request!r} is not available or visible; "
                f"Torch reports {visible_device_count} visible CUDA device(s)"
            )
    if backend == "onnx" or device == "cpu":
        resolved_device = "cpu"
    elif explicit_cuda:
        resolved_device = f"cuda:{_cuda_device_index(device_request)}"
    else:
        resolved_device = "cuda:0" if _cuda_available() else "cpu"
    return _ModelSelection(backend, path, device_request, resolved_device)


def _normalize_names(names: Any) -> dict[int, str] | None:
    if names is None or names == {} or names == []:
        return None
    if isinstance(names, dict):
        try:
            normalized = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError("forbidden symbols model has invalid class-name metadata") from exc
        if len(normalized) != len(names):
            raise RuntimeError("forbidden symbols model has duplicate class IDs in class-name metadata")
        return normalized
    if isinstance(names, (list, tuple)):
        return {idx: str(value) for idx, value in enumerate(names)}
    raise RuntimeError("forbidden symbols model has unsupported class-name metadata")


def _validate_names(names: Any, source: str) -> None:
    normalized = _normalize_names(names)
    if normalized is not None and normalized != FORBIDDEN_SYMBOL_CLASSES:
        raise RuntimeError(f"incompatible forbidden symbols class mapping from {source}: expected {FORBIDDEN_SYMBOL_CLASSES}, got {normalized}")


def _validate_model(model: Any) -> None:
    task = getattr(model, "task", None)
    if task is None:
        task = getattr(getattr(model, "model", None), "task", None)
    if task is not None and str(task).strip().lower() != "detect":
        raise RuntimeError(f"incompatible forbidden symbols model task: expected detect, got {task!r}")
    class_count = getattr(getattr(model, "model", None), "nc", None)
    if class_count is not None:
        class_count_value = float(class_count)
        if not math.isfinite(class_count_value) or not class_count_value.is_integer() or int(class_count_value) != len(FORBIDDEN_SYMBOL_CLASSES):
            raise RuntimeError(f"incompatible forbidden symbols model class count: expected 7, got {class_count!r}")
    _validate_names(getattr(model, "names", None), "model.names")


def _load_model(selection: _ModelSelection) -> Any:
    with _FORBIDDEN_SYMBOLS_YOLO_CACHE_LOCK:
        if selection.cache_key in _FORBIDDEN_SYMBOLS_YOLO_CACHE:
            return _FORBIDDEN_SYMBOLS_YOLO_CACHE[selection.cache_key]
        if not _module_available("ultralytics"):
            raise ImportError("ultralytics not available for forbidden symbols YOLO")
        # The application ships local model files and never permits Ultralytics to
        # install packages or fetch substitute assets during moderation.
        os.environ["YOLO_AUTOINSTALL"] = "false"
        ultralytics = importlib.import_module("ultralytics")
        for module_name in ("ultralytics.utils", "ultralytics.utils.checks"):
            module = sys.modules.get(module_name)
            if module is not None and hasattr(module, "AUTOINSTALL"):
                setattr(module, "AUTOINSTALL", False)
        model = getattr(ultralytics, "YOLO")(str(selection.path), task="detect")
        _validate_model(model)
        _FORBIDDEN_SYMBOLS_YOLO_CACHE[selection.cache_key] = model
        _FORBIDDEN_SYMBOLS_YOLO_INFERENCE_LOCKS.setdefault(selection.cache_key, threading.RLock())
        return model


def _inference_lock(key: str) -> threading.RLock:
    with _FORBIDDEN_SYMBOLS_YOLO_CACHE_LOCK:
        return _FORBIDDEN_SYMBOLS_YOLO_INFERENCE_LOCKS.setdefault(key, threading.RLock())


def _tolist(value: Any) -> list[Any]:
    if value is None:
        return []
    for method in ("detach", "cpu", "numpy"):
        if hasattr(value, method):
            value = getattr(value, method)()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


def _predict(model: Any, source: Any, *, conf: float, iou: float, max_det: int, device: str, lock: threading.RLock) -> Any:
    with lock:
        return model.predict(
            source,
            conf=conf,
            iou=iou,
            imgsz=FORBIDDEN_SYMBOL_INPUT_SIZE,
            max_det=max_det,
            device=device,
            verbose=False,
        )


def _results_list(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, list):
        return results
    try:
        return list(results)
    except TypeError:
        return [results]


def _normalize_result(result: Any, frame: Frame, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    _validate_names(getattr(result, "names", None), "result.names")
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        raise RuntimeError("forbidden symbols YOLO returned a detection result without boxes")
    class_ids = _tolist(getattr(boxes, "cls", None))
    confidences = _tolist(getattr(boxes, "conf", None))
    xyxy_values = _tolist(getattr(boxes, "xyxy", None))
    if len(class_ids) != len(confidences) or len(class_ids) != len(xyxy_values):
        raise RuntimeError(
            "forbidden symbols YOLO returned inconsistent detection arrays: "
            f"{len(class_ids)} classes, {len(confidences)} confidences, {len(xyxy_values)} boxes"
        )
    width, height = image_size
    if width <= 0 or height <= 0:
        raise RuntimeError(f"forbidden symbols YOLO received invalid source image dimensions: {width}x{height}")
    detections: list[dict[str, Any]] = []
    for raw_cid, raw_conf, raw_box in zip(class_ids, confidences, xyxy_values, strict=True):
        cid_value = float(raw_cid)
        if not math.isfinite(cid_value) or not cid_value.is_integer():
            raise RuntimeError(f"forbidden symbols YOLO returned invalid class ID: {raw_cid!r}")
        class_id = int(cid_value)
        if class_id not in FORBIDDEN_SYMBOL_CLASSES:
            raise RuntimeError(f"forbidden symbols YOLO returned unknown class ID {class_id}; expected 0..6")
        confidence = float(raw_conf)
        values = [float(value) for value in _tolist(raw_box)]
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise RuntimeError(f"forbidden symbols YOLO returned invalid confidence {confidence!r}; expected a finite value in [0, 1]")
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise RuntimeError("forbidden symbols YOLO returned an invalid bounding box")
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"forbidden symbols YOLO returned non-positive bounding-box geometry: {values}")
        boundary_tolerance = max(BOUNDING_BOX_ABSOLUTE_TOLERANCE_PX, max(width, height) * 1e-6)
        if not (
            -boundary_tolerance <= x1 <= width + boundary_tolerance
            and -boundary_tolerance <= x2 <= width + boundary_tolerance
            and -boundary_tolerance <= y1 <= height + boundary_tolerance
            and -boundary_tolerance <= y2 <= height + boundary_tolerance
        ):
            raise RuntimeError(
                f"forbidden symbols YOLO returned bounding-box coordinates outside the source image tolerance: {values}"
            )
        x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
        y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError(f"forbidden symbols YOLO bounding box falls outside the source image after clamping: {values}")
        box = [x1, y1, x2, y2]
        norm = [x1 / width if width else 0.0, y1 / height if height else 0.0, x2 / width if width else 0.0, y2 / height if height else 0.0]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(width * height or 1)
        detections.append(
            {
                "frame_idx": int(frame.idx),
                "class_id": class_id,
                "label": FORBIDDEN_SYMBOL_CLASSES[class_id],
                "confidence": safe_float01(confidence),
                "bbox_xyxy": box,
                "bbox_norm_xyxy": [safe_float01(value) for value in norm],
                "area_ratio": safe_float01(area),
                "image_size": [int(width), int(height)],
            }
        )
    return detections


def _threshold(label: str, default: float, overrides: dict[str, float]) -> float:
    return float(overrides.get(label.strip().lower(), default))


def _strict_probability_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a number in [0, 1]") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


def _strict_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _strict_label_thresholds(name: str) -> dict[str, float]:
    raw = os.getenv(name, "") or ""
    allowed = {label.lower() for label in FORBIDDEN_SYMBOL_CLASSES.values()}
    parsed: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"{name} contains invalid entry {item!r}; expected label:threshold")
        label, raw_value = item.split(":", 1)
        label = label.strip().lower()
        if label not in allowed:
            raise ValueError(f"{name} contains unknown forbidden-symbol label {label!r}")
        if label in parsed:
            raise ValueError(f"{name} contains duplicate label {label!r}")
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{name} contains an invalid threshold for {label!r}") from exc
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} threshold for {label!r} must be a finite number in [0, 1]")
        parsed[label] = value
    return parsed


def _validate_policy_thresholds(
    review_conf: float,
    block_conf: float,
    label_review_conf: dict[str, float],
    label_block_conf: dict[str, float],
) -> None:
    for label in FORBIDDEN_SYMBOL_CLASSES.values():
        normalized = label.lower()
        review = label_review_conf.get(normalized, review_conf)
        block = label_block_conf.get(normalized, block_conf)
        if review > block:
            raise ValueError(
                f"forbidden-symbol review threshold for {label!r} ({review}) must not exceed its block threshold ({block})"
            )


def evaluate_forbidden_symbol_policy(
    detections: list[dict[str, Any]],
    *,
    review_conf: float | None = None,
    block_conf: float | None = None,
    label_review_conf: dict[str, float] | None = None,
    label_block_conf: dict[str, float] | None = None,
    ignore_labels: set[str] | None = None,
) -> dict[str, Any]:
    """Apply runtime moderation policy to normalized backend-independent detections."""
    review = review_conf if review_conf is not None else _strict_probability_env("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", 0.30)
    block = block_conf if block_conf is not None else _strict_probability_env("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", 0.90)
    review_map = label_review_conf if label_review_conf is not None else _strict_label_thresholds("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF")
    block_map = label_block_conf if label_block_conf is not None else _strict_label_thresholds("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF")
    ignored = ignore_labels if ignore_labels is not None else env_label_set("FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS", "")
    _validate_policy_thresholds(review, block, review_map, block_map)
    eligible = [d for d in detections if str(d.get("label", "")).strip().lower() not in ignored]
    block_hits = [d for d in eligible if float(d["confidence"]) >= _threshold(str(d["label"]), block, block_map)]
    review_hits = [d for d in eligible if float(d["confidence"]) >= _threshold(str(d["label"]), review, review_map)]

    def strongest(values: list[dict[str, Any]]) -> dict[str, Any] | None:
        return max(values, key=lambda d: float(d["confidence"])) if values else None

    return {
        "review_hit": bool(review_hits or block_hits),
        "block_hit": bool(block_hits),
        "review_detection": strongest(review_hits),
        "block_detection": strongest(block_hits),
        "ignored_detection_count": len(detections) - len(eligible),
    }


def _zero_scores() -> dict[str, float]:
    return {
        "forbidden_symbols_detected": 0.0,
        "forbidden_symbols_max_conf": 0.0,
        "forbidden_symbols_review_hit": 0.0,
        "forbidden_symbols_block_hit": 0.0,
        "forbidden_symbols_detection_count": 0.0,
        "forbidden_symbols_top_conf": 0.0,
    }


class YOLOForbiddenSymbolsEngine(Engine):
    """Local Ultralytics YOLO26s forbidden-symbol object detector."""

    name = "YOLO forbidden symbols"

    def available(self):
        if not env_bool("FORBIDDEN_SYMBOLS_YOLO_ENABLE", True):
            return False, "FORBIDDEN_SYMBOLS_YOLO_ENABLE=0"
        return True, "ok"

    def _skipped(self, start: int, message: str, details: dict[str, Any]) -> EngineResult:
        return EngineResult(
            name=self.name,
            status=EngineStatus.SKIPPED,
            error=message,
            scores=_zero_scores(),
            details={"inference_skipped": True, "detections": [], "detection_count": 0, **details},
            took_ms=now_ms() - start,
        )

    def run(self, path: str, frames: List[Frame], max_api_frames: int = 3) -> EngineResult:
        start = now_ms()
        conf = _strict_probability_env("FORBIDDEN_SYMBOLS_YOLO_CONF", 0.25)
        iou = _strict_probability_env("FORBIDDEN_SYMBOLS_YOLO_IOU", 0.45)
        imgsz = _strict_int_env("FORBIDDEN_SYMBOLS_YOLO_IMGSZ", FORBIDDEN_SYMBOL_INPUT_SIZE)
        if imgsz != FORBIDDEN_SYMBOL_INPUT_SIZE:
            raise ValueError(f"FORBIDDEN_SYMBOLS_YOLO_IMGSZ must be {FORBIDDEN_SYMBOL_INPUT_SIZE} for the YOLO26s models")
        max_det = _strict_int_env("FORBIDDEN_SYMBOLS_YOLO_MAX_DET", 20)
        if max_det <= 0:
            raise ValueError("FORBIDDEN_SYMBOLS_YOLO_MAX_DET must be greater than zero")
        max_frames = _strict_int_env("FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES", 2)
        batch_requested = env_bool("FORBIDDEN_SYMBOLS_YOLO_BATCH_ENABLE", True)
        stop_after_block = env_bool("FORBIDDEN_SYMBOLS_YOLO_STOP_AFTER_BLOCK", True)
        base_details = {
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "max_frames": max_frames,
            "batch_requested": batch_requested,
        }
        if max_frames <= 0:
            return self._skipped(
                start,
                "inference disabled via FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0",
                {**base_details, "skip_reason": "FORBIDDEN_SYMBOLS_YOLO_MAX_FRAMES<=0", "batch_enabled": False},
            )
        review_conf = _strict_probability_env("FORBIDDEN_SYMBOLS_YOLO_REVIEW_CONF", 0.30)
        block_conf = _strict_probability_env("FORBIDDEN_SYMBOLS_YOLO_BLOCK_CONF", 0.90)
        label_review_conf = _strict_label_thresholds("FORBIDDEN_SYMBOLS_YOLO_LABEL_REVIEW_CONF")
        label_block_conf = _strict_label_thresholds("FORBIDDEN_SYMBOLS_YOLO_LABEL_BLOCK_CONF")
        _validate_policy_thresholds(review_conf, block_conf, label_review_conf, label_block_conf)
        ignore_labels = env_label_set("FORBIDDEN_SYMBOLS_YOLO_IGNORE_LABELS", "")
        policy_kwargs = {
            "review_conf": review_conf,
            "block_conf": block_conf,
            "label_review_conf": label_review_conf,
            "label_block_conf": label_block_conf,
            "ignore_labels": ignore_labels,
        }
        base_details.update({
            "review_conf": review_conf,
            "block_conf": block_conf,
            "label_review_conf": label_review_conf,
            "label_block_conf": label_block_conf,
            "ignore_labels": sorted(ignore_labels),
        })
        try:
            selection = _select_model()
        except OptionalModelUnavailable as exc:
            return self._skipped(start, str(exc), base_details)
        try:
            model = _load_model(selection)
        except (OptionalModelUnavailable, ImportError) as exc:
            return self._skipped(start, str(exc), {**base_details, "backend": selection.backend, "model_path": str(selection.path)})

        chosen = frames[:max_frames]
        lock = _inference_lock(selection.cache_key)
        detections: list[dict[str, Any]] = []
        processed: list[int] = []
        result_count = 0
        batch_enabled = bool(batch_requested and selection.backend == "pt" and len(chosen) > 1)
        batch_reason = "onnx model has fixed batch size 1" if batch_requested and selection.backend == "onnx" else ""
        early_stopped = False

        if batch_enabled:
            images = [frame.pil.convert("RGB") for frame in chosen]
            try:
                try:
                    results = _results_list(_predict(model, images, conf=conf, iou=iou, max_det=max_det, device=selection.device_resolved, lock=lock))
                except TypeError:
                    batch_enabled = False
                    batch_reason = "PT predictor rejected batched source; used sequential inference"
                    results = []
                    for image in images:
                        one = _results_list(_predict(model, image, conf=conf, iou=iou, max_det=max_det, device=selection.device_resolved, lock=lock))
                        if len(one) != 1:
                            raise RuntimeError(f"forbidden symbols YOLO returned {len(one)} results for one frame")
                        results.extend(one)
                if len(results) != len(chosen):
                    raise RuntimeError(f"forbidden symbols YOLO returned {len(results)} results for {len(chosen)} frames")
                result_count = len(results)
                for frame, image, result in zip(chosen, images, results, strict=True):
                    processed.append(int(frame.idx))
                    detections.extend(_normalize_result(result, frame, image.size))
            finally:
                for image in images:
                    image.close()
        else:
            for frame in chosen:
                with frame.pil.convert("RGB") as image:
                    results = _results_list(_predict(model, image, conf=conf, iou=iou, max_det=max_det, device=selection.device_resolved, lock=lock))
                    if len(results) != 1:
                        raise RuntimeError(f"forbidden symbols YOLO returned {len(results)} results for one frame")
                    result_count += 1
                    processed.append(int(frame.idx))
                    frame_detections = _normalize_result(results[0], frame, image.size)
                    detections.extend(frame_detections)
                if stop_after_block and evaluate_forbidden_symbol_policy(frame_detections, **policy_kwargs)["block_hit"]:
                    early_stopped = True
                    break

        policy = evaluate_forbidden_symbol_policy(detections, **policy_kwargs)
        top = max(detections, key=lambda d: float(d["confidence"])) if detections else None
        max_conf = float(top["confidence"]) if top else 0.0
        details: dict[str, Any] = {
            "backend": selection.backend,
            "runtime": "ultralytics",
            "model_path": str(selection.path),
            "model_format": selection.backend,
            "model_exists": True,
            "model_size_bytes": int(selection.path.stat().st_size),
            "model_pointer_detected": False,
            "device_requested": selection.device_requested,
            "device_resolved": selection.device_resolved,
            **base_details,
            "class_count": len(FORBIDDEN_SYMBOL_CLASSES),
            "class_names": {str(key): value for key, value in FORBIDDEN_SYMBOL_CLASSES.items()},
            "detection_count": len(detections),
            "top_label": str(top["label"]) if top else "",
            "top_confidence": safe_float01(max_conf),
            "detections": detections,
            "processed_frames": processed,
            "result_count": result_count,
            "batch_enabled": batch_enabled,
            "inference_skipped": False,
            "early_stop_after_block": bool(stop_after_block and not batch_enabled),
            "early_stopped": early_stopped,
            "policy_evaluated": True,
            **policy,
        }
        if batch_reason:
            details["batch_disabled_reason"] = batch_reason
        return EngineResult(
            name=self.name,
            status=EngineStatus.OK,
            scores={
                "forbidden_symbols_detected": 1.0 if detections else 0.0,
                "forbidden_symbols_max_conf": safe_float01(max_conf),
                "forbidden_symbols_review_hit": 1.0 if policy["review_hit"] else 0.0,
                "forbidden_symbols_block_hit": 1.0 if policy["block_hit"] else 0.0,
                "forbidden_symbols_detection_count": float(len(detections)),
                "forbidden_symbols_top_conf": safe_float01(max_conf),
            },
            details=details,
            took_ms=now_ms() - start,
        )
