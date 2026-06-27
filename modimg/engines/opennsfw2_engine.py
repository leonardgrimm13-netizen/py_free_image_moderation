from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from typing import Any, List, Tuple, Optional
from PIL import Image

from ..enums import EngineStatus
from ..types import Engine, EngineResult
from ..utils import env_bool, now_ms

class OpenNSFW2Engine(Engine):
    """Offline NSFW probability via OpenNSFW2.

    Supports the PyPI package **opennsfw2** (recommended) and also keeps compatibility
    with older code that imports **open_nsfw2**.
    """
    name = "OpenNSFW2"

    _BACKEND = None  # (name, module)
    _BACKEND_LOCK = threading.RLock()

    @staticmethod
    def _in_process_mode() -> str:
        raw = os.getenv("OPENNSFW2_IN_PROCESS")
        if raw is None or str(raw).strip() == "":
            return "0"
        value = str(raw).strip().lower()
        if value == "auto":
            return "auto"
        if value in ("1", "true", "yes", "on"):
            return "1"
        if value in ("0", "false", "no", "off"):
            return "0"
        return "0"

    @staticmethod
    def _is_missing_backend_error(exc: Exception) -> bool:
        msg = f"{type(exc).__name__}: {exc}".lower()
        backend_markers = (
            "tensorflow",
            "tf_keras",
            "tf-keras",
            "keras",
            "backend",
            "no module named",
        )
        return isinstance(exc, (ImportError, ModuleNotFoundError)) or any(marker in msg for marker in backend_markers)

    def _import_backend(self):
        with OpenNSFW2Engine._BACKEND_LOCK:
            if OpenNSFW2Engine._BACKEND is not None:
                return OpenNSFW2Engine._BACKEND
            # Preferred official package name on PyPI is `opennsfw2`.
            try:
                import opennsfw2 as n2  # type: ignore
                OpenNSFW2Engine._BACKEND = ("opennsfw2", n2)
                return OpenNSFW2Engine._BACKEND
            except Exception:
                pass
            # Back-compat name some projects use:
            import open_nsfw2 as n2  # type: ignore
            OpenNSFW2Engine._BACKEND = ("open_nsfw2", n2)
            return OpenNSFW2Engine._BACKEND

    def available(self) -> Tuple[bool, str]:
        if env_bool("OPENNSFW2_DISABLE", False):
            return False, "disabled via OPENNSFW2_DISABLE=1"
        try:
            if "opennsfw2" in sys.modules or "open_nsfw2" in sys.modules:
                return True, "ok"
            if importlib.util.find_spec("opennsfw2") is not None or importlib.util.find_spec("open_nsfw2") is not None:
                return True, "ok"
            return False, "opennsfw2/open_nsfw2 not installed"
        except Exception as e:
            return False, f"opennsfw2/open_nsfw2 availability check failed: {type(e).__name__}: {e}"

    def _result_from_probability(self, prob: Any, backend_name: str, start: int) -> EngineResult:
        if prob is None:
            return EngineResult(
                name=self.name,
                status=EngineStatus.ERROR,
                error=f"{backend_name} installed but no compatible predict_* function found",
                took_ms=now_ms() - start,
            )
        try:
            p = float(prob)
        except Exception as exc:
            return EngineResult(
                name=self.name,
                status=EngineStatus.ERROR,
                error=f"{backend_name} returned non-numeric probability: {type(exc).__name__}: {exc}",
                took_ms=now_ms() - start,
            )
        p = float(max(0.0, min(1.0, p)))
        return EngineResult(name=self.name, status=EngineStatus.OK, scores={"nsfw_probability": p}, took_ms=now_ms() - start)

    def _predict_in_process(self, path: str, frames: List[Any], start: int) -> EngineResult:
        try:
            backend_name, n2 = self._import_backend()
        except Exception as exc:
            return EngineResult(
                name=self.name,
                status=EngineStatus.SKIPPED,
                error=f"opennsfw2 backend unavailable: {type(exc).__name__}: {exc}",
                took_ms=now_ms() - start,
            )

        def _to_pil(x: Any) -> Image.Image:
            if hasattr(x, "pil"):
                return getattr(x, "pil")
            return x

        im = _to_pil(frames[0]).convert("RGB")

        prob = None
        try:
            # Official opennsfw2 API exposes predict_image/predict_images for image paths.
            if hasattr(n2, "predict_image"):
                prob = n2.predict_image(path)
            elif hasattr(n2, "predict_images"):
                prob = (n2.predict_images([path]) or [0.0])[0]
            elif hasattr(n2, "predict"):
                prob = n2.predict(im)
        except Exception as e:
            status = EngineStatus.SKIPPED if self._is_missing_backend_error(e) else EngineStatus.ERROR
            return EngineResult(
                name=self.name,
                status=status,
                error=f"{backend_name} prediction failed: {type(e).__name__}: {e}",
                took_ms=now_ms() - start,
            )

        return self._result_from_probability(prob, backend_name, start)

    def _predict_in_subprocess(self, path: str, start: int) -> EngineResult:
        script = r"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

try:
    try:
        import opennsfw2 as n2
        backend_name = "opennsfw2"
    except Exception:
        import open_nsfw2 as n2
        backend_name = "open_nsfw2"

    image_path = sys.argv[1]
    if hasattr(n2, "predict_image"):
        prob = n2.predict_image(image_path)
    elif hasattr(n2, "predict_images"):
        values = n2.predict_images([image_path]) or [0.0]
        prob = values[0]
    else:
        raise RuntimeError("installed backend has no predict_image/predict_images function")

    print(json.dumps({"ok": True, "backend": backend_name, "probability": float(prob)}))
except Exception as exc:
    print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": str(exc)}))
    raise SystemExit(2)
"""
        try:
            timeout = max(1.0, float(os.getenv("OPENNSFW2_TIMEOUT_SEC", "120")))
        except Exception:
            timeout = 120.0
        env = os.environ.copy()
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        env.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script, path],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return EngineResult(
                name=self.name,
                status=EngineStatus.ERROR,
                error=f"opennsfw2 prediction timed out after {timeout:.0f}s",
                took_ms=now_ms() - start,
            )
        except Exception as exc:
            return EngineResult(
                name=self.name,
                status=EngineStatus.ERROR,
                error=f"opennsfw2 subprocess failed: {type(exc).__name__}: {exc}",
                took_ms=now_ms() - start,
            )

        payload = None
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    payload = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

        if proc.returncode == 0 and isinstance(payload, dict) and payload.get("ok") is True:
            return self._result_from_probability(payload.get("probability"), str(payload.get("backend") or "opennsfw2"), start)

        stderr_tail = (proc.stderr or "").strip()[-1200:]
        if proc.returncode < 0:
            return EngineResult(
                name=self.name,
                status=EngineStatus.ERROR,
                error=f"opennsfw2 subprocess terminated by signal {-proc.returncode}: {stderr_tail}",
                took_ms=now_ms() - start,
            )
        if isinstance(payload, dict):
            err_type = str(payload.get("error_type") or "Error")
            err_msg = str(payload.get("error") or "")
            probe_exc = ModuleNotFoundError(err_msg) if "no module named" in err_msg.lower() else RuntimeError(err_msg)
            status = EngineStatus.SKIPPED if err_type in {"ImportError", "ModuleNotFoundError"} or self._is_missing_backend_error(probe_exc) else EngineStatus.ERROR
            return EngineResult(
                name=self.name,
                status=status,
                error=f"opennsfw2 prediction failed: {err_type}: {err_msg}",
                took_ms=now_ms() - start,
            )

        return EngineResult(
            name=self.name,
            status=EngineStatus.ERROR,
            error=f"opennsfw2 subprocess exited {proc.returncode}: {stderr_tail}",
            took_ms=now_ms() - start,
        )

    def run(self, path: str, frames: List[Any], max_api_frames: Optional[int] = None) -> EngineResult:
        start = now_ms()
        if not frames:
            return EngineResult(name=self.name, status=EngineStatus.SKIPPED, error="no frames", took_ms=now_ms() - start)
        mode = self._in_process_mode()
        if mode == "1":
            return self._predict_in_process(path, frames, start)
        if mode == "auto":
            in_process_result = self._predict_in_process(path, frames, start)
            if in_process_result.status == EngineStatus.ERROR:
                return self._predict_in_subprocess(path, start)
            return in_process_result
        return self._predict_in_subprocess(path, start)
