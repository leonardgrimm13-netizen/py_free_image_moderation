"""Small utilities shared across the project."""
from __future__ import annotations

import os
import io
import json
import math
import re
import mimetypes
import tempfile
import time
import ssl
import urllib.parse
import urllib.request
from typing import Any, Tuple
from PIL import Image

def env_int(name: str, default: int) -> int:
    """Read an int from env, returning default on missing/invalid."""
    try:
        v = os.getenv(name)
        if v is None:
            return default
        v = str(v).strip()
        if v == "":
            return default
        if re.fullmatch(r"[+-]?\d+(?:\.0+)?", v):
            return int(float(v))
        return int(v)
    except Exception:
        return default


def env_int_any(names: tuple[str, ...], default: int) -> int:
    """Read the first defined env var in `names` as an int."""
    for n in names:
        if os.getenv(n) is not None:
            return env_int(n, default)
    return default


def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return default


def env_float(name: str, default: float, *, min_value: float | None = None, max_value: float | None = None) -> float:
    """Read a float from env, returning default on missing/invalid values.

    Optional min/max bounds clamp the parsed value.
    """
    raw = os.getenv(name)
    try:
        value = float(str(raw).strip()) if raw is not None and str(raw).strip() != "" else float(default)
    except Exception:
        value = float(default)
    if not math.isfinite(value):
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value


def env_label_set(name: str, default: str = "") -> set[str]:
    """Read a comma-separated env var as a normalized lowercase label set."""
    raw = os.getenv(name, default) or default
    return {x.strip().lower() for x in str(raw).split(",") if x.strip()}


def parse_label_thresholds(raw: str | None) -> dict[str, float]:
    """Parse comma-separated ``label:threshold`` pairs into normalized thresholds.

    Invalid entries are ignored. Threshold values are clamped into [0, 1],
    matching ``safe_float01`` semantics for other moderation confidence values.
    """
    thresholds: dict[str, float] = {}
    for entry in str(raw or "").split(","):
        if ":" not in entry:
            continue
        label, value = entry.split(":", 1)
        label = label.strip().lower()
        value = value.strip()
        if not label or not value:
            continue
        try:
            parsed = float(value)
        except Exception:
            continue
        if not math.isfinite(parsed):
            continue
        thresholds[label] = safe_float01(parsed)
    return thresholds


def env_label_thresholds(name: str, default: str = "") -> dict[str, float]:
    """Read an env var as normalized ``label:threshold`` confidence overrides."""
    raw = os.getenv(name, default) or default
    return parse_label_thresholds(raw)


def status_value(status: Any) -> str:
    """Return a lowercase engine status string for enums and legacy strings."""
    return str(status.value if hasattr(status, "value") else status).lower()


def safe_float01(v: Any, default: float = 0.0) -> float:
    """Convert to float in [0,1]. NaN/inf/invalid -> default."""
    try:
        f = float(v)
        if not math.isfinite(f):
            return float(default)
        if f < 0.0:
            return 0.0
        if f > 1.0:
            return 1.0
        return f
    except Exception:
        return float(default)

def is_url(s: str) -> bool:
    try:
        p = urllib.parse.urlparse(s)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

def now_ms() -> int:
    return int(time.time() * 1000)

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()

def guess_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "application/octet-stream"

def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"]

def _sniff_image(data0: bytes) -> Tuple[str, str]:
    if data0.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if data0.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if data0[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif", "image/gif"
    if len(data0) >= 12 and data0[:4] == b"RIFF" and data0[8:12] == b"WEBP":
        return ".webp", "image/webp"
    return "", ""

def download_url_to_temp(url: str, max_bytes: int = 25_000_000, timeout_sec: int = 20) -> tuple[str, str]:
    """Download an image from URL to a temp file; returns (temp_path, display_name)."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "image-moderator/1.0", "Accept": "image/*,*/*;q=0.8"},
        method="GET",
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout_sec) as resp:
        cl = resp.headers.get("Content-Length")
        if cl is not None:
            try:
                if int(cl) > max_bytes:
                    raise RuntimeError(f"URL too large: {int(cl)} bytes (limit {max_bytes})")
            except ValueError:
                pass
        ctype = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        data = resp.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError(f"URL too large: downloaded > {max_bytes} bytes")

    sniff_ext, sniff_mime = _sniff_image(data)
    if ctype and (not ctype.startswith("image/")):
        if sniff_mime:
            ctype = sniff_mime
        else:
            raise RuntimeError(f"URL did not return an image (content-type={ctype})")
    if (not ctype) and sniff_mime:
        ctype = sniff_mime

    ext = ""
    if ctype in ("image/jpeg", "image/jpg"):
        ext = ".jpg"
    elif ctype == "image/png":
        ext = ".png"
    elif ctype == "image/webp":
        ext = ".webp"
    elif ctype == "image/gif":
        ext = ".gif"
    else:
        path_ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower()
        if path_ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            ext = ".jpg" if path_ext == ".jpeg" else path_ext
        elif sniff_ext:
            ext = sniff_ext
        else:
            raise RuntimeError("URL does not look like a supported image format (jpeg/png/webp/gif).")

    display = os.path.basename(urllib.parse.urlparse(url).path) or ("downloaded" + ext)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(data)
    tmp.close()
    return tmp.name, display

def safe_model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return str(obj)
