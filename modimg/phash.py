"""Perceptual hash helpers + allow/block list management."""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

from .resources import resolve_bundled_resource_path, resolve_resource_path
from .utils import atomic_write_text, env_int


# --- pHash helpers (optional ImageHash, otherwise numpy DCT implementation) ---
try:
    import imagehash as _imagehash  # type: ignore
except Exception:
    _imagehash = None  # type: ignore

_PHASH_DCT_CACHE: Dict[int, np.ndarray] = {}
_PHASH_LIST_CACHE: Dict[str, Tuple[tuple[int, int], List[Tuple[str, str, int, int]]]] = {}
_PHASH_EXACT_CACHE: Dict[str, Tuple[tuple[int, int], Dict[int, Dict[int, Tuple[str, str]]]]] = {}
_PHASH_CACHE_LOCK = threading.RLock()
_HEX_RE = re.compile(r"^[0-9a-f]+$")


def _optional_imagehash_phash(img: Image.Image) -> Optional[str]:
    """Use ImageHash when possible; its pure-numpy fallback remains authoritative."""
    if _imagehash is None:
        return None
    try:
        return str(_imagehash.phash(img)).lower()
    except Exception:
        # Optional ImageHash versions can reject Pillow modes accepted by the
        # built-in implementation. Falling back keeps pHash available offline.
        return None

def project_root() -> str:
    # parent of modimg
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_list_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return p
    path = Path(p).expanduser()
    if path.is_absolute():
        return str(path)
    return str(resolve_resource_path(path))

def get_allowlist_path() -> str:
    default = os.path.join("data", "phash_allowlist.txt")
    configured = os.getenv("PHASH_ALLOWLIST")
    return resolve_list_path(configured) if configured and configured.strip() else str(resolve_bundled_resource_path(default))

def get_blocklist_path() -> str:
    default = os.path.join("data", "phash_blocklist.txt")
    configured = os.getenv("PHASH_BLOCKLIST")
    return resolve_list_path(configured) if configured and configured.strip() else str(resolve_bundled_resource_path(default))

def _phash_cache_invalidate(path: str) -> None:
    try:
        p = resolve_list_path(path)
    except Exception:
        p = path
    with _PHASH_CACHE_LOCK:
        _PHASH_LIST_CACHE.pop(p, None)
        _PHASH_EXACT_CACHE.pop(p, None)

def _append_phash_to_list(phash_hex: str, list_path: str, label: str) -> bool:
    list_path = resolve_list_path(list_path)
    phash_hex = (phash_hex or "").strip().lower()
    max_hex_length = max(1, env_int("PHASH_MAX_HEX_LENGTH", 256))
    if not phash_hex or len(phash_hex) > max_hex_length or _HEX_RE.fullmatch(phash_hex) is None:
        return False
    safe_label = " ".join(str(label or "unknown").replace(",", " ").split())[:200] or "unknown"
    with _PHASH_CACHE_LOCK:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(list_path)), exist_ok=True)
            max_bytes = max(1, env_int("PHASH_LIST_MAX_BYTES", 10_000_000))
            existing: set[str] = set()
            existing_text = ""
            if os.path.exists(list_path):
                if os.path.getsize(list_path) > max_bytes:
                    return False
                with open(list_path, "rb") as f:
                    raw = f.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    return False
                existing_text = raw.decode("utf-8")
                for line in existing_text.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    h = line.split(",", 1)[0].strip().lower()
                    if h:
                        existing.add(h)
            if phash_hex in existing:
                return False
            separator = "" if not existing_text or existing_text.endswith(("\n", "\r")) else "\n"
            updated_text = f"{existing_text}{separator}{phash_hex},{safe_label}\n"
            if len(updated_text.encode("utf-8")) > max_bytes:
                return False
            atomic_write_text(list_path, updated_text)
            _phash_cache_invalidate(list_path)
            return True
        except (OSError, UnicodeError):
            return False


def append_phash_to_allowlist(phash_hex: str, allowlist_path: str, label: str) -> bool:
    return _append_phash_to_list(phash_hex, allowlist_path, label)


def append_phash_to_blocklist(phash_hex: str, blocklist_path: str, label: str) -> bool:
    return _append_phash_to_list(phash_hex, blocklist_path, label)

def _dct_matrix(n: int) -> np.ndarray:
    with _PHASH_CACHE_LOCK:
        m = _PHASH_DCT_CACHE.get(n)
        if m is not None:
            return m
    x = np.arange(n, dtype=np.float32)
    k = x.reshape((n, 1))
    mat = np.cos((np.pi * (2.0 * x + 1.0) * k) / (2.0 * n)).astype(np.float32)
    mat[0, :] *= (1.0 / np.sqrt(n))
    mat[1:, :] *= (np.sqrt(2.0 / n))
    with _PHASH_CACHE_LOCK:
        cached = _PHASH_DCT_CACHE.get(n)
        if cached is not None:
            return cached
        _PHASH_DCT_CACHE[n] = mat
        return mat

def phash_hex_from_pil(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    imagehash_result = _optional_imagehash_phash(img)
    if imagehash_result is not None:
        return imagehash_result
    size = int(hash_size) * int(highfreq_factor)
    try:
        resample = Image.Resampling.LANCZOS  # type: ignore[attr-defined]
    except Exception:
        resample = Image.LANCZOS  # type: ignore[attr-defined]
    with img.convert("L") as grayscale:
        with grayscale.resize((size, size), resample=resample) as resized:
            pixels = np.array(resized, dtype=np.float32, copy=True)
    n = pixels.shape[0]
    C = _dct_matrix(n)
    dct = C @ pixels @ C.T
    dctlow = dct[:hash_size, :hash_size]
    med = float(np.median(dctlow[1:, :])) if hash_size > 1 else float(np.median(dctlow))
    bits = (dctlow > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    width = (hash_size * hash_size) // 4
    return f"{val:0{width}x}"

def frame_phash_hex_int(frame: object) -> Tuple[str, int]:
    lock_factory = getattr(frame, "cache_lock", None)
    lock = lock_factory() if callable(lock_factory) else _PHASH_CACHE_LOCK
    with lock:
        hx = getattr(frame, "_phash_hex", None)
        iv = getattr(frame, "_phash_int", None)
        if hx is None or iv is None:
            pil = getattr(frame, "pil")
            hx = phash_hex_from_pil(pil)
            iv = int(hx, 16)
            setattr(frame, "_phash_hex", hx)
            setattr(frame, "_phash_int", iv)
        return str(hx), int(iv)


def _file_signature(path: str) -> tuple[int, int] | None:
    try:
        stat_result = os.stat(path)
    except OSError:
        return None
    return stat_result.st_mtime_ns, stat_result.st_size

def load_phash_list(path: str, default_label: str) -> List[Tuple[str, str, int, int]]:
    path = resolve_list_path(path)
    signature = _file_signature(path)
    if signature is None:
        return []
    max_bytes = max(1, env_int("PHASH_LIST_MAX_BYTES", 10_000_000))
    if signature[1] > max_bytes:
        return []
    with _PHASH_CACHE_LOCK:
        cached = _PHASH_LIST_CACHE.get(path)
        if cached and cached[0] == signature:
            return cached[1]
    out: List[Tuple[str, str, int, int]] = []
    max_entries = max(1, env_int("PHASH_LIST_MAX_ENTRIES", 100_000))
    max_hex_length = max(1, env_int("PHASH_MAX_HEX_LENGTH", 256))
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return []
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",", 1)]
            hx = parts[0].lower()
            if not hx or len(hx) > max_hex_length or _HEX_RE.fullmatch(hx) is None:
                continue
            label = parts[1] if len(parts) > 1 and parts[1] else default_label
            iv = int(hx, 16)
            out.append((hx, label, iv, len(hx)))
            if len(out) >= max_entries:
                break
    except (OSError, UnicodeError):
        out = []
    with _PHASH_CACHE_LOCK:
        _PHASH_LIST_CACHE[path] = (signature, out)
    return out


def load_phash_exact_map(path: str, default_label: str) -> Dict[int, Dict[int, Tuple[str, str]]]:
    """Return map[hex_len][int] -> (hex,label) for O(1) exact matches."""
    path = resolve_list_path(path)
    signature = _file_signature(path)
    if signature is None:
        return {}
    with _PHASH_CACHE_LOCK:
        cached = _PHASH_EXACT_CACHE.get(path)
        if cached and cached[0] == signature:
            return cached[1]
    mp: Dict[int, Dict[int, Tuple[str, str]]] = {}
    entries = load_phash_list(path, default_label=default_label)
    for hx, label, iv, hlen in entries:
        mp.setdefault(hlen, {})[iv] = (hx, label)
    with _PHASH_CACHE_LOCK:
        _PHASH_EXACT_CACHE[path] = (signature, mp)
    return mp

def best_match_distance(phash_int: int, phash_hex_len: int, entries: List[Tuple[str, str, int, int]], max_distance: int) -> Optional[Tuple[int, str, str]]:
    """Return (dist, hex, label) for best match within max_distance."""
    best: Optional[Tuple[int, str, str]] = None
    for hx, label, iv, hlen in entries:
        if hlen != phash_hex_len:
            continue
        d = (phash_int ^ iv).bit_count()
        if d <= max_distance and (best is None or d < best[0]):
            best = (d, hx, label)
    return best
