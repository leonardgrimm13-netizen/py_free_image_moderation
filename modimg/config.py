"""Configuration and .env loading utilities."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from .utils import env_bool, env_int, safe_float01


def _parse_env_line(line: str) -> Optional[tuple[str, str]]:
    """Parse a single env line supporting KEY=VALUE and export/set prefixes."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    lower = s.lower()
    if lower.startswith("export "):
        s = s[7:].lstrip()
    elif lower.startswith("set "):
        s = s[4:].lstrip()
    if "=" not in s:
        return None
    k, v = s.split("=", 1)
    k = k.strip().lstrip("\ufeff")
    v = v.strip()
    if not k:
        return None
    if v and not ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
        if " #" in v:
            v = v.split(" #", 1)[0].rstrip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return k, v


def load_dotenv(path: str, *, override: bool | None = None) -> list[str]:
    """Load a .env file into environment variables and return loaded keys."""
    loaded: list[str] = []
    if override is None:
        override = env_bool("DOTENV_OVERRIDE", False)

    try:
        p = Path(path)
        if not p.exists():
            return loaded
        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                parsed = _parse_env_line(line)
                if not parsed:
                    continue
                k, v = parsed
                if (not override) and (k in os.environ):
                    continue
                os.environ[k] = v
                loaded.append(k)
    except OSError:
        return loaded
    return loaded


def load_dotenv_candidates() -> tuple[str | None, list[str]]:
    """Try loading .env/.env.txt/.env.example in the project root."""
    root = Path(__file__).resolve().parent.parent
    candidates = [root / ".env", root / ".env.txt", root / ".env.example"]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate), load_dotenv(str(candidate), override=False if candidate.name == ".env.example" else None)
    return None, []


@dataclass
class Config:
    """Typed runtime configuration loaded from process environment."""

    ocr_enable: bool
    opennsfw2_disable: bool
    openai_disable: bool
    sightengine_disable: bool
    short_circuit_phash: bool
    sample_frames: int
    final_block_threshold: float
    parallel_engines: bool
    parallel_workers: int
    debug: bool

    @classmethod
    def from_env(cls) -> "Config":
        """Build configuration from environment variables."""
        return cls(
            ocr_enable=env_bool("OCR_ENABLE", False),
            opennsfw2_disable=env_bool("OPENNSFW2_DISABLE", False),
            openai_disable=env_bool("OPENAI_DISABLE", False),
            sightengine_disable=env_bool("SIGHTENGINE_DISABLE", False),
            short_circuit_phash=env_bool("SHORT_CIRCUIT_PHASH", True),
            sample_frames=max(1, env_int("SAMPLE_FRAMES", 12)),
            final_block_threshold=safe_float01(os.getenv("FINAL_BLOCK_THRESHOLD", "0.80"), default=0.80),
            parallel_engines=env_bool("MODIMG_PARALLEL_ENGINES", False),
            parallel_workers=max(1, env_int("MODIMG_PARALLEL_WORKERS", 4)),
            debug=env_bool("MODIMG_DEBUG", False) or env_bool("DEBUG", False),
        )


_CURRENT_CONFIG: Config | None = None


def get_config(*, reload: bool = False) -> Config:
    """Return cached configuration with optional reload."""
    global _CURRENT_CONFIG
    if _CURRENT_CONFIG is None or reload:
        _CURRENT_CONFIG = Config.from_env()
    return _CURRENT_CONFIG


_USED_DOTENV_PATH, _LOADED_KEYS = load_dotenv_candidates()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def project_root() -> str:
    """Absolute path to project root."""
    return str(Path(__file__).resolve().parent.parent)
