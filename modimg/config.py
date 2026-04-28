"""Configuration and .env loading (no external dependency required)."""
from __future__ import annotations

import os


def _parse_env_line(line: str):
    """Parse a single env line: KEY=VALUE, export KEY=VALUE, set KEY=VALUE."""
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
    k = k.strip().lstrip("\ufeff")  # strip BOM if present
    v = v.strip()
    if not k:
        return None

    # Strip inline comment for unquoted values: KEY=VAL # comment
    if v and not ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
        if " #" in v:
            v = v.split(" #", 1)[0].rstrip()

    # Strip wrapping quotes
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        v = v[1:-1]
    return k, v


def load_dotenv(path: str, *, override: bool | None = None) -> list[str]:
    """Load a .env file into environment variables. Returns list of loaded keys."""
    loaded: list[str] = []
    if override is None:
        override = (os.getenv("DOTENV_OVERRIDE", "0").strip() == "1")

    try:
        if not os.path.exists(path):
            return loaded
        with open(path, "r", encoding="utf-8-sig") as f:
            for line in f:
                parsed = _parse_env_line(line)
                if not parsed:
                    continue
                k, v = parsed
                if (not override) and (k in os.environ):
                    continue
                os.environ[k] = v
                loaded.append(k)
    except Exception:
        return loaded
    return loaded


def load_dotenv_candidates() -> tuple[str | None, list[str]]:
    """Try loading .env next to the entry script. Returns (used_path, loaded_keys)."""
    base = os.path.dirname(os.path.abspath(__file__))
    # package file: modimg/config.py -> project root is parent of modimg
    root = os.path.dirname(base)

    env_path = os.path.join(root, ".env")
    env_txt_path = os.path.join(root, ".env.txt")
    example_path = os.path.join(root, ".env.example")

    used: str | None = None
    loaded_keys: list[str] = []

    # STRICT: decide by file existence
    if os.path.exists(env_path):
        used = env_path
        loaded_keys = load_dotenv(env_path)
    elif os.path.exists(env_txt_path):
        used = env_txt_path
        loaded_keys = load_dotenv(env_txt_path)
    elif os.path.exists(example_path):
        used = example_path
        loaded_keys = load_dotenv(example_path, override=False)

    # Helpful debug prints
    if os.getenv("DEBUG", "0").strip() == "1":
        checked = [env_path, env_txt_path, example_path]
        if used and loaded_keys:
            uniq = ", ".join(sorted(set(loaded_keys)))
            print(f"[dotenv] loaded from {used}: {uniq}")
        elif used:
            print(f"[dotenv] found {used} but no keys were loaded (check formatting: KEY=VALUE).")
        else:
            print(f"[dotenv] no env file found in project root (checked: {', '.join(checked)}).")

        if (not os.path.exists(env_path)) and os.path.exists(env_txt_path):
            print("[dotenv] NOTE: You have .env.txt — rename it to .env so editors/tools behave correctly.")

    return used, loaded_keys


# Load .env automatically on import (matches old behavior)
_USED_DOTENV_PATH, _LOADED_KEYS = load_dotenv_candidates()

# Reduce noisy TensorFlow logs if OpenNSFW2 loads TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def project_root() -> str:
    """Absolute path of project root (parent of the modimg package)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
