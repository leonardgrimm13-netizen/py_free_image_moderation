"""Central logging setup for modimg."""
from __future__ import annotations

import logging
import os
import sys


def configure_logging() -> None:
    """Configure root logger once, honoring MODIMG_LOG_LEVEL and MODIMG_DEBUG."""
    root = logging.getLogger("modimg")
    if root.handlers:
        return

    level_name = os.getenv("MODIMG_LOG_LEVEL", "DEBUG" if os.getenv("MODIMG_DEBUG", "0").strip() == "1" else "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s" if level <= logging.DEBUG else "%(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    root.setLevel(level)
    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger under the modimg namespace."""
    configure_logging()
    return logging.getLogger(f"modimg.{name}")
