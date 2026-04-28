"""Enumerations shared across moderation modules."""
from __future__ import annotations

from enum import Enum


class EngineStatus(str, Enum):
    """Execution status for an engine run."""

    OK = "ok"
    SKIPPED = "skipped"
    ERROR = "error"


class VerdictLabel(str, Enum):
    """Final moderation verdict labels."""

    OK = "OK"
    REVIEW = "REVIEW"
    BLOCK = "BLOCK"
