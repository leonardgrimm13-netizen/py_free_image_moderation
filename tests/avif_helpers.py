"""Small in-memory AVIF fixtures shared by AVIF tests."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Sequence

from PIL import Image


def make_avif_bytes(
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (16, 12),
    color: int | tuple[int, ...] = (30, 80, 160),
    quality: int = 90,
) -> bytes:
    """Encode one small AVIF image using Pillow's built-in AVIF support."""
    image = Image.new(mode, size, color=color)
    output = BytesIO()
    try:
        image.save(output, format="AVIF", quality=quality)
        return output.getvalue()
    finally:
        image.close()
        output.close()


def write_avif(
    path: Path,
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (16, 12),
    color: int | tuple[int, ...] = (30, 80, 160),
    quality: int = 90,
) -> Path:
    """Write one small AVIF test image and return its path."""
    path.write_bytes(make_avif_bytes(mode=mode, size=size, color=color, quality=quality))
    return path


def make_animated_avif_bytes(
    colors: Sequence[tuple[int, int, int]],
    *,
    size: tuple[int, int] = (16, 12),
    quality: int = 90,
) -> bytes:
    """Encode a deterministic AVIF sequence without storing a fixture binary."""
    if not colors:
        raise ValueError("at least one animation frame is required")
    frames = [Image.new("RGB", size, color=color) for color in colors]
    output = BytesIO()
    try:
        frames[0].save(
            output,
            format="AVIF",
            save_all=True,
            append_images=frames[1:],
            duration=40,
            loop=0,
            quality=quality,
        )
        return output.getvalue()
    finally:
        for frame in frames:
            frame.close()
        output.close()
