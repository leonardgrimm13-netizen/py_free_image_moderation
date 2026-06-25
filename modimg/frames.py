"""Frame loading and sampling (supports GIFs)."""
from __future__ import annotations

from typing import List
from PIL import Image

from .types import Frame

def load_frames(path: str, sample_frames: int = 12) -> List[Frame]:
    try:
        sample_limit = max(1, int(sample_frames))
    except Exception:
        sample_limit = 1
    frames: List[Frame] = []
    with Image.open(path) as img:
        is_animated = bool(getattr(img, "is_animated", False)) or getattr(img, "n_frames", 1) > 1
        if is_animated:
            n = getattr(img, "n_frames", 1)
            if sample_limit <= 1:
                indices = [0]
            elif n <= sample_limit:
                indices = list(range(n))
            else:
                indices = [int(round(i)) for i in [0] + [j * (n - 1) / (sample_limit - 1) for j in range(1, sample_limit - 1)] + [n - 1]]
                indices = sorted({max(0, min(n - 1, k)) for k in indices})

            for idx in indices:
                try:
                    img.seek(idx)
                except Exception:
                    continue
                frame_rgb = img.convert("RGB")
                # Do not eagerly encode JPEG; compute lazily only if needed.
                frames.append(Frame(idx=idx, pil=frame_rgb.copy()))
        else:
            frame_rgb = img.convert("RGB")
            frames.append(Frame(idx=0, pil=frame_rgb.copy()))
    if not frames:
        raise ValueError("no readable image frames")
    return frames
