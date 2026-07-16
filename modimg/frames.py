"""Frame loading and sampling (supports GIFs)."""
from __future__ import annotations

import warnings

from PIL import Image

from .types import Frame
from .utils import env_int


def _validate_dimensions(img: Image.Image) -> int:
    width, height = img.size
    max_dimension = max(1, env_int("MODIMG_MAX_IMAGE_DIMENSION", 32_768))
    max_pixels = max(1, env_int("MODIMG_MAX_IMAGE_PIXELS", 64_000_000))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image dimensions: {width}x{height}")
    if width > max_dimension or height > max_dimension:
        raise ValueError(f"image dimensions exceed limit: {width}x{height} (max dimension {max_dimension})")
    pixels = width * height
    if pixels > max_pixels:
        raise ValueError(f"image pixel count exceeds limit: {pixels} (limit {max_pixels})")
    return pixels


def load_frames(path: str, sample_frames: int = 12) -> list[Frame]:
    try:
        sample_limit = max(1, int(sample_frames))
    except (TypeError, ValueError, OverflowError):
        sample_limit = 1
    sample_limit = min(sample_limit, max(1, env_int("MODIMG_MAX_SAMPLED_FRAMES", 100)))
    frames: list[Frame] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                pixels_per_frame = _validate_dimensions(img)
                n = int(getattr(img, "n_frames", 1) or 1)
                is_animated = bool(getattr(img, "is_animated", False)) or n > 1
                if is_animated:
                    max_animation_frames = max(1, env_int("MODIMG_MAX_ANIMATION_FRAMES", 5_000))
                    if n > max_animation_frames:
                        raise ValueError(f"animation frame count exceeds limit: {n} (limit {max_animation_frames})")
                    if sample_limit <= 1:
                        indices = [0]
                    elif n <= sample_limit:
                        indices = list(range(n))
                    else:
                        positions = [0] + [j * (n - 1) / (sample_limit - 1) for j in range(1, sample_limit - 1)] + [n - 1]
                        indices = sorted({max(0, min(n - 1, int(round(position)))) for position in positions})

                    max_decoded_pixels = max(1, env_int("MODIMG_MAX_DECODED_PIXELS", 256_000_000))
                    decoded_pixels = pixels_per_frame * len(indices)
                    if decoded_pixels > max_decoded_pixels:
                        raise ValueError(
                            f"sampled frames exceed decoded-pixel budget: {decoded_pixels} (limit {max_decoded_pixels})"
                        )

                    for idx in indices:
                        try:
                            img.seek(idx)
                            frame_rgb = img.convert("RGB")
                        except (EOFError, OSError, ValueError) as exc:
                            raise ValueError(f"failed to decode selected animation frame {idx}") from exc
                        frames.append(Frame(idx=idx, pil=frame_rgb))
                else:
                    frames.append(Frame(idx=0, pil=img.convert("RGB")))
    except BaseException:
        for frame in frames:
            frame.pil.close()
        raise
    if not frames:
        raise ValueError("no readable image frames")
    return frames
