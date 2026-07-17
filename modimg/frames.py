"""Raster image frame loading and sampling, including GIF/WebP/AVIF sequences."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import warnings

from PIL import Image, features

from .types import Frame
from .utils import _IMAGE_SNIFF_BYTES, _sniff_image, env_int


@lru_cache(maxsize=1)
def pillow_avif_codec_available() -> bool:
    """Return whether this Pillow build exposes its compiled AVIF module."""
    try:
        return bool(features.check_module("avif"))
    except (ImportError, OSError, ValueError):
        return False


def _sniff_path_format(path: str) -> str:
    try:
        with Path(path).open("rb") as source:
            extension, _mime = _sniff_image(source.read(_IMAGE_SNIFF_BYTES))
    except OSError:
        return ""
    return extension.lstrip(".")


def validate_image_dimensions(img: Image.Image) -> int:
    """Validate decoded image dimensions against the global input limits."""
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
    detected_format = _sniff_path_format(path)
    avif_detected = detected_format == "avif"
    if avif_detected:
        max_avif_bytes = max(1, env_int("MODIMG_MAX_AVIF_BYTES", 100_000_000))
        try:
            source_bytes = Path(path).stat().st_size
        except OSError:
            source_bytes = -1
        if source_bytes > max_avif_bytes:
            raise ValueError(
                f"AVIF source exceeds byte limit: {source_bytes} (limit {max_avif_bytes})"
            )
    if avif_detected and not pillow_avif_codec_available():
        raise ValueError(
            "AVIF image detected, but this Pillow build has no working AVIF codec; "
            "Pillow>=11.3.0 with AVIF support is required"
        )
    try:
        sample_limit = max(1, int(sample_frames))
    except (TypeError, ValueError, OverflowError):
        sample_limit = 1
    sample_limit = min(sample_limit, max(1, env_int("MODIMG_MAX_SAMPLED_FRAMES", 100)))
    frames: list[Frame] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            try:
                opened_image = Image.open(path)
            except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
                if avif_detected:
                    raise ValueError(
                        "failed to decode AVIF image: invalid data or an unavailable AVIF decoder; "
                        "Pillow>=11.3.0 with AVIF support is required"
                    ) from exc
                raise
            with opened_image as img:
                source_format = str(getattr(img, "format", "") or detected_format).strip().lower()
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
                    decoded_pixels = 0
                    for idx in indices:
                        try:
                            img.seek(idx)
                        except (EOFError, OSError, ValueError) as exc:
                            raise ValueError(f"failed to decode selected animation frame {idx}") from exc
                        frame_pixels = validate_image_dimensions(img)
                        decoded_pixels += frame_pixels
                        if decoded_pixels > max_decoded_pixels:
                            raise ValueError(
                                f"sampled frames exceed decoded-pixel budget: {decoded_pixels} "
                                f"(limit {max_decoded_pixels})"
                            )
                        try:
                            frame_rgb = img.convert("RGB")
                        except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
                            raise ValueError(f"failed to decode selected animation frame {idx}") from exc
                        frames.append(Frame(idx=idx, pil=frame_rgb, source_format=source_format))
                else:
                    pixels_per_frame = validate_image_dimensions(img)
                    max_decoded_pixels = max(1, env_int("MODIMG_MAX_DECODED_PIXELS", 256_000_000))
                    if pixels_per_frame > max_decoded_pixels:
                        raise ValueError(
                            f"decoded image exceeds decoded-pixel budget: {pixels_per_frame} "
                            f"(limit {max_decoded_pixels})"
                        )
                    try:
                        frame_rgb = img.convert("RGB")
                    except (OSError, RuntimeError, SyntaxError, ValueError) as exc:
                        if avif_detected:
                            raise ValueError(
                                "failed to decode AVIF image: invalid data or an unavailable AVIF decoder; "
                                "Pillow>=11.3.0 with AVIF support is required"
                            ) from exc
                        raise
                    frames.append(Frame(idx=0, pil=frame_rgb, source_format=source_format))
    except BaseException as exc:
        for frame in frames:
            frame.pil.close()
        if avif_detected and isinstance(exc, (OSError, RuntimeError, SyntaxError)):
            raise ValueError(
                "failed to decode AVIF image: invalid data or an unavailable AVIF decoder; "
                "Pillow>=11.3.0 with AVIF support is required"
            ) from exc
        raise
    if not frames:
        raise ValueError("no readable image frames")
    return frames
