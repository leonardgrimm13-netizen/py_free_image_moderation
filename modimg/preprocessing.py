"""Input normalization before Pillow decoding and moderation engine execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .svg import SvgConfigurationError, SvgError, max_svg_bytes, rasterize_svg_bytes, read_svg_file, validate_svg_bytes
from .utils import _IMAGE_SNIFF_BYTES, _RASTER_IMAGE_EXTENSIONS, _sniff_image


@dataclass
class PreparedImage:
    """An input path normalized for Pillow and frame-based moderation engines."""

    processing_path: str
    source_format: str
    normalized_format: str
    temporary_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _looks_like_xml_candidate(prefix: bytes, suffix: str) -> bool:
    if suffix == ".svg":
        return True
    if prefix.startswith((b"\xef\xbb\xbf", b"\xff\xfe", b"\xfe\xff")):
        return True
    return prefix.lstrip().startswith(b"<")


def prepare_image(path: str) -> PreparedImage:
    """Return the path that Pillow and every moderation engine must process.

    Raster and animation paths, including AVIF, pass through unchanged. SVG
    candidates are read under a source limit, securely validated, and rendered
    to a temporary RGB PNG. Active legacy path engines may lazily request a
    shared JPEG from the decoded AVIF frame. The caller owns every returned
    ``temporary_paths`` entry.
    """
    source_path = Path(path)
    with source_path.open("rb") as source:
        prefix = source.read(_IMAGE_SNIFF_BYTES)
    sniff_extension, _mime = _sniff_image(prefix)
    if sniff_extension:
        source_format = sniff_extension.lstrip(".")
        return PreparedImage(
            processing_path=str(source_path),
            source_format=source_format,
            normalized_format=source_format,
        )

    suffix = source_path.suffix.lower()
    source_format = suffix.lstrip(".") if suffix in _RASTER_IMAGE_EXTENSIONS else "unknown"
    xml_candidate = _looks_like_xml_candidate(prefix, suffix)
    if suffix == ".avif" and not xml_candidate:
        raise ValueError("file does not contain a valid AVIF image")
    if suffix in _RASTER_IMAGE_EXTENSIONS and not xml_candidate:
        return PreparedImage(str(source_path), source_format, source_format)
    if not xml_candidate and source_path.stat().st_size > max_svg_bytes():
        return PreparedImage(str(source_path), source_format, source_format)

    svg_bytes = read_svg_file(source_path)
    try:
        validated = validate_svg_bytes(svg_bytes)
    except SvgConfigurationError:
        raise
    except SvgError:
        if xml_candidate:
            raise
        return PreparedImage(str(source_path), source_format, source_format)
    rasterized = rasterize_svg_bytes(svg_bytes, validated=validated)
    return PreparedImage(
        processing_path=rasterized.path,
        source_format="svg",
        normalized_format="png",
        temporary_paths=[rasterized.path],
        metadata=rasterized.metadata,
    )
