"""Secure validation and bounded SVG-to-PNG rasterization."""
from __future__ import annotations

import base64
import binascii
import importlib
import io
import math
import os
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import ParseError

from PIL import Image, ImageColor, UnidentifiedImageError

from .frames import validate_image_dimensions
from .logging_utils import get_logger
from .utils import env_bool, env_int


LOGGER = get_logger("svg")

_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_UTF8_BOM = b"\xef\xbb\xbf"
_UTF16_LE_BOM = b"\xff\xfe"
_UTF16_BE_BOM = b"\xfe\xff"
_UTF32_LE_BOM = b"\xff\xfe\x00\x00"
_UTF32_BE_BOM = b"\x00\x00\xfe\xff"
_MAX_INTRINSIC_DIMENSION = 1_000_000_000.0

_XML_DECLARATION_RE = re.compile(r"\A\s*<\?xml\s+.*?\?>", re.IGNORECASE | re.DOTALL)
_XML_ENCODING_RE = re.compile(r"\bencoding\s*=\s*(['\"])([^'\"]+)\1", re.IGNORECASE)
_DOCTYPE_RE = re.compile(r"<!\s*DOCTYPE\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"<!\s*ENTITY\b", re.IGNORECASE)
_XML_STYLESHEET_RE = re.compile(r"<\?\s*xml-stylesheet\b", re.IGNORECASE)
_LENGTH_RE = re.compile(
    r"\A([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Z%]*)\Z"
)
_INTERNAL_REFERENCE_RE = re.compile(r"\A#[A-Za-z_][A-Za-z0-9_.:-]*\Z")
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_URL_START_RE = re.compile(r"url\s*\(", re.IGNORECASE)
_CSS_URL_RE = re.compile(
    r"url\s*\(\s*(?:(['\"])(.*?)\1|([^)]*?))\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_DATA_IMAGE_RE = re.compile(
    r"\Adata:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/]*={0,2})\Z",
    re.IGNORECASE,
)

_FORBIDDEN_ELEMENTS = {"script", "handler", "listener", "foreignobject", "iframe", "object", "embed"}
_REFERENCE_ATTRIBUTES = {"href", "src", "data", "poster"}
_DATA_IMAGE_ELEMENTS = {"image", "feimage"}
_RELATIVE_LENGTH_UNITS = {"%", "em", "ex", "ch", "rem", "vw", "vh", "vmin", "vmax"}
_LENGTH_FACTORS = {
    "": 1.0,
    "px": 1.0,
    "in": 96.0,
    "cm": 96.0 / 2.54,
    "mm": 96.0 / 25.4,
    "q": 96.0 / 101.6,
    "pt": 96.0 / 72.0,
    "pc": 16.0,
}
_EMBEDDED_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}


class SvgError(ValueError):
    """Base class for controlled SVG preprocessing failures."""


class SvgNotSvgError(SvgError):
    """Raised when parsed XML does not have an SVG root element."""


class SvgConfigurationError(SvgError):
    """Raised when SVG runtime configuration is unsafe or invalid."""


@dataclass(frozen=True)
class ValidatedSvg:
    """Validated SVG text and its bounded target render configuration."""

    svg_text: str
    render_width: int
    render_height: int
    background: str
    background_rgb: tuple[int, int, int]
    embedded_image_bytes: int


@dataclass(frozen=True)
class RasterizedSvg:
    """A temporary RGB PNG produced from a validated SVG document."""

    path: str
    metadata: dict[str, Any]


@dataclass
class _EmbeddedBudget:
    total_bytes: int = 0
    total_pixels: int = 0


def max_svg_bytes() -> int:
    """Return the configured upper bound for an uncompressed SVG source."""
    return max(1, env_int("MODIMG_MAX_SVG_BYTES", 10_000_000))


def read_svg_file(path: str | Path) -> bytes:
    """Read an SVG candidate without ever exceeding the configured byte limit."""
    limit = max_svg_bytes()
    with Path(path).open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise SvgError(f"SVG source exceeds byte limit ({limit} bytes)")
    return data


def _decode_svg_bytes(data: bytes) -> str:
    if not data:
        raise SvgError("SVG parsing failed: empty document")
    if data.startswith((_UTF32_LE_BOM, _UTF32_BE_BOM)):
        raise SvgError("SVG parsing failed: unsupported text encoding")

    encoding_kind = "utf-8"
    payload = data
    codec = "utf-8"
    if data.startswith(_UTF8_BOM):
        payload = data[len(_UTF8_BOM) :]
        encoding_kind = "utf-8-bom"
    elif data.startswith(_UTF16_LE_BOM):
        payload = data[len(_UTF16_LE_BOM) :]
        codec = "utf-16-le"
        encoding_kind = "utf-16-le"
    elif data.startswith(_UTF16_BE_BOM):
        payload = data[len(_UTF16_BE_BOM) :]
        codec = "utf-16-be"
        encoding_kind = "utf-16-be"

    try:
        text = payload.decode(codec, errors="strict")
    except UnicodeDecodeError as exc:
        raise SvgError("SVG parsing failed: unsupported or invalid text encoding") from exc
    if "\x00" in text:
        raise SvgError("SVG parsing failed: unsupported or invalid text encoding")

    declaration = _XML_DECLARATION_RE.match(text)
    if declaration is not None:
        match = _XML_ENCODING_RE.search(declaration.group(0))
        if match is not None:
            declared = match.group(2).strip().lower().replace("_", "-")
            compatible = {
                "utf-8": {"utf-8", "utf8", "us-ascii", "ascii"},
                "utf-8-bom": {"utf-8", "utf8"},
                "utf-16-le": {"utf-16", "utf16", "utf-16le", "utf16le"},
                "utf-16-be": {"utf-16", "utf16", "utf-16be", "utf16be"},
            }[encoding_kind]
            if declared not in compatible:
                raise SvgError("SVG parsing failed: encoding declaration conflicts with document bytes")
    return text


def _local_name(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    return name.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _parse_length(value: str | None, *, name: str) -> float | None:
    if value is None:
        return None
    raw = value.strip()
    if raw.lower() == "auto":
        return None
    match = _LENGTH_RE.fullmatch(raw)
    if match is None:
        raise SvgError(f"SVG rejected: invalid {name} dimension")
    try:
        number = float(match.group(1))
    except ValueError as exc:
        raise SvgError(f"SVG rejected: invalid {name} dimension") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise SvgError(f"SVG rejected: {name} dimension must be positive and finite")
    unit = match.group(2).lower()
    if unit in _RELATIVE_LENGTH_UNITS:
        return None
    factor = _LENGTH_FACTORS.get(unit)
    if factor is None:
        raise SvgError(f"SVG rejected: unsupported {name} dimension unit")
    pixels = number * factor
    if not math.isfinite(pixels) or pixels > _MAX_INTRINSIC_DIMENSION:
        raise SvgError(f"SVG rejected: {name} dimension is too large")
    return pixels


def _parse_view_box(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    if len(parts) != 4:
        raise SvgError("SVG rejected: invalid viewBox")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise SvgError("SVG rejected: invalid viewBox") from exc
    if not all(math.isfinite(number) for number in numbers):
        raise SvgError("SVG rejected: viewBox values must be finite")
    width, height = numbers[2], numbers[3]
    if width <= 0.0 or height <= 0.0:
        raise SvgError("SVG rejected: viewBox dimensions must be positive")
    if width > _MAX_INTRINSIC_DIMENSION or height > _MAX_INTRINSIC_DIMENSION:
        raise SvgError("SVG rejected: viewBox dimensions are too large")
    return width, height


def _effective_render_limits() -> tuple[int, int]:
    dimension = min(
        max(1, env_int("MODIMG_SVG_MAX_RENDER_DIMENSION", 4_096)),
        max(1, env_int("MODIMG_MAX_IMAGE_DIMENSION", 32_768)),
    )
    pixels = min(
        max(1, env_int("MODIMG_SVG_MAX_RENDER_PIXELS", 16_000_000)),
        max(1, env_int("MODIMG_MAX_IMAGE_PIXELS", 64_000_000)),
        max(1, env_int("MODIMG_MAX_DECODED_PIXELS", 256_000_000)),
    )
    return dimension, pixels


def _render_dimensions(root: Any) -> tuple[int, int]:
    width = _parse_length(root.attrib.get("width"), name="width")
    height = _parse_length(root.attrib.get("height"), name="height")
    view_box = _parse_view_box(root.attrib.get("viewBox"))

    if width is not None and height is not None:
        intrinsic_width, intrinsic_height = width, height
    elif view_box is not None and width is not None:
        intrinsic_width = width
        intrinsic_height = width * view_box[1] / view_box[0]
    elif view_box is not None and height is not None:
        intrinsic_height = height
        intrinsic_width = height * view_box[0] / view_box[1]
    elif view_box is not None:
        intrinsic_width, intrinsic_height = view_box
    else:
        intrinsic_width = float(max(1, env_int("MODIMG_SVG_DEFAULT_WIDTH", 1_024)))
        intrinsic_height = float(max(1, env_int("MODIMG_SVG_DEFAULT_HEIGHT", 1_024)))

    if not math.isfinite(intrinsic_width) or not math.isfinite(intrinsic_height):
        raise SvgError("SVG rejected: calculated render dimensions are not finite")
    if intrinsic_width <= 0.0 or intrinsic_height <= 0.0:
        raise SvgError("SVG rejected: calculated render dimensions must be positive")
    if intrinsic_width > _MAX_INTRINSIC_DIMENSION or intrinsic_height > _MAX_INTRINSIC_DIMENSION:
        raise SvgError("SVG rejected: calculated render dimensions are too large")

    max_dimension, max_pixels = _effective_render_limits()
    intrinsic_pixels = intrinsic_width * intrinsic_height
    pixel_scale = 1.0 if intrinsic_pixels <= max_pixels else math.sqrt(max_pixels / intrinsic_pixels)
    scale = min(
        1.0,
        max_dimension / intrinsic_width,
        max_dimension / intrinsic_height,
        pixel_scale,
    )
    render_width = max(1, int(math.floor(intrinsic_width * scale)))
    render_height = max(1, int(math.floor(intrinsic_height * scale)))
    if render_width > max_dimension or render_height > max_dimension or render_width * render_height > max_pixels:
        raise SvgError("SVG rejected: calculated render size exceeds configured limits")
    return render_width, render_height


def _background() -> tuple[str, tuple[int, int, int]]:
    raw = (str(os.getenv("MODIMG_SVG_BACKGROUND", "#ffffff")) or "#ffffff").strip()
    if len(raw) > 128:
        raise SvgConfigurationError("Invalid MODIMG_SVG_BACKGROUND: color value is too long")
    try:
        red, green, blue, alpha = ImageColor.getcolor(raw, "RGBA")
    except (TypeError, ValueError) as exc:
        raise SvgConfigurationError("Invalid MODIMG_SVG_BACKGROUND: expected a CSS color") from exc
    if alpha != 255:
        raise SvgConfigurationError("Invalid MODIMG_SVG_BACKGROUND: background must be opaque")
    return f"#{red:02x}{green:02x}{blue:02x}", (red, green, blue)


def _validate_css_references(css: str) -> None:
    cleaned = _CSS_COMMENT_RE.sub("", css)
    if "\\" in cleaned:
        raise SvgError("SVG rejected: CSS escapes are not allowed")
    if re.search(r"@\s*import\b", cleaned, re.IGNORECASE):
        raise SvgError("SVG rejected: external stylesheets are not allowed")
    if re.search(r"@\s*font-face\b", cleaned, re.IGNORECASE):
        raise SvgError("SVG rejected: external fonts are not allowed")

    cursor = 0
    while True:
        start = _CSS_URL_START_RE.search(cleaned, cursor)
        if start is None:
            return
        match = _CSS_URL_RE.match(cleaned, start.start())
        if match is None:
            raise SvgError("SVG rejected: malformed CSS url() reference")
        reference = (match.group(2) if match.group(1) else match.group(3) or "").strip()
        if _INTERNAL_REFERENCE_RE.fullmatch(reference) is None:
            raise SvgError("SVG rejected: external resource reference is not allowed")
        cursor = match.end()


def _validate_embedded_image(reference: str, element_name: str, budget: _EmbeddedBudget) -> None:
    if element_name not in _DATA_IMAGE_ELEMENTS:
        raise SvgError("SVG rejected: data URI is not allowed on this element")
    if not env_bool("MODIMG_SVG_ALLOW_DATA_IMAGES", True):
        raise SvgError("SVG rejected: embedded data images are disabled")
    match = _DATA_IMAGE_RE.fullmatch(reference)
    if match is None:
        raise SvgError("SVG rejected: unsupported or malformed data image")
    mime_type = match.group(1).lower()
    payload = match.group(2)
    per_image_limit = max(1, env_int("MODIMG_SVG_MAX_EMBEDDED_IMAGE_BYTES", 5_000_000))
    total_limit = max(1, env_int("MODIMG_SVG_MAX_TOTAL_EMBEDDED_BYTES", 10_000_000))
    max_encoded_length = ((per_image_limit + 2) // 3) * 4 + 4
    if len(payload) > max_encoded_length:
        raise SvgError("SVG rejected: embedded image exceeds byte limit")
    try:
        decoded = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SvgError("SVG rejected: embedded image contains invalid base64") from exc
    if not decoded:
        raise SvgError("SVG rejected: embedded image is empty")
    if len(decoded) > per_image_limit:
        raise SvgError("SVG rejected: embedded image exceeds byte limit")
    if budget.total_bytes + len(decoded) > total_limit:
        raise SvgError("SVG rejected: total embedded image bytes exceed limit")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(decoded)) as embedded:
                expected_format = _EMBEDDED_FORMATS[mime_type]
                if str(embedded.format or "").upper() != expected_format:
                    raise SvgError("SVG rejected: embedded image MIME type does not match its content")
                pixels = validate_image_dimensions(embedded)
                frame_count = int(getattr(embedded, "n_frames", 1) or 1)
                max_frames = max(1, env_int("MODIMG_MAX_ANIMATION_FRAMES", 5_000))
                if frame_count > max_frames:
                    raise SvgError("SVG rejected: embedded animation frame count exceeds limit")
                embedded.verify()
    except SvgError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise SvgError("SVG rejected: embedded image is not a valid supported raster image") from exc

    decoded_pixel_limit = max(1, env_int("MODIMG_MAX_DECODED_PIXELS", 256_000_000))
    decoded_pixels = pixels * frame_count
    if decoded_pixels > decoded_pixel_limit or budget.total_pixels + decoded_pixels > decoded_pixel_limit:
        raise SvgError("SVG rejected: embedded images exceed decoded-pixel budget")
    budget.total_bytes += len(decoded)
    budget.total_pixels += decoded_pixels


def _validate_reference(reference: str, element_name: str, budget: _EmbeddedBudget) -> None:
    value = reference.strip()
    if not value:
        return
    if _INTERNAL_REFERENCE_RE.fullmatch(value) is not None:
        return
    if value.lower().startswith("data:"):
        _validate_embedded_image(value, element_name, budget)
        return
    raise SvgError("SVG rejected: external resource reference is not allowed")


def _validate_tree(root: Any) -> _EmbeddedBudget:
    budget = _EmbeddedBudget()
    for element in root.iter():
        element_name = _local_name(element.tag).lower()
        if not element_name:
            continue
        if element_name in _FORBIDDEN_ELEMENTS:
            if element_name == "script":
                raise SvgError("SVG rejected: script elements are not allowed")
            if element_name == "foreignobject":
                raise SvgError("SVG rejected: foreignObject elements are not allowed")
            raise SvgError(f"SVG rejected: {element_name} elements are not allowed")

        for attribute, raw_value in element.attrib.items():
            attribute_name = _local_name(attribute).lower()
            value = str(raw_value)
            if attribute_name == "base":
                raise SvgError("SVG rejected: external resource base paths are not allowed")
            if attribute_name.startswith("on"):
                raise SvgError("SVG rejected: event handler attributes are not allowed")
            if attribute_name in _REFERENCE_ATTRIBUTES:
                _validate_reference(value, element_name, budget)
            _validate_css_references(value)

        if element_name == "style":
            _validate_css_references("".join(element.itertext()))
    return budget


def validate_svg_bytes(data: bytes) -> ValidatedSvg:
    """Parse and validate an untrusted, complete SVG document without rendering it."""
    if len(data) > max_svg_bytes():
        raise SvgError(f"SVG source exceeds byte limit ({max_svg_bytes()} bytes)")
    text = _decode_svg_bytes(data)
    if _DOCTYPE_RE.search(text):
        raise SvgError("SVG rejected: DOCTYPE declarations are not allowed")
    if _ENTITY_RE.search(text):
        raise SvgError("SVG rejected: ENTITY declarations are not allowed")
    if _XML_STYLESHEET_RE.search(text):
        raise SvgError("SVG rejected: external stylesheets are not allowed")

    try:
        safe_element_tree = importlib.import_module("defusedxml.ElementTree")
        common = importlib.import_module("defusedxml.common")
    except (ImportError, ModuleNotFoundError) as exc:
        raise SvgError("SVG parser unavailable: install defusedxml>=0.7.1,<1.0") from exc
    try:
        root = safe_element_tree.fromstring(
            data,
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
    except common.DTDForbidden as exc:
        raise SvgError("SVG rejected: DOCTYPE declarations are not allowed") from exc
    except common.EntitiesForbidden as exc:
        raise SvgError("SVG rejected: entity declarations are not allowed") from exc
    except common.ExternalReferenceForbidden as exc:
        raise SvgError("SVG rejected: external XML entities are not allowed") from exc
    except ParseError as exc:
        raise SvgError("SVG parsing failed: malformed XML") from exc

    if _local_name(root.tag) != "svg":
        raise SvgNotSvgError("SVG rejected: root element is not svg")
    namespace = root.tag[1:].split("}", 1)[0] if isinstance(root.tag, str) and root.tag.startswith("{") else ""
    if namespace and namespace != _SVG_NAMESPACE:
        raise SvgNotSvgError("SVG rejected: root element is not in the SVG namespace")

    embedded_budget = _validate_tree(root)
    render_width, render_height = _render_dimensions(root)
    background, background_rgb = _background()
    renderer_text = _XML_DECLARATION_RE.sub("", text, count=1)
    return ValidatedSvg(
        svg_text=renderer_text,
        render_width=render_width,
        render_height=render_height,
        background=background,
        background_rgb=background_rgb,
        embedded_image_bytes=embedded_budget.total_bytes,
    )


def is_svg_bytes(data: bytes) -> bool:
    """Return whether complete bytes form a valid SVG under the security policy."""
    try:
        validate_svg_bytes(data)
    except SvgConfigurationError:
        return True
    except SvgError:
        return False
    return True


def _validate_rendered_dimensions(image: Image.Image) -> int:
    pixels = validate_image_dimensions(image)
    max_dimension, max_pixels = _effective_render_limits()
    width, height = image.size
    if width > max_dimension or height > max_dimension or pixels > max_pixels:
        raise SvgError("SVG renderer produced an image exceeding configured limits")
    return pixels


def _normalize_renderer_png(
    png_bytes: bytes,
    background_rgb: tuple[int, int, int],
    expected_size: tuple[int, int],
) -> tuple[bytes, int, int]:
    if not isinstance(png_bytes, bytes) or not png_bytes.startswith(_PNG_SIGNATURE):
        raise SvgError("SVG renderer produced invalid PNG data")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(png_bytes)) as probe:
                if str(probe.format or "").upper() != "PNG":
                    raise SvgError("SVG renderer produced a non-PNG image")
                if probe.size != expected_size:
                    raise SvgError("SVG renderer produced unexpected output dimensions")
                _validate_rendered_dimensions(probe)
                probe.verify()
            with Image.open(io.BytesIO(png_bytes)) as rendered:
                if rendered.size != expected_size:
                    raise SvgError("SVG renderer produced unexpected output dimensions")
                _validate_rendered_dimensions(rendered)
                rendered.load()
                with rendered.convert("RGBA") as foreground:
                    with Image.new("RGBA", rendered.size, (*background_rgb, 255)) as canvas:
                        canvas.alpha_composite(foreground)
                        with canvas.convert("RGB") as rgb:
                            output = io.BytesIO()
                            rgb.save(output, format="PNG")
                            normalized = output.getvalue()
                width, height = rendered.size
    except SvgError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise SvgError("SVG renderer produced invalid PNG data") from exc
    if not normalized.startswith(_PNG_SIGNATURE):
        raise SvgError("SVG renderer produced invalid PNG data")
    return normalized, int(width), int(height)


def _verify_temporary_png(path: str, expected_size: tuple[int, int]) -> None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as probe:
                if str(probe.format or "").upper() != "PNG":
                    raise SvgError("SVG renderer output is not a PNG file")
                if probe.size != expected_size:
                    raise SvgError("SVG renderer output dimensions changed while writing PNG")
                _validate_rendered_dimensions(probe)
                probe.verify()
            with Image.open(path) as decoded:
                _validate_rendered_dimensions(decoded)
                decoded.load()
                if decoded.mode != "RGB":
                    raise SvgError("SVG renderer output is not an RGB PNG")
    except SvgError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise SvgError("SVG renderer produced an invalid PNG file") from exc


def rasterize_svg_bytes(data: bytes, *, validated: ValidatedSvg | None = None) -> RasterizedSvg:
    """Validate and render SVG bytes to a unique, verified temporary RGB PNG."""
    validated = validated or validate_svg_bytes(data)
    try:
        renderer = importlib.import_module("resvg_py")
    except (ImportError, ModuleNotFoundError) as exc:
        raise SvgError("SVG renderer unavailable: install resvg_py>=0.3,<1.0") from exc
    svg_to_bytes = getattr(renderer, "svg_to_bytes", None)
    if not callable(svg_to_bytes):
        raise SvgError("SVG renderer unavailable: resvg_py.svg_to_bytes is missing")

    try:
        rendered_bytes = svg_to_bytes(
            svg_string=validated.svg_text,
            width=validated.render_width,
            height=validated.render_height,
            background=validated.background,
            dpi=96.0,
            resources_dir=None,
            log_information=False,
        )
    except (ValueError, RuntimeError, OSError, OverflowError) as exc:
        raise SvgError("SVG rendering failed: renderer rejected the document") from exc

    expected_size = (validated.render_width, validated.render_height)
    normalized_png, actual_width, actual_height = _normalize_renderer_png(
        rendered_bytes,
        validated.background_rgb,
        expected_size,
    )
    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temporary:
            temporary_path = temporary.name
            temporary.write(normalized_png)
        _verify_temporary_png(temporary_path, expected_size)
    except BaseException:
        if temporary_path:
            try:
                Path(temporary_path).unlink(missing_ok=True)
            except OSError as cleanup_error:
                LOGGER.warning("failed to remove SVG temporary PNG after an error: %s", type(cleanup_error).__name__)
        raise

    metadata: dict[str, Any] = {
        "source_format": "svg",
        "normalized_format": "png",
        "renderer": "resvg_py",
        "render_width": actual_width,
        "render_height": actual_height,
        "background": validated.background,
    }
    if validated.embedded_image_bytes:
        metadata["embedded_image_bytes"] = validated.embedded_image_bytes
    return RasterizedSvg(path=temporary_path, metadata=metadata)
