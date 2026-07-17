from __future__ import annotations

import base64
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from modimg import svg as svg_module
from modimg.preprocessing import prepare_image
from modimg.svg import (
    SvgConfigurationError,
    SvgError,
    SvgNotSvgError,
    is_svg_bytes,
    rasterize_svg_bytes,
    read_svg_file,
    validate_svg_bytes,
)
from modimg.utils import is_image_file


SVG_NAMESPACE = "http://www.w3.org/2000/svg"


def _svg(*, attributes: str = 'width="64" height="32"', body: str = "") -> bytes:
    return f'<svg xmlns="{SVG_NAMESPACE}" {attributes}>{body}</svg>'.encode()


def _png_bytes(*, size: tuple[int, int] = (2, 2), mode: str = "RGBA") -> bytes:
    output = io.BytesIO()
    Image.new(mode, size, color=(10, 20, 30, 128) if mode == "RGBA" else (10, 20, 30)).save(output, format="PNG")
    return output.getvalue()


def _data_uri(data: bytes, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"


def _patch_renderer(monkeypatch: pytest.MonkeyPatch, renderer: object) -> None:
    real_import_module = svg_module.importlib.import_module

    def import_module(name: str):
        if name == "resvg_py":
            if isinstance(renderer, BaseException):
                raise renderer
            return renderer
        return real_import_module(name)

    monkeypatch.setattr(svg_module.importlib, "import_module", import_module)


def test_real_svg_render_is_verified_rgb_png_with_expected_dimensions_and_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODIMG_SVG_BACKGROUND", "#fefdfc")
    data = _svg(body='<rect width="32" height="32" fill="red"/>')

    rasterized = rasterize_svg_bytes(data)
    path = Path(rasterized.path)
    try:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        with Image.open(path) as image:
            image.load()
            assert image.format == "PNG"
            assert image.mode == "RGB"
            assert image.size == (64, 32)
            assert image.getpixel((4, 4)) == (255, 0, 0)
            assert image.getpixel((63, 31)) == (254, 253, 252)
        assert rasterized.metadata == {
            "source_format": "svg",
            "normalized_format": "png",
            "renderer": "resvg_py",
            "render_width": 64,
            "render_height": 32,
            "background": "#fefdfc",
        }
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "data",
    [
        _svg(),
        b"\xef\xbb\xbf<?xml version='1.0' encoding='UTF-8'?>\n"
        + _svg(),
        b" \n<!-- before the root -->\n" + _svg(),
        b"\xff\xfe"
        + (f'<?xml version="1.0" encoding="UTF-16"?><svg xmlns="{SVG_NAMESPACE}" width="64" height="32"/>').encode(
            "utf-16-le"
        ),
        b"\xfe\xff"
        + (f'<?xml version="1.0" encoding="UTF-16BE"?><svg xmlns="{SVG_NAMESPACE}" width="64" height="32"/>').encode(
            "utf-16-be"
        ),
    ],
    ids=["utf8", "utf8-bom-declaration", "leading-comment", "utf16-le", "utf16-be"],
)
def test_supported_svg_encodings_and_prefixes_are_accepted(data: bytes) -> None:
    validated = validate_svg_bytes(data)

    assert is_svg_bytes(data) is True
    assert (validated.render_width, validated.render_height) == (64, 32)


@pytest.mark.parametrize(
    "data",
    [
        b"\xff\xfe\x00\x00<svg/>",
        b"\x00\x00\xfe\xff<svg/>",
        b"\xff\xfe" + '<?xml version="1.0" encoding="UTF-8"?><svg/>'.encode("utf-16-le"),
        b"\xff\xfe<\x00s",
        b"\xff\xfe\x00\xd8",
    ],
    ids=["utf32-le", "utf32-be", "conflicting-declaration", "truncated-utf16", "invalid-utf16"],
)
def test_unsupported_invalid_or_conflicting_encodings_are_rejected(data: bytes) -> None:
    with pytest.raises(SvgError, match="encoding|malformed"):
        validate_svg_bytes(data)


def test_content_detection_accepts_uppercase_extension_and_extensionless_svg(tmp_path: Path) -> None:
    uppercase = tmp_path / "IMAGE.SVG"
    extensionless = tmp_path / "image-data"
    uppercase.write_bytes(_svg())
    extensionless.write_bytes(b"\xef\xbb\xbf<!--comment-->" + _svg())

    assert is_image_file(str(uppercase)) is True
    assert is_image_file(str(extensionless)) is True

    prepared = prepare_image(str(extensionless))
    try:
        assert prepared.source_format == "svg"
        assert prepared.normalized_format == "png"
        assert Path(prepared.processing_path).exists()
    finally:
        Path(prepared.processing_path).unlink(missing_ok=True)


def test_svg_suffix_alone_does_not_make_invalid_content_an_image(tmp_path: Path) -> None:
    path = tmp_path / "not-really.svg"
    path.write_text("<html></html>", encoding="utf-8")

    assert is_image_file(str(path)) is False


@pytest.mark.parametrize("data", [b"<svg>", b"<svg></svG>", b"not xml", b""])
def test_malformed_xml_is_rejected(data: bytes) -> None:
    with pytest.raises(SvgError, match="malformed XML|empty document"):
        validate_svg_bytes(data)
    assert is_svg_bytes(data) is False


@pytest.mark.parametrize(
    "data",
    [
        b"<html></html>",
        b"<xml></xml>",
        b'<svg xmlns="https://example.test/not-svg"></svg>',
    ],
)
def test_non_svg_root_or_wrong_namespace_is_rejected(data: bytes) -> None:
    with pytest.raises(SvgNotSvgError, match="root element"):
        validate_svg_bytes(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b'<!DOCTYPE svg><svg xmlns="http://www.w3.org/2000/svg"/>', "DOCTYPE"),
        (
            b'<!DOCTYPE svg [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
            b'<svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>',
            "DOCTYPE|ENTITY",
        ),
        (
            b'<!DOCTYPE svg [<!ENTITY a "ha"><!ENTITY b "&a;&a;&a;&a;&a;&a;&a;&a;">]>'
            b'<svg xmlns="http://www.w3.org/2000/svg"><text>&b;&b;&b;&b;&b;&b;&b;&b;</text></svg>',
            "DOCTYPE|ENTITY",
        ),
    ],
    ids=["doctype", "xxe", "entity-expansion"],
)
def test_dtd_xxe_and_entity_expansion_are_rejected(data: bytes, message: str) -> None:
    with pytest.raises(SvgError, match=message):
        validate_svg_bytes(data)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("<script>alert(1)</script>", "script elements"),
        ("<foreignObject><div>unsafe</div></foreignObject>", "foreignObject elements"),
        ('<rect onclick="alert(1)"/>', "event handler"),
        ('<rect onload="alert(1)"/>', "event handler"),
    ],
)
def test_active_content_is_rejected(body: str, message: str) -> None:
    with pytest.raises(SvgError, match=message):
        validate_svg_bytes(_svg(body=body))


@pytest.mark.parametrize(
    "body",
    [
        '<image href="https://example.org/image.png"/>',
        '<image href="file:///etc/passwd"/>',
        '<image href="../secret.png"/>',
        '<image href="/var/tmp/secret.png"/>',
        '<image href="C:\\temp\\secret.png"/>',
        '<image href="//example.org/image.png"/>',
        '<use href="https://example.org/file.svg#id"/>',
        '<g xml:base="https://example.org/external.svg"><use href="#symbol"/></g>',
        '<rect style="fill: url(https://example.org/value)"/>',
        '<style>rect { fill: url("file:///etc/passwd"); }</style>',
        '<style>@import "https://example.org/external.css"; rect { fill: red; }</style>',
        '<style>@font-face { font-family: evil; src: url(#embedded-font); }</style>',
    ],
    ids=[
        "https-image",
        "file-image",
        "relative-image",
        "absolute-image",
        "windows-image",
        "protocol-relative-image",
        "external-use",
        "external-xml-base",
        "style-attribute-url",
        "style-element-url",
        "css-import",
        "font-face",
    ],
)
def test_external_file_network_css_and_font_references_are_rejected(body: str) -> None:
    with pytest.raises(SvgError, match="external|CSS escapes"):
        validate_svg_bytes(_svg(body=body))


def test_external_xml_stylesheet_processing_instruction_is_rejected() -> None:
    data = b'<?xml-stylesheet href="https://example.org/external.css"?>' + _svg()

    with pytest.raises(SvgError, match="external stylesheets"):
        validate_svg_bytes(data)


def test_internal_gradients_masks_and_use_references_remain_supported() -> None:
    body = """
      <defs>
        <linearGradient id="gradient"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient>
        <clipPath id="clip"><rect width="32" height="32"/></clipPath>
        <symbol id="symbol"><circle cx="8" cy="8" r="8" fill="green"/></symbol>
      </defs>
      <rect width="64" height="32" fill="url(#gradient)" clip-path="url(#clip)"/>
      <use href="#symbol" x="32"/>
    """

    validated = validate_svg_bytes(_svg(body=body))
    rasterized = rasterize_svg_bytes(_svg(body=body))
    try:
        assert (validated.render_width, validated.render_height) == (64, 32)
        with Image.open(rasterized.path) as image:
            image.load()
            assert image.mode == "RGB"
            assert image.size == (64, 32)
    finally:
        Path(rasterized.path).unlink(missing_ok=True)


def test_small_valid_embedded_png_data_uri_is_accepted_and_accounted() -> None:
    embedded = _png_bytes()
    body = f'<image href="{_data_uri(embedded)}" width="2" height="2"/>'

    validated = validate_svg_bytes(_svg(body=body))
    rasterized = rasterize_svg_bytes(_svg(body=body))
    try:
        assert validated.embedded_image_bytes == len(embedded)
        assert rasterized.metadata["embedded_image_bytes"] == len(embedded)
        with Image.open(rasterized.path) as image:
            assert image.mode == "RGB"
            assert image.size == (64, 32)
    finally:
        Path(rasterized.path).unlink(missing_ok=True)


def test_embedded_data_images_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODIMG_SVG_ALLOW_DATA_IMAGES", "0")
    body = f'<image href="{_data_uri(_png_bytes())}"/>'

    with pytest.raises(SvgError, match="disabled"):
        validate_svg_bytes(_svg(body=body))


def test_embedded_data_image_per_item_byte_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    embedded = _png_bytes()
    monkeypatch.setenv("MODIMG_SVG_MAX_EMBEDDED_IMAGE_BYTES", str(len(embedded) - 1))
    body = f'<image href="{_data_uri(embedded)}"/>'

    with pytest.raises(SvgError, match="embedded image exceeds byte limit"):
        validate_svg_bytes(_svg(body=body))


def test_embedded_data_image_total_byte_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    embedded = _png_bytes()
    monkeypatch.setenv("MODIMG_SVG_MAX_EMBEDDED_IMAGE_BYTES", str(len(embedded)))
    monkeypatch.setenv("MODIMG_SVG_MAX_TOTAL_EMBEDDED_BYTES", str(len(embedded) * 2 - 1))
    uri = _data_uri(embedded)
    body = f'<image href="{uri}"/><image href="{uri}" x="4"/>'

    with pytest.raises(SvgError, match="total embedded image bytes"):
        validate_svg_bytes(_svg(body=body))


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("data:image/png;base64,A", "invalid base64"),
        ("data:text/html;base64,PGgxPm5vPC9oMT4=", "unsupported or malformed"),
        ("data:image/svg+xml;base64,PHN2Zy8+", "unsupported or malformed"),
        (_data_uri(_png_bytes(), "image/jpeg"), "MIME type does not match"),
    ],
    ids=["invalid-base64", "html", "nested-svg", "mime-mismatch"],
)
def test_unsafe_or_invalid_data_uris_are_rejected(reference: str, message: str) -> None:
    with pytest.raises(SvgError, match=message):
        validate_svg_bytes(_svg(body=f'<image href="{reference}"/>'))


def test_data_uri_is_rejected_on_non_image_elements() -> None:
    with pytest.raises(SvgError, match="not allowed on this element"):
        validate_svg_bytes(_svg(body=f'<use href="{_data_uri(_png_bytes())}"/>'))


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ('width="0" height="10"', "width dimension must be positive"),
        ('width="-1" height="10"', "width dimension must be positive"),
        ('width="NaN" height="10"', "invalid width"),
        ('width="10foo" height="10"', "unsupported width dimension unit"),
        ('width="1000000001" height="1"', "width dimension is too large"),
        ('viewBox="0 0 0 10"', "viewBox dimensions must be positive"),
        ('viewBox="0 0 -1 10"', "viewBox dimensions must be positive"),
        ('viewBox="0 0 inf 10"', "viewBox values must be finite"),
        ('viewBox="0 0 1 2 3"', "invalid viewBox"),
        ('viewBox="0 0 1000000001 1"', "viewBox dimensions are too large"),
    ],
)
def test_invalid_dimensions_are_rejected(attributes: str, message: str) -> None:
    with pytest.raises(SvgError, match=message):
        validate_svg_bytes(_svg(attributes=attributes))


@pytest.mark.parametrize(
    "attributes",
    [
        'width="1e-300" height="1e-300"',
        'viewBox="0 0 1e-300 1e-300"',
    ],
)
def test_tiny_positive_dimensions_do_not_underflow_the_pixel_scaling(attributes: str) -> None:
    validated = validate_svg_bytes(_svg(attributes=attributes))

    assert (validated.render_width, validated.render_height) == (1, 1)


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ('width="1in" height="12pt"', (96, 16)),
        ('viewBox="0 0 320 180"', (320, 180)),
        ('width="160" viewBox="0 0 320 180"', (160, 90)),
        ('height="90" viewBox="0 0 320 180"', (160, 90)),
        ('width="100%" height="100%" viewBox="0 0 80 40"', (80, 40)),
    ],
)
def test_physical_units_viewbox_and_aspect_ratio_determine_render_size(
    attributes: str,
    expected: tuple[int, int],
) -> None:
    validated = validate_svg_bytes(_svg(attributes=attributes))

    assert (validated.render_width, validated.render_height) == expected


def test_missing_or_percentage_only_dimensions_use_configured_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODIMG_SVG_DEFAULT_WIDTH", "23")
    monkeypatch.setenv("MODIMG_SVG_DEFAULT_HEIGHT", "17")

    missing = validate_svg_bytes(_svg(attributes=""))
    percentages = validate_svg_bytes(_svg(attributes='width="100%" height="50%"'))

    assert (missing.render_width, missing.render_height) == (23, 17)
    assert (percentages.render_width, percentages.render_height) == (23, 17)


def test_large_allowed_svg_is_downscaled_by_dimension_while_preserving_aspect_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODIMG_SVG_MAX_RENDER_DIMENSION", "1000")
    monkeypatch.setenv("MODIMG_SVG_MAX_RENDER_PIXELS", "1000000")

    validated = validate_svg_bytes(_svg(attributes='width="10000" height="5000"'))

    assert (validated.render_width, validated.render_height) == (1000, 500)


def test_svg_is_downscaled_by_pixel_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODIMG_SVG_MAX_RENDER_DIMENSION", "5000")
    monkeypatch.setenv("MODIMG_SVG_MAX_RENDER_PIXELS", "1000000")

    validated = validate_svg_bytes(_svg(attributes='width="4000" height="4000"'))

    assert (validated.render_width, validated.render_height) == (1000, 1000)


def test_svg_limits_can_never_exceed_global_image_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODIMG_SVG_MAX_RENDER_DIMENSION", "1000")
    monkeypatch.setenv("MODIMG_SVG_MAX_RENDER_PIXELS", "1000000")
    monkeypatch.setenv("MODIMG_MAX_IMAGE_DIMENSION", "20")
    monkeypatch.setenv("MODIMG_MAX_IMAGE_PIXELS", "200")
    monkeypatch.setenv("MODIMG_MAX_DECODED_PIXELS", "200")

    validated = validate_svg_bytes(_svg(attributes='width="100" height="50"'))

    assert validated.render_width <= 20
    assert validated.render_height <= 20
    assert validated.render_width * validated.render_height <= 200
    assert (validated.render_width, validated.render_height) == (20, 10)


def test_svg_source_byte_limit_is_enforced_for_bytes_and_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = _svg()
    path = tmp_path / "oversized.svg"
    path.write_bytes(data)
    monkeypatch.setenv("MODIMG_MAX_SVG_BYTES", str(len(data) - 1))

    with pytest.raises(SvgError, match="source exceeds byte limit"):
        validate_svg_bytes(data)
    with pytest.raises(SvgError, match="source exceeds byte limit"):
        read_svg_file(path)


@pytest.mark.parametrize("background", ["not-a-color", "#ffffff80", "rgba(1, 2, 3, 0.5)"])
def test_invalid_or_translucent_background_is_a_clear_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
    background: str,
) -> None:
    monkeypatch.setenv("MODIMG_SVG_BACKGROUND", background)

    with pytest.raises(SvgConfigurationError, match="MODIMG_SVG_BACKGROUND"):
        validate_svg_bytes(_svg())


def test_renderer_value_error_is_wrapped_without_creating_a_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_render(**kwargs: object) -> bytes:
        raise ValueError("renderer internals must not leak")

    _patch_renderer(monkeypatch, SimpleNamespace(svg_to_bytes=fail_render))
    created: list[str] = []
    real_named_temporary_file = svg_module.tempfile.NamedTemporaryFile

    def named_temporary_file(*args: object, **kwargs: object):
        temporary = real_named_temporary_file(*args, **kwargs)
        created.append(temporary.name)
        return temporary

    monkeypatch.setattr(svg_module.tempfile, "NamedTemporaryFile", named_temporary_file)

    with pytest.raises(SvgError, match="SVG rendering failed"):
        rasterize_svg_bytes(_svg())
    assert created == []


def test_renderer_receives_validated_text_bounded_dimensions_and_no_resource_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def render(**kwargs: object) -> bytes:
        calls.append(kwargs)
        return _png_bytes(size=(64, 32))

    _patch_renderer(monkeypatch, SimpleNamespace(svg_to_bytes=render))

    rasterized = rasterize_svg_bytes(
        b'<?xml version="1.0" encoding="UTF-8"?>' + _svg(body='<rect width="64" height="32"/>')
    )
    try:
        assert len(calls) == 1
        call = calls[0]
        assert isinstance(call["svg_string"], str)
        assert not str(call["svg_string"]).lstrip().startswith("<?xml")
        assert call["width"] == 64
        assert call["height"] == 32
        assert call["background"] == "#ffffff"
        assert call["dpi"] == 96.0
        assert call["resources_dir"] is None
        assert call["log_information"] is False
        assert "svg_path" not in call
    finally:
        Path(rasterized.path).unlink(missing_ok=True)


def test_missing_renderer_has_actionable_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_renderer(monkeypatch, ModuleNotFoundError("No module named 'resvg_py'"))

    with pytest.raises(SvgError, match=r"renderer unavailable: install resvg_py>=0\.3,<1\.0"):
        rasterize_svg_bytes(_svg())


def test_missing_renderer_entrypoint_has_actionable_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_renderer(monkeypatch, SimpleNamespace())

    with pytest.raises(SvgError, match="svg_to_bytes is missing"):
        rasterize_svg_bytes(_svg())


@pytest.mark.parametrize("renderer_output", [b"not a png", b"\x89PNG\r\n\x1a\ntruncated"])
def test_invalid_renderer_png_is_rejected_before_temp_file_creation(
    monkeypatch: pytest.MonkeyPatch,
    renderer_output: bytes,
) -> None:
    _patch_renderer(monkeypatch, SimpleNamespace(svg_to_bytes=lambda **kwargs: renderer_output))
    created: list[str] = []
    real_named_temporary_file = svg_module.tempfile.NamedTemporaryFile

    def named_temporary_file(*args: object, **kwargs: object):
        temporary = real_named_temporary_file(*args, **kwargs)
        created.append(temporary.name)
        return temporary

    monkeypatch.setattr(svg_module.tempfile, "NamedTemporaryFile", named_temporary_file)

    with pytest.raises(SvgError, match="invalid PNG"):
        rasterize_svg_bytes(_svg())
    assert created == []


def test_renderer_png_must_match_the_validated_output_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_renderer(monkeypatch, SimpleNamespace(svg_to_bytes=lambda **kwargs: _png_bytes(size=(1, 1))))

    with pytest.raises(SvgError, match="unexpected output dimensions"):
        rasterize_svg_bytes(_svg())


def test_temp_png_is_removed_if_post_write_verification_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_renderer(monkeypatch, SimpleNamespace(svg_to_bytes=lambda **kwargs: _png_bytes(size=(64, 32))))
    created: list[str] = []
    real_named_temporary_file = svg_module.tempfile.NamedTemporaryFile

    def named_temporary_file(*args: object, **kwargs: object):
        temporary = real_named_temporary_file(*args, **kwargs)
        created.append(temporary.name)
        return temporary

    def fail_verification(path: str, expected_size: tuple[int, int]) -> None:
        assert Path(path).exists()
        raise SvgError("forced post-write verification failure")

    monkeypatch.setattr(svg_module.tempfile, "NamedTemporaryFile", named_temporary_file)
    monkeypatch.setattr(svg_module, "_verify_temporary_png", fail_verification)

    with pytest.raises(SvgError, match="forced post-write"):
        rasterize_svg_bytes(_svg())
    assert len(created) == 1
    assert Path(created[0]).exists() is False
