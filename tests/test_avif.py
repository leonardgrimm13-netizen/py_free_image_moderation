from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from modimg.frames import load_frames
from modimg.preprocessing import prepare_image
from modimg.utils import _IMAGE_SNIFF_BYTES, _sniff_image, is_image_file
from tests.avif_helpers import make_avif_bytes, write_avif


def _ftyp_box(
    major_brand: bytes,
    *compatible_brands: bytes,
    extended_size: bool = False,
) -> bytes:
    assert len(major_brand) == 4
    assert all(len(brand) == 4 for brand in compatible_brands)
    payload = major_brand + b"\x00\x00\x00\x00" + b"".join(compatible_brands)
    if extended_size:
        box_size = 16 + len(payload)
        return b"\x00\x00\x00\x01ftyp" + box_size.to_bytes(8, "big") + payload
    box_size = 8 + len(payload)
    return box_size.to_bytes(4, "big") + b"ftyp" + payload


@pytest.mark.parametrize(
    "header",
    [
        _ftyp_box(b"avif"),
        _ftyp_box(b"avis"),
        _ftyp_box(b"mif1", b"avif"),
        _ftyp_box(b"mif1", b"avis"),
        _ftyp_box(b"avif", b"mif1", extended_size=True),
        _ftyp_box(b"mif1", b"avif", extended_size=True),
    ],
)
def test_sniff_image_accepts_explicit_avif_and_avis_brands(header: bytes) -> None:
    assert _sniff_image(header) == (".avif", "image/avif")


def test_sniff_image_checks_compatible_brand_at_bounded_ftyp_boundary() -> None:
    # 16-byte header plus 60 compatible brands fills the complete 256-byte
    # sniff window. The AVIF brand deliberately appears in the last slot.
    compatible = [b"mif1"] * 59 + [b"avif"]
    header = _ftyp_box(b"heic", *compatible)

    assert len(header) == _IMAGE_SNIFF_BYTES
    assert _sniff_image(header) == (".avif", "image/avif")


@pytest.mark.parametrize(
    "header",
    [
        b"",
        b"\x00\x00\x00\x10ftypavif",
        b"\x00\x00\x00\x00ftypavif\x00\x00\x00\x00",
        b"\x00\x00\x00\x0cftypavif",
        b"\x00\x00\x00\x14ftypavif\x00\x00\x00\x00",
        b"\x00\x00\x00\x11ftypavif\x00\x00\x00\x00x",
        b"\x00\x00\x00\x01ftyp" + (23).to_bytes(8, "big") + b"avif\x00\x00\x00\x00",
        b"\x00\x00\x00\x01ftyp" + (300).to_bytes(8, "big") + b"avif\x00\x00\x00\x00",
        _ftyp_box(b"heic", b"heix", b"mif1"),
        _ftyp_box(b"mif1", b"heic"),
        _ftyp_box(b"mif1") + b"outside-avif-brand",
        b"random bytes containing avif but no ISO BMFF box",
        b"\x00\x00\x00\x10junkavif\x00\x00\x00\x00",
    ],
)
def test_sniff_image_rejects_truncated_corrupt_and_non_avif_ftyp(header: bytes) -> None:
    assert _sniff_image(header) == ("", "")


def test_sniff_image_rejects_ftyp_box_larger_than_bounded_prefix() -> None:
    compatible = [b"mif1"] * 60 + [b"avif"]
    header = _ftyp_box(b"heic", *compatible)

    assert len(header) > _IMAGE_SNIFF_BYTES
    assert _sniff_image(header) == ("", "")


def test_pillow_generated_avif_has_expected_content_signature() -> None:
    encoded = make_avif_bytes()

    assert _sniff_image(encoded[:_IMAGE_SNIFF_BYTES]) == (".avif", "image/avif")


@pytest.mark.parametrize("filename", ["image.avif", "image.AVIF", "image.AvIf", "image", "image.bin"])
def test_is_image_file_detects_real_avif_by_suffix_or_content(tmp_path: Path, filename: str) -> None:
    path = write_avif(tmp_path / filename)

    assert is_image_file(str(path)) is True


def test_is_image_file_keeps_known_suffix_scan_semantics_but_decoder_remains_authoritative(tmp_path: Path) -> None:
    invalid_known_suffix = tmp_path / "broken.avif"
    invalid_unknown_suffix = tmp_path / "broken.bin"
    invalid_known_suffix.write_bytes(b"not an image")
    invalid_unknown_suffix.write_bytes(b"not an image")

    assert is_image_file(str(invalid_known_suffix)) is True
    assert is_image_file(str(invalid_unknown_suffix)) is False
    with pytest.raises((OSError, ValueError)):
        load_frames(str(invalid_known_suffix), sample_frames=1)


def test_prepare_image_rejects_non_avif_pillow_format_with_avif_suffix(tmp_path: Path) -> None:
    misleading = tmp_path / "misleading.avif"
    Image.new("RGB", (3, 2), color=(1, 2, 3)).save(misleading, format="PPM")

    with pytest.raises(ValueError, match="does not contain a valid AVIF image"):
        prepare_image(str(misleading))


def test_windows_style_mixed_case_avif_suffix_is_a_scan_candidate() -> None:
    assert is_image_file(r"C:\Users\me\Pictures\photo.AvIf") is True


@pytest.mark.parametrize("filename", ["native.avif", "native", "native.dat"])
def test_prepare_image_keeps_avif_native_without_temporary_conversion(tmp_path: Path, filename: str) -> None:
    path = write_avif(tmp_path / filename)

    prepared = prepare_image(str(path))

    assert prepared.processing_path == str(path)
    assert prepared.source_format == "avif"
    assert prepared.normalized_format == "avif"
    assert prepared.temporary_paths == []
    assert prepared.metadata == {}
