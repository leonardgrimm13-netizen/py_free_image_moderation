from __future__ import annotations

import base64
import io
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from PIL import Image

from modimg.frames import load_frames
from modimg.svg import SvgError, _normalize_renderer_png, _verify_temporary_png, validate_svg_bytes


def _png_bytes(size: tuple[int, int]) -> bytes:
    output = io.BytesIO()
    with Image.new("RGB", size) as image:
        image.save(output, format="PNG")
    return output.getvalue()


def _read_image(kind: str, path: Path, png: bytes, size: tuple[int, int]) -> None:
    if kind == "frames":
        frames = load_frames(str(path))
        for frame in frames:
            frame.close()
    elif kind == "embedded":
        encoded = base64.b64encode(png).decode("ascii")
        validate_svg_bytes(
            ('<svg xmlns="http://www.w3.org/2000/svg" width="2" height="2">'
             f'<image href="data:image/png;base64,{encoded}"/></svg>').encode()
        )
    elif kind == "normalize":
        _normalize_renderer_png(png, (255, 255, 255), size)
    else:
        _verify_temporary_png(str(path), size)


@pytest.mark.parametrize("first_kind", ["frames", "embedded", "normalize", "temporary"])
@pytest.mark.parametrize("second_kind", ["frames", "embedded", "normalize", "temporary"])
@pytest.mark.parametrize("bomb", [False, True], ids=["normal", "bomb-warning"])
def test_overlapping_image_reads_preserve_filters_and_bomb_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    first_kind: str,
    second_kind: str,
    bomb: bool,
) -> None:
    sizes = [(2, 2), (3, 2) if bomb else (2, 2)]
    images = [_png_bytes(size) for size in sizes]
    paths = [tmp_path / f"{index}.png" for index in range(2)]
    for path, png in zip(paths, images):
        path.write_bytes(png)
    if bomb:
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)

    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    local = threading.local()
    original_open = Image.open
    initial_filters = warnings.filters
    original_filters = initial_filters[:]
    original_showwarning = warnings.showwarning

    def controlled_open(*args, **kwargs):
        if not local.opened:
            local.opened = True
            entered[local.index].set()
            assert release[local.index].wait(timeout=5)
        return original_open(*args, **kwargs)

    def read(index: int, kind: str) -> None:
        local.index = index
        local.opened = False
        _read_image(kind, paths[index], images[index], sizes[index])

    monkeypatch.setattr(Image, "open", controlled_open)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(read, 0, first_kind)
            second = None
            try:
                assert entered[0].wait(timeout=5)
                second = executor.submit(read, 1, second_kind)
                # Both decoders must enter before either is released.
                assert entered[1].wait(timeout=5)
                release[0].set()
                first.result(timeout=5)
                release[1].set()
                if bomb:
                    expected = Image.DecompressionBombWarning if second_kind == "frames" else SvgError
                    with pytest.raises(expected) as raised:
                        second.result(timeout=5)
                    if second_kind != "frames":
                        assert isinstance(raised.value.__cause__, Image.DecompressionBombWarning)
                else:
                    second.result(timeout=5)
            finally:
                for event in release:
                    event.set()

        assert warnings.filters is initial_filters
        assert warnings.filters == original_filters
        assert warnings.showwarning is original_showwarning
    finally:
        # Leave the test process usable when running against the broken code.
        warnings.filters = initial_filters
        initial_filters[:] = original_filters
        warnings.showwarning = original_showwarning


@pytest.mark.parametrize("kind", ["frames", "embedded", "normalize", "temporary"])
@pytest.mark.parametrize("max_pixels", [None, 6, 4, 2], ids=["disabled", "at-limit", "warning", "error"])
def test_pillow_pixel_limits_remain_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    max_pixels: int | None,
) -> None:
    size = (3, 2)
    png = _png_bytes(size)
    path = tmp_path / "pixels.png"
    path.write_bytes(png)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", max_pixels)
    filters = warnings.filters
    original_filters = filters[:]

    if max_pixels is None or max_pixels >= 6:
        _read_image(kind, path, png, size)
    else:
        bomb_error = Image.DecompressionBombWarning if max_pixels == 4 else Image.DecompressionBombError
        with pytest.raises(bomb_error if kind == "frames" else SvgError) as raised:
            _read_image(kind, path, png, size)
        if kind != "frames":
            assert isinstance(raised.value.__cause__, bomb_error)

    assert warnings.filters is filters
    assert warnings.filters == original_filters
