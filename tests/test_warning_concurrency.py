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
    if kind in {"frames", "animation"}:
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
@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
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
@pytest.mark.parametrize("max_pixels", [None, 6, 4, 3, 2], ids=["disabled", "at-limit", "warning", "twice-limit", "error"])
@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
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
        bomb_error = Image.DecompressionBombWarning if max_pixels in {3, 4} else Image.DecompressionBombError
        with pytest.raises(bomb_error if kind == "frames" else SvgError) as raised:
            _read_image(kind, path, png, size)
        if kind != "frames":
            assert isinstance(raised.value.__cause__, bomb_error)

    assert warnings.filters is filters
    assert warnings.filters == original_filters


@pytest.mark.parametrize("kind", ["frames", "animation", "embedded", "normalize", "temporary"])
@pytest.mark.parametrize("max_pixels", [4, 2], ids=["warning", "error"])
def test_unrelated_warning_context_cannot_disable_pixel_limits_or_lose_its_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
    max_pixels: int,
) -> None:
    normal = tmp_path / "normal.png"
    normal.write_bytes(_png_bytes((1, 1)))
    size = (3, 2)
    png = _png_bytes(size)
    path = tmp_path / ("pixels.tiff" if kind == "animation" else "pixels.png")
    if kind == "animation":
        with Image.new("RGB", (1, 1)) as first, Image.new("RGB", size) as second:
            first.save(path, format="TIFF", save_all=True, append_images=[second])
    else:
        path.write_bytes(png)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", max_pixels)

    reader_entered = threading.Event()
    unrelated_entered = threading.Event()
    release_reader = threading.Event()
    release_unrelated = threading.Event()
    original_open = Image.open
    initial_filters = warnings.filters
    original_filters = initial_filters[:]
    original_showwarning = warnings.showwarning
    original_showwarnmsg_impl = warnings._showwarnmsg_impl

    def controlled_open(source, *args, **kwargs):
        if str(source) == str(normal):
            reader_entered.set()
            assert release_reader.wait(timeout=5)
        return original_open(source, *args, **kwargs)

    def unrelated_library() -> list[str]:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            warnings.simplefilter("always", UserWarning)
            unrelated_entered.set()
            assert release_unrelated.wait(timeout=5)
            warnings.warn("unrelated library warning", UserWarning)
        return [str(warning.message) for warning in captured]

    monkeypatch.setattr(Image, "open", controlled_open)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            reader = executor.submit(_read_image, "frames", normal, b"", (1, 1))
            unrelated = None
            try:
                assert reader_entered.wait(timeout=5)
                unrelated = executor.submit(unrelated_library)
                assert unrelated_entered.wait(timeout=5)
                bomb_error = Image.DecompressionBombWarning if max_pixels == 4 else Image.DecompressionBombError
                expected = bomb_error if kind in {"frames", "animation"} else SvgError
                with pytest.raises(expected) as raised:
                    _read_image(kind, path, png, size)
                if expected is SvgError:
                    assert isinstance(raised.value.__cause__, bomb_error)
                release_reader.set()
                reader.result(timeout=5)
            finally:
                release_reader.set()
                release_unrelated.set()
                reader.result(timeout=5)
            assert unrelated is not None
            assert unrelated.result(timeout=5) == ["unrelated library warning"]

        assert warnings.filters is initial_filters
        assert warnings.filters == original_filters
        assert warnings.showwarning is original_showwarning
        assert warnings._showwarnmsg_impl is original_showwarnmsg_impl
    finally:
        # A failing regression must not leave pytest's warning capture corrupted.
        warnings.filters = initial_filters
        initial_filters[:] = original_filters
        warnings.showwarning = original_showwarning
        warnings._showwarnmsg_impl = original_showwarnmsg_impl
