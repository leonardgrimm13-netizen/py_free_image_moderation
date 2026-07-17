from __future__ import annotations

import pytest
from PIL import Image

from modimg.frames import load_frames
from tests.avif_helpers import make_animated_avif_bytes, write_avif


def test_static_image_loads_one_frame(tmp_path) -> None:
    img = tmp_path / "static.png"
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(img)

    frames = load_frames(str(img), sample_frames=12)

    assert len(frames) == 1
    assert frames[0].idx == 0


def test_animated_gif_sample_frames_zero_loads_first_frame_only(tmp_path) -> None:
    gif = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (8, 8), color=(i * 20, 0, 0)) for i in range(5)]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=20, loop=0)

    sampled = load_frames(str(gif), sample_frames=0)

    assert [fr.idx for fr in sampled] == [0]


def test_animated_gif_sampling_includes_first_and_last(tmp_path) -> None:
    gif = tmp_path / "animated.gif"
    frames = [Image.new("RGB", (8, 8), color=(i * 20, 0, 0)) for i in range(6)]
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=20, loop=0)

    sampled = load_frames(str(gif), sample_frames=3)

    assert sampled[0].idx == 0
    assert sampled[-1].idx == 5
    assert len(sampled) == 3


def test_corrupted_image_raises_controlled_loader_exception(tmp_path) -> None:
    bad = tmp_path / "bad.gif"
    bad.write_bytes(b"not a gif")

    with pytest.raises(Exception):
        load_frames(str(bad), sample_frames=2)


@pytest.mark.parametrize(
    ("mode", "suffix", "save_format"),
    [("L", ".png", "PNG"), ("RGBA", ".png", "PNG"), ("CMYK", ".jpg", "JPEG")],
)
def test_static_image_modes_are_detached_as_rgb_frames(tmp_path, mode: str, suffix: str, save_format: str) -> None:
    path = tmp_path / f"mode{suffix}"
    Image.new(mode, (8, 8)).save(path, format=save_format)

    frames = load_frames(str(path), sample_frames=1)

    assert frames[0].pil.mode == "RGB"
    assert frames[0].pil.getpixel((0, 0)) is not None
    frames[0].close()


def test_image_pixel_limit_is_enforced(monkeypatch, tmp_path) -> None:
    path = tmp_path / "large.png"
    Image.new("RGB", (10, 10)).save(path)
    monkeypatch.setenv("MODIMG_MAX_IMAGE_PIXELS", "99")

    with pytest.raises(ValueError, match="pixel count exceeds limit"):
        load_frames(str(path), sample_frames=1)


@pytest.mark.parametrize(("suffix", "image_format"), [(".png", "PNG"), (".jpg", "JPEG")])
def test_static_raster_decoded_pixel_budget_is_enforced(
    monkeypatch,
    tmp_path,
    suffix: str,
    image_format: str,
) -> None:
    path = tmp_path / f"budget{suffix}"
    Image.new("RGB", (10, 10)).save(path, format=image_format)
    monkeypatch.setenv("MODIMG_MAX_IMAGE_PIXELS", "1000")
    monkeypatch.setenv("MODIMG_MAX_DECODED_PIXELS", "99")

    with pytest.raises(ValueError, match="decoded image exceeds decoded-pixel budget"):
        load_frames(str(path), sample_frames=1)


def test_animation_frame_limit_is_enforced(monkeypatch, tmp_path) -> None:
    path = tmp_path / "many.gif"
    images = [Image.new("RGB", (4, 4), color=(idx * 40, 0, 0)) for idx in range(3)]
    images[0].save(path, save_all=True, append_images=images[1:], duration=20, loop=0)
    monkeypatch.setenv("MODIMG_MAX_ANIMATION_FRAMES", "2")

    with pytest.raises(ValueError, match="animation frame count exceeds limit"):
        load_frames(str(path), sample_frames=3)


def test_sampled_frame_decoded_pixel_budget_is_enforced(monkeypatch, tmp_path) -> None:
    path = tmp_path / "budget.gif"
    images = [Image.new("RGB", (10, 10), color=(idx * 40, 0, 0)) for idx in range(2)]
    images[0].save(path, save_all=True, append_images=images[1:], duration=20, loop=0)
    monkeypatch.setenv("MODIMG_MAX_DECODED_PIXELS", "199")

    with pytest.raises(ValueError, match="decoded-pixel budget"):
        load_frames(str(path), sample_frames=2)


def test_each_selected_multiframe_page_obeys_dimension_limits(monkeypatch, tmp_path) -> None:
    path = tmp_path / "varying-pages.tiff"
    first = Image.new("RGB", (1, 1))
    second = Image.new("RGB", (20, 20))
    first.save(path, format="TIFF", save_all=True, append_images=[second])
    first.close()
    second.close()
    monkeypatch.setenv("MODIMG_MAX_IMAGE_DIMENSION", "10")

    with pytest.raises(ValueError, match="image dimensions exceed limit: 20x20"):
        load_frames(str(path), sample_frames=2)


def test_multiframe_decoded_pixel_budget_uses_each_selected_page_size(monkeypatch, tmp_path) -> None:
    path = tmp_path / "varying-budget.tiff"
    first = Image.new("RGB", (5, 5))
    second = Image.new("RGB", (10, 10))
    first.save(path, format="TIFF", save_all=True, append_images=[second])
    first.close()
    second.close()
    monkeypatch.setenv("MODIMG_MAX_IMAGE_PIXELS", "1000")
    monkeypatch.setenv("MODIMG_MAX_DECODED_PIXELS", "120")

    with pytest.raises(ValueError, match=r"sampled frames exceed decoded-pixel budget: 125 .*limit 120"):
        load_frames(str(path), sample_frames=2)


def test_selected_animation_decode_failure_is_not_silently_ignored(monkeypatch) -> None:
    class BrokenAnimation:
        size = (2, 2)
        n_frames = 2
        is_animated = True

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def seek(self, index: int) -> None:
            if index == 1:
                raise EOFError("truncated frame")

        def convert(self, mode: str):
            return Image.new(mode, self.size)

    monkeypatch.setattr("modimg.frames.Image.open", lambda path: BrokenAnimation())

    with pytest.raises(ValueError, match="failed to decode selected animation frame 1"):
        load_frames("broken.gif", sample_frames=2)


def test_avif_loads_natively_as_detached_rgb_frame(tmp_path) -> None:
    path = write_avif(tmp_path / "sample.avif", size=(13, 9), color=(30, 80, 160), quality=95)

    frames = load_frames(str(path), sample_frames=12)
    try:
        assert len(frames) == 1
        assert frames[0].idx == 0
        assert frames[0].source_format == "avif"
        assert frames[0].pil.mode == "RGB"
        assert frames[0].pil.size == (13, 9)
        actual = frames[0].pil.getpixel((6, 4))
        assert all(abs(observed - expected) <= 8 for observed, expected in zip(actual, (30, 80, 160)))
    finally:
        for frame in frames:
            frame.close()


def test_avif_with_alpha_is_decoded_to_pipeline_rgb(tmp_path) -> None:
    path = write_avif(
        tmp_path / "alpha.avif",
        mode="RGBA",
        size=(8, 6),
        color=(100, 50, 200, 128),
        quality=95,
    )

    frames = load_frames(str(path), sample_frames=1)
    try:
        assert frames[0].pil.mode == "RGB"
        assert frames[0].pil.size == (8, 6)
        actual = frames[0].pil.getpixel((0, 0))
        assert all(abs(observed - expected) <= 10 for observed, expected in zip(actual, (100, 50, 200)))
    finally:
        for frame in frames:
            frame.close()


def test_animated_avif_uses_existing_safe_frame_sampling(tmp_path) -> None:
    path = tmp_path / "sequence.avif"
    path.write_bytes(
        make_animated_avif_bytes(
            [(10, 20, 30), (70, 20, 30), (130, 20, 30), (190, 20, 30)],
            quality=95,
        )
    )

    frames = load_frames(str(path), sample_frames=3)
    try:
        assert [frame.idx for frame in frames] == [0, 2, 3]
        assert all(frame.pil.mode == "RGB" for frame in frames)
        assert all(frame.source_format == "avif" for frame in frames)
        observed_red = [frame.pil.getpixel((0, 0))[0] for frame in frames]
        assert observed_red == pytest.approx([10, 130, 190], abs=10)
    finally:
        for frame in frames:
            frame.close()


def test_animated_avif_frame_limit_is_enforced(monkeypatch, tmp_path) -> None:
    path = tmp_path / "too-many.avif"
    path.write_bytes(make_animated_avif_bytes([(10, 0, 0), (80, 0, 0), (160, 0, 0)]))
    monkeypatch.setenv("MODIMG_MAX_ANIMATION_FRAMES", "2")

    with pytest.raises(ValueError, match="animation frame count exceeds limit"):
        load_frames(str(path), sample_frames=3)


def test_animated_avif_decoded_pixel_budget_is_enforced(monkeypatch, tmp_path) -> None:
    path = tmp_path / "budget.avif"
    path.write_bytes(make_animated_avif_bytes([(10, 0, 0), (80, 0, 0)], size=(10, 10)))
    monkeypatch.setenv("MODIMG_MAX_DECODED_PIXELS", "199")

    with pytest.raises(ValueError, match="decoded-pixel budget"):
        load_frames(str(path), sample_frames=2)


@pytest.mark.parametrize(
    ("environment_name", "environment_value", "message"),
    [
        ("MODIMG_MAX_IMAGE_DIMENSION", "11", "dimensions exceed limit"),
        ("MODIMG_MAX_IMAGE_PIXELS", "119", "pixel count exceeds limit"),
        ("MODIMG_MAX_DECODED_PIXELS", "119", "decoded image exceeds decoded-pixel budget"),
    ],
)
def test_static_avif_obeys_all_image_resource_limits(
    monkeypatch,
    tmp_path,
    environment_name: str,
    environment_value: str,
    message: str,
) -> None:
    path = write_avif(tmp_path / "limited.avif", size=(12, 10))
    monkeypatch.setenv(environment_name, environment_value)

    with pytest.raises(ValueError, match=message):
        load_frames(str(path), sample_frames=1)


def test_recognized_but_invalid_avif_has_controlled_decode_error(tmp_path) -> None:
    path = tmp_path / "invalid.avif"
    path.write_bytes(b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00mif1avif" + b"invalid payload")

    with pytest.raises(
        ValueError,
        match=r"failed to decode AVIF image: invalid data.*Pillow>=11\.3\.0 with AVIF support is required",
    ):
        load_frames(str(path), sample_frames=1)


def test_recognized_avif_without_runtime_codec_has_actionable_error(monkeypatch, tmp_path) -> None:
    path = write_avif(tmp_path / "codec.avif")
    monkeypatch.setattr("modimg.frames.pillow_avif_codec_available", lambda: False)

    with pytest.raises(
        ValueError,
        match=r"AVIF image detected.*Pillow>=11\.3\.0 with AVIF support is required",
    ):
        load_frames(str(path), sample_frames=1)


def test_recognized_avif_decoder_value_error_is_normalized(monkeypatch, tmp_path) -> None:
    path = write_avif(tmp_path / "codec-choice.avif")
    monkeypatch.setattr("modimg.frames.pillow_avif_codec_available", lambda: True)

    def invalid_codec(_path):
        raise ValueError("Invalid opening codec")

    monkeypatch.setattr("modimg.frames.Image.open", invalid_codec)

    with pytest.raises(
        ValueError,
        match=r"failed to decode AVIF image.*Pillow>=11\.3\.0 with AVIF support is required",
    ):
        load_frames(str(path), sample_frames=1)


def test_local_avif_encoded_source_byte_limit_is_enforced(monkeypatch, tmp_path) -> None:
    path = write_avif(tmp_path / "source-limit.avif")
    monkeypatch.setenv("MODIMG_MAX_AVIF_BYTES", str(path.stat().st_size - 1))

    with pytest.raises(ValueError, match="AVIF source exceeds byte limit"):
        load_frames(str(path), sample_frames=1)


def test_non_avif_never_performs_avif_codec_probe(monkeypatch, tmp_path) -> None:
    # Content wins over a misleading .avif suffix, so ordinary images neither
    # pay for nor depend upon the AVIF feature check.
    path = tmp_path / "actually-png.avif"
    Image.new("RGB", (4, 3), color=(1, 2, 3)).save(path, format="PNG")

    def unexpected_probe() -> bool:
        raise AssertionError("AVIF codec probe must be lazy and content-gated")

    monkeypatch.setattr("modimg.frames.pillow_avif_codec_available", unexpected_probe)
    frames = load_frames(str(path), sample_frames=1)
    try:
        assert frames[0].pil.size == (4, 3)
        assert frames[0].source_format == "png"
    finally:
        for frame in frames:
            frame.close()


def test_partially_decoded_avif_frames_are_closed_after_later_frame_failure(monkeypatch, tmp_path) -> None:
    path = tmp_path / "broken-sequence.avif"
    path.write_bytes(b"\x00\x00\x00\x18ftypavif\x00\x00\x00\x00mif1avif")
    produced = Image.new("RGB", (2, 2), color=(1, 2, 3))
    close_calls: list[bool] = []
    original_close = produced.close

    def tracked_close() -> None:
        close_calls.append(True)
        original_close()

    produced.close = tracked_close  # type: ignore[method-assign]

    class BrokenAvifAnimation:
        size = (2, 2)
        format = "AVIF"
        n_frames = 2
        is_animated = True
        current = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def seek(self, index: int) -> None:
            self.current = index

        def convert(self, mode: str):
            assert mode == "RGB"
            if self.current == 1:
                raise OSError("truncated second frame")
            return produced

    monkeypatch.setattr("modimg.frames.pillow_avif_codec_available", lambda: True)
    monkeypatch.setattr("modimg.frames.Image.open", lambda _path: BrokenAvifAnimation())

    with pytest.raises(ValueError, match="failed to decode selected animation frame 1"):
        load_frames(str(path), sample_frames=2)

    assert close_calls == [True]
