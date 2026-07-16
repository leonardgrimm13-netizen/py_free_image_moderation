from __future__ import annotations

import pytest
from PIL import Image

from modimg.frames import load_frames


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
