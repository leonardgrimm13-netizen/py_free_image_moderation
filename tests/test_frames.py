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
