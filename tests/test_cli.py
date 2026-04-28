from __future__ import annotations

import os
import subprocess
import sys


def test_cli_help() -> None:
    proc = subprocess.run(
        [sys.executable, "moderate_image.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode == 0
    help_text = f"{proc.stdout}\n{proc.stderr}"
    assert "--no-apis" in help_text


def test_main_import_smoke() -> None:
    __import__("modimg.cli")
    __import__("modimg.pipeline")


def test_cli_help_with_invalid_sample_frames_env_does_not_crash() -> None:
    env = os.environ.copy()
    env["SAMPLE_FRAMES"] = "not_an_int"

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    assert proc.returncode == 0


def test_cli_with_invalid_verdict_threshold_env_does_not_crash(tmp_path) -> None:
    from PIL import Image

    img_path = tmp_path / "sample.png"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(img_path)

    env = os.environ.copy()
    env["FINAL_BLOCK_THRESHOLD"] = "not_a_float"

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img_path), "--no-apis"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode in (0, 2)
    assert "Traceback (most recent call last)" not in combined
