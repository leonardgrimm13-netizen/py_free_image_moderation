from __future__ import annotations

import json
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
    env["OPENNSFW2_DISABLE"] = "1"
    env["NUDENET_DISABLE"] = "1"
    env["OCR_ENABLE"] = "0"
    env["FORBIDDEN_SYMBOLS_YOLO_ENABLE"] = "0"
    env["YOLO_WEAPON_MODEL"] = str(tmp_path / "missing-weapons.pt")

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


def test_cli_empty_directory_returns_exit_code_2_without_traceback(tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(tmp_path), "--recursive", "--no-apis"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 2
    assert "Traceback (most recent call last)" not in combined


def test_cli_file_workers_preserve_json_order_and_count(tmp_path) -> None:
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for idx, color in enumerate(((10, 20, 30), (40, 50, 60), (70, 80, 90)), start=1):
        Image.new("RGB", (16, 16), color=color).save(img_dir / f"{idx:02d}.png")

    env = os.environ.copy()
    env.update(
        {
            "OPENNSFW2_DISABLE": "1",
            "NUDENET_DISABLE": "1",
            "OCR_ENABLE": "0",
            "FORBIDDEN_SYMBOLS_YOLO_ENABLE": "0",
            "YOLO_WEAPON_MODEL": str(tmp_path / "missing-weapons.pt"),
        }
    )

    outputs = []
    for workers in (1, 2):
        json_path = tmp_path / f"reports-{workers}.json"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "modimg.cli",
                str(img_dir),
                "--recursive",
                "--no-apis",
                "--sample-frames",
                "1",
                "--file-workers",
                str(workers),
                "--json",
                str(json_path),
            ],
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
        outputs.append(json.loads(json_path.read_text(encoding="utf-8")))

    expected_names = [str(img_dir / f"{idx:02d}.png") for idx in range(1, 4)]
    assert [[item["name"] for item in output] for output in outputs] == [expected_names, expected_names]
    assert len(outputs[0]) == len(outputs[1]) == 3
