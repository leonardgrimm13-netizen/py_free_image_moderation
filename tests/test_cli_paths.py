from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from modimg.utils import is_image_file


def _fast_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OCR_ENABLE": "0",
            "OPENNSFW2_DISABLE": "1",
            "NUDENET_DISABLE": "1",
            "FORBIDDEN_SYMBOLS_YOLO_ENABLE": "0",
            "API_POLICY": "never",
            "NO_CHECKS_POLICY": "ok",
        }
    )
    return env


def _make_image(path: Path) -> None:
    Image.new("RGB", (12, 12), color=(20, 40, 60)).save(path)


def _load_report(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def test_cli_no_input_returns_usage_error() -> None:
    proc = subprocess.run(
        [sys.executable, "moderate_image.py"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 2
    assert "input is required" in combined
    assert "Traceback (most recent call last)" not in combined


def test_directory_scan_nonrecursive_and_recursive(tmp_path) -> None:
    root_img = tmp_path / "Root Image.PNG"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_img = nested_dir / "nested.png"
    _make_image(root_img)
    _make_image(nested_img)

    nonrecursive_json = tmp_path / "out" / "nonrecursive.json"
    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(tmp_path), "--no-apis", "--json", str(nonrecursive_json)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    reports = _load_report(nonrecursive_json)
    assert [Path(item["path"]).name for item in reports] == [root_img.name]

    recursive_json = tmp_path / "out" / "recursive.json"
    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(tmp_path), "--recursive", "--no-apis", "--json", str(recursive_json)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    reports = _load_report(recursive_json)
    assert {Path(item["path"]).name for item in reports} == {root_img.name, nested_img.name}


def test_file_path_with_spaces_unicode_and_uppercase_extension(tmp_path) -> None:
    img = tmp_path / "spaced ünicode IMAGE.PNG"
    report = tmp_path / "reports" / "report.json"
    _make_image(img)

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img), "--no-apis", "--json", str(report)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert Path(payload["path"]).name == img.name


def test_cli_expands_tilde_for_single_image_path(tmp_path) -> None:
    fake_home = tmp_path / "home user"
    images_dir = fake_home / "pictures"
    images_dir.mkdir(parents=True)
    img = images_dir / "tilde image.png"
    report = tmp_path / "reports" / "tilde-report.json"
    _make_image(img)

    env = _fast_env()
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", "~/pictures/tilde image.png", "--no-apis", "--json", str(report)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0
    assert "Traceback (most recent call last)" not in combined
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert Path(payload["path"]) == img


def test_invalid_image_returns_loader_error_json(tmp_path) -> None:
    bad = tmp_path / "broken.png"
    report = tmp_path / "broken-report.json"
    bad.write_bytes(b"not an image")

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(bad), "--no-apis", "--json", str(report)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["verdict"]["label"] == "REVIEW"
    assert payload["results"][0]["name"] == "Loader"
    assert payload["results"][0]["status"] == "error"


def test_windows_style_uppercase_image_path_detection() -> None:
    assert is_image_file(r"C:\Users\me\Pictures\Test Image.PNG") is True
