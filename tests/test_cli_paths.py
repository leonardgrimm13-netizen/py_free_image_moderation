from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

from modimg.utils import is_image_file


VALID_SVG_TEXT = """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="12">
  <rect width="24" height="12" fill="red"/>
</svg>
"""


def _fast_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OCR_ENABLE": "0",
            "OPENNSFW2_DISABLE": "1",
            "NUDENET_DISABLE": "1",
            "FORBIDDEN_SYMBOLS_YOLO_ENABLE": "0",
            "YOLO_BACKEND": "disabled",
            "API_POLICY": "never",
            "NO_CHECKS_POLICY": "ok",
        }
    )
    return env


def _make_image(path: Path) -> None:
    Image.new("RGB", (12, 12), color=(20, 40, 60)).save(path)


def _make_svg(path: Path) -> None:
    path.write_text(VALID_SVG_TEXT, encoding="utf-8")


def _make_avif(path: Path) -> None:
    with Image.new("RGB", (12, 10), color=(30, 80, 160)) as image:
        image.save(path, format="AVIF", quality=90)


def _load_report(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else [payload]


def test_cli_no_input_returns_usage_error() -> None:
    proc = subprocess.run(
        [sys.executable, "moderate_image.py"],
        check=False,
        timeout=60,
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
    root_svg = tmp_path / "02-vector.svg"
    root_upper_svg = tmp_path / "03-vector.SVG"
    root_extensionless_svg = tmp_path / "04-vector"
    fake_svg = tmp_path / "05-not-really-svg.svg"
    root_avif = tmp_path / "06-image.avif"
    root_upper_avif = tmp_path / "07-image.AVIF"
    root_mixed_avif = tmp_path / "08-image.AvIf"
    root_extensionless_avif = tmp_path / "09-avif-without-extension"
    root_wrong_extension_avif = tmp_path / "10-avif-content.bin"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested_img = nested_dir / "nested.png"
    nested_svg = nested_dir / "nested.svg"
    nested_avif = nested_dir / "nested.avif"
    _make_image(root_img)
    _make_svg(root_svg)
    _make_svg(root_upper_svg)
    _make_svg(root_extensionless_svg)
    fake_svg.write_text("<html></html>", encoding="utf-8")
    for path in (root_avif, root_upper_avif, root_mixed_avif, root_extensionless_avif, root_wrong_extension_avif):
        _make_avif(path)
    _make_image(nested_img)
    _make_svg(nested_svg)
    _make_avif(nested_avif)

    assert is_image_file(str(root_svg)) is True
    assert is_image_file(str(root_upper_svg)) is True
    assert is_image_file(str(root_extensionless_svg)) is True
    assert is_image_file(str(fake_svg)) is False
    assert is_image_file(str(root_avif)) is True
    assert is_image_file(str(root_upper_avif)) is True
    assert is_image_file(str(root_mixed_avif)) is True
    assert is_image_file(str(root_extensionless_avif)) is True
    assert is_image_file(str(root_wrong_extension_avif)) is True

    nonrecursive_json = tmp_path / "out" / "nonrecursive.json"
    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(tmp_path), "--no-apis", "--json", str(nonrecursive_json)],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    reports = _load_report(nonrecursive_json)
    assert [item["path"] for item in reports] == sorted(
        str(path)
        for path in (
            root_img,
            root_svg,
            root_upper_svg,
            root_extensionless_svg,
            root_avif,
            root_upper_avif,
            root_mixed_avif,
            root_extensionless_avif,
            root_wrong_extension_avif,
        )
    )
    for item in reports:
        if item["path"] in {
            str(root_avif),
            str(root_upper_avif),
            str(root_mixed_avif),
            str(root_extensionless_avif),
            str(root_wrong_extension_avif),
        }:
            assert item["preprocessing"] == {"source_format": "avif", "native_decode": True}

    recursive_json = tmp_path / "out" / "recursive.json"
    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(tmp_path), "--recursive", "--no-apis", "--json", str(recursive_json)],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    reports = _load_report(recursive_json)
    assert [item["path"] for item in reports] == sorted(
        str(path)
        for path in (
            root_img,
            root_svg,
            root_upper_svg,
            root_extensionless_svg,
            root_avif,
            root_upper_avif,
            root_mixed_avif,
            root_extensionless_avif,
            root_wrong_extension_avif,
            nested_img,
            nested_svg,
            nested_avif,
        )
    )


def test_file_path_with_spaces_unicode_and_uppercase_extension(tmp_path) -> None:
    img = tmp_path / "spaced ünicode IMAGE.PNG"
    report = tmp_path / "reports" / "report.json"
    _make_image(img)

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img), "--no-apis", "--json", str(report)],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert Path(payload["path"]).name == img.name


def test_avif_path_with_spaces_unicode_and_mixed_case_extension(tmp_path) -> None:
    image = tmp_path / "spaced ünicode IMAGE.AvIf"
    report = tmp_path / "reports" / "avif-report.json"
    _make_avif(image)

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(image), "--no-apis", "--json", str(report)],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert Path(payload["path"]).name == image.name
    assert payload["preprocessing"] == {"source_format": "avif", "native_decode": True}


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
        timeout=60,
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
        timeout=60,
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
    assert is_image_file(r"C:\Users\me\Pictures\Test Image.AVIF") is True


def test_directory_scan_includes_extensionless_image(tmp_path) -> None:
    image = tmp_path / "extensionless"
    report = tmp_path / "report.json"
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(image, format="PNG")

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(tmp_path), "--no-apis", "--json", str(report)],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["path"] == str(image)


def test_cli_accepts_multiple_explicit_inputs_and_deduplicates(tmp_path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.AVIF"
    third = tmp_path / "third-extensionless"
    report = tmp_path / "report.json"
    _make_image(first)
    _make_avif(second)
    _make_avif(third)

    proc = subprocess.run(
        [
            sys.executable,
            "moderate_image.py",
            str(first),
            str(second),
            str(first),
            str(third),
            str(second),
            "--no-apis",
            "--json",
            str(report),
        ],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    assert proc.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert [item["path"] for item in payload] == [str(first), str(second), str(third)]
    assert payload[1]["preprocessing"]["source_format"] == "avif"
    assert payload[2]["preprocessing"]["source_format"] == "avif"


def test_cli_regression_supported_raster_formats(tmp_path) -> None:
    image_dir = tmp_path / "formats"
    image_dir.mkdir()
    formats = [
        ("01.jpg", "JPEG"),
        ("02.png", "PNG"),
        ("03.gif", "GIF"),
        ("04.webp", "WEBP"),
        ("05.bmp", "BMP"),
        ("06.tiff", "TIFF"),
        ("07.AVIF", "AVIF"),
    ]
    paths: list[Path] = []
    for index, (filename, image_format) in enumerate(formats):
        path = image_dir / filename
        Image.new("RGB", (12, 10), color=(20 + index, 40, 60)).save(path, format=image_format)
        paths.append(path)

    report = tmp_path / "raster-report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "moderate_image.py",
            str(image_dir),
            "--no-apis",
            "--file-workers",
            "2",
            "--json",
            str(report),
        ],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_fast_env(),
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0
    assert "Traceback (most recent call last)" not in combined
    payload = _load_report(report)
    assert [item["path"] for item in payload] == [str(path) for path in paths]
