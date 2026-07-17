from __future__ import annotations

import json
import os
import subprocess
import sys


VALID_SVG_TEXT = """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="16">
  <rect width="32" height="16" fill="red"/>
</svg>
"""


def _make_avif(path) -> None:
    from PIL import Image

    with Image.new("RGB", (16, 12), color=(30, 80, 160)) as image:
        image.save(path, format="AVIF", quality=90)


def test_cli_help() -> None:
    for command in (
        [sys.executable, "moderate_image.py", "--help"],
        [sys.executable, "-m", "modimg.cli", "--help"],
    ):
        proc = subprocess.run(
            command,
            check=False,
            timeout=60,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.returncode == 0
        help_text = f"{proc.stdout}\n{proc.stderr}"
        assert "--no-apis" in help_text
        assert "SVG" in help_text
        assert "AVIF" in help_text


def test_main_import_smoke() -> None:
    __import__("modimg.cli")
    __import__("modimg.pipeline")


def test_cli_help_with_invalid_sample_frames_env_does_not_crash() -> None:
    env = os.environ.copy()
    env["SAMPLE_FRAMES"] = "not_an_int"
    env["MODIMG_MAX_SVG_BYTES"] = "not_an_int"
    env["MODIMG_SVG_DEFAULT_WIDTH"] = "not_an_int"
    env["MODIMG_SVG_MAX_RENDER_PIXELS"] = "not_an_int"

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", "--help"],
        check=False,
        timeout=60,
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
    env["YOLO_BACKEND"] = "disabled"
    env["YOLO_WEAPON_MODEL"] = str(tmp_path / "missing-weapons.pt")
    env["API_POLICY"] = "never"
    env["NO_CHECKS_POLICY"] = "ok"

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img_path), "--no-apis"],
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


def test_cli_empty_directory_returns_exit_code_2_without_traceback(tmp_path) -> None:
    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(tmp_path), "--recursive", "--no-apis"],
        check=False,
        timeout=60,
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
    first = img_dir / "01.png"
    second = img_dir / "02.avif"
    third = img_dir / "03.AVIF"
    fourth = img_dir / "04.AvIf"
    fifth = img_dir / "05.svg"
    nested_dir = img_dir / "nested"
    nested_dir.mkdir()
    sixth = nested_dir / "06-extensionless-avif"
    seventh = nested_dir / "07.svg"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(first)
    for path in (second, third, fourth, sixth):
        _make_avif(path)
    for path in (fifth, seventh):
        path.write_text(VALID_SVG_TEXT, encoding="utf-8")

    env = os.environ.copy()
    env.update(
        {
            "OPENNSFW2_DISABLE": "1",
            "NUDENET_DISABLE": "1",
            "OCR_ENABLE": "0",
            "FORBIDDEN_SYMBOLS_YOLO_ENABLE": "0",
            "YOLO_BACKEND": "disabled",
            "YOLO_WEAPON_MODEL": str(tmp_path / "missing-weapons.pt"),
            "API_POLICY": "never",
            "NO_CHECKS_POLICY": "ok",
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
        outputs.append(json.loads(json_path.read_text(encoding="utf-8")))

    expected_names = sorted(str(path) for path in (first, second, third, fourth, fifth, sixth, seventh))
    assert [[item["name"] for item in output] for output in outputs] == [expected_names, expected_names]
    assert len(outputs[0]) == len(outputs[1]) == 7
    for output in outputs:
        by_path = {item["path"]: item for item in output}
        for avif_path in (second, third, fourth, sixth):
            assert by_path[str(avif_path)]["preprocessing"] == {
                "source_format": "avif",
                "native_decode": True,
            }
        for svg_path in (fifth, seventh):
            assert by_path[str(svg_path)]["preprocessing"]["normalized_format"] == "png"


def test_cli_real_svg_smoke_for_script_and_module(tmp_path) -> None:
    svg_path = tmp_path / "local-vector.svg"
    svg_path.write_text(VALID_SVG_TEXT, encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "OPENNSFW2_DISABLE": "1",
            "NUDENET_DISABLE": "1",
            "OCR_ENABLE": "0",
            "FORBIDDEN_SYMBOLS_YOLO_ENABLE": "0",
            "YOLO_BACKEND": "disabled",
            "YOLO_WEAPON_MODEL": str(tmp_path / "missing-weapons.pt"),
            "API_POLICY": "never",
            "NO_CHECKS_POLICY": "ok",
        }
    )

    for index, command in enumerate(
        (
            [sys.executable, "moderate_image.py"],
            [sys.executable, "-m", "modimg.cli"],
        )
    ):
        json_path = tmp_path / f"svg-report-{index}.json"
        proc = subprocess.run(
            [*command, str(svg_path), "--no-apis", "--json", str(json_path)],
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
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["path"] == str(svg_path)
        assert payload["preprocessing"]["source_format"] == "svg"
        assert payload["preprocessing"]["normalized_format"] == "png"


def test_cli_real_avif_smoke_for_script_and_module_json(tmp_path) -> None:
    avif_path = tmp_path / "local-image.AVIF"
    _make_avif(avif_path)
    env = os.environ.copy()
    env.update(
        {
            "OPENNSFW2_DISABLE": "1",
            "NUDENET_DISABLE": "1",
            "OCR_ENABLE": "0",
            "FORBIDDEN_SYMBOLS_YOLO_ENABLE": "0",
            "YOLO_BACKEND": "disabled",
            "YOLO_WEAPON_MODEL": str(tmp_path / "missing-weapons.pt"),
            "API_POLICY": "never",
            "NO_CHECKS_POLICY": "ok",
        }
    )

    for index, command in enumerate(
        (
            [sys.executable, "moderate_image.py"],
            [sys.executable, "-m", "modimg.cli"],
        )
    ):
        json_path = tmp_path / f"avif-report-{index}.json"
        proc = subprocess.run(
            [*command, str(avif_path), "--no-apis", "--json", str(json_path)],
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
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["name"] == str(avif_path)
        assert payload["path"] == str(avif_path)
        assert payload["preprocessing"] == {"source_format": "avif", "native_decode": True}
        assert "<temporary-file>" not in repr(payload)
