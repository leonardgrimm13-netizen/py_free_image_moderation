from __future__ import annotations

import json
import subprocess
import sys

from PIL import Image


def _make_image(path) -> None:
    Image.new("RGB", (16, 16), color=(50, 100, 150)).save(path)


def test_cli_benchmark_json_file(tmp_path) -> None:
    img_path = tmp_path / "sample.png"
    bench_path = tmp_path / "benchmark.json"
    _make_image(img_path)

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img_path), "--no-apis", "--benchmark-json", str(bench_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode in (0, 2)
    assert "Traceback (most recent call last)" not in combined
    assert bench_path.exists()
    payload = json.loads(bench_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["total_files"] == 1
    assert "total_ms" in payload
    assert "total_wall_ms" in payload
    assert "avg_file_ms" in payload
    assert "verdict_counts" in payload
    assert "engine_stats" in payload
    assert "slowest_files" in payload
    assert "slowest_engines" in payload


def test_cli_benchmark_console_does_not_crash(tmp_path) -> None:
    img_path = tmp_path / "sample.png"
    _make_image(img_path)

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img_path), "--no-apis", "--benchmark"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    combined = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode in (0, 2)
    assert "Traceback (most recent call last)" not in combined
    assert "BENCHMARK" in combined
    assert "Files" in combined
    assert "Per engine" in combined


def test_cli_benchmark_does_not_change_old_json_single_report(tmp_path) -> None:
    img_path = tmp_path / "sample.png"
    json_path = tmp_path / "moderation.json"
    _make_image(img_path)

    proc = subprocess.run(
        [sys.executable, "moderate_image.py", str(img_path), "--no-apis", "--json", str(json_path)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode in (0, 2)
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "name" in payload
    assert "path" in payload
    assert "verdict" in payload
    assert "results" in payload
    assert "benchmark" not in payload
    assert "reports" not in payload


def test_cli_benchmark_still_does_not_change_old_json_single_report(tmp_path) -> None:
    img_path = tmp_path / "sample.png"
    moderation_json_path = tmp_path / "moderation.json"
    benchmark_json_path = tmp_path / "benchmark.json"
    _make_image(img_path)

    proc = subprocess.run(
        [
            sys.executable,
            "moderate_image.py",
            str(img_path),
            "--no-apis",
            "--json",
            str(moderation_json_path),
            "--benchmark-json",
            str(benchmark_json_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert proc.returncode in (0, 2)
    assert moderation_json_path.exists()
    assert benchmark_json_path.exists()
    moderation_payload = json.loads(moderation_json_path.read_text(encoding="utf-8"))
    benchmark_payload = json.loads(benchmark_json_path.read_text(encoding="utf-8"))
    assert isinstance(moderation_payload, dict)
    assert "benchmark" not in moderation_payload
    assert "reports" not in moderation_payload
    assert benchmark_payload["version"] == 1


def test_cli_benchmark_help_lists_flags() -> None:
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
    assert "--benchmark" in help_text
    assert "--benchmark-json" in help_text
