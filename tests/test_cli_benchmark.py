from __future__ import annotations

import json
import subprocess
import sys

from PIL import Image


def _make_image(path) -> None:
    Image.new("RGB", (16, 16), color=(50, 100, 150)).save(path)


def _light_env(tmp_path) -> dict[str, str]:
    import os

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
    return env


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
        env=_light_env(tmp_path),
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
        env=_light_env(tmp_path),
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
        env=_light_env(tmp_path),
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
        env=_light_env(tmp_path),
    )

    assert proc.returncode in (0, 2)
    assert moderation_json_path.exists()
    assert benchmark_json_path.exists()
    moderation_payload = json.loads(moderation_json_path.read_text(encoding="utf-8"))
    benchmark_payload = json.loads(benchmark_json_path.read_text(encoding="utf-8"))
    assert isinstance(moderation_payload, dict)
    assert "benchmark" not in moderation_payload
    assert "reports" not in moderation_payload
    assert "total_wall_ms" in benchmark_payload
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


def test_cli_benchmark_wall_time_is_measured_for_all_file_worker_modes(monkeypatch, tmp_path) -> None:
    from modimg import cli
    from modimg.enums import EngineStatus, VerdictLabel
    from modimg.types import EngineResult, Verdict

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for idx in range(2):
        (img_dir / f"{idx}.png").write_bytes(b"")

    seen_run_states = []

    def fake_process_input(
        idx,
        inp,
        *,
        no_apis,
        sample_frames,
        benchmark_enabled,
        openai_run_state,
        sightengine_run_state,
    ):
        seen_run_states.append((openai_run_state, sightengine_run_state))
        rep = {
            "name": inp,
            "path": inp,
            "verdict": Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, []),
            "results": [EngineResult(name="fake", status=EngineStatus.OK, took_ms=0)],
            "auto_learn": "",
        }
        benchmark_item = {
            "name": inp,
            "path": inp,
            "verdict_label": "OK",
            "total_ms": 1000,
            "engine_total_ms": 0,
            "unattributed_ms": 1000,
            "engine_count": 1,
            "slowest_engine": "fake",
            "slowest_engine_ms": 0,
            "engines": [{"name": "fake", "status": "ok", "took_ms": 0}],
        }
        return idx, rep, benchmark_item if benchmark_enabled else None

    monkeypatch.setattr(cli, "_process_input", fake_process_input)
    dotenv_cwd_flags = []
    monkeypatch.setattr(
        cli,
        "load_dotenv_candidates",
        lambda *, include_cwd=False: dotenv_cwd_flags.append(include_cwd) or (None, []),
    )

    states_by_cli_run = []
    for workers in (1, 2):
        state_start = len(seen_run_states)
        bench_path = tmp_path / f"benchmark-{workers}.json"
        rc = cli.main([str(img_dir), "--file-workers", str(workers), "--benchmark-json", str(bench_path)])

        payload = json.loads(bench_path.read_text(encoding="utf-8"))
        assert rc == 0
        assert payload["total_ms"] == 2000
        assert payload["total_wall_ms"] < 1000
        states_for_run = seen_run_states[state_start:]
        assert len({id(openai_state) for openai_state, _ in states_for_run}) == 1
        assert len({id(sightengine_state) for _, sightengine_state in states_for_run}) == 1
        states_by_cli_run.append(states_for_run[0])

    assert states_by_cli_run[0][0] is not states_by_cli_run[1][0]
    assert states_by_cli_run[0][1] is not states_by_cli_run[1][1]
    assert dotenv_cwd_flags == [True, True]
