from __future__ import annotations

import json
import subprocess
import sys
import types

from PIL import Image

from modimg.config import get_config
from modimg.pipeline import maybe_auto_learn, run_engines
from modimg.types import Engine, EngineResult
from modimg.verdict import compute_verdict, pick_file_dialog, pick_folder_dialog
from modimg.enums import EngineStatus, VerdictLabel
from modimg.types import Frame, Verdict


class SlowEngine(Engine):
    def __init__(self, name: str, delay: float = 0.05) -> None:
        super().__init__()
        self.name = name
        self.delay = delay

    def run(self, path, frames, max_api_frames=3):
        import time

        time.sleep(self.delay)
        return EngineResult(name=self.name, status="ok", scores={"x": 1.0})


class _FakeDialogRoot:
    def __init__(self, *, fail_withdraw: bool = False) -> None:
        self.destroyed = False
        self.fail_withdraw = fail_withdraw

    def withdraw(self) -> None:
        if self.fail_withdraw:
            raise RuntimeError("withdraw failed")

    def destroy(self) -> None:
        self.destroyed = True


def _install_fake_tk(monkeypatch, root: _FakeDialogRoot, filedialog: types.ModuleType) -> None:
    tkinter = types.ModuleType("tkinter")
    tkinter.Tk = lambda: root
    tkinter.filedialog = filedialog
    monkeypatch.setitem(sys.modules, "tkinter", tkinter)
    monkeypatch.setitem(sys.modules, "tkinter.filedialog", filedialog)


def test_file_dialog_filter_supports_uppercase_svg_and_cleans_root(monkeypatch) -> None:
    root = _FakeDialogRoot()
    captured: dict[str, object] = {}
    filedialog = types.ModuleType("tkinter.filedialog")

    def askopenfilename(**kwargs):
        captured.update(kwargs)
        return ""

    filedialog.askopenfilename = askopenfilename
    _install_fake_tk(monkeypatch, root, filedialog)

    assert pick_file_dialog() is None
    assert "*.[sS][vV][gG]" in str(captured["filetypes"])
    assert root.destroyed is True


def test_folder_dialog_cleans_root_when_the_dialog_raises(monkeypatch) -> None:
    root = _FakeDialogRoot()
    filedialog = types.ModuleType("tkinter.filedialog")

    def askdirectory():
        raise RuntimeError("dialog failed")

    filedialog.askdirectory = askdirectory
    _install_fake_tk(monkeypatch, root, filedialog)

    assert pick_folder_dialog() is None
    assert root.destroyed is True


def test_file_dialog_cleans_root_when_withdraw_raises(monkeypatch) -> None:
    root = _FakeDialogRoot(fail_withdraw=True)
    filedialog = types.ModuleType("tkinter.filedialog")
    filedialog.askopenfilename = lambda **kwargs: "unexpected"
    _install_fake_tk(monkeypatch, root, filedialog)

    assert pick_file_dialog() is None
    assert root.destroyed is True


def test_get_config_disable_flag_parsing(monkeypatch) -> None:
    monkeypatch.setenv("OPENNSFW2_DISABLE", "1")
    cfg = get_config(reload=True)
    assert cfg.opennsfw2_disable is True


def test_run_engines_parallel_preserves_order(monkeypatch) -> None:
    monkeypatch.setenv("MODIMG_PARALLEL_ENGINES", "1")
    monkeypatch.setenv("MODIMG_PARALLEL_WORKERS", "2")
    get_config(reload=True)

    engines = [SlowEngine("a"), SlowEngine("b")]
    results = run_engines("dummy", [], engines)

    assert [r.name for r in results] == ["a", "b"]


def test_run_engines_parallel_preserves_order_with_out_of_order_completion(monkeypatch) -> None:
    monkeypatch.setenv("MODIMG_PARALLEL_ENGINES", "1")
    monkeypatch.setenv("MODIMG_PARALLEL_WORKERS", "2")
    get_config(reload=True)

    engines = [SlowEngine("slow", 0.05), SlowEngine("fast", 0.0)]
    results = run_engines("dummy", [], engines)

    assert [r.name for r in results] == ["slow", "fast"]


def test_maybe_auto_learn_processes_all_hashes(monkeypatch) -> None:
    monkeypatch.setenv("PHASH_AUTO_LEARN_ENABLE", "1")
    monkeypatch.setenv("PHASH_GIF_LEARN_FIRST_LAST", "1")
    monkeypatch.setenv("PHASH_AUTO_ALLOW_APPEND", "1")
    calls: list[str] = []

    monkeypatch.setattr("modimg.pipeline.frame_phash_hex_int", lambda fr: (f"h{fr.idx}", 1))

    def fake_append(hx: str, path: str, label: str) -> bool:
        calls.append(hx)
        return hx == "h0"

    monkeypatch.setattr("modimg.pipeline.append_phash_to_allowlist", fake_append)
    monkeypatch.setattr("modimg.pipeline.get_allowlist_path", lambda: "allow.txt")

    frames = [Frame(idx=0, pil=Image.new("RGB", (2, 2))), Frame(idx=5, pil=Image.new("RGB", (2, 2)))]
    verdict = Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, [])

    msg = maybe_auto_learn(verdict, frames)

    assert calls == ["h0", "h5"]
    assert msg == "Auto-added pHash to allowlist (allow.txt)"


def test_maybe_auto_learn_master_switch_off_ignores_legacy_flags(monkeypatch) -> None:
    monkeypatch.setenv("PHASH_AUTO_LEARN_ENABLE", "0")
    monkeypatch.setenv("PHASH_AUTO_ALLOW_APPEND", "1")
    monkeypatch.setenv("PHASH_AUTO_APPEND", "1")
    monkeypatch.setattr("modimg.pipeline.frame_phash_hex_int", lambda fr: (f"h{fr.idx}", 1))

    def fail_append(*args, **kwargs):
        raise AssertionError("append_phash_to_allowlist should not be called when master switch is off")

    monkeypatch.setattr("modimg.pipeline.append_phash_to_allowlist", fail_append)

    frames = [Frame(idx=0, pil=Image.new("RGB", (2, 2)))]
    verdict = Verdict(VerdictLabel.OK, 0.0, 0.0, 0.0, [])

    assert maybe_auto_learn(verdict, frames) is None


def test_cli_json_serializes_enum_values(tmp_path) -> None:
    import os

    img_path = tmp_path / "sample.png"
    out_path = tmp_path / "report.json"
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(img_path)
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

    proc = subprocess.run(
        [sys.executable, "-m", "modimg.cli", str(img_path), "--no-apis", "--json", str(out_path)],
        check=False,
        timeout=60,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert proc.returncode == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data["verdict"]["label"], str)
    assert data["verdict"]["label"] in {"OK", "REVIEW", "BLOCK"}
    assert all(isinstance(r["status"], str) for r in data["results"])


def test_final_block_threshold_uses_config(monkeypatch) -> None:
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.95")
    get_config(reload=True)
    verdict = compute_verdict([EngineResult(name="OpenNSFW2", status="ok", scores={"nsfw_probability": 0.90})])
    assert verdict.label == VerdictLabel.REVIEW


def test_no_checks_policy_default_review(monkeypatch) -> None:
    monkeypatch.delenv("NO_CHECKS_POLICY", raising=False)
    verdict = compute_verdict([EngineResult(name="OpenNSFW2", status="skipped", error="disabled")])
    assert verdict.label == VerdictLabel.REVIEW
    assert "No checks ran (all engines skipped/disabled)." in verdict.reasons


def test_no_checks_policy_ok(monkeypatch) -> None:
    monkeypatch.setenv("NO_CHECKS_POLICY", "ok")
    verdict = compute_verdict([EngineResult(name="OpenNSFW2", status="skipped", error="disabled")])
    assert verdict.label == VerdictLabel.OK
    assert "No checks ran (all engines skipped/disabled)." in verdict.reasons


def test_no_checks_policy_block(monkeypatch) -> None:
    monkeypatch.setenv("NO_CHECKS_POLICY", "block")
    verdict = compute_verdict([EngineResult(name="OpenNSFW2", status="skipped", error="disabled")])
    assert verdict.label == VerdictLabel.BLOCK
    assert "No checks ran (all engines skipped/disabled)." in verdict.reasons


def test_compute_verdict_accepts_enum_and_legacy_string_statuses(monkeypatch) -> None:
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.85")
    get_config(reload=True)
    enum_verdict = compute_verdict([EngineResult(name="OpenNSFW2", status=EngineStatus.OK, scores={"nsfw_probability": 0.90})])
    string_verdict = compute_verdict([EngineResult(name="OpenNSFW2", status="ok", scores={"nsfw_probability": 0.90})])

    assert enum_verdict.label == string_verdict.label == VerdictLabel.BLOCK


def test_compute_verdict_core_error_policy_values(monkeypatch) -> None:
    result = EngineResult(name="OCR text", status=EngineStatus.ERROR, error="boom")

    monkeypatch.setenv("ENGINE_ERROR_POLICY", "ignore")
    assert compute_verdict([result]).label == VerdictLabel.REVIEW  # no successful checks fallback still applies

    monkeypatch.setenv("ENGINE_ERROR_POLICY", "review")
    assert compute_verdict([result]).label == VerdictLabel.REVIEW

    monkeypatch.setenv("ENGINE_ERROR_POLICY", "block")
    assert compute_verdict([result]).label == VerdictLabel.BLOCK


def test_compute_verdict_specific_rules(monkeypatch) -> None:
    assert compute_verdict([EngineResult(name="pHash allowlist", status="ok", scores={"phash_allow_match": 1.0})]).label == VerdictLabel.OK
    assert compute_verdict([EngineResult(name="pHash blocklist", status="ok", scores={"phash_block_match": 1.0})]).label == VerdictLabel.BLOCK
    assert compute_verdict([EngineResult(name="OCR text", status="ok", scores={"ocr_match": 1.0})]).label == VerdictLabel.BLOCK
    assert compute_verdict([EngineResult(name="OpenAI Moderation", status="ok", scores={"sexual/minors": 0.02})]).label == VerdictLabel.BLOCK


def test_phash_allow_does_not_override_later_block_when_short_circuit_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SHORT_CIRCUIT_PHASH", "0")
    results = [
        EngineResult(name="pHash allowlist", status="ok", scores={"phash_allow_match": 1.0}),
        EngineResult(name="OCR text", status="ok", scores={"ocr_match": 1.0}),
    ]

    verdict = compute_verdict(results)

    assert verdict.label == VerdictLabel.BLOCK
    assert "pHash allowlist match" in verdict.reasons
    assert "OCR text blocked" in verdict.reasons


def test_conflicting_phash_lists_block_regardless_of_result_order() -> None:
    allow = EngineResult(name="pHash allowlist", status="ok", scores={"phash_allow_match": 1.0})
    block = EngineResult(name="pHash blocklist", status="ok", scores={"phash_block_match": 1.0})

    assert compute_verdict([allow, block]).label == VerdictLabel.BLOCK
    assert compute_verdict([block, allow]).label == VerdictLabel.BLOCK


def test_openai_sexual_minors_threshold_is_inclusive(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_SEXUAL_MINORS_BLOCK_THRESHOLD", "0.01")

    verdict = compute_verdict(
        [EngineResult(name="OpenAI Moderation", status="ok", scores={"sexual/minors": 0.01})]
    )

    assert verdict.label == VerdictLabel.BLOCK


def test_openai_flagged_unmapped_category_requires_review(monkeypatch) -> None:
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.85")
    monkeypatch.setenv("FINAL_REVIEW_THRESHOLD", "0.40")
    get_config(reload=True)

    verdict = compute_verdict(
        [EngineResult(name="OpenAI Moderation", status="ok", scores={"flagged": 1.0})]
    )

    assert verdict.label == VerdictLabel.REVIEW
    assert "outside mapped score categories" in verdict.reasons[0]


def test_openai_high_self_harm_score_can_block(monkeypatch) -> None:
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.85")
    get_config(reload=True)

    verdict = compute_verdict(
        [EngineResult(name="OpenAI Moderation", status="ok", scores={"self-harm/intent": 0.90})]
    )

    assert verdict.label == VerdictLabel.BLOCK
    assert verdict.violence_risk == 0.90
    assert "OpenAI self-harm=0.90" in verdict.reasons


def test_openai_harassment_and_illicit_scores_affect_verdict(monkeypatch) -> None:
    monkeypatch.setenv("FINAL_BLOCK_THRESHOLD", "0.85")
    get_config(reload=True)

    harassment = compute_verdict(
        [EngineResult(name="OpenAI Moderation", status="ok", scores={"harassment/threatening": 0.60})]
    )
    illicit = compute_verdict(
        [EngineResult(name="OpenAI Moderation", status="ok", scores={"illicit/violent": 0.60})]
    )

    assert harassment.label == VerdictLabel.REVIEW
    assert harassment.hate_risk == 0.60
    assert illicit.label == VerdictLabel.REVIEW
    assert illicit.violence_risk == 0.60


def test_dotenv_example_is_not_loaded_as_runtime_defaults(monkeypatch) -> None:
    from pathlib import Path

    from modimg import config as config_mod

    monkeypatch.setenv("DOTENV_OVERRIDE", "0")
    monkeypatch.setenv("SAMPLE_FRAMES", "12")
    monkeypatch.setenv("FORBIDDEN_SYMBOLS_YOLO_CONF", "sentinel-from-test")

    path, keys = config_mod.load_dotenv_candidates()

    assert path is None or Path(path).name in {".env", ".env.txt"}
    assert path is None or Path(path).name != ".env.example"
    assert "FORBIDDEN_SYMBOLS_YOLO_CONF" not in keys
    assert config_mod.os.environ["FORBIDDEN_SYMBOLS_YOLO_CONF"] == "sentinel-from-test"
    assert config_mod.get_config(reload=True).sample_frames == 12


def test_cli_dotenv_search_can_use_cwd_without_changing_import_default(monkeypatch, tmp_path) -> None:
    from modimg import config as config_mod

    package_root = tmp_path / "installed"
    package_file = package_root / "modimg" / "config.py"
    package_file.parent.mkdir(parents=True)
    package_file.touch()
    cwd = tmp_path / "working-directory"
    cwd.mkdir()
    (cwd / ".env").write_text("MODIMG_TEST_CWD_DOTENV=loaded\n", encoding="utf-8")
    (cwd / ".env.example").write_text("MODIMG_TEST_EXAMPLE=must-not-load\n", encoding="utf-8")

    monkeypatch.chdir(cwd)
    monkeypatch.setattr(config_mod, "__file__", str(package_file))
    monkeypatch.delenv("MODIMG_TEST_CWD_DOTENV", raising=False)
    monkeypatch.delenv("MODIMG_TEST_EXAMPLE", raising=False)

    import_path, import_keys = config_mod.load_dotenv_candidates()
    cli_path, cli_keys = config_mod.load_dotenv_candidates(include_cwd=True)

    assert import_path is None
    assert import_keys == []
    assert cli_path == str(cwd / ".env")
    assert cli_keys == ["MODIMG_TEST_CWD_DOTENV"]
    assert config_mod.os.environ["MODIMG_TEST_CWD_DOTENV"] == "loaded"
    assert "MODIMG_TEST_EXAMPLE" not in config_mod.os.environ


def test_json_safe_handles_paths_numpy_and_non_finite_values(tmp_path) -> None:
    import math
    import numpy as np

    from modimg.utils import json_dumps_safe, json_safe

    payload = {
        "path": tmp_path / "x.png",
        "np_int": np.int64(7),
        "np_float": np.float32(0.25),
        "np_array": np.array([1, 2, 3]),
        "nan": float("nan"),
        "bytes": b"secret-bytes",
    }

    normalized = json_safe(payload)
    assert normalized["path"].endswith("x.png")
    assert normalized["np_int"] == 7
    assert normalized["np_array"] == [1, 2, 3]
    assert normalized["nan"] is None
    assert normalized["bytes"] == "<bytes:12>"
    dumped = json_dumps_safe(payload, allow_nan=False)
    assert "NaN" not in dumped
    assert math.isclose(normalized["np_float"], 0.25, rel_tol=1e-6)


def test_parse_label_float_map_ignores_invalid_values() -> None:
    from modimg.utils import parse_label_float_map

    parsed = parse_label_float_map("isis:0.75, broken, swastika:1.4, bad:nan, antifa:-1")

    assert parsed == {"isis": 0.75, "swastika": 1.0, "antifa": 0.0}
