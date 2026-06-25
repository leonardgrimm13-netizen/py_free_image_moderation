from __future__ import annotations

from PIL import Image

from modimg.engines.phash_allow import PHashAllowlistEngine
from modimg.engines.phash_block import PHashBlocklistEngine
from modimg.enums import EngineStatus, VerdictLabel
from modimg.phash import frame_phash_hex_int, resolve_list_path
from modimg.types import Frame
from modimg.verdict import compute_verdict


def _frame() -> Frame:
    return Frame(idx=0, pil=Image.new("RGB", (16, 16), color=(1, 2, 3)))


def test_phash_allowlist_match_returns_ok_verdict(tmp_path) -> None:
    frame = _frame()
    hx, _ = frame_phash_hex_int(frame)
    allowlist = tmp_path / "allow list.txt"
    allowlist.write_text(f"{hx},known-ok\n", encoding="utf-8")

    result = PHashAllowlistEngine(allowlist_path=str(allowlist)).execute("sample.png", [frame])
    verdict = compute_verdict([result])

    assert result.status == EngineStatus.OK
    assert result.scores["phash_allow_match"] == 1.0
    assert result.details["match_label"] == "known-ok"
    assert verdict.label == VerdictLabel.OK


def test_phash_blocklist_match_returns_block_verdict(tmp_path) -> None:
    frame = _frame()
    hx, _ = frame_phash_hex_int(frame)
    blocklist = tmp_path / "block list.txt"
    blocklist.write_text(f"{hx},known-block\n", encoding="utf-8")

    result = PHashBlocklistEngine(blocklist_path=str(blocklist), max_distance=0).execute("sample.png", [frame])
    verdict = compute_verdict([result])

    assert result.status == EngineStatus.OK
    assert result.scores["phash_block_match"] == 1.0
    assert result.details["match_label"] == "known-block"
    assert verdict.label == VerdictLabel.BLOCK


def test_resolve_list_path_expands_user(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = resolve_list_path("~/allow.txt")

    assert resolved == str(tmp_path / "allow.txt")
