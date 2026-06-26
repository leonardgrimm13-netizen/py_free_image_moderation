from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from modimg.engines.phash_allow import PHashAllowlistEngine
from modimg.engines.phash_block import PHashBlocklistEngine
from modimg.enums import EngineStatus, VerdictLabel
from modimg.phash import append_phash_to_allowlist, frame_phash_hex_int, load_phash_exact_map, load_phash_list, resolve_list_path
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


def test_empty_phash_blocklist_skips_without_hashing(monkeypatch, tmp_path) -> None:
    frame = _frame()
    blocklist = tmp_path / "block.txt"
    blocklist.write_text("# empty\n", encoding="utf-8")

    def fail_hash(_frame: Frame) -> tuple[str, int]:
        raise AssertionError("empty blocklist should skip before hashing frames")

    monkeypatch.setattr("modimg.engines.phash_block.ph.frame_phash_hex_int", fail_hash)

    result = PHashBlocklistEngine(blocklist_path=str(blocklist)).execute("sample.png", [frame])

    assert result.status == EngineStatus.SKIPPED
    assert result.error == "blocklist empty"


def test_empty_phash_allowlist_skips_without_hashing(monkeypatch, tmp_path) -> None:
    frame = _frame()
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("# empty\n", encoding="utf-8")

    def fail_hash(_frame: Frame) -> tuple[str, int]:
        raise AssertionError("empty allowlist should skip before hashing frames")

    monkeypatch.setattr("modimg.engines.phash_allow.ph.frame_phash_hex_int", fail_hash)

    result = PHashAllowlistEngine(allowlist_path=str(allowlist)).execute("sample.png", [frame])

    assert result.status == EngineStatus.SKIPPED
    assert result.error == "allowlist empty"


def test_resolve_list_path_expands_user(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = resolve_list_path("~/allow.txt")

    assert resolved == str(tmp_path / "allow.txt")


def test_phash_caches_can_be_used_from_multiple_threads(tmp_path) -> None:
    frame = _frame()
    hx, _ = frame_phash_hex_int(frame)
    phash_list = tmp_path / "list.txt"
    phash_list.write_text(f"{hx},known\n", encoding="utf-8")

    def work(i: int) -> tuple[str, int, int]:
        local_frame = Frame(idx=i, pil=Image.new("RGB", (16, 16), color=(1, 2, 3)))
        local_hx, _ = frame_phash_hex_int(local_frame)
        entries = load_phash_list(str(phash_list), default_label="known")
        exact = load_phash_exact_map(str(phash_list), default_label="known")
        return local_hx, len(entries), len(exact)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(work, range(32)))

    assert all(local_hx == hx for local_hx, _, _ in results)
    assert all(entry_count == 1 for _, entry_count, _ in results)
    assert all(exact_count == 1 for _, _, exact_count in results)


def test_parallel_phash_append_does_not_duplicate_entries(tmp_path) -> None:
    allowlist = tmp_path / "allow.txt"
    hx = "0123456789abcdef"

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: append_phash_to_allowlist(hx, str(allowlist), "known"), range(32)))

    lines = [line for line in allowlist.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines == [f"{hx},known"]
    assert results.count(True) == 1
